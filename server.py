from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from dotenv import load_dotenv
import os, uuid, base64, logging
from openai import OpenAI

load_dotenv()

MONGO_URL = os.environ['MONGO_URL']
DB_NAME = os.environ.get('DB_NAME', 'vocalacademy')
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

client_db = AsyncIOMotorClient(MONGO_URL)
db = client_db[DB_NAME]
openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LANG_GREETING = {
    "Telugu": "Namaskaram", "Hindi": "Namaste", "Tamil": "Vanakkam",
    "Kannada": "Namaskara", "Malayalam": "Namaskaram", "Marathi": "Namaskar",
    "Bengali": "Nomoshkar", "Gujarati": "Namaste", "Punjabi": "Sat Sri Akal",
    "English": "Hello", "Spanish": "Hola", "French": "Bonjour",
    "Arabic": "Marhaba", "Japanese": "Konnichiwa", "German": "Hallo",
    "Mandarin": "Ni hao", "Korean": "Annyeong"
}

class StartSessionRequest(BaseModel):
    mother_tongue: str
    target_language: str
    level: str
    device_id: str

class ChatRequest(BaseModel):
    session_id: str
    audio_base64: str
    device_id: str

class EndSessionRequest(BaseModel):
    session_id: str
    device_id: str

@api_router.get("/health")
async def health():
    return {"status": "ok"}

@api_router.post("/session/start")
async def start_session(req: StartSessionRequest):
    try:
        session_id = str(uuid.uuid4())
        greeting = LANG_GREETING.get(req.mother_tongue, "Hello")
        
        system_prompt = f"""You are Voca, a friendly multilingual language tutor. 
The student's mother tongue is {req.mother_tongue} and they are learning {req.target_language} at {req.level} level.
Always teach in {req.target_language} but when introducing new words or when the student seems confused, explain the meaning in {req.mother_tongue}.
Keep all responses under 2 sentences strictly. This is a real-time voice app.
Be warm, energetic and encouraging. Remember what words you have taught and don't repeat them."""

        intro = f"{greeting}! I am Voca, your {req.target_language} tutor. Let's start learning together! Tell me your name."
        
        tts_response = openai_client.audio.speech.create(
            model="tts-1", voice="nova", input=intro, speed=1.15
        )
        audio_b64 = base64.b64encode(tts_response.content).decode()
        
        await db.sessions.insert_one({
            "session_id": session_id,
            "device_id": req.device_id,
            "mother_tongue": req.mother_tongue,
            "target_language": req.target_language,
            "level": req.level,
            "system_prompt": system_prompt,
            "messages": [{"role": "assistant", "content": intro}],
            "words_learned": [],
            "created_at": datetime.now(timezone.utc).isoformat()
        })
        
        return {"session_id": session_id, "audio_base64": audio_b64, "text": intro}
    except Exception as e:
        logger.error(f"Start session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/session/chat")
async def chat(req: ChatRequest):
    try:
        session = await db.sessions.find_one({"session_id": req.session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        audio_bytes = base64.b64decode(req.audio_base64)
        
        import io
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav"
        
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )
        user_text = transcript.text
        
        messages = session.get("messages", [])
        messages.append({"role": "user", "content": user_text})
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": session["system_prompt"]}] + messages[-10:],
            max_tokens=150
        )
        assistant_text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_text})
        
        tts_response = openai_client.audio.speech.create(
            model="tts-1", voice="nova", input=assistant_text, speed=1.15
        )
        audio_b64 = base64.b64encode(tts_response.content).decode()
        
        await db.sessions.update_one(
            {"session_id": req.session_id},
            {"$set": {"messages": messages}}
        )
        
        return {"audio_base64": audio_b64, "text": assistant_text, "user_text": user_text}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/session/end")
async def end_session(req: EndSessionRequest):
    try:
        session = await db.sessions.find_one({"session_id": req.session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        summary = f"Great session! You practiced {session['target_language']} today. Keep it up!"
        
        tts_response = openai_client.audio.speech.create(
            model="tts-1", voice="nova", input=summary, speed=1.15
        )
        audio_b64 = base64.b64encode(tts_response.content).decode()
        
        await db.sessions.update_one(
            {"session_id": req.session_id},
            {"$set": {"ended_at": datetime.now(timezone.utc).isoformat()}}
        )
        
        return {"summary_text": summary, "audio_base64": audio_b64}
    except Exception as e:
        logger.error(f"End session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown():
    client_db.close()
