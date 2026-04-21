from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
from dotenv import load_dotenv
import os, uuid, logging
from groq import Groq

load_dotenv()

MONGO_URL = os.environ['MONGO_URL']
DB_NAME = os.environ.get('DB_NAME', 'vocalacademy')
GROQ_API_KEY = os.environ['GROQ_API_KEY']

client_db = AsyncIOMotorClient(MONGO_URL)
db = client_db[DB_NAME]
groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StartSessionRequest(BaseModel):
    mother_tongue: str
    target_language: str
    level: str
    device_id: str

class ChatRequest(BaseModel):
    session_id: str
    user_text: str
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

        system_prompt = f"""You are Voca, a friendly multilingual language tutor.
The student's mother tongue is {req.mother_tongue} and they are learning {req.target_language} at {req.level} level.
Always teach in {req.target_language} but when introducing new words or when the student seems confused, explain the meaning in {req.mother_tongue}.
Keep all responses under 2 sentences strictly. This is a real-time voice app.
Be warm, energetic and encouraging. Remember what words you have taught and dont repeat them.
After every 5 exchanges give a short encouragement in {req.mother_tongue}."""

        intro = f"Hello! I am Voca, your {req.target_language} tutor. Let us start learning together! Tell me your name."

        await db.sessions.insert_one({
            "session_id": session_id,
            "device_id": req.device_id,
            "mother_tongue": req.mother_tongue,
            "target_language": req.target_language,
            "level": req.level,
            "system_prompt": system_prompt,
            "messages": [{"role": "assistant", "content": intro}],
            "words_learned": [],
            "exchange_count": 0,
            "created_at": datetime.now(timezone.utc).isoformat()
        })

        return {"session_id": session_id, "text": intro}
    except Exception as e:
        logger.error(f"Start session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/session/chat")
async def chat(req: ChatRequest):
    try:
        session = await db.sessions.find_one({"session_id": req.session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        messages = session.get("messages", [])
        messages.append({"role": "user", "content": req.user_text})

        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "system", "content": session["system_prompt"]}] + messages[-10:],
            max_tokens=150
        )
        assistant_text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_text})

        exchange_count = session.get("exchange_count", 0) + 1

        await db.sessions.update_one(
            {"session_id": req.session_id},
            {"$set": {"messages": messages, "exchange_count": exchange_count}}
        )

        return {"text": assistant_text, "user_text": req.user_text}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/session/end")
async def end_session(req: EndSessionRequest):
    try:
        session = await db.sessions.find_one({"session_id": req.session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        messages = session.get("messages", [])
        exchange_count = session.get("exchange_count", 0)

        summary = f"Great session! You had {exchange_count} exchanges practicing {session['target_language']}. Keep it up!"

        await db.sessions.update_one(
            {"session_id": req.session_id},
            {"$set": {"ended_at": datetime.now(timezone.utc).isoformat()}}
        )

        return {"summary_text": summary}
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
