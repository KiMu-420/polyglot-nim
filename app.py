import os
import json
import hashlib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import redis
import jwt
import requests
from openai import OpenAI
from dotenv import load_dotenv
from nemoguardrails import RailsConfig, LLMRails
from models import SessionLocal, User, Manifesto

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgres://", "postgresql://", 1)

EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL = "meta/llama-3.1-8b-instruct"
MANIFESTO_FILE = "manifesto.txt"  # fallback global manifesto
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CLERK_JWKS_URL = "https://api.clerk.com/v1/jwks"

# Globals
redis_client = None
guardrails = None
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)

# ---------- Auth ----------
def verify_clerk_token(authorization: str = None) -> str | None:
    if not CLERK_SECRET_KEY or not authorization:
        return None
    token = authorization.replace("Bearer ", "")
    try:
        jwks = requests.get(CLERK_JWKS_URL).json()
        public_keys = {key["kid"]: jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key)) for key in jwks["keys"]}
        kid = jwt.get_unverified_header(token)["kid"]
        payload = jwt.decode(token, key=public_keys[kid], algorithms=["RS256"], options={"verify_exp": True})
        return payload.get("sub")
    except:
        return None

# ---------- Helper functions ----------
def get_embedding(text: str, input_type: str = "passage") -> np.ndarray:
    resp = client.embeddings.create(
        input=[text], model=EMBED_MODEL, encoding_format="float", extra_body={"input_type": input_type}
    )
    return np.array(resp.data[0].embedding)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def chunk_text(text: str, chunk_size: int = 300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]).strip() for i in range(0, len(words), chunk_size) if " ".join(words[i:i+chunk_size]).strip()]

def load_manifesto_from_db(user_id: str) -> tuple:
    """Load the user's manifesto from database, return (chunks, embeddings) or (None, None)."""
    if not DATABASE_URL:
        return None, None
    db: Session = SessionLocal()
    try:
        user = db.query(User).filter(User.clerk_id == user_id).first()
        if not user or not user.manifestos:
            return None, None
        manifesto = user.manifestos[0]  # use the first manifesto
        chunks = chunk_text(manifesto.content)
        if not chunks:
            return None, None
        embeddings = np.array([get_embedding(ch, "passage") for ch in chunks])
        return chunks, embeddings
    finally:
        db.close()

def load_global_manifesto() -> tuple:
    """Fallback global manifesto from file."""
    try:
        with open(MANIFESTO_FILE, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)
        embeddings = np.array([get_embedding(ch, "passage") for ch in chunks])
        return chunks, embeddings
    except FileNotFoundError:
        return [], None

# ---------- Startup ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, guardrails
    try:
        redis_client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2)
        redis_client.ping()
        print("Connected to Redis.")
    except:
        redis_client = None

    try:
        guardrails_config = RailsConfig.from_path("./config")
        guardrails = LLMRails(guardrails_config)
        print("Guardrails loaded.")
    except:
        guardrails = None

    if DATABASE_URL:
        from models import Base, engine
        Base.metadata.create_all(bind=engine)
        print("Database tables created.")
    yield

app = FastAPI(title="Manifesto AI", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    mode: str
    similarity: float | None = None
    user_id: str | None = None

class UploadRequest(BaseModel):
    content: str
    title: str = "My Manifesto"

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, authorization: str = Header(None)):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing question")

    user_id = verify_clerk_token(authorization) if CLERK_SECRET_KEY else None

    if guardrails:
        try:
            safety_result = await guardrails.generate_async(prompt=question)
            if "refuse" in safety_result.lower():
                return AskResponse(answer="I can only answer questions about the manifesto.", mode="safety_blocked", user_id=user_id)
        except:
            pass

    # Load manifesto (user-specific first, then global fallback)
    chunks, embeddings = None, None
    if user_id and DATABASE_URL:
        chunks, embeddings = load_manifesto_from_db(user_id)
    if not chunks:
        chunks, embeddings = load_global_manifesto()

    context, score = None, None
    if chunks and embeddings is not None:
        q_emb = get_embedding(question, "query")
        sims = [cosine_similarity(q_emb, emb) for emb in embeddings]
        best_idx = int(np.argmax(sims))
        context = chunks[best_idx]
        score = float(sims[best_idx])

    if context:
        prompt = f"""You are an AI that answers based STRICTLY on the following manifesto.

MANIFESTO EXCERPT:
{context}

QUESTION: {question}
ANSWER:"""
    else:
        prompt = f"Question: {question}\nAnswer:"

    completion = client.chat.completions.create(
        model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=300
    )
    answer = completion.choices[0].message.content

    return AskResponse(answer=answer, mode="manifesto" if context else "general", similarity=score, user_id=user_id)

@app.post("/upload")
async def upload_manifesto(req: UploadRequest, authorization: str = Header(None)):
    user_id = verify_clerk_token(authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    db = next(get_db())
    try:
        # Upsert user
        user = db.query(User).filter(User.clerk_id == user_id).first()
        if not user:
            user = User(clerk_id=user_id)
            db.add(user)
            db.commit()
            db.refresh(user)

        # Create/replace manifesto
        manifesto = db.query(Manifesto).filter(Manifesto.owner_id == user_id).first()
        if manifesto:
            manifesto.content = req.content
            manifesto.title = req.title
        else:
            manifesto = Manifesto(owner_id=user_id, content=req.content, title=req.title)
            db.add(manifesto)
        db.commit()
        return {"status": "ok", "message": "Manifesto uploaded successfully"}
    finally:
        db.close()

@app.get("/health")
async def health():
    return {"status": "ok"}