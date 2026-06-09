import os
import json
import hashlib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not set in .env file")

EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL = "meta/llama-3.1-8b-instruct"
MANIFESTO_FILE = "manifesto.txt"

# Redis configuration (optional – set REDIS_URL in Railway if you add Redis)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Globals (these will be initialised at startup)
MANIFESTO_CHUNKS = []
MANIFESTO_EMBEDDINGS = None
MANIFESTO_HASH = None
redis_client = None

# ---------- Embedding helpers ----------
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

def get_embedding(text: str, input_type: str = "passage") -> np.ndarray:
    resp = client.embeddings.create(
        input=[text],
        model=EMBED_MODEL,
        encoding_format="float",
        extra_body={"input_type": input_type}
    )
    return np.array(resp.data[0].embedding)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ---------- Manifesto loading + Redis caching ----------
def chunk_text(text: str, chunk_size: int = 300) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def load_manifesto():
    global MANIFESTO_CHUNKS, MANIFESTO_EMBEDDINGS, MANIFESTO_HASH

    try:
        with open(MANIFESTO_FILE, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print("No manifesto.txt found. Running in general mode.")
        MANIFESTO_CHUNKS = []
        MANIFESTO_EMBEDDINGS = None
        return

    # Hash to detect changes
    new_hash = hashlib.md5(text.encode()).hexdigest()
    if new_hash == MANIFESTO_HASH and MANIFESTO_EMBEDDINGS is not None:
        print("Manifesto unchanged, using cached embeddings.")
        return

    MANIFESTO_HASH = new_hash
    MANIFESTO_CHUNKS = chunk_text(text)
    print(f"Manifesto split into {len(MANIFESTO_CHUNKS)} chunks.")

    # Try Redis cache first
    if redis_client:
        cached = redis_client.get(f"manifesto:{MANIFESTO_HASH}")
        if cached:
            MANIFESTO_EMBEDDINGS = np.frombuffer(cached).reshape(-1, 1024)  # adjust dim if needed
            print("Loaded embeddings from Redis.")
            return

    print("Computing manifesto embeddings (this may take ~30s)...")
    MANIFESTO_EMBEDDINGS = np.array([get_embedding(ch, "passage") for ch in MANIFESTO_CHUNKS])
    print("Embeddings computed.")

    # Store in Redis if available
    if redis_client:
        redis_client.set(f"manifesto:{MANIFESTO_HASH}", MANIFESTO_EMBEDDINGS.tobytes())
        print("Stored embeddings in Redis.")

# ---------- Startup event ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global redis_client
    try:
        redis_client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2)
        redis_client.ping()
        print("Connected to Redis.")
    except:
        print("Redis not available – embeddings will be recomputed on restart.")
        redis_client = None

    load_manifesto()
    yield
    # Shutdown (nothing needed)

# ---------- FastAPI app ----------
app = FastAPI(title="Manifesto AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    mode: str
    similarity: float | None = None

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing question")

    # Search manifesto
    context = None
    score = None
    if MANIFESTO_CHUNKS and MANIFESTO_EMBEDDINGS is not None:
        q_emb = get_embedding(question, "query")
        sims = [cosine_similarity(q_emb, c_emb) for c_emb in MANIFESTO_EMBEDDINGS]
        best_idx = int(np.argmax(sims))
        context = MANIFESTO_CHUNKS[best_idx]
        score = float(sims[best_idx])

    # Build prompt
    if context:
        prompt = f"""You are an AI that answers questions based STRICTLY on the following manifesto.
Do not bring in outside knowledge. If the answer isn't in the manifesto, say so.

MANIFESTO EXCERPT:
{context}

QUESTION: {question}
ANSWER:"""
    else:
        prompt = f"Question: {question}\nAnswer:"

    # Call NIM
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )
    answer = completion.choices[0].message.content

    return AskResponse(
        answer=answer,
        mode="manifesto" if context else "general",
        similarity=score
    )

# Health check
@app.get("/health")
async def health():
    return {"status": "ok"}