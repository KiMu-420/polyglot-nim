import os
import json
import hashlib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis
import jwt
import requests
from openai import OpenAI
from dotenv import load_dotenv
from nemoguardrails import RailsConfig, LLMRails

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")  # optional for now

EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL = "meta/llama-3.1-8b-instruct"
MANIFESTO_FILE = "manifesto.txt"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Clerk JWKS endpoint (used to verify tokens)
CLERK_JWKS_URL = "https://api.clerk.com/v1/jwks"

# Globals
MANIFESTO_CHUNKS = []
MANIFESTO_EMBEDDINGS = None
MANIFESTO_HASH = None
redis_client = None
guardrails = None

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# ---------- Auth helper ----------
def verify_clerk_token(authorization: str = None) -> str | None:
    """If Clerk is configured, verify the JWT. Returns user ID or None."""
    if not CLERK_SECRET_KEY or not authorization:
        return None  # auth not required yet

    token = authorization.replace("Bearer ", "")
    try:
        # Fetch Clerk's public keys
        jwks = requests.get(CLERK_JWKS_URL).json()
        public_keys = {}
        for key in jwks["keys"]:
            public_keys[key["kid"]] = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key))

        # Decode and verify
        kid = jwt.get_unverified_header(token)["kid"]
        payload = jwt.decode(
            token,
            key=public_keys[kid],
            algorithms=["RS256"],
            options={"verify_exp": True}
        )
        return payload.get("sub")  # Clerk user ID
    except Exception:
        return None  # invalid token

# ---------- Manifesto helpers ----------
def chunk_text(text: str, chunk_size: int = 300) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

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

def load_manifesto():
    global MANIFESTO_CHUNKS, MANIFESTO_EMBEDDINGS, MANIFESTO_HASH
    try:
        with open(MANIFESTO_FILE, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        MANIFESTO_CHUNKS = []
        MANIFESTO_EMBEDDINGS = None
        return

    new_hash = hashlib.md5(text.encode()).hexdigest()
    if new_hash == MANIFESTO_HASH and MANIFESTO_EMBEDDINGS is not None:
        return

    MANIFESTO_HASH = new_hash
    MANIFESTO_CHUNKS = chunk_text(text)
    print(f"Manifesto split into {len(MANIFESTO_CHUNKS)} chunks.")

    if redis_client:
        cached = redis_client.get(f"manifesto:{MANIFESTO_HASH}")
        if cached:
            MANIFESTO_EMBEDDINGS = np.frombuffer(cached).reshape(-1, 1024)
            return

    print("Computing manifesto embeddings...")
    MANIFESTO_EMBEDDINGS = np.array([get_embedding(ch, "passage") for ch in MANIFESTO_CHUNKS])

    if redis_client:
        redis_client.set(f"manifesto:{MANIFESTO_HASH}", MANIFESTO_EMBEDDINGS.tobytes())

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

    load_manifesto()
    yield

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
    user_id: str | None = None

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, authorization: str = Header(None)):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing question")

    # Optional auth (doesn't block, just identifies user)
    user_id = verify_clerk_token(authorization) if CLERK_SECRET_KEY else None

    # Guardrails
    if guardrails:
        try:
            safety_result = await guardrails.generate_async(prompt=question)
            if "refuse" in safety_result.lower():
                return AskResponse(answer="I can only answer questions about the manifesto.", mode="safety_blocked", user_id=user_id)
        except:
            pass

    # Manifesto search
    context, score = None, None
    if MANIFESTO_CHUNKS and MANIFESTO_EMBEDDINGS is not None:
        q_emb = get_embedding(question, "query")
        sims = [cosine_similarity(q_emb, c_emb) for c_emb in MANIFESTO_EMBEDDINGS]
        best_idx = int(np.argmax(sims))
        context = MANIFESTO_CHUNKS[best_idx]
        score = float(sims[best_idx])

    # Prompt
    if context:
        prompt = f"""You are an AI that answers based STRICTLY on the following manifesto.

MANIFESTO EXCERPT:
{context}

QUESTION: {question}
ANSWER:"""
    else:
        prompt = f"Question: {question}\nAnswer:"

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )
    answer = completion.choices[0].message.content

    return AskResponse(answer=answer, mode="manifesto" if context else "general", similarity=score, user_id=user_id)

@app.get("/health")
async def health():
    return {"status": "ok"}