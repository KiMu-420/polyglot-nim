import os
import json
import numpy as np
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
import hashlib

load_dotenv()
app = Flask(__name__)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not set in .env file")

EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL = "meta/llama-3.1-8b-instruct"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# Store for loaded manifesto
MANIFESTO_CHUNKS = []
MANIFESTO_EMBEDDINGS = None
MANIFESTO_HASH = None

def get_embedding(text, input_type="passage"):
    resp = client.embeddings.create(
        input=[text],
        model=EMBED_MODEL,
        encoding_format="float",
        extra_body={"input_type": input_type}
    )
    return np.array(resp.data[0].embedding)

def load_manifesto(filepath="manifesto.txt"):
    global MANIFESTO_CHUNKS, MANIFESTO_EMBEDDINGS, MANIFESTO_HASH
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Check if manifesto changed
        new_hash = hashlib.md5(text.encode()).hexdigest()
        if new_hash == MANIFESTO_HASH and MANIFESTO_EMBEDDINGS is not None:
            print("Manifesto unchanged, using cached embeddings.")
            return
        
        MANIFESTO_HASH = new_hash
        
        # Chunk the manifesto
        words = text.split()
        MANIFESTO_CHUNKS = []
        chunk_size = 300
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            if chunk.strip():
                MANIFESTO_CHUNKS.append(chunk)
        
        print(f"Manifesto split into {len(MANIFESTO_CHUNKS)} chunks.")
        print("Embedding manifesto (this takes ~30 seconds)...")
        MANIFESTO_EMBEDDINGS = np.array([get_embedding(ch, "passage") for ch in MANIFESTO_CHUNKS])
        print("Manifesto loaded and embedded.")
        
    except FileNotFoundError:
        print("No manifesto.txt found. Using general AI mode.")
        MANIFESTO_CHUNKS = []
        MANIFESTO_EMBEDDINGS = None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_manifesto(query):
    if MANIFESTO_EMBEDDINGS is None or len(MANIFESTO_CHUNKS) == 0:
        return None, 0
    
    q_emb = get_embedding(query, "query")
    sims = [cosine_similarity(q_emb, c_emb) for c_emb in MANIFESTO_EMBEDDINGS]
    best_idx = int(np.argmax(sims))
    return MANIFESTO_CHUNKS[best_idx], float(sims[best_idx])

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing question"}), 400
    
    question = data["question"]
    context, score = search_manifesto(question)
    
    if context:
        prompt = f"""You are an AI that answers questions based STRICTLY on the following manifesto. 
Do not bring in outside knowledge. If the answer isn't in the manifesto, say so.

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
    
    return jsonify({
        "answer": answer,
        "mode": "manifesto" if context else "general",
        "similarity": score if context else None
    })

@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_json()
    if not data or "manifesto" not in data:
        return jsonify({"error": "Missing manifesto text"}), 400
    
    with open("manifesto.txt", "w", encoding="utf-8") as f:
        f.write(data["manifesto"])
    
    load_manifesto()
    return jsonify({"status": "ok", "chunks": len(MANIFESTO_CHUNKS)})

if __name__ == "__main__":
    load_manifesto()
    app.run(host="0.0.0.0", port=5000)