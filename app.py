import os
import json
import numpy as np
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not set")

EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL  = "meta/llama-3.1-8b-instruct"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# Load chunks from C++ output
CHUNKS_FILE = "data/chunks.json"
try:
    with open(CHUNKS_FILE, "r") as f:
        CHUNKS = json.load(f)["chunks"]
    print(f"Loaded {len(CHUNKS)} chunks from {CHUNKS_FILE}")
except FileNotFoundError:
    CHUNKS = ["Fallback: no document found."]
    print("chunks.json not found, using fallback.")

def get_embedding(text, input_type="passage"):
    resp = client.embeddings.create(
        input=[text],
        model=EMBED_MODEL,
        encoding_format="float",
        extra_body={"input_type": input_type}
    )
    return np.array(resp.data[0].embedding)

print("Embedding chunks...")
chunk_embeddings = np.array([get_embedding(ch, "passage") for ch in CHUNKS])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query):
    q_emb = get_embedding(query, "query")
    sims = [cosine_similarity(q_emb, c_emb) for c_emb in chunk_embeddings]
    best_idx = int(np.argmax(sims))
    return CHUNKS[best_idx], float(sims[best_idx])

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing question"}), 400
    question = data["question"]
    context, score = retrieve(question)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )
    answer = completion.choices[0].message.content
    return jsonify({"answer": answer, "context": context, "similarity": score})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
