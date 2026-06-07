import os
import json
import time
from http.server import BaseHTTPRequestHandler

# Simple in-memory rate limiter (per IP, 10 requests per minute)
RATE_LIMIT = {}
MAX_REQUESTS = 10
WINDOW = 60  # seconds

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Rate limit check
        client_ip = self.client_address[0]
        now = time.time()
        if client_ip in RATE_LIMIT:
            requests = [t for t in RATE_LIMIT[client_ip] if now - t < WINDOW]
            RATE_LIMIT[client_ip] = requests
            if len(requests) >= MAX_REQUESTS:
                self.send_response(429)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Too many requests. Try again in a minute."}).encode())
                return
        else:
            RATE_LIMIT[client_ip] = []
        RATE_LIMIT[client_ip].append(now)

        # Parse request
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = json.loads(body)

        import requests
        NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }

        nim_response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers,
            json={
                "model": "meta/llama-3.1-8b-instruct",
                "messages": [{"role": "user", "content": data.get("question", "")}],
                "max_tokens": 300
            },
            timeout=15
        )

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        answer = nim_response.json()["choices"][0]["message"]["content"]
        self.wfile.write(json.dumps({"answer": answer}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()SS