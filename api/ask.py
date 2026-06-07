import os
import json
import requests

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

def handler(request):
    if request.get("method") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        }
    
    body = json.loads(request.get("body", "{}"))
    question = body.get("question", "")
    
    if not question:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Missing question"})
        }
    
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    nim_response = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers=headers,
        json={
            "model": "meta/llama-3.1-8b-instruct",
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 300
        },
        timeout=15
    )
    
    data = nim_response.json()
    answer = data["choices"][0]["message"]["content"]
    
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({"answer": answer})
    }