import requests

resp = requests.post(
    "http://localhost:8000/infer",
    json={"prompt": "Hello AI infrastructure", "max_tokens": 16},
)
print(resp.json())