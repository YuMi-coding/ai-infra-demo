import requests
for q in [
    "What does AI infrastructure mean?",
    "Explain what AI infrastructure means in one sentence.",
    "What is 1+1?"
]:
    r = requests.post(
        "http://127.0.0.1:8000/answer",
        json={"question": q, "max_tokens": 32},
        timeout=120
    )
    print(q, "->", r.json())
