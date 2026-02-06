from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import time

from server.answer import AnswerEngine, AnswerConfig

app = FastAPI()

llm = LLM(
    model="facebook/opt-125m",  # small, fast, reliable on Colab
    gpu_memory_utilization=0.3,
)

engine = AnswerEngine(AnswerConfig())

class Req(BaseModel):
    question: str
    max_tokens: int | None = None

@app.post("/infer")
def infer(req: Req):
    start = time.time()
    params = SamplingParams(max_tokens=req.max_tokens or 64)
    outputs = llm.generate([req.question], params)
    latency = time.time() - start

    return {
        "text": outputs[0].outputs[0].text,
        "latency_sec": latency
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/answer")
def answer(req: Req):
    start = time.time()
    text = engine.answer(req.question, max_new_tokens=req.max_tokens)
    return {"answer": text, "latency_sec": time.time() - start}
