from fastapi import FastAPI, Response
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

from server.answer import AnswerEngine, AnswerConfig

REQS = Counter("http_requests_total", "Total number of HTTP requests", ["path", "status"])
LAT = Histogram("http_request_latency_seconds", "Latency of HTTP requests in seconds", ["path"])

app = FastAPI()

llm = LLM(
    model="facebook/opt-125m",
    gpu_memory_utilization=0.75,
)

engine = AnswerEngine(AnswerConfig())

class Req(BaseModel):
    question: str
    max_tokens: int | None = None

@app.post("/infer")
def infer(req: Req):
    path = "/infer"
    start = time.time()
    status = 200
    try:
        params = SamplingParams(max_tokens=req.max_tokens or 64)
        outputs = llm.generate([req.question], params)
        latency = time.time() - start
        return {
            "text": outputs[0].outputs[0].text,
            "latency_sec": latency
        }
    except Exception as e:
        status = 500
        return {"error": str(e)}
    finally:
        latency = time.time() - start
        LAT.labels(path=path).observe(latency)
        REQS.labels(path=path, status=status).inc()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/answer")
def answer(req: Req):
    path = "/answer"
    start = time.time()
    status = 200
    try:
        text = engine.answer(req.question, max_new_tokens=req.max_tokens)
        latency = time.time() - start
        return {"answer": text, "latency_sec": latency}
    except Exception as e:
        status = 500
        return {"error": str(e)}
    finally:
        latency = time.time() - start
        LAT.labels(path=path).observe(latency)
        REQS.labels(path=path, status=status).inc()


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)