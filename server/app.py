from fastapi import FastAPI, Response
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time, gc, torch, asyncio

from server.answer import AnswerEngine, AnswerConfig

REQS = Counter("http_requests_total", "Total number of HTTP requests", ["path", "status"])
LAT = Histogram("http_request_latency_seconds", "Latency of HTTP requests in seconds", ["path"])

LLM_LOCK = asyncio.Lock()
llm: LLM | None = None
engine: AnswerEngine | None = None

app = FastAPI()

@app.on_event("startup")
def _startup():
    global llm, engine
    llm = LLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        gpu_memory_utilization=0.6, # Changed per suggestion in PagedAttention
        enable_sleep_mode=True,   # important: allows llm.sleep() to actually free VRAM :contentReference[oaicite:0]{index=0}
    )
    engine = AnswerEngine(AnswerConfig())

@app.on_event("shutdown")
def _shutdown():
    global llm, engine

    # 1) Stop/free vLLM
    if llm is not None:
        try:
            # releases most GPU memory (weights + KV) when sleep mode is enabled :contentReference[oaicite:1]{index=1}
            llm.sleep(level=2)
        except Exception:
            pass
        llm = None

    # 2) Stop/free Transformers engine
    if engine is not None:
        try:
            engine.close()
        except Exception:
            pass
        engine = None

    gc.collect()
    torch.cuda.empty_cache()

class InferReq(BaseModel):
    prompt: str
    max_tokens: int | None = None

class AnswerReq(BaseModel):
    question: str
    max_tokens: int | None = None

@app.post("/infer")
async def infer(req: InferReq):
    path = "/infer"
    start = time.time()
    status = 200
    try:
        async with LLM_LOCK:
            params = SamplingParams(max_tokens=req.max_tokens or 64)
            outputs = llm.generate([req.prompt], params)
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
def answer(req: AnswerReq):
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