# new file: server/proxy.py
import asyncio
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response

VLLM_BASE = "http://127.0.0.1:8000"
MAX_INFLIGHT = 8
UPSTREAM_TIMEOUT_SEC = 20.0

REQUESTS_TOTAL = Counter(
    "proxy_requests_total",
    "Total requests seen by proxy",
    ["route", "status"]
)

REQUEST_LATENCY = Histogram(
    "proxy_request_latency_seconds",
    "End-to-end request latency",
    ["route"]
)

INFLIGHT = Gauge(
    "proxy_inflight_requests",
    "Current inflight requests"
)

REJECTED_TOTAL = Counter(
    "proxy_rejected_total",
    "Requests rejected due to concurrency limit",
    ["route"]
)

semaphore = asyncio.Semaphore(MAX_INFLIGHT)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_SEC)
    yield
    await app.state.client.aclose()

app = FastAPI(lifespan=lifespan)

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    route = "/v1/chat/completions"
    body = await request.json()

    # immediate reject if full
    if semaphore.locked() and semaphore._value == 0:
        REJECTED_TOTAL.labels(route=route).inc()
        REQUESTS_TOTAL.labels(route=route, status="rejected").inc()
        return JSONResponse(
            status_code=429,
            content={"error": {"message": "server overloaded, try again later"}}
        )

    start = time.perf_counter()

    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=0.01)
    except asyncio.TimeoutError:
        REJECTED_TOTAL.labels(route=route).inc()
        REQUESTS_TOTAL.labels(route=route, status="rejected").inc()
        return JSONResponse(
            status_code=429,
            content={"error": {"message": "server overloaded, try again later"}}
        )

    INFLIGHT.inc()
    try:
        resp = await app.state.client.post(f"{VLLM_BASE}{route}", json=body)
        duration = time.perf_counter() - start
        REQUEST_LATENCY.labels(route=route).observe(duration)
        REQUESTS_TOTAL.labels(route=route, status=str(resp.status_code)).inc()
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.TimeoutException:
        duration = time.perf_counter() - start
        REQUEST_LATENCY.labels(route=route).observe(duration)
        REQUESTS_TOTAL.labels(route=route, status="timeout").inc()
        raise HTTPException(status_code=504, detail="upstream timeout")
    finally:
        INFLIGHT.dec()
        semaphore.release()