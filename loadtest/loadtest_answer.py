import asyncio, time, statistics
import httpx
import argparse
parser = argparse.ArgumentParser(description="Load test script")
parser.add_argument("-c", "--concurrency", type=int, default=4)
parser.add_argument("-t", "--total", type=int, default=40)

URL = "http://localhost:8000/answer"
PAYLOAD = {"question": "Explain AI infrastructure in one sentence.", "max_tokens": 64}

async def one(client):
    t0 = time.time()
    r = await client.post(URL, json = PAYLOAD, timeout = 120)
    r.raise_for_status()
    return time.time() - t0

async def run(concurrency = 4, total = 40):
    times = []
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrency)

        async def task():
            async with sem:
                dt = await one(client)
                times.append(dt)
        
        await asyncio.gather(*[task() for _ in range(total)])

    times.sort()
    def pct(p): return times[int(p*(len(times) - 1))]
    print(f"n = {len(times)} conc = {concurrency}")
    print(f"p50={pct(0.5):.3f} p95={pct(0.95):.3f} p99={pct(0.99):.3f}")

if __name__ == "__main__":
    args = parser.parse_args()
    asyncio.run(run(concurrency=args.concurrency, total=args.total))