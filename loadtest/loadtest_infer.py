import asyncio, time, statistics, os, csv
import httpx
import argparse
parser = argparse.ArgumentParser(description="Load test script")
parser.add_argument("-c", "--concurrency", type=int, default=4)
parser.add_argument("-t", "--total", type=int, default=40)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--max-tokens", type=int, default=16)
parser.add_argument("--csv", type=str, default="/workspace/loadtest_results.csv")
parser.add_argument("--sweep", action="store_true", help="Run a sweep over conc/max_tokens")



URL = "http://localhost:8000/infer"
# PAYLOAD = {"prompt": "Hello AI infrastructure", "max_tokens": 64}
PROMPT = "What does AI infrastructure mean?"
def append_csv(path: str, row: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

async def one(client, max_tokens:int = 64):
    t0 = time.time()
    r = await client.post(URL, 
                          json = {"prompt": PROMPT, "max_tokens": max_tokens}, 
                          timeout = 120)
    r.raise_for_status()
    return time.time() - t0

async def run(concurrency=4, total=40, warmup=10, max_tokens=16):
    times = []
    
    def pct(p):
        if not times:
            return float("nan")
        k = int(round(p * (len(times) - 1)))
        k = max(0, min(k, len(times) - 1))
        return times[k]
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrency)

        async def task(record: bool):
            async with sem:
                dt = await one(client, max_tokens=max_tokens)
                if record:
                    times.append(dt)
        if warmup > 0:
            await asyncio.gather(*[task(record=False) for _ in range(warmup)])
        
        await asyncio.gather(*[task(record=True) for _ in range(total)])

    times.sort()
    row = {
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "n": len(times),
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "mean": statistics.mean(times) if times else float("nan"),
    }
    print(f"n={row['n']} conc={concurrency} max_tokens={max_tokens}")
    print(f"p50={row['p50']:.3f} p95={row['p95']:.3f} p99={row['p99']:.3f} mean={row['mean']:.3f}")
    return row

if __name__ == "__main__":
    args = parser.parse_args()

    async def main():
        if not args.sweep:
            row = await run(
                concurrency=args.concurrency,
                total=args.total,
                warmup=args.warmup,
                max_tokens=args.max_tokens,
            )
            append_csv(args.csv, row)
            return

        # Minimum grid sweep
        conc_levels = [1, 2, 4, 8]
        tok_levels = [16, 64]
        for mt in tok_levels:
            for c in conc_levels:
                row = await run(concurrency=c, total=args.total, warmup=args.warmup, max_tokens=mt)
                append_csv(args.csv, row)

    asyncio.run(main())
