# Load Test Guide

This folder contains the load generator used to benchmark the vLLM serving path, either directly against the vLLM OpenAI-compatible API server or through the local FastAPI proxy.

The current goal of this script is not to be a full benchmarking framework. It is a lightweight tool for answering a focused set of systems questions:

- How does latency change as concurrency increases?
- Where is the throughput knee?
- What happens when requests are sent through the proxy instead of directly to vLLM?
- How do prompt length and `max_tokens` affect service behavior?
- When overload controls are enabled, do requests fail cleanly?

---

## File

- `loadtest_llm_infer.py`: asyncio-based HTTP load generator for OpenAI-compatible text generation endpoints

---

## What the script measures

For each test run, the script records:

- request count successfully completed
- p50 latency
- p95 latency
- p99 latency
- p99.9 latency
- mean latency
- wall-clock runtime
- requests per second (RPS)
- approximate output tokens per second
- number of failed requests

Results are appended to a CSV file so that multiple runs can be compared later.

---

## Endpoint modes

The script can target either of these:

### Proxy mode
Use the FastAPI proxy in front of vLLM:

```python
URL = "http://localhost:8080/v1/chat/completions"
```

This is the preferred mode for service-level benchmarking, because proxy metrics and overload behavior are visible.

### Direct mode
Hit vLLM directly:

```python
URL = "http://localhost:8000/v1/chat/completions"
```

This is useful as a baseline for comparing proxy overhead.

---

## Request format

The script uses the OpenAI chat completions API format.

The request body should follow this shape:

```python
json={
    "model": MODEL,
    "messages": [
        {"role": "user", "content": PROMPT}
    ],
    "max_tokens": max_tokens,
    "temperature": 0.0,
}
```

Do not use a top-level `prompt` field with `/v1/chat/completions`. That request shape belongs to `/v1/completions` and will produce `400 Bad Request` if sent to the chat endpoint.

---

## Basic usage

### Single run

```bash
python3 ./loadtest_llm_infer.py -c 4 --warmup 10 --max-tokens 16 --total 40
```

### Sweep mode

```bash
python3 ./loadtest_llm_infer.py --sweep --warmup 10 --total 40
```

In sweep mode, the script currently iterates over:

- concurrency levels: `1, 2, 4, 8, 24, 32, 48, 56, 64`
- output lengths: `16, 64, 256`

This produces a small grid of runs that is useful for finding saturation points.

---

## Command-line arguments

| Argument | Meaning | Default |
|---|---|---:|
| `-c`, `--concurrency` | Number of concurrent client tasks | `4` |
| `-t`, `--total` | Number of measured requests in the main run | `40` |
| `--warmup` | Number of warmup requests before recording results | `10` |
| `--max-tokens` | Maximum number of generated output tokens | `16` |
| `--csv` | Output CSV path | `/workspace/ai-infra-demo/report/loadtest_results.csv` |
| `--sweep` | Run multiple concurrency / token combinations | disabled |

---

## Warmup behavior

Warmup requests are sent before the measured run begins. Their latency is not appended to the main `times` array.

This is useful because first-request effects can distort results:

- tokenizer setup
- model graph warmup
- cache population
- lazy memory allocation

A small warmup count is usually enough for this project.

---

## CSV output

Each run appends a single row to the CSV file with fields like:

- `concurrency`
- `max_tokens`
- `n`
- `p50`
- `p95`
- `p99`
- `p999`
- `mean`
- `wall_sec`
- `rps`
- `tok_per_sec`
- `errors`

This file is intended to support later plotting and report writing.

---

## Recommended experiment flow

Use the following order when benchmarking:

### 1. Sanity check
Run a tiny test with low concurrency:

```bash
python3 ./loadtest_llm_infer.py -c 1 --warmup 2 --max-tokens 16 --total 5
```

Confirm:

- all requests succeed
- latency numbers look plausible
- proxy and vLLM metrics move during the run

### 2. Small scale comparison
Run a few fixed points:

```bash
python3 ./loadtest_llm_infer.py -c 1 --warmup 5 --max-tokens 16 --total 20
python3 ./loadtest_llm_infer.py -c 4 --warmup 5 --max-tokens 16 --total 20
python3 ./loadtest_llm_infer.py -c 8 --warmup 5 --max-tokens 16 --total 20
```

This helps identify whether the system is still scaling or already queueing.

### 3. Sweep
Use `--sweep` only after the endpoint is confirmed healthy. Otherwise you end up filling the CSV with garbage runs.

---

## How to interpret results

### Good scaling region
You will usually see:

- RPS increasing with concurrency
- latency increasing slowly
- low error count

### Queueing knee
You will often see:

- RPS flattening
- p95 and p99 rising sharply
- inflight requests staying elevated
- eventual rejects or timeouts if overload control is active

This knee matters more than peak throughput alone.

### Fast failures
If many requests complete extremely quickly and the error count rises, that often indicates request validation errors rather than genuine model latency.

---

## Debugging tips

### 400 Bad Request
Most likely causes:

- sending `prompt` to `/v1/chat/completions`
- wrong model name in request body
- malformed JSON payload

Print the response body when an `HTTPStatusError` occurs.

### No proxy metrics movement
Most likely cause:

- the script is still targeting port `8000` instead of `8080`

### Unrealistically low latency
Most likely cause:

- requests are failing fast and being included in the run

---

## Suggested improvement to the script

The current script counts errors, but debugging becomes easier if HTTP response bodies are printed for non-2xx responses.

The task error handling should include something like:

```python
except httpx.HTTPStatusError as e:
    errors += 1
    print("HTTP error:", e.response.status_code, e.response.text)
except Exception as e:
    errors += 1
    print("Other error:", repr(e))
```

This is especially useful during early experiment setup.

---

## Current limitations

- token throughput is approximated as `successful_requests * max_tokens / wall_sec`
- prompt token count is fixed by the script and not logged explicitly yet
- results are point measurements, not a long-duration soak test
- no per-request metadata is stored beyond aggregated latency statistics

These limitations are acceptable for the current stage of the project.

---

## Next improvements

Good next upgrades for this folder include:

1. logging actual prompt token count and response token count
2. separating successful latency from failed-request latency
3. exporting benchmark plots automatically from the CSV
4. adding request IDs and per-run labels
5. comparing direct-vLLM runs against proxy runs in one report

---

## Role of this folder in the repo

This folder is the measurement harness for the whole project.

The serving layer can be changed, the proxy can evolve, and observability can grow, but none of that matters unless the benchmark harness can reliably show what changed and whether it improved the system.