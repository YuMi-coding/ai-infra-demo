# AI Infra Demo

A hands-on AI infrastructure project focused on **LLM serving, load testing, and observability**.

This repo is built to show practical infrastructure skills around AI workloads rather than model training research. The current workflow runs a **vLLM OpenAI-compatible API server** behind a small **FastAPI proxy**, drives traffic with an **async load generator**, and scrapes metrics with **Prometheus**. The goal is to measure where the service saturates, how latency behaves under concurrency, and how to make overload behavior explicit and controlled.

Please note this repository is built with extensive AI assistance, except this line itself.

## Current focus

- Serve an instruct model through **vLLM**
- Put a lightweight **proxy layer** in front of the model server
- Add **service-level metrics** for requests, latency, inflight load, and rejection behavior
- Run **repeatable load tests** across concurrency and output-length settings
- Use **Prometheus** to collect metrics during experiments

## Architecture

```text
loadtest client
    -> FastAPI proxy (:8080)
        -> vLLM OpenAI-compatible API server (:8000)

Prometheus (:9090)
    -> scrapes proxy /metrics
    -> scrapes vLLM /metrics
```

### Component roles

- **vLLM API server**
  - Hosts the model
  - Exposes OpenAI-compatible endpoints and internal metrics
- **Proxy**
  - Sits in front of vLLM
  - Adds request-level instrumentation
  - Provides a place to enforce concurrency limits, timeouts, and rejection policy
- **Load test**
  - Generates concurrent requests
  - Measures end-to-end latency and throughput
  - Saves experiment outputs for later comparison
- **Prometheus**
  - Scrapes metrics from both proxy and vLLM
  - Supports debugging and experiment analysis

## Why this repo exists

Many AI infrastructure roles are not asking whether you can fine-tune a model. They are asking whether you can make model-serving systems **fast, measurable, reliable, and predictable under pressure**.

This repo is meant to demonstrate exactly that:

- service-oriented thinking instead of notebook-only experimentation
- performance analysis instead of vague benchmarking
- observability and overload handling instead of raw “it runs” demos
- AI workload context combined with systems and networking instincts

## Repo structure

```text
ai-infra-demo/
├── README.md                  # Project overview and top-level workflow
├── server/                    # Serving components such as the proxy
├── loadtest/                  # Async benchmarking scripts and CSV output
├── observability/             # Prometheus config and metrics notes
├── report/                    # Experiment outputs, plots, and summaries
└── .venv/                     # Local virtual environment (not committed)
```

## Current workflow

### 1) Start vLLM

Example:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --gpu-memory-utilization 0.60 \
  --max-model-len 512 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --disable-log-requests
```

Quick checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics | head
```

### 2) Start the proxy

Example:

```bash
uvicorn server.proxy:app --host 0.0.0.0 --port 8080
```

Quick checks:

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/metrics | head
```

### 3) Start Prometheus

Example:

```bash
prometheus --config.file=observability/prometheus.yml --web.listen-address=0.0.0.0:9090
```

Quick check:

```bash
curl http://127.0.0.1:9090/-/healthy
```

### 4) Run the load test against the proxy

Example:

```bash
python3 ./loadtest/loadtest_llm_infer.py -c 4 --warmup 10 --max-tokens 16 --total 40
```

Important:
- the load test should target the **proxy** on port `8080`
- for `/v1/chat/completions`, requests must use **`messages`**, not `prompt`

## Example request format

For chat completions, the request body should look like this:

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "messages": [
    {"role": "user", "content": "What does AI infrastructure mean?"}
  ],
  "max_tokens": 16,
  "temperature": 0.0
}
```

## Metrics currently exposed by the proxy

The proxy exports Prometheus metrics intended to capture service-level behavior:

- `proxy_requests_total{route, status}`
  - total requests seen by the proxy
- `proxy_request_latency_seconds{route}`
  - end-to-end latency histogram
- `proxy_inflight_requests`
  - number of requests currently in progress
- `proxy_rejected_total{route}`
  - requests rejected due to concurrency or admission control

### Useful PromQL examples

Request rate by status:

```promql
sum by (status) (rate(proxy_requests_total[1m]))
```

Recent request volume by status:

```promql
sum by (status) (increase(proxy_requests_total[1m]))
```

Proxy p95 latency:

```promql
histogram_quantile(0.95, sum(rate(proxy_request_latency_seconds_bucket[1m])) by (le))
```

Proxy p99 latency:

```promql
histogram_quantile(0.99, sum(rate(proxy_request_latency_seconds_bucket[1m])) by (le))
```

Current inflight requests:

```promql
proxy_inflight_requests
```

Rejected request rate:

```promql
sum(rate(proxy_rejected_total[1m]))
```

## What has been validated so far

- vLLM can be launched successfully on Runpod
- a model server can be reached through the OpenAI-compatible API
- concurrent request testing works with asyncio and httpx
- proxy-level counters and latency histograms are being exported
- Prometheus can be used to scrape both proxy and vLLM metrics
- request schema mismatches on chat-completions were identified and corrected

## Current limitations

This repo is still in active development. At the current stage:

- the proxy is intentionally minimal
- overload policy is not yet fully hardened
- dashboards are still lightweight
- the benchmark suite is focused on inference, not distributed training
- experiment management is still manual rather than automated

## Roadmap

### Stage 1: basic observability
- [x] Launch vLLM and expose metrics
- [x] Add a proxy with service-level metrics
- [x] Point the load test at the proxy
- [x] Scrape metrics with Prometheus
- [ ] Build a stable dashboard view for recent runs

### Stage 2: overload control
- [ ] Enforce bounded concurrency in the proxy
- [ ] Add explicit timeout behavior
- [ ] Reject excess requests quickly instead of allowing unbounded queue growth
- [ ] Measure the latency-throughput knee under controlled overload

### Stage 3: experiment discipline
- [ ] Standardize experiment matrices
- [ ] Save benchmark outputs and plots into `report/`
- [ ] Compare behavior across models and batching settings
- [ ] Separate successful latency from fast-fail error latency in analysis

### Stage 4: production-style evolution
- [ ] Add Grafana dashboards
- [ ] Add tracing selectively for debugging
- [ ] Package the workflow more cleanly for repeatable deployment
- [ ] Extend to Kubernetes and autoscaling experiments

## What this repo is meant to show

This project is meant to demonstrate practical AI infrastructure capability in areas such as:

- LLM serving systems
- performance measurement and tail-latency analysis
- observability with Prometheus-style metrics
- overload handling and service protection
- systems-level thinking applied to AI workloads

The emphasis is not on model novelty. The emphasis is on building a system that is **observable, measurable, and defensible under load**.

## Notes

- This repo should not store real tokens or secrets in committed scripts.
- Environment-specific values such as Hugging Face tokens should be provided through environment variables or secret management.
- Counters in Prometheus are monotonic by design; use `rate()` or `increase()` for recent-run analysis rather than trying to “clear” them.

## Next documentation to add

After this root README, the next useful docs are:

- `loadtest/README.md` for benchmark usage and CSV fields
- `observability/README.md` for Prometheus setup and common queries
- `server/README.md` for proxy behavior and serving notes
