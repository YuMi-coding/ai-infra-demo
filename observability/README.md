# Observability

This folder contains the metrics and scraping configuration for the AI inference demo.

The current observability path is intentionally simple:

```text
load generator -> FastAPI proxy -> vLLM API server
                               -> /metrics scraped by Prometheus
```

At this stage, the observability goal is not full production dashboards. It is to make service behavior visible under load so we can answer a few basic questions with data:

- Is the proxy receiving requests?
- Are requests succeeding or failing?
- How does latency change as concurrency rises?
- Is the service saturating or rejecting traffic?
- Do proxy metrics and vLLM metrics move together?

---

## Current components

- **FastAPI proxy** on port `8080`
  - forwards requests to vLLM
  - exposes custom Prometheus metrics at `/metrics`
- **vLLM API server** on port `8000`
  - serves OpenAI-compatible requests
  - exposes internal metrics at `/metrics`
- **Prometheus** on port `9090`
  - scrapes both endpoints

---

## Ports

| Component | Port | Notes |
|---|---:|---|
| vLLM API server | 8000 | upstream inference server |
| FastAPI proxy | 8080 | request shaping and service-level metrics |
| Prometheus | 9090 | scraping and query UI |

---

## File in this folder

```text
observability/
  prometheus.yml
```

---

## Prometheus configuration

A minimal config is enough for the current setup:

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "proxy"
    static_configs:
      - targets: ["127.0.0.1:8080"]

  - job_name: "vllm"
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

This assumes all processes are running inside the same Runpod container.

---

## How to run Prometheus

If Prometheus is already installed:

```bash
prometheus --config.file=observability/prometheus.yml --web.listen-address=0.0.0.0:9090
```

Then verify health:

```bash
curl http://127.0.0.1:9090/-/healthy
```

Open the Prometheus UI and check:

- **Status -> Targets**

Both of these should be `UP`:

- `proxy`
- `vllm`

---

## Verify the metrics endpoints first

Before debugging Prometheus, make sure the endpoints themselves work:

```bash
curl http://127.0.0.1:8080/metrics | head
curl http://127.0.0.1:8000/metrics | head
```

If these do not return Prometheus text exposition format, fix that first.

---

## Proxy metrics

The proxy currently exports the following service-level metrics.

### `proxy_requests_total`
Counter for all requests seen by the proxy.

Labels:
- `route`
- `status`

Examples:
- `status="200"`
- `status="400"`
- `status="429"`
- `status="504"`

Use this to measure request rate and success/error mix.

### `proxy_request_latency_seconds`
Histogram for end-to-end proxy request latency.

Label:
- `route`

This includes all observed requests for that route. In the current implementation, this may include both successful requests and fast-fail requests, so interpret latency carefully if many 4xx responses are present.

### `proxy_inflight_requests`
Gauge for current in-flight requests being handled by the proxy.

Use this to observe queue pressure and concurrency.

### `proxy_rejected_total`
Counter for requests rejected due to concurrency limits or overload policy.

Label:
- `route`

This should stay at zero unless the proxy admission policy is actively rejecting excess load.

---

## Why counters are not “cleared”

Prometheus counters are monotonic by design. You normally do **not** reset them.

Instead, use:
- `rate(...)` for per-second trends
- `increase(...)` for activity during a recent time window

If you need a fresh counter value for the proxy, restart the proxy process. The metric objects live in that process’s memory.

---

## Useful PromQL queries

### Request rate
```promql
sum(rate(proxy_requests_total[1m]))
```

### Request rate by status
```promql
sum by (status) (rate(proxy_requests_total[1m]))
```

### Requests during the last 1 minute
```promql
sum by (status) (increase(proxy_requests_total[1m]))
```

### Success ratio
```promql
sum(rate(proxy_requests_total{status="200"}[1m]))
/
sum(rate(proxy_requests_total[1m]))
```

### p95 latency
```promql
histogram_quantile(
  0.95,
  sum(rate(proxy_request_latency_seconds_bucket[1m])) by (le)
)
```

### p99 latency
```promql
histogram_quantile(
  0.99,
  sum(rate(proxy_request_latency_seconds_bucket[1m])) by (le)
)
```

### In-flight requests
```promql
proxy_inflight_requests
```

### Rejected request rate
```promql
sum(rate(proxy_rejected_total[1m]))
```

---

## Recommended Session 1 workflow

### 1. Start all three processes
- vLLM on `8000`
- proxy on `8080`
- Prometheus on `9090`

### 2. Confirm both metrics endpoints work
```bash
curl http://127.0.0.1:8000/metrics | head
curl http://127.0.0.1:8080/metrics | head
```

### 3. Confirm Prometheus targets are `UP`
In the Prometheus UI:
- **Status -> Targets**

### 4. Run load through the proxy
Point the load test to:

```text
http://127.0.0.1:8080/v1/chat/completions
```

Do not hit vLLM directly if you want proxy metrics to move.

### 5. Query the metrics
At minimum, check:
- request rate by status
- p95/p99 latency
- in-flight requests
- rejected requests

---

## Interpreting results correctly

### If many requests return `400`
Do not trust latency percentiles yet. Fast-fail invalid requests can dominate the histogram and make the service look artificially fast.

Fix request formatting first, then rerun.

### If request rate rises but p99 explodes
This usually means queueing is growing and you are approaching the saturation knee.

### If request rate flattens while latency keeps rising
The system is overloaded and no longer scaling with added concurrency.

### If rejections appear quickly and latency stays bounded
That is often healthier than letting tail latency grow without control.

---

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Prometheus target is DOWN | wrong port or process not running | verify with `ss -ltnp` and `curl` |
| Proxy metrics stay flat | load test still hitting port `8000` | point load test to `8080` |
| No histogram query result | no traffic yet or wrong metric name | run a longer test and inspect metric names |
| Latency looks unrealistically tiny | many 4xx fast-fail requests | fix payload shape and rerun |
| Counters look “dirty” from previous runs | expected counter behavior | use `rate()` / `increase()` or restart proxy |

---

## Minimal smoke test

```bash
curl http://127.0.0.1:8000/metrics | head
curl http://127.0.0.1:8080/metrics | head
curl http://127.0.0.1:9090/-/healthy
```

Then run a tiny load test and verify:

```promql
sum by (status) (increase(proxy_requests_total[1m]))
```

If that query moves and both targets are `UP`, Session 1 observability is alive.

---

## Future extensions

This folder can later grow to include:

- Grafana dashboards
- alert rules
- recording rules
- service-level success/latency dashboards
- tracing integration
- GPU utilization dashboards
- per-model latency and throughput views

For now, keep it simple and reliable.