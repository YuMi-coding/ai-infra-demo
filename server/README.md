# Server

This folder contains the serving layer for the AI infrastructure demo.

The current design separates model serving from service-level control:

```text
client / load generator
    -> FastAPI proxy
        -> vLLM OpenAI-compatible API server
```

This split is deliberate.

- **vLLM** handles model execution, batching, and token generation.
- **The proxy** handles service behavior: request admission, timeouts, forwarding, and service-level metrics.

That separation makes the system easier to reason about under load and easier to evolve into a more production-like inference stack.

---

## Current components

```text
server/
  proxy.py
```

At the moment, `proxy.py` is the custom layer in this folder. The vLLM server is launched separately with a shell command rather than from Python code in this directory.

---

## Ports

| Component | Port | Role |
|---|---:|---|
| vLLM API server | 8000 | upstream model serving endpoint |
| FastAPI proxy | 8080 | service entry point for load tests and observability |

---

## Request flow

The current request path is:

```text
POST /v1/chat/completions  -> proxy:8080
                           -> forwarded to vllm:8000
                           -> response returned to client
```

For the current Llama-3.1-Instruct setup, requests should use the **chat completions** format.

Example request body:

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

Using a `prompt` field with `/v1/chat/completions` will produce `400 Bad Request`. If you want to use `prompt`, the endpoint should instead be `/v1/completions`.

---

## Why the proxy exists

vLLM already serves the model, so why add a proxy?

Because real inference systems need more than “the model can answer.”

The proxy is the place to implement:

- bounded concurrency
- overload rejection
- upstream timeouts
- service-level request metrics
- request logging
- authentication or policy checks later
- routing logic later, if multiple models are added

This is the layer where the repo starts to look like AI infrastructure rather than just model serving.

---

## Current proxy responsibilities

`proxy.py` currently provides:

- health check endpoint
- Prometheus metrics endpoint
- request forwarding to the vLLM server
- request latency measurement
- in-flight request tracking
- overload rejection counter
- upstream timeout handling

---

## Proxy metrics

The proxy exports these metrics on `/metrics`:

- `proxy_requests_total`
- `proxy_request_latency_seconds`
- `proxy_inflight_requests`
- `proxy_rejected_total`

These are documented in more detail in `observability/README.md`.

---

## How to run the serving stack

## 1. Start vLLM

Example launch command:

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

## 2. Start the proxy

From the repo root:

```bash
uvicorn server.proxy:app --host 0.0.0.0 --port 8080
```

Quick checks:

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/metrics | head
```

## 3. Send traffic to the proxy, not directly to vLLM

The client or load test should point to:

```text
http://127.0.0.1:8080/v1/chat/completions
```

If you hit port `8000` directly, proxy metrics will not move and the service-level controls in `proxy.py` will be bypassed.

---

## Local sanity test

You can test the proxy directly with `curl`:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "max_tokens": 16,
    "temperature": 0.0
  }'
```

If this request succeeds through the proxy, the serving path is functioning.

---

## Failure modes to watch for

| Symptom | Likely cause | Fix |
|---|---|---|
| `400 Bad Request` from both proxy and vLLM | request schema mismatch | use `messages` for `/v1/chat/completions` |
| proxy metrics stay flat | client still hitting port `8000` | point client to `8080` |
| `504` from proxy | upstream timeout | increase timeout or reduce load |
| all requests rejected | concurrency gate too strict | raise the proxy in-flight limit |
| requests succeed but no `/metrics` data | metrics endpoint not scraped or wrong target | verify Prometheus configuration |

---

## Design notes

This repo intentionally keeps the proxy simple.

The goal is not to build a full production gateway immediately. The goal is to add just enough control and observability to turn a raw model server into a measurable service.

That means the server layer should evolve in small, defensible steps:

1. forwarding
2. metrics
3. admission control
4. timeouts
5. logging
6. tracing
7. policy/routing
8. autoscaling integration

---

## Current limitations

The current proxy is still a prototype.

Known limitations include:

- simple forwarding path
- minimal request validation
- no authentication
- no structured request logging yet
- latency histogram may mix successful and fast-fail requests
- concurrency control is intentionally basic
- only one upstream model endpoint is assumed

These are acceptable for the current phase of the project.

---

## Suggested next improvements

Good next improvements for this folder:

- add structured JSON request logs
- separate latency metrics by status class
- make admission control cleaner and less ad hoc
- expose configuration via environment variables
- support both `/v1/chat/completions` and `/v1/completions` cleanly
- add request IDs for traceability
- add optional tracing hooks

---

## Relationship to the rest of the repo

- `server/` defines service behavior
- `loadtest/` generates pressure and measures outcomes
- `observability/` makes those outcomes visible
- `report/` stores benchmark results and experiment notes

Taken together, these pieces form the basis of an inference-platform demo rather than a single model-serving script.