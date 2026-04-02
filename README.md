<div align="center">

# LLM Gateway

### One API for LLM providers, with cost and reliability controls built in.

Route application requests across OpenAI, Anthropic, Gemini, and local OpenAI-compatible models. Track token usage, estimated spend, failures, and fallbacks by application.

</div>

---

## Contents

- [Why LLM Gateway?](#why-llm-gateway)
- [Features](#features)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [API](#api)
- [Cost and reliability reporting](#cost-and-reliability-reporting)
- [Security and privacy](#security-and-privacy)
- [Run with Docker](#run-with-docker)
- [Current limits](#current-limits)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Why LLM Gateway?

An application that calls model vendors directly has to handle vendor-specific APIs, retries, credentials, and spend reporting in every service. LLM Gateway puts those concerns behind one endpoint and uses **logical routes** (`simple`, `reasoning`, `local`) so an application does not need to know which provider serves a request.

The goal is a small, understandable infrastructure service: start with one process and SQLite, then replace the in-process pieces with shared infrastructure as traffic grows.

## Features

| Capability | What it does |
|---|---|
| Provider-neutral chat API | Accepts a common chat-completions request and translates it for OpenAI-compatible APIs, Anthropic Messages, and Gemini `generateContent`. |
| Configurable model routes | Map a logical route to a provider and model in YAML. |
| Retry and fallback | Retry timeouts, network errors, HTTP 429, and upstream 5xx responses with bounded exponential backoff; then try configured fallbacks in order. |
| Traffic controls | Apply request timeouts, a maximum in-flight request count, and per-provider requests-per-minute limits. |
| Usage and cost ledger | Persist provider, route, model, token usage, estimated USD cost, status, latency, fallback, and cache-hit metadata in SQLite. |
| Application reporting | Group usage and reliability totals by the caller-supplied application name. |
| Response cache | Optionally cache a route's completion for a configured TTL. Cache hits have zero upstream usage and cost in the ledger. |
| PII and secret redaction | Pattern-redact common email addresses, phone-like strings, and API-key forms in message text before sending upstream. |
| Optional gateway authentication | Require a bearer key for route, completion, and report endpoints by setting `GATEWAY_API_KEY`. |
| Local model support | Connect to a local inference server that implements the OpenAI chat-completions API. |

## Architecture

```text
┌──────────────────────┐
│ Application          │
│ support-router       │
└──────────┬───────────┘
           │ POST /v1/chat/completions
           ▼
┌─────────────────────────────────────────┐
│ LLM Gateway                             │
│ route · redact · limit · retry · cache  │
│ usage ledger · cost/reliability report  │
└───────┬──────────┬───────────┬──────────┘
        │          │           │
        ▼          ▼           ▼
     OpenAI    Anthropic    Gemini / Local
```

The current application is a FastAPI service. Provider translation and response normalization live in adapters; route, retry, and cache policy is read from `config.yaml`; usage metadata is stored in SQLite. Request and response bodies are not stored.

## Quick start

**Requirements:** Python 3.11 or newer and an API key for each hosted provider you plan to call.

```bash
git clone <your-repository-url>
cd llm_gateway
python -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
llm-gateway
```

The service starts at `http://localhost:8000`. The example config includes OpenAI, Anthropic, Gemini, and a local OpenAI-compatible provider. Only configure credentials for providers that are enabled in a route or fallback.

Useful local URLs:

- `GET http://localhost:8000/health` — liveness check
- `GET http://localhost:8000/docs` — interactive API docs
- `GET http://localhost:8000/v1/routes` — active logical route configuration

To require a key from gateway clients, set `GATEWAY_API_KEY` in the gateway process and send it as a bearer token. The health endpoint remains public for liveness probes.

## Configuration

Edit `config.yaml`. Provider credentials are read from environment variables named by each provider's `api_key_env`; do not put API keys in the YAML file.

```yaml
server:
  request_timeout_seconds: 45
  max_retries: 2
  retry_base_delay_seconds: 0.25
  max_concurrent_requests: 100
  database: ./gateway.sqlite3

security:
  redact_pii: true

providers:
  openai:
    kind: openai
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    requests_per_minute: 600
    models:
      gpt-4o-mini:
        input_cost_per_1m: 0.15
        output_cost_per_1m: 0.60

routes:
  simple:
    provider: openai
    model: gpt-4o-mini
    fallbacks:
      - provider: gemini
        model: gemini-2.0-flash
    cache:
      enabled: true
      ttl_seconds: 60
```

Each route has a primary `provider` and `model`, an ordered optional `fallbacks` list, and optional cache settings. Each provider's `kind` must be `openai`, `anthropic`, or `gemini`. The local example uses `kind: openai` with an OpenAI-compatible base URL and an optional key.

### Pricing configuration

`input_cost_per_1m` and `output_cost_per_1m` are USD per million tokens. The gateway multiplies these rates by the provider-reported token counts. A model without an explicit price entry is treated as zero cost, so add a price entry for every billable model. Update prices when provider pricing changes. The report is an estimate, not an invoice; it does not include provider-specific discounts, taxes, or other billing adjustments.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `GATEWAY_CONFIG` | Path to YAML configuration | `config.yaml` |
| `GATEWAY_DATABASE` | SQLite database path | `./gateway.sqlite3` |
| `GATEWAY_API_KEY` | Optional required gateway bearer key | Unset (gateway auth disabled) |
| `OPENAI_API_KEY` | OpenAI credential when configured | Unset |
| `ANTHROPIC_API_KEY` | Anthropic credential when configured | Unset |
| `GOOGLE_API_KEY` | Gemini credential when configured | Unset |
| `PORT` | HTTP listen port | `8000` |
| `LOG_LEVEL` | Python logging level | `INFO` |

## API

### `POST /v1/chat/completions`

The request shape is similar to the OpenAI Chat Completions API. Here, `model` is the logical route name from `config.yaml`.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer gateway-secret' \
  -d '{
    "model": "simple",
    "application": "support-router",
    "messages": [
      {"role": "user", "content": "Summarize this support ticket."}
    ],
    "max_tokens": 300
  }'
```

The response includes the normalized completion, usage, actual provider/model selection, and gateway cost metadata:

```json
{
  "id": "gw_...",
  "object": "chat.completion",
  "model": "simple",
  "choices": [
    {"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}
  ],
  "usage": {"prompt_tokens": 42, "completion_tokens": 55, "total_tokens": 97},
  "gateway": {
    "provider": "openai",
    "upstream_model": "gpt-4o-mini",
    "fallback_used": false,
    "cached": false,
    "cost_usd": 0.0000393
  }
}
```

Supported request fields are `model`, `messages`, `application`, `temperature`, `max_tokens`, `top_p`, `user`, and `metadata`. The current implementation forwards message text and generation settings; `user` and `metadata` are accepted for compatibility but are not forwarded or persisted.

### Other endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Public liveness response. |
| `GET` | `/v1/routes` | Configured route names, providers, models, fallbacks, and cache policy. |
| `POST` | `/v1/chat/completions` | Make a routed chat completion. |
| `GET` | `/v1/admin/report` | Aggregate token, cost, and reliability data. |
| `GET` | `/docs` | FastAPI Swagger UI. |

Unless otherwise noted, endpoints under `/v1` require the configured gateway bearer key when `GATEWAY_API_KEY` is set.

## Cost and reliability reporting

Query the aggregate report by time range and/or application:

```bash
curl -H 'Authorization: Bearer gateway-secret' \
  'http://localhost:8000/v1/admin/report?start=2026-10-01T00:00:00Z&end=2026-11-01T00:00:00Z&application=support-router'
```

`start` is inclusive, `end` is exclusive, and both accept ISO-8601 timestamps. The response includes aggregate requests, input/output/total tokens, total and average cost per request, failure and fallback rates, average latency, cache hits, and a `by_application` breakdown.

```json
{
  "requests": 18421,
  "prompt_tokens": 8200000,
  "completion_tokens": 4200000,
  "total_tokens": 12400000,
  "cost_usd": 83.2,
  "avg_cost_usd": 0.0045,
  "failures": 147,
  "failure_rate": 0.0079,
  "fallbacks": 239,
  "fallback_rate": 0.013,
  "cache_hits": 410,
  "avg_latency_ms": 642,
  "by_application": [
    {"application": "support-router", "requests": 18421, "total_tokens": 12400000, "cost_usd": 83.2, "failures": 147, "fallbacks": 239}
  ]
}
```

The report is backed by a SQLite ledger keyed by gateway request ID. It records usage and operational metadata, not prompt or completion content. Failed requests may have zero token and cost values when the provider did not return usage.

## Security and privacy

- Set `GATEWAY_API_KEY` to protect all `/v1` endpoints with bearer authentication.
- Keep provider credentials in environment variables or a secrets manager. Do not commit credentials or `.env` files.
- With `security.redact_pii: true`, the gateway pattern-redacts common email, phone-like, and API-key strings in message text **before forwarding it to the model**.
- Redaction is pattern-based and is not a full data loss prevention system. Review whether this protection is sufficient for your data and use case.
- Prompts and completions are not written to the usage database or application logs. Avoid enabling HTTP wire-level logging in production.
- Put the service behind TLS and restrict access to `/v1/admin/report` using gateway authentication and network policy.

## Run with Docker

Build and run from the repository root:

```bash
docker build -t llm-gateway .
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e GATEWAY_API_KEY="gateway-secret" \
  -v "$(pwd)/data:/data" \
  -e GATEWAY_DATABASE=/data/gateway.sqlite3 \
  llm-gateway
```

For persistent data, mount a durable volume at the configured database directory. The included Docker image is a simple single-process deployment starting point.

## Current limits

- Streaming responses are not implemented.
- Rate limits, concurrency limits, and response cache are local to one process; they are not shared across replicas.
- SQLite is intended for a single instance and modest event volume. Multi-replica deployments need a shared database and coordinated migrations.
- The current gateway accepts text messages, not image/audio/video payloads or tool/function calls.
- Pricing uses configured per-token rates and provider-reported token counts; it is an estimate and depends on maintaining the model price map.
- PII redaction covers common patterns only and can miss sensitive information or alter phone-like text.

## Roadmap

Suggested next milestones, ordered by production value:

- [ ] Add automated unit and integration coverage for providers, routing, retries, redaction, caching, pricing, and failure accounting.
- [ ] Add OpenTelemetry traces and Prometheus metrics for latency, retries, provider health, cost, and cache behavior.
- [ ] Add streaming support while preserving usage accounting and fallback behavior before the first response chunk.
- [ ] Add shared Redis rate limits/cache and PostgreSQL-backed event storage for multi-replica deployments.
- [ ] Add per-application budgets, hard spend limits, quotas, and alerts.
- [ ] Add circuit breakers and provider health-aware routing to avoid repeated calls to unhealthy upstreams.
- [ ] Add more adapters: Azure OpenAI, AWS Bedrock, Vertex AI, and vLLM-native endpoints.
- [ ] Add model allowlists, per-route policies, request size limits, and stronger authentication options.
- [ ] Add tool calling, structured outputs, and multimodal request support.
- [ ] Add a dashboard and export endpoints for finance and operations workflows.

## Contributing

Issues and pull requests are welcome. A useful contribution includes a clear problem statement, configuration or API examples when relevant, and notes about provider-specific behavior. For a new provider adapter, normalize text, finish reasons, and usage into the gateway's shared response format, and avoid logging request or response content.

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

## Reference projects

The documentation structure and gateway concepts were informed by public documentation from [LiteLLM](https://github.com/BerriAI/litellm), [Kong Gateway](https://github.com/Kong/kong), and [Envoy](https://github.com/envoyproxy/envoy). This project is independent and is not affiliated with those projects.
