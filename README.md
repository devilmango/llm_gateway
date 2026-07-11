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
| Provider-neutral chat API | Accepts a common chat-completions request and translates it for OpenAI-compatible APIs, Anthropic Messages, Gemini, Azure OpenAI, Vertex AI, and Amazon Bedrock Converse. |
| Provider adapters | Supports OpenAI-compatible, Azure OpenAI, Anthropic, Gemini, Vertex AI, Amazon Bedrock Converse, and vLLM-native/OpenAI-compatible endpoints. |
| Provider capability registry | Advertises and enforces streaming, streaming tools, tools, structured output, images, and audio input/output per provider, with optional provider-level capability overrides. |
| Configurable model routes | Map a logical route to a provider and model in YAML. |
| Retry and fallback | Retry timeouts, network errors, HTTP 429, and upstream 5xx responses with bounded exponential backoff; then try configured fallbacks in order. |
| Traffic controls | Apply request timeouts, a maximum in-flight request count, and per-provider requests-per-minute limits. Use Redis to share request limits across replicas. |
| Usage and cost ledger | Persist provider, route, model, token usage, estimated USD cost, status, latency, fallback, and cache-hit metadata in SQLite or PostgreSQL. |
| Application reporting | Group usage and reliability totals by application, provider, model, route, and cost center. |
| Application budgets and quotas | Enforce UTC calendar-month spend caps, per-application request rates, and request size/output bounds. Reserve estimated maximum cost before provider calls to limit concurrent overspend. |
| Budget alerts | Emit structured warning logs and optional signed webhook events when configured spend thresholds (80% and 100% by default) are crossed. |
| Observability | Export Prometheus request, token, cost, fallback, cache, in-flight, end-to-end latency, provider outcome/latency, and retry metrics; emit nested OpenTelemetry request/provider-attempt spans and optionally export them over OTLP/HTTP. |
| Provider health | Open a circuit after repeated transient failures, skip open providers, and allow a single half-open probe after the recovery period. |
| Latency-aware routing | Optionally order a route's primary and fallback candidates using shared provider EWMA latency and failure-rate signals. |
| Streaming | Stream OpenAI-compatible, Azure, Anthropic, Gemini, and Vertex output as server-sent events, including tool-call and audio deltas where supported; account usage at completion. |
| Tenant access controls | Configure per-tenant bearer credentials, route allowlists, application allowlists, and tenant-specific budget/limit policies. |
| Tool calling and structured outputs | Pass OpenAI-style function definitions and JSON response schemas; adapters translate them for OpenAI-compatible, Anthropic, and Gemini providers. Tool execution stays in the calling application. |
| Durable budget alert webhooks | Persist threshold events in an outbox, retry transient failures with backoff, optionally sign with HMAC-SHA256, and inspect or redrive dead letters. |
| Response cache | Optionally cache a route's completion for a configured TTL. Cache hits have zero upstream usage and cost in the ledger. |
| PII and secret redaction | Pattern-redact common email addresses, phone-like strings, and API-key forms in message text before sending upstream. |
| Optional gateway authentication | Require a bearer key for route, completion, and report endpoints by setting `GATEWAY_API_KEY`; support OIDC/JWT for admin and client access. |
| Scoped admin keys | Configure multiple admin keys with report, budget, provider-health, or webhook scopes; overlap keys during rotation and disable old entries to revoke them. |
| Cost-center reporting and exports | Tag requests with a cost center, view provider/model/route breakdowns, filter reports, and export request-level records as CSV. |
| Multimodal image input | Accept OpenAI-style image content parts and translate data URIs or image URLs for supported providers. |
| Audio input and output | Accept base64 audio input parts and request generated audio for providers whose capability registry enables it. |
| OIDC/JWT and audit log | Validate admin JWTs against configured issuer/audience/JWKS, enforce scopes, and persist admin request records. |
| Operations dashboard | Serve a responsive dashboard at `/admin` for cost, budgets, provider health, and webhook dead letters. |
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

The current application is a FastAPI service. Provider translation, response normalization, and capability checks live in adapters; route, retry, and cache policy is read from `config.yaml`; usage and admin audit metadata are stored in SQLite or PostgreSQL. Request and response bodies are not written to the usage ledger or application logs. When route caching is enabled, completion responses are held in local memory or Redis until their configured TTL expires.

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
- `GET http://localhost:8000/metrics` — Prometheus metrics

To require a key from gateway clients, set `GATEWAY_API_KEY` in the gateway process and send it as a bearer token. The health endpoint remains public for liveness probes.

For tenant-level access, define `tenants` in the YAML configuration. Each tenant maps an API key environment variable to allowed logical routes and application policy overrides. When tenants are configured, `/v1/chat/completions` and `/v1/routes` require a matching tenant key; admin endpoints require a separate `GATEWAY_API_KEY` or configured scoped admin key.

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
  # Optional scoped admin credentials support overlap during rotation.
  # admin_api_keys:
  #   - id: finance-reporting
  #     api_key_env: GATEWAY_FINANCE_KEY
  #     scopes: [report]
  #   - id: platform-operations
  #     api_key_env: GATEWAY_OPERATIONS_KEY
  #     scopes: [budgets, providers, webhooks, audit]
  # oidc:
  #   enabled: true
  #   issuer: https://identity.example.com/
  #   audience: llm-gateway
  #   jwks_url: https://identity.example.com/.well-known/jwks.json
  #   algorithms: [RS256]

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

# Optional: enable shared latency/failure EWMA ordering among route candidates.
routing:
  strategy: primary_first # or latency_aware

# Optional tenant configuration. Keep key values in the process environment.
tenants:
  support-team:
    api_key_env: SUPPORT_TEAM_GATEWAY_KEY
    allowed_routes: [simple]
    default_policy:
      max_requests_per_minute: 120
      max_completion_tokens: 1000
    applications:
      support-router:
        monthly_limit_usd: 50
        max_input_bytes: 32768
```

Each route has a primary `provider` and `model`, an ordered optional `fallbacks` list, and optional cache settings. Each provider's `kind` must be `openai`, `anthropic`, or `gemini`. The local example uses `kind: openai` with an OpenAI-compatible base URL and an optional key.

### Pricing configuration

`input_cost_per_1m` and `output_cost_per_1m` are USD per million tokens. The gateway multiplies these rates by the provider-reported token counts. A model without an explicit price entry is treated as zero cost, so add a price entry for every billable model. Update prices when provider pricing changes. The report is an estimate, not an invoice; it does not include provider-specific discounts, taxes, or other billing adjustments.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `GATEWAY_CONFIG` | Path to YAML configuration | `config.yaml` |
| `GATEWAY_DATABASE` | SQLite path or PostgreSQL DSN | `./gateway.sqlite3` |
| `REDIS_URL` | Optional shared cache and distributed rate-limit store | Unset (process-local backends) |
| `GATEWAY_API_KEY` | Optional required gateway bearer key | Unset (gateway auth disabled) |
| Tenant key variables | Configure each tenant's `api_key_env` with a unique secret variable | Required for each configured tenant |
| `OPENAI_API_KEY` | OpenAI credential when configured | Unset |
| `ANTHROPIC_API_KEY` | Anthropic credential when configured | Unset |
| `GOOGLE_API_KEY` | Gemini credential when configured | Unset |
| `PORT` | HTTP listen port | `8000` |
| `LOG_LEVEL` | Python logging level | `INFO` |
| `OTEL_SERVICE_NAME` | OpenTelemetry service resource name | `llm-gateway` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Optional OTLP/HTTP traces endpoint | Unset (spans are not exported) |

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

Supported request fields include `model`, `messages`, `application`, `cost_center`, `temperature`, `max_tokens`, `top_p`, `tools`, `tool_choice`, `response_format`, `modalities`, `audio`, and `stream`. `user` and `metadata` are accepted for compatibility but are not forwarded or persisted.

For tool calling, send OpenAI-style function declarations. The gateway returns normalized assistant `tool_calls`; the application remains responsible for authorizing and executing the functions, then sending their results as `role: "tool"` messages in a follow-up request.

```json
{
  "model": "simple",
  "messages": [{"role": "user", "content": "Look up order A-17"}],
  "tools": [{"type": "function", "function": {
    "name": "lookup_order",
    "description": "Look up an order by its ID",
    "parameters": {
      "type": "object",
      "properties": {"order_id": {"type": "string"}},
      "required": ["order_id"]
    }
  }}],
  "tool_choice": "auto"
}
```

Use `response_format: {"type":"json_schema","json_schema":{"name":"order_status","strict":true,"schema":{"type":"object","properties":{"status":{"type":"string"}},"required":["status"],"additionalProperties":false}}}` for schema-constrained output. `json_object` is also accepted when a specific schema is not needed. The gateway translates these options for OpenAI-compatible, Anthropic, and Gemini APIs; Anthropic JSON-object mode uses an explicit instruction, while Anthropic schema mode and Gemini modes use provider formatting controls. Each provider may impose its own JSON Schema limits.

### Other endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Public liveness response. |
| `GET` | `/v1/routes` | Configured route names, providers, models, fallbacks, and cache policy. |
| `POST` | `/v1/chat/completions` | Make a routed chat completion. |
| `GET` | `/admin` | Operations dashboard shell; data loads through authenticated admin APIs. |
| `GET` | `/v1/admin/report` | Aggregate token, cost, and reliability data. |
| `GET` | `/v1/admin/report.csv` | Export filtered request-level usage and cost records as CSV (up to 100,000 rows). |
| `GET` | `/v1/admin/budgets` | Current-month spend, limit, and utilization for configured applications. |
| `GET` | `/v1/admin/providers` | Provider circuit state and consecutive failure counts. |
| `GET` | `/v1/admin/audit` | Recent authenticated admin API access events. |
| `GET` | `/v1/admin/webhooks` | Inspect pending, delivered, and dead-letter budget alert events. |
| `POST` | `/v1/admin/webhooks/{event_id}/redrive` | Retry a dead-letter alert event. |
| `GET` | `/metrics` | Prometheus metrics in exposition format. |
| `GET` | `/docs` | FastAPI Swagger UI. |

When tenant credentials are configured, `/v1/chat/completions` and `/v1/routes` require a tenant key, while `/v1/admin/*` requires a `GATEWAY_API_KEY`, scoped admin key, or configured OIDC bearer token. Without tenant configuration, `GATEWAY_API_KEY` optionally protects `/v1` endpoints.

## Cost and reliability reporting

Query the aggregate report by time range and/or application:

```bash
curl -H 'Authorization: Bearer gateway-secret' \
  'http://localhost:8000/v1/admin/report?start=2026-10-01T00:00:00Z&end=2026-11-01T00:00:00Z&application=support-router'
```

`start` is inclusive, `end` is exclusive, and both accept ISO-8601 timestamps. The response includes aggregate requests, input/output/total tokens, total and average cost per request, failure and fallback rates, average latency, cache hits, and breakdowns by application, provider, model, route, and cost center. Optional `cost_center`, `provider`, and `route` filters can be combined with time and application filters. `GET /v1/admin/report.csv` accepts the same filters and exports at most 100,000 request rows.

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

### Application budget policy

Budgets are configured under `budgets`. Policies in `applications` override fields from `default`. Monthly limits use UTC calendar months. For a configured monthly limit, requests are capped by `max_input_bytes` and `max_completion_tokens`; if `max_tokens` is omitted, the policy's completion cap is applied. Before calling providers, the gateway reserves a conservative estimate based on UTF-8 input bytes, the completion cap, and the highest configured price among the route's primary and fallback models. Database transactions include active reservations when checking the cap, which bounds concurrent overspend. Correct price entries are required for accurate enforcement. Reservations expire after five minutes to recover from process crashes.

`max_requests_per_minute` is a per-application quota (shared across replicas with Redis). Configure `alert_thresholds` as fractions of the monthly cap (for example `[0.5, 0.8, 1.0]`) to emit one warning log and optional webhook event at each threshold per app and month. `GET /v1/admin/budgets` reports current-month spend and utilization.

Tenant application names are scoped in the ledger as `<tenant-id>:<application>`, so same-named applications belonging to different tenants do not share budget accounting. Tenant application policies override global application policies, and each tenant may also define a `default_policy`.

Add `alert_webhook_url` to the matching application policy to put each newly crossed `alert_thresholds` event in the persistent outbox. Optionally set `alert_webhook_secret_env` to the name of an environment variable containing a shared secret. A worker POSTs compact JSON with the `X-LLM-Gateway-Signature: sha256=<hex>` header and a stable `Idempotency-Key`. Transient HTTP 429/5xx and network failures are retried with exponential backoff; after eight failed attempts an event enters the dead-letter queue. Permanent 4xx responses go directly to dead letter. Delivery does not block or fail the model request. Inspect `/v1/admin/webhooks` and redrive dead letters after fixing the receiver. Configure webhook URLs only in trusted gateway configuration.

Set `"stream": true` on a chat request to receive OpenAI-compatible SSE frames (`data: ...`) followed by `[DONE]`. Native streaming is supported for OpenAI-compatible, Anthropic, and Gemini APIs. The gateway can move to a fallback before sending text. After the first text delta, provider failures are returned as an SSE `error` event because replaying a second provider response could duplicate output. Token usage is included when the upstream sends final usage metadata.

Requests may include OpenAI-style `tools`, `tool_choice`, and `response_format`. The gateway translates function declarations, prior assistant tool calls, and tool-result messages for Anthropic and Gemini as well as OpenAI-compatible endpoints. Tool calls are returned in the normalized assistant response; the gateway never runs the requested function. `response_format` accepts `json_object` or `json_schema`. Streaming tool calls are currently rejected. Requests containing tool definitions or tool-call history bypass response caching to avoid replaying stale actions.

For image input, use OpenAI-compatible content parts such as `{"type":"text","text":"Describe this"}` and `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`. OpenAI-compatible providers receive the parts unchanged; Anthropic and Gemini receive native image blocks. Remote image URLs are forwarded as provider URL references; Gemini deployments should use provider-supported file URIs or inline data URIs.

For audio input, include `{"type":"input_audio","input_audio":{"data":"<base64>","format":"wav"}}` in the message content. Audio output can be requested with `"modalities":["text","audio"]` and `"audio":{"voice":"alloy","format":"wav"}`. The provider capability registry rejects unsupported combinations before an upstream request and allows the route to try a compatible fallback. Provider-level `capabilities` mappings can override defaults.

With `stream: true` and `tools`, tool-call chunks use OpenAI-compatible `delta.tool_calls` fragments. The application still owns tool authorization and execution. Audio-producing routes may emit provider-native audio deltas when streaming is enabled and supported.

### Provider adapter configuration

Use `kind: azure_openai` with an Azure resource `base_url`, deployment name as the route model, `api_key_env`, and optional `api_version`. Use `kind: bedrock` with `region`; requests are signed with AWS Signature Version 4 using `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optional `AWS_SESSION_TOKEN`. Use `kind: vertex` with `project_id`, `region`, and an access-token environment variable (default `GOOGLE_CLOUD_ACCESS_TOKEN`). Use `kind: vllm` for a vLLM OpenAI-compatible endpoint. Bedrock supports Converse requests but not streaming; capability checks allow compatible fallbacks to serve unsupported requests.

Provider capabilities appear in `GET /v1/routes` and `GET /v1/admin/providers`. A provider-level override can capture a known limitation, for example `capabilities: {audio_output: false}`. Keep overrides conservative and verify them against the selected upstream model.

Prometheus counters and histograms are exposed at `/metrics`. OpenTelemetry request and provider-attempt spans are created; set `OTEL_EXPORTER_OTLP_ENDPOINT` to export traces with OTLP over HTTP. Keep `/metrics` on a private network or protect it at the ingress if metrics are sensitive in your environment.

## Multi-replica deployment

Set `REDIS_URL` to enable shared Redis response caching, atomic per-provider/per-application sliding-window quotas, and provider health/latency signals used by latency-aware routing. Redis must be available when the service starts; the gateway verifies the connection and fails startup if the configured backend is unavailable. Cache reads/writes and provider health writes fail open; quota checks fail closed with HTTP 503 if Redis becomes unavailable, so traffic is not sent without enforcing configured limits. Use a private Redis network and `rediss://` when TLS is required; cached completions remain readable to Redis operators until their TTL expires.

Set `GATEWAY_DATABASE` to a PostgreSQL DSN such as `postgresql://gateway:password@db:5432/llm_gateway` to use the PostgreSQL ledger and coordinate concurrent budget reservations across replicas. The service creates the required tables and indexes at startup. Use SQLite for single-instance local development.

Provider circuit breakers remain process-local. Each replica independently skips providers that have crossed its failure threshold; recovery probes are limited to one per provider per process. Latency and success/failure summaries are shared in Redis when configured. Set `routing.strategy: latency_aware` to order configured route candidates by their shared EWMA latency, penalized by failure rate. Use `/v1/admin/providers` to inspect circuit and performance signals. Deploy behind TLS, keep Redis and PostgreSQL private, and protect admin and metrics endpoints at the network edge.

## Security and privacy

- Set `GATEWAY_API_KEY` to protect all `/v1` endpoints with bearer authentication.
- Configure `security.admin_api_keys` for scoped admin access. Supported scopes are `report`, `budgets`, `providers`, `webhooks`, and `audit`. Rotate by enabling a replacement key before disabling the old entry; entries take effect on process restart.
- Configure `security.oidc` with issuer, audience, and JWKS URL to validate signed JWTs. Admin tokens need `llm-gateway:<scope>` or `admin:*`; client access uses the `client` scope when OIDC is enabled without tenant keys. Access tokens are validated offline, so revocation takes effect when the token expires unless the identity provider issues a replacement or rejection is reflected by a new JWKS key set.
- Authenticated admin calls are recorded in the audit table and exposed at `/v1/admin/audit`. The audit log stores actor, path, status, and time; it omits tokens, prompts, and request bodies.
- When using tenant credentials, configure separate admin credentials; tenant keys are restricted to allowed routes and applications.
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

- Bedrock streaming is not implemented. Gemini image URL inputs must use provider-supported file URIs or inline data URIs.
- Without Redis, rate limits, cache, and provider performance signals are process-local. Concurrency limits and circuit-breaker health remain process-local even when Redis is configured.
- PostgreSQL supports shared usage storage and monthly budget reservations across replicas. SQLite remains intended for a single instance and modest event volume.
- The gateway accepts text, image, and supported audio messages, but not video payloads. It returns tool calls for the application to execute and does not run them itself.
- Scoped admin key changes require a process restart. The dashboard shell is public; data APIs retain configured authentication and should be served behind TLS.
- Pricing uses configured per-token rates and provider-reported token counts; it is an estimate and depends on maintaining the model price map.
- PII redaction covers common patterns only and can miss sensitive information or alter phone-like text.

## Roadmap

Suggested next milestones, ordered by production value:

- [x] Add automated endpoint and provider-adapter coverage for routing, retries, redaction, caching, pricing, failure accounting, budgets, and quotas.
- [x] Add OpenTelemetry traces and Prometheus metrics for latency, retries, provider health, cost, and cache behavior.
- [x] Add streaming support while preserving usage accounting and fallback behavior before the first text chunk.
- [x] Add shared Redis rate limits/cache and PostgreSQL-backed event storage for multi-replica deployments.
- [x] Add per-application budgets, estimated hard spend limits, quotas, and log-based alerts.
- [x] Add circuit breakers and provider health-aware routing to avoid repeated calls to unhealthy upstreams.
- [x] Add tenant credentials with route/application scopes and tenant-specific budget policies.
- [x] Add normalized tool calling and structured JSON output formats across configured providers.
- [x] Deliver budget threshold alerts to optional signed webhooks.
- [x] Add a durable webhook outbox with retry backoff, delivery status, and dead-letter redrive.
- [x] Add cost-center tags, provider/model/route report breakdowns, filters, and CSV usage exports.
- [x] Add native Anthropic and Gemini streaming and multimodal image input.
- [x] Add configurable scoped admin keys with rotation overlap and revocation.
- [x] Add Azure OpenAI, AWS Bedrock Converse, Vertex AI, and vLLM adapters with a capability registry.
- [x] Add OIDC/JWT validation, scoped admin access, and audit logs.
- [x] Add an operations dashboard, audio input/output, and streaming tool-call chunks.

## Contributing

Issues and pull requests are welcome. A useful contribution includes a clear problem statement, configuration or API examples when relevant, and notes about provider-specific behavior. For a new provider adapter, normalize text, finish reasons, and usage into the gateway's shared response format, and avoid logging request or response content.

Install development dependencies and run the same checks as CI:

```bash
pip install -e '.[dev]'
ruff check llm_gateway tests
pytest
```

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

## Reference projects

The documentation structure and gateway concepts were informed by public documentation from [LiteLLM](https://github.com/BerriAI/litellm), [Kong Gateway](https://github.com/Kong/kong), and [Envoy](https://github.com/envoyproxy/envoy). This project is independent and is not affiliated with those projects.
