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
| Provider adapters | Supports OpenAI-compatible, Azure OpenAI, Anthropic, Gemini, Vertex AI, Amazon Bedrock Converse/ConverseStream, and vLLM-native/OpenAI-compatible endpoints. |
| Provider capability registry | Advertises and enforces streaming, streaming tools, tools, structured output, images, and audio input/output per provider, with optional provider-level capability overrides. |
| Configurable model routes | Map a logical route to a provider and model in YAML. |
| Retry and fallback | Retry timeouts, network errors, HTTP 429, and upstream 5xx responses with bounded exponential backoff; then try configured fallbacks in order. |
| Traffic controls | Apply request timeouts, a maximum in-flight request count, and per-provider requests-per-minute limits. Use Redis to share rate limits and renewable concurrency leases across replicas. |
| Usage and cost ledger | Persist provider, route, model, token usage, estimated USD cost, status, latency, fallback, cache-hit, and canary metadata in SQLite or PostgreSQL. |
| Application reporting | Group usage and reliability totals by application, provider, model, route, and cost center. |
| Application budgets and quotas | Enforce UTC calendar-month spend caps, per-application request rates, and request size/output bounds. Reserve estimated maximum cost before provider calls to limit concurrent overspend. |
| Budget alerts | Emit structured warning logs and optional signed webhook events when configured spend thresholds (80% and 100% by default) are crossed. |
| Observability | Export Prometheus request, token, cost, fallback, cache, in-flight, end-to-end latency, provider outcome/latency, retry, SLO, and GenAI semantic-convention metrics; emit OpenTelemetry GenAI request/provider-attempt attributes and optionally export spans over OTLP/HTTP. Message content is excluded from telemetry. |
| Provider health | Open a circuit after repeated transient failures, skip open providers, and allow a single half-open probe after the recovery period. Redis shares circuit state across replicas. |
| Latency-aware routing | Optionally order a route's primary and fallback candidates using shared provider EWMA latency and failure-rate signals. |
| Canary routing and evaluation | Deterministically send a configured share of requests to a candidate provider/model; evaluate JSONL prompt sets by pass rate, latency, and estimated cost using `llm-gateway-eval`. |
| Context-aware canary flags | Optionally evaluate an OpenFeature Remote Evaluation Protocol boolean flag with application and tenant context. Evaluation errors fail closed to the normal route. |
| Route and application SLOs | Report availability and latency objective compliance and error-budget burn; ship example Prometheus recording and alerting rules. |
| Streaming | Stream OpenAI-compatible, Azure, Anthropic, Gemini, Vertex, and Bedrock output as server-sent events, including tool-call and audio deltas where supported; account usage at completion. Fallback stops as soon as caller-visible content is emitted. |
| Tenant data policies | Restrict tenants to allowed routes, providers, models, and tagged processing regions; automatically expire tenant usage metadata on a per-tenant retention schedule. |
| Dynamic credential rotation | Re-read mounted provider, tenant, gateway, and scoped-admin secret files on each use so CSI/agent-managed secret rotations take effect without restarting the gateway. |
| Tool calling and structured outputs | Pass OpenAI-style function definitions and JSON response schemas; adapters translate them for OpenAI-compatible, Anthropic, and Gemini providers. Tool execution stays in the calling application. |
| Durable budget alert webhooks | Persist threshold events in an outbox, retry transient failures with backoff, optionally sign with HMAC-SHA256, and inspect or redrive dead letters. |
| Response cache | Optionally cache a route's completion for a configured TTL. Cache hits have zero upstream usage and cost in the ledger. |
| PII and secret redaction | Pattern-redact common email addresses, phone-like strings, and API-key forms in message text before sending upstream. |
| Optional gateway authentication | Require a bearer key for route, completion, and report endpoints by setting `GATEWAY_API_KEY`; support OIDC/JWT for admin and client access. |
| Scoped admin keys | Configure multiple admin keys with report, budget, provider-health, or webhook scopes; overlap keys during rotation and disable old entries to revoke them. |
| Cost-center reporting and exports | Tag requests with a cost center, view provider/model/route breakdowns, filter reports, and export request-level records as CSV. |
| Cost reconciliation | Import actual provider charges by gateway request ID and compare totals and variance with the original per-request estimates. |
| Cost anomaly detection | Compare recent per-request and per-1,000-token costs with the previous equal-length window by application, provider, model, and pricing version. |
| Routing decision history | Inspect selected, fallback, and skipped provider/model candidates with safe reason codes, retry counts, and report aggregates. |
| Saved evaluation comparisons | Compare persisted evaluation runs by route, including pass-rate confidence intervals, cost and latency regressions, failure rates, and optional promotion gates. |
| Tool execution timeout visibility | Inspect generated tool calls without reported outcomes, flag overdue calls, and scrape pending/overdue counts for Prometheus alerting. |
| Tool approval workflow | Set approval TTLs, revoke active approvals, notify configured applications through the signed webhook outbox, and use the Python client helper to check approval immediately before execution. |
| Shadow evaluation and promotion | Sample live requests to a non-returned candidate, compare validity-based quality, cost, latency, and failures, then combine shadow thresholds with saved evaluation and live SLO gates. |
| Provider billing correlation metadata | Map the gateway request ID into provider-specific headers and nested request-body fields for invoice matching. |
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

For tenant-level access, define `tenants` in the YAML configuration. Each tenant can use an environment variable or a mounted `secret_file` for its bearer key, then define route, provider, model, region, and application allowlists. When tenants are configured, `/v1/chat/completions` and `/v1/routes` require a matching tenant key; admin endpoints require a separate `GATEWAY_API_KEY` or configured scoped admin key.

## Configuration

Edit `config.yaml`. Provider credentials can come from environment variables named by `api_key_env`, or from a mounted secret file named by `api_key_file`; do not put API keys in the YAML file. File-backed credentials are re-read for each provider request and authentication check.

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

# Optional tenant configuration. Use mounted secret files for live rotation.
tenants:
  support-team:
    secret_file: /run/secrets/support-team-gateway-key
    allowed_routes: [simple]
    allowed_providers: [openai, gemini]
    allowed_models: [gpt-4o-mini, gemini-2.0-flash]
    allowed_regions: [global]
    data_policy:
      usage_retention_days: 90
    default_policy:
      max_requests_per_minute: 120
      max_completion_tokens: 1000
    applications:
      support-router:
        monthly_limit_usd: 50
        max_input_bytes: 32768
```

Each route has a primary `provider` and `model`, an ordered optional `fallbacks` list, and optional cache settings. A route can also define a `canary` with `provider`, `model`, and `traffic_percent` from 0 to 100. A request ID hash makes sampling stable for that request; selected canary traffic bypasses route caching and falls back to the configured primary/fallback candidates if needed. Provider `kind` values include `openai`, `azure_openai`, `vllm`, `anthropic`, `gemini`, `vertex`, and `bedrock`.

`allowed_providers`, `allowed_models`, and `allowed_regions` are optional tenant restrictions and apply to the route's primary, canary, and fallback candidates. Set each provider's `data_region` to the region where that deployment processes tenant data; untagged providers are treated as `global`. The gateway uses these tags as operator-supplied policy metadata and does not independently verify provider residency. Tenant `data_policy.usage_retention_days` removes old usage metadata every 24 hours. The minimum is 35 days to preserve full current-month budget accounting; prompt and completion content is not stored in the ledger.

### Pricing configuration

`input_cost_per_1m` and `output_cost_per_1m` are USD per million tokens. The gateway multiplies these rates by the provider-reported token counts. A model without an explicit price entry is treated as zero cost, so add a price entry for every billable model. Set `pricing_catalog` on a provider and `pricing_version` per model (or on the provider as a default) whenever you update rates. The request ledger preserves the catalog and version used at request time, so `/v1/admin/report` can compare estimates, invoice actuals, and variance by provider/model/catalog without rewriting historical rates. The report is an estimate, not an invoice; it does not include provider-specific discounts, taxes, or other billing adjustments.

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

Generated tool-call argument strings must parse as JSON objects and satisfy the declared function's JSON Schema before they are returned. The gateway checks local schema references only; remote `$ref` values are rejected to prevent implicit network fetches. For streaming responses, tool-call chunks are withheld until their complete arguments pass schema validation.

Each upstream attempt receives the gateway's generated request ID in `X-Gateway-Request-ID`, including retries, fallbacks, and streaming calls. Per-provider `request_id_header` and `request_id_body_field` settings can override the header name and insert the same ID into a provider-supported metadata field. The body path is dot-separated and created when missing. For example, set `request_id_header: X-Invoice-Request` and `request_id_body_field: metadata.gateway_request_id` on one provider. Use only metadata fields supported by that provider; the gateway cannot guarantee the provider retains custom body fields in its invoice export.

Optional tool governance restricts declared and generated function names, caps generated calls per request, and can apply a per-tenant/application per-minute call limit. Configure global `tool_governance` and/or tenant `allowed_tools`, `default_policy.max_tool_calls_per_request`, and `default_policy.max_tool_calls_per_minute`; application policies can further narrow the allowlist and limits. Global and tenant allowlists intersect, and configured limits use the most restrictive value. Disallowed declarations are rejected before a provider call. Disallowed or excessive generated calls are withheld and recorded as denied metadata. Successful tool names are audited without arguments. Admins with the `audit` scope can inspect `/v1/admin/tool-audit?tenant_id=support-team`.

For high-impact functions, add their names to `tool_governance.require_approval_tools` or a tenant's `require_approval_tools` / application policy. A generated call is audited as `approval_required` and returned with its ID in `gateway.approval_required_tool_call_ids`. Before invoking it, query `GET /v1/tool-approvals/{request_id}/{tool_call_id}?application=...`; only `approved` and `not_required` allow execution. Approvers with the `tool_approvals` scope submit `approved`, `denied`, or `revoked` to `POST /v1/admin/tool-approvals`. Approval expires after `tool_governance.approval_ttl_seconds` (default 3,600 seconds), or the request's `expires_in_seconds`; expiry and revocation immediately block new execution reports. Each decision is retained as an audit event with the authenticated actor. Configure `tool_governance.approval_webhooks.<application>.url` to notify an application through the durable, optionally HMAC-signed webhook outbox.

The Python helper checks approval immediately before calling the application's async tool function and reports its outcome without sending tool arguments to the gateway:

```python
from llm_gateway.client import execute_tool_with_approval

result = await execute_tool_with_approval(
    gateway_client,
    request_id=request_id,
    tool_call_id=tool_call_id,
    application="support-router",
    execute=lambda: delete_account(account_id),
)
```

The gateway returns tool calls for the application to execute; the application and downstream service must still authorize the side effect. The helper supports async callables and requires `gateway_client` to have the gateway base URL and tenant authorization configured.

The application still executes tools itself. After execution, report only the outcome metadata; arguments, results, and exception text are not accepted or stored. The gateway verifies the request, tool-call ID, application, and tenant against the generated-call audit record. Repeating the same outcome is idempotent; changing an already recorded outcome is rejected.

Set `tool_governance.execution_timeout_seconds` to control when an unreported outcome is considered overdue (default: 300 seconds). Admins with the `audit` scope can inspect pending calls at `/v1/admin/tool-executions/pending`; the endpoint returns age, timeout, pending count, and overdue count. `/metrics` refreshes `llm_gateway_tool_executions_pending` and `llm_gateway_tool_executions_overdue` on each scrape, so Prometheus can alert when the overdue gauge is above zero.

```bash
curl -X POST http://localhost:8000/v1/tool-executions \
  -H 'Authorization: Bearer tenant-secret' -H 'Content-Type: application/json' \
  -d '{"request_id":"gw_abc","tool_call_id":"call_123","application":"support-router","outcome":"failed","error_code":"order_service_timeout"}'
```

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
| `GET` | `/v1/admin/slos` | Report route/application availability and latency objectives with error-budget burn. |
| `GET` | `/v1/admin/cost-anomalies` | Compare recent cost rates with the prior window and flag threshold breaches. |
| `GET` | `/v1/admin/routing-decisions` | Inspect per-request provider/model selection, fallback, and skip reasons. |
| `GET` | `/v1/admin/report.csv` | Export filtered request-level usage and cost records as CSV (up to 100,000 rows). |
| `POST` | `/v1/admin/cost-reconciliation` | Atomically import actual costs keyed by gateway request ID. |
| `POST` | `/v1/admin/cost-reconciliation/import` | Normalize CSV, JSON, or JSONL billing rows and reconcile atomically. |
| `GET` | `/v1/admin/budgets` | Current-month spend, limit, and utilization for configured applications. |
| `GET` | `/v1/admin/providers` | Provider circuit state and consecutive failure counts. |
| `GET` | `/v1/admin/audit` | Recent authenticated admin API access events. |
| `GET` | `/v1/admin/tool-audit` | Metadata-only tool-call decisions, optionally filtered by tenant. |
| `POST` | `/v1/admin/tool-approvals` | Record approval, denial, or revocation for a gated tool call. Requires the `tool_approvals` scope. |
| `GET` | `/v1/tool-approvals/{request_id}/{tool_call_id}` | Query approval state before the application executes a gated tool call. |
| `POST` | `/v1/tool-executions` | Report a caller-executed tool outcome for an authorized generated call. |
| `GET` | `/v1/admin/tool-executions` | Inspect reported execution outcomes, optionally filtered by tenant. |
| `GET` | `/v1/admin/tool-executions/pending` | Inspect generated tool calls without outcomes and identify overdue calls. |
| `POST` | `/v1/admin/evaluations` | Register an immutable evaluation dataset version and save a report. |
| `GET` | `/v1/admin/evaluations` | List saved evaluation reports by dataset and version. |
| `POST` | `/v1/admin/evaluations/compare` | Compare two saved runs by route and persist cost, latency, pass-rate, and failure gates. |
| `GET` | `/v1/admin/evaluations/comparisons` | List saved comparisons. |
| `GET` | `/v1/admin/shadow-reports` | Compare sampled candidate quality, cost, latency, failures, and shadow promotion gates. |
| `POST` | `/v1/admin/promotions/evaluate` | Combine a saved evaluation comparison, shadow gates, and current route/application SLOs. |
| `GET` | `/v1/admin/webhooks` | Inspect pending, delivered, and dead-letter budget alert events. |
| `POST` | `/v1/admin/webhooks/{event_id}/redrive` | Retry a dead-letter alert event. |
| `GET` | `/metrics` | Prometheus metrics in exposition format. |
| `GET` | `/docs` | FastAPI Swagger UI. |

When tenant credentials are configured, `/v1/chat/completions`, `/v1/routes`, `/v1/tool-approvals/*`, and `/v1/tool-executions` require a tenant key, while `/v1/admin/*` requires a `GATEWAY_API_KEY`, scoped admin key, or configured OIDC bearer token. Without tenant configuration, `GATEWAY_API_KEY` optionally protects `/v1` endpoints.

## Cost and reliability reporting

Query the aggregate report by time range and/or application:

```bash
curl -H 'Authorization: Bearer gateway-secret' \
  'http://localhost:8000/v1/admin/report?start=2026-10-01T00:00:00Z&end=2026-11-01T00:00:00Z&application=support-router'
```

`start` is inclusive, `end` is exclusive, and both accept ISO-8601 timestamps. The response includes aggregate requests, input/output/total tokens, estimated and reconciled cost totals, cost variance, failure and fallback rates, canary request counts, average latency, cache hits, and breakdowns by application, provider, model, route, cost center, and provider/model/pricing catalog version. Optional `cost_center`, `provider`, and `route` filters can be combined with time and application filters. `GET /v1/admin/report.csv` accepts the same filters and exports at most 100,000 request rows. `cost_usd` remains the estimate used for budgets; actual charges are stored separately.

### Route and application SLOs

Configure defaults and optional per-route or per-application overrides in `slos`:

```yaml
slos:
  default:
    availability_target: 0.99
    latency_target_ms: 1000
    latency_objective: 0.95
  applications:
    support-router:
      availability_target: 0.999
  routes:
    reasoning:
      latency_target_ms: 3000
```

`GET /v1/admin/slos?window_hours=24` reports observed availability, latency compliance, and each objective's error-budget burn rate and remaining percentage by application and route. Latency compliance measures requests within `latency_target_ms`; availability counts requests with status `success`. Example Prometheus rules are in [`deploy/prometheus/llm-gateway-slo.rules.yml`](deploy/prometheus/llm-gateway-slo.rules.yml). Load the file using Prometheus `rule_files`; the example alerts use a 14.4x short-window burn threshold.

### Context-aware canary routing

Deterministic `traffic_percent` canaries continue to work without a flag service. For dynamic targeting, configure an OFREP-compatible endpoint:

```yaml
routing:
  strategy: primary_first
  feature_flags:
    enabled: true
    base_url: https://flags.example.com
    flag_key: llm-gateway-canary
    bearer_token_env: OPENFEATURE_OFREP_BEARER_TOKEN
    timeout_seconds: 0.5
```

For each request with a configured route canary, the gateway posts to `/ofrep/v1/evaluate/flags/{flag_key}` with `targetingKey`, `application`, and (for authenticated tenants) `tenant_id`. A boolean `true` selects the canary. False values, timeouts, malformed responses, and provider errors retain the normal primary/fallback order and appear in routing-decision history. Request messages and user content are excluded from flag context and telemetry.

### GenAI telemetry

Spans include `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, and token usage attributes where available. Prometheus exposes the normalized GenAI inference metrics `gen_ai_client_inference_duration_seconds`, `gen_ai_client_inference_usage_input_tokens_total`, and `gen_ai_client_inference_usage_output_tokens_total`, with operation, provider, and model labels. The gateway does not emit GenAI input/output message attributes.

Cost anomaly detection compares successful requests in the latest window with the immediately preceding window of equal length. It groups by application, provider, model, and pricing version, and returns both per-request and USD per 1,000-token changes. Defaults are a 24-hour window, a 50% absolute-change threshold, and at least five requests in each window. Small samples are omitted; zero-cost baselines are flagged when the current cost becomes nonzero.

```bash
curl -H 'Authorization: Bearer gateway-secret' \
  'http://localhost:8000/v1/admin/cost-anomalies?window_hours=24&threshold_percent=50&min_requests=5'
```

Each completed request also records routing outcomes without prompt or response content. The `/v1/admin/report` response includes an aggregate breakdown by decision reason; `GET /v1/admin/routing-decisions?request_id=...` returns the ordered provider/model candidates, whether each was selected, used as a fallback, or skipped, the safe reason code, and retry count. Other supported filters include application, route, provider, and decision.

Import provider invoice rows as an atomic batch. Every request ID must exist in the ledger; unknown IDs reject the full batch. Re-importing a request replaces its prior actual cost and records the reconciliation source.

```bash
curl -X POST http://localhost:8000/v1/admin/cost-reconciliation \
  -H 'Authorization: Bearer gateway-secret' -H 'Content-Type: application/json' \
  -d '{"source":"invoice-2026-10","records":[{"request_id":"chatcmpl-123","actual_cost_usd":0.0047}]}'
```

For provider billing exports, use `/v1/admin/cost-reconciliation/import` with `format` set to `csv`, `json`, or `jsonl` and the file text in `content`. The importer recognizes `request_id`, `gateway_request_id`, or `llm_gateway_request_id` and `actual_cost_usd`, `cost_usd`, `amount_usd`, or `cost`. Provider-specific headers or nested JSON fields can be mapped explicitly; dot-separated JSON paths are supported. `cost_multiplier` converts values such as integer micro-USD to USD. If an export includes currency, only USD rows are accepted. Every row must carry the gateway request ID (for example as a provider metadata tag); provider-side request IDs cannot be matched automatically. Rows are validated as a batch before reconciliation.

```bash
curl -X POST http://localhost:8000/v1/admin/cost-reconciliation/import \
  -H 'Authorization: Bearer gateway-secret' -H 'Content-Type: application/json' \
  -d '{"source":"provider-invoice-2026-10","format":"csv","mapping":{"request_id":"gateway_request_id","actual_cost_usd":"billed_micros","currency":"currency"},"cost_multiplier":0.000001,"content":"gateway_request_id,billed_micros,currency\\ngw_abc,1250,USD"}'
```

Run an evaluation set against named logical routes to compare response checks, latency, and estimated spend. Cases without an expected text check contribute to latency/cost but not pass rate.

```jsonl
{"id":"refund-policy","route":"simple","messages":[{"role":"user","content":"When can I request a refund?"}],"expected_contains":["30 days"]}
{"id":"refund-reasoning","route":"reasoning","messages":[{"role":"user","content":"When can I request a refund?"}],"expected_contains":["30 days"]}
```

```bash
llm-gateway-eval prompts.jsonl --url http://localhost:8000 --output eval-report.json
```

The runner prints per-route pass rate, average and p95 latency, failures, and estimated cost. Expected-text checks are a simple deterministic baseline; use a curated dataset or external judge for semantic quality evaluation.

Give a dataset a stable version and save each run to the gateway for history. The dataset name/version pair is immutable: trying to reuse it with different file contents is rejected. The gateway stores the file SHA-256 and report, not the prompt set itself. Set `GATEWAY_ADMIN_API_KEY` or pass `--admin-api-key` with a key that has the `report` scope.

```bash
llm-gateway-eval prompts.jsonl --compare simple reasoning \
  --api-key "$GATEWAY_TENANT_KEY" --admin-api-key "$GATEWAY_ADMIN_API_KEY" \
  --dataset-name support-prompts --dataset-version 2026-10 \
  --persist --output route-comparison.json
```

Compare a candidate route with a baseline on the same prompts and fail the command if promotion thresholds are missed:

```bash
llm-gateway-eval prompts.jsonl --compare simple reasoning \
  --min-pass-rate 0.95 --max-cost-regression-percent 20 \
  --max-latency-regression-percent 15 --max-failure-rate 0.01 \
  --output route-comparison.json
```

The candidate is evaluated against the dataset's `expected_contains` and `expected_not_contains` checks. Cost and average latency regression are measured against the baseline; the failure-rate gate counts gateway/API errors. The command exits nonzero when a candidate check or configured gate fails. These deterministic checks are useful for rollout regression control, but do not replace semantic evaluation by a human or a separately managed judge.

### Shadow traffic and promotion gates

Configure a route's `shadow` candidate and `traffic_percent` to sample requests using a stable request-ID hash. The gateway sends a separate non-streaming request to the candidate after the primary response is selected; the shadow response is never returned to the caller and response content is not persisted. Reports compare paired output validity (non-empty output plus valid declared tool arguments or requested JSON shape), estimated cost, latency, and failure rate. This quality signal is a format/validity check, not a semantic judge.

`GET /v1/admin/shadow-reports?application=support-router&route=simple` returns the sample report and configured shadow gates. Configure `shadow.promotion.min_samples`, `max_quality_loss`, `max_cost_regression_percent`, `max_latency_regression_percent`, and `max_failure_rate` on the route. Candidate promotion combines those results with a saved evaluation comparison and live application/route SLOs:

```bash
curl -X POST http://localhost:8000/v1/admin/promotions/evaluate \
  -H 'Authorization: Bearer gateway-secret' -H 'Content-Type: application/json' \
  -d '{"comparison_id":"comparison-id","application":"support-router","route":"simple","window_hours":24}'
```

Promotion is fail-closed: it requires a passing saved evaluation comparison for the route, enough shadow samples with all configured gates passing, and observed requests meeting the current availability and latency objectives. Shadow observations contain only IDs, dimensions, counts, estimated costs, latency, and validity/failure metadata.

### Idempotent requests

Send an `Idempotency-Key` header to prevent caller retries from generating another completion or charge:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: support-ticket-9842' \
  -d '{"model":"simple","application":"support-router","messages":[{"role":"user","content":"Summarize this ticket"}]}'
```

The key is scoped to the tenant and application, and bound to the complete request body. Reusing it with a different body returns `409`. A concurrent or interrupted request also returns `409` on retry for the remainder of the configured TTL (`server.idempotency_ttl_seconds`, default 86,400 seconds), avoiding a second potentially billable generation. Completed JSON and SSE responses are replayed. Without Redis, records are process-local; configure Redis to share idempotency state across replicas. Because response bodies are retained for replay, choose a TTL and Redis retention policy appropriate for the sensitivity of completions.

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

Set `"stream": true` on a chat request to receive OpenAI-compatible SSE frames (`data: ...`) followed by `[DONE]`. Native streaming is supported for OpenAI-compatible, Anthropic, and Gemini APIs. The gateway can move to a fallback before sending text. After the first text delta, provider failures are returned as an SSE `error` event because replaying a second provider response could duplicate output. Token usage is included when the upstream sends usage metadata. If the client disconnects, the gateway cancels upstream iteration where supported, records the request as `cancelled`, and stores usage metadata received before cancellation; providers that only report usage at the end may have unknown or zero partial usage.

Requests may include OpenAI-style `tools`, `tool_choice`, and `response_format`. The gateway translates function declarations, prior assistant tool calls, and tool-result messages for Anthropic and Gemini as well as OpenAI-compatible endpoints. Tool calls are returned in the normalized assistant response; the gateway never runs the requested function. `response_format` accepts `json_object` or `json_schema`. Streaming tool calls are currently rejected. Requests containing tool definitions or tool-call history bypass response caching to avoid replaying stale actions.

For image input, use OpenAI-compatible content parts such as `{"type":"text","text":"Describe this"}` and `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`. OpenAI-compatible providers receive the parts unchanged; Anthropic and Gemini receive native image blocks. Remote image URLs are forwarded as provider URL references; Gemini deployments should use provider-supported file URIs or inline data URIs.

For audio input, include `{"type":"input_audio","input_audio":{"data":"<base64>","format":"wav"}}` in the message content. Audio output can be requested with `"modalities":["text","audio"]` and `"audio":{"voice":"alloy","format":"wav"}`. The provider capability registry rejects unsupported combinations before an upstream request and allows the route to try a compatible fallback. Provider-level `capabilities` mappings can override defaults.

With `stream: true` and `tools`, tool-call chunks use OpenAI-compatible `delta.tool_calls` fragments. The application still owns tool authorization and execution. Audio-producing routes may emit provider-native audio deltas when streaming is enabled and supported.

### Provider adapter configuration

Use `kind: azure_openai` with an Azure resource `base_url`, deployment name as the route model, `api_key_env`, and optional `api_version`. Use `kind: bedrock` with `region`; requests are signed with AWS Signature Version 4 using `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optional `AWS_SESSION_TOKEN`. Use `kind: vertex` with `project_id`, `region`, and an access-token environment variable (default `GOOGLE_CLOUD_ACCESS_TOKEN`). Use `kind: vllm` for a vLLM OpenAI-compatible endpoint. Bedrock supports Converse and ConverseStream requests, including normalized streaming tool calls and usage. If a streaming provider fails before any caller-visible text, tool, or audio delta, the gateway can use the next configured provider; after output begins, it stops rather than risk duplicated or conflicting content.

Provider capabilities appear in `GET /v1/routes` and `GET /v1/admin/providers`. A provider-level override can capture a known limitation, for example `capabilities: {audio_output: false}`. Keep overrides conservative and verify them against the selected upstream model.

Prometheus counters and histograms are exposed at `/metrics`. OpenTelemetry request and provider-attempt spans are created; set `OTEL_EXPORTER_OTLP_ENDPOINT` to export traces with OTLP over HTTP. Keep `/metrics` on a private network or protect it at the ingress if metrics are sensitive in your environment.

## Multi-replica deployment

Set `REDIS_URL` to enable shared Redis response caching, atomic per-provider/per-application sliding-window quotas, global in-flight concurrency leases, shared circuit-breaker state, and provider health/latency signals for routing. Redis must be available when the service starts; the gateway verifies the connection and fails startup if the configured backend is unavailable. Cache reads/writes and provider health writes fail open; quota and concurrency acquisition fail closed with HTTP 503 if Redis becomes unavailable. Concurrency leases renew while requests run and expire automatically after a process crash. Use a private Redis network and `rediss://` when TLS is required; cached completions remain readable to Redis operators until their TTL expires.

Set `GATEWAY_DATABASE` to a PostgreSQL DSN such as `postgresql://gateway:password@db:5432/llm_gateway` to use the PostgreSQL ledger and coordinate concurrent budget reservations across replicas. The service creates the required tables and indexes at startup. Use SQLite for single-instance local development.

When Redis is configured, circuit states and the single half-open recovery probe are shared across replicas. Latency and success/failure summaries are also shared. Set `routing.strategy: latency_aware` to order configured route candidates by their shared EWMA latency, penalized by failure rate. Use `/v1/admin/providers` to inspect circuit and performance signals. Deploy behind TLS, keep Redis and PostgreSQL private, and protect admin and metrics endpoints at the network edge.

## Security and privacy

- Set `GATEWAY_API_KEY` to protect all `/v1` endpoints with bearer authentication.
- Configure `security.admin_api_keys` for scoped admin access. Supported scopes include `report`, `budgets`, `providers`, `webhooks`, `audit`, and `tool_approvals`. Rotate by enabling a replacement key before disabling the old entry; changing the configured key list requires a restart, while changing a referenced secret file takes effect on the next request.
- Configure `security.oidc` with issuer, audience, and JWKS URL to validate signed JWTs. Admin tokens need `llm-gateway:<scope>` or `admin:*`; client access uses the `client` scope when OIDC is enabled without tenant keys. Access tokens are validated offline, so revocation takes effect when the token expires unless the identity provider issues a replacement or rejection is reflected by a new JWKS key set.
- Authenticated admin calls are recorded in the audit table and exposed at `/v1/admin/audit`. The audit log stores actor, path, status, and time; it omits tokens, prompts, and request bodies.
- When using tenant credentials, configure separate admin credentials; tenant keys are restricted to allowed routes and applications.
- Keep provider credentials in environment variables or a secrets manager. Do not commit credentials or `.env` files.
- For live rotation, configure `api_key_file` on providers/admin keys, `access_token_file` for Vertex, `access_key_file`/`secret_key_file`/`session_token_file` for Bedrock, `secret_file` on tenants, or `security.gateway_api_key_file`. Point these at files maintained by a secrets manager integration such as a CSI secret mount or Vault Agent. File values take precedence over environment values and are re-read at each use; the gateway does not cache credentials.
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

- Gemini image URL inputs must use provider-supported file URIs or inline data URIs.
- Without Redis, rate limits, cache, concurrency limits, and circuit-breaker health are process-local.
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
- [x] Share concurrency leases and circuit-breaker state through Redis.
- [x] Add deterministic canary routing and a JSONL route evaluation CLI.
- [x] Import actual provider charges and report reconciled cost variance.
- [x] Re-read provider, tenant, gateway, and admin credentials from rotated secret files.
- [x] Enforce tenant provider/model/region allowlists and usage metadata retention policies.
- [x] Version evaluation datasets and persist route-comparison reports.
- [x] Record caller-reported tool execution outcomes against audited generated calls.
- [x] Compare saved evaluation runs by route with conservative pass-rate intervals and configurable promotion gates.
- [x] Track pending and overdue tool execution outcomes and expose Prometheus alert metrics.
- [x] Map gateway request IDs into provider-specific headers and nested body metadata for invoice correlation.
- [x] Detect per-request and per-token cost anomalies by application, provider, model, and pricing version.
- [x] Persist provider/model routing decisions and expose selection, fallback, and skip reasons in reports.
- [x] Add OpenTelemetry GenAI semantic attributes and token/duration metrics without message content.
- [x] Support optional context-aware canary selection through OpenFeature OFREP with fail-closed defaults.
- [x] Report route/application SLOs and provide Prometheus recording and alert rules.
- [x] Require external approval for configured high-impact tool calls and retain approver decisions in the audit trail.
- [x] Add a provider adapter conformance suite for streaming, tool calls, structured output, usage, and request ID propagation.
- [x] Stream Bedrock ConverseStream text, tool-call, and usage events; test fallback boundaries before and after caller-visible output.
- [x] Add expiring and revocable tool approvals, application webhooks, and a Python helper that checks approval before execution.
- [x] Sample shadow traffic and compare output validity, cost, latency, and failures; combine shadow, evaluation, and SLO gates for promotion.

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
