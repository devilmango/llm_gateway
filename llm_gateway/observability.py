from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()
REQUESTS = Counter("llm_gateway_requests_total", "Gateway requests by route and outcome", ["route", "status"], registry=REGISTRY)
TOKENS = Counter("llm_gateway_tokens_total", "Provider-reported tokens", ["route", "kind"], registry=REGISTRY)
COST = Counter("llm_gateway_cost_usd_total", "Estimated provider spend in USD", ["route"], registry=REGISTRY)
FALLBACKS = Counter("llm_gateway_fallbacks_total", "Requests served by a fallback provider", ["route"], registry=REGISTRY)
CACHE_HITS = Counter("llm_gateway_cache_hits_total", "Responses served from gateway cache", ["route"], registry=REGISTRY)
PROVIDER_REQUESTS = Counter("llm_gateway_provider_requests_total", "Provider request attempts by outcome", ["provider", "outcome"], registry=REGISTRY)
PROVIDER_RETRIES = Counter("llm_gateway_provider_retries_total", "Retries attempted against providers", ["provider"], registry=REGISTRY)
PROVIDER_CIRCUIT_STATE = Gauge("llm_gateway_provider_circuit_state", "Provider circuit state (closed=0, half_open=1, open=2)", ["provider", "operation"], registry=REGISTRY)
IN_FLIGHT = Gauge("llm_gateway_requests_in_flight", "Requests currently being handled", registry=REGISTRY)
TOOL_EXECUTIONS_PENDING = Gauge("llm_gateway_tool_executions_pending", "Generated tool calls without a reported outcome", registry=REGISTRY)
TOOL_EXECUTIONS_OVERDUE = Gauge("llm_gateway_tool_executions_overdue", "Generated tool calls past the execution outcome deadline", registry=REGISTRY)
LATENCY = Histogram("llm_gateway_request_duration_seconds", "End-to-end request duration", ["route", "status"], registry=REGISTRY,
                    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60))
PROVIDER_LATENCY = Histogram("llm_gateway_provider_request_duration_seconds", "Provider request duration by outcome", ["provider", "outcome"], registry=REGISTRY,
                             buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60))
WEBHOOK_DELIVERIES = Counter(
    "llm_gateway_webhook_deliveries_total", "Webhook delivery attempts by endpoint and outcome",
    ["endpoint", "event_type", "outcome"], registry=REGISTRY,
)
WEBHOOK_DELIVERY_LATENCY = Histogram(
    "llm_gateway_webhook_delivery_duration_seconds", "Webhook request duration by endpoint and outcome",
    ["endpoint", "event_type", "outcome"], registry=REGISTRY,
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10),
)
WEBHOOK_CIRCUIT_STATE = Gauge(
    "llm_gateway_webhook_circuit_state", "Webhook endpoint circuit state (closed=0, open=1)",
    ["endpoint"], registry=REGISTRY,
)
WEBHOOK_CIRCUIT_STATE.labels(endpoint="uninitialized").set(0)

# GenAI semantic-convention metric names and bounded attributes. These mirror the
# corresponding OTel instruments so Prometheus dashboards can use the same concepts.
GEN_AI_OPERATION_DURATION = Histogram(
    "gen_ai_client_inference_duration_seconds",
    "Duration of GenAI client inference operations",
    ["gen_ai_operation_name", "gen_ai_provider_name", "gen_ai_request_model"],
    registry=REGISTRY, buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
GEN_AI_INPUT_TOKENS = Counter(
    "gen_ai_client_inference_usage_input_tokens_total",
    "Number of input tokens used by GenAI client inference operations",
    ["gen_ai_operation_name", "gen_ai_provider_name", "gen_ai_request_model"],
    registry=REGISTRY,
)
GEN_AI_OUTPUT_TOKENS = Counter(
    "gen_ai_client_inference_usage_output_tokens_total",
    "Number of output tokens used by GenAI client inference operations",
    ["gen_ai_operation_name", "gen_ai_provider_name", "gen_ai_request_model"],
    registry=REGISTRY,
)
SLO_REQUESTS = Counter(
    "llm_gateway_slo_requests_total", "Requests counted toward configured SLOs",
    ["application", "route", "outcome"], registry=REGISTRY,
)
SLO_LATENCY_REQUESTS = Counter(
    "llm_gateway_slo_latency_requests_total", "Requests meeting or missing the configured latency objective",
    ["application", "route", "outcome"], registry=REGISTRY,
)
SLO_AVAILABILITY_TARGET = Gauge(
    "llm_gateway_slo_availability_target_ratio", "Configured availability objective",
    ["application", "route"], registry=REGISTRY,
)
SLO_LATENCY_TARGET = Gauge(
    "llm_gateway_slo_latency_target_seconds", "Configured latency objective threshold in seconds",
    ["application", "route"], registry=REGISTRY,
)
SLO_LATENCY_OBJECTIVE = Gauge(
    "llm_gateway_slo_latency_objective_ratio", "Configured fraction of requests within the latency threshold",
    ["application", "route"], registry=REGISTRY,
)


def gen_ai_provider_name(provider_config: dict) -> str:
    """Map gateway adapter kinds to the OpenTelemetry GenAI provider vocabulary."""
    kind = str(provider_config.get("kind", "")).lower()
    if kind in {"gemini", "google_genai"}:
        return "gcp.gemini"
    if kind == "vertex":
        return "gcp.vertex_ai"
    if kind in {"azure_openai", "azure"}:
        return "azure.ai.openai"
    if kind == "bedrock":
        return "aws.bedrock"
    return {
        "openai": "openai", "anthropic": "anthropic", "vllm": "openai",
    }.get(kind, kind or "unknown")


def initialize_tracing() -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "llm-gateway")}))
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    return provider


def metrics_response() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
