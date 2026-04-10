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
IN_FLIGHT = Gauge("llm_gateway_requests_in_flight", "Requests currently being handled", registry=REGISTRY)
LATENCY = Histogram("llm_gateway_request_duration_seconds", "End-to-end request duration", ["route", "status"], registry=REGISTRY,
                    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60))
PROVIDER_LATENCY = Histogram("llm_gateway_provider_request_duration_seconds", "Provider request duration by outcome", ["provider", "outcome"], registry=REGISTRY,
                             buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60))


def initialize_tracing() -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "llm-gateway")}))
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    return provider


def metrics_response() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
