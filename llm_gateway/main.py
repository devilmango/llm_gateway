from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from opentelemetry import trace

from .config import load_config
from .distributed import RateLimitBackendUnavailable
from .limiting import RateLimitExceeded
from .observability import (
    CACHE_HITS,
    COST,
    FALLBACKS,
    IN_FLIGHT,
    LATENCY,
    REQUESTS,
    TOKENS,
    initialize_tracing,
    metrics_response,
)
from .providers import ProviderError
from .redaction import redact
from .schemas import ChatChoice, ChatRequest, ChatResponse, Usage
from .service import GatewayService
from .storage import BudgetExceeded, create_usage_store

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("llm_gateway")
TRACER_PROVIDER = initialize_tracing()
TRACER = TRACER_PROVIDER.get_tracer("llm_gateway")


def create_app(config: dict[str, Any] | None = None,
               http_transport: httpx.AsyncBaseTransport | None = None,
               shared_redis: Any | None = None) -> FastAPI:
    cfg = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        timeout = httpx.Timeout(float(cfg["server"].get("request_timeout_seconds", 45)))
        app.state.http = httpx.AsyncClient(timeout=timeout, transport=http_transport)
        redis_client = shared_redis
        owns_redis = False
        app.state.store = None
        try:
            redis_url = cfg["server"].get("redis_url") or os.getenv("REDIS_URL")
            if redis_client is None and redis_url:
                from redis.asyncio import Redis

                redis_client = Redis.from_url(redis_url, decode_responses=True)
                owns_redis = True
                await redis_client.ping()
            app.state.redis = redis_client
            app.state.gateway = GatewayService(cfg, app.state.http, shared_redis=redis_client)
            database = os.getenv("GATEWAY_DATABASE", cfg["server"].get("database", "./gateway.sqlite3"))
            app.state.store = create_usage_store(database)
            yield
        finally:
            await app.state.http.aclose()
            if owns_redis and redis_client:
                await redis_client.aclose()
            close_store = getattr(app.state.store, "close", None) if app.state.store else None
            if close_store:
                await asyncio.to_thread(close_store)
            TRACER_PROVIDER.force_flush()

    app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)

    def configured_tenants() -> dict[str, Any]:
        return cfg.get("tenants", cfg.get("security", {}).get("tenants", {}))

    async def authenticate_admin(authorization: str | None = Header(default=None)) -> None:
        expected = os.getenv("GATEWAY_API_KEY")
        if configured_tenants() and not expected:
            raise HTTPException(status_code=503, detail="Admin API key is not configured")
        if expected:
            provided = authorization.removeprefix("Bearer ") if authorization else ""
            if not hmac.compare_digest(provided, expected):
                raise HTTPException(status_code=401, detail="Invalid gateway credentials")

    async def authenticate_client(authorization: str | None = Header(default=None)) -> dict[str, Any] | None:
        tenants = configured_tenants()
        if tenants:
            provided = authorization.removeprefix("Bearer ") if authorization else ""
            for tenant_id, tenant in tenants.items():
                env_name = tenant.get("api_key_env")
                expected = os.getenv(env_name, "") if env_name else ""
                if expected and hmac.compare_digest(provided, expected):
                    return {"id": tenant_id, "config": tenant}
            raise HTTPException(status_code=401, detail="Invalid tenant credentials")
        await authenticate_admin(authorization)
        return None

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        body, content_type = metrics_response()
        return Response(content=body, media_type=content_type)

    @app.get("/v1/routes")
    async def routes(tenant: dict[str, Any] | None = Depends(authenticate_client)) -> dict[str, Any]:  # noqa: B008
        allowed_routes = set(tenant["config"].get("allowed_routes", [])) if tenant else set(cfg["routes"])
        return {name: {"provider": route["provider"], "model": route["model"],
                       "fallbacks": route.get("fallbacks", []), "cache": route.get("cache", {})}
                for name, route in cfg["routes"].items() if name in allowed_routes}

    @app.post("/v1/chat/completions", response_model=None)
    async def chat(body: ChatRequest, request: Request,
                   tenant: dict[str, Any] | None = Depends(authenticate_client)) -> Any:  # noqa: B008
        request_id = "gw_" + uuid.uuid4().hex
        started = time.perf_counter()
        route_name = body.model
        messages = [{"role": message.role, "content": message.content} for message in body.messages]
        should_redact = bool(cfg.get("security", {}).get("redact_pii", True))
        if should_redact:
            messages = [{**message, "content": redact(message["content"])} for message in messages]
        upstream_payload = {"messages": messages}
        for source, destination in (("temperature", "temperature"), ("max_tokens", "max_tokens"), ("top_p", "top_p")):
            value = getattr(body, source)
            if value is not None:
                upstream_payload[destination] = value
        app_name = body.application.strip() or "default"
        if tenant:
            tenant_cfg = tenant["config"]
            if route_name not in tenant_cfg.get("allowed_routes", []):
                raise HTTPException(status_code=403, detail="Tenant is not permitted to use this route")
            tenant_apps = tenant_cfg.get("applications", {})
            if app_name not in tenant_apps:
                raise HTTPException(status_code=403, detail="Tenant is not permitted to use this application")
            ledger_app_name = f"{tenant['id']}:{app_name}"
        else:
            tenant_cfg, tenant_apps, ledger_app_name = {}, {}, app_name
        IN_FLIGHT.inc()
        span = TRACER.start_span("llm.gateway.request")
        span.set_attribute("llm.gateway.request_id", request_id)
        result: dict[str, Any] | None = None
        error_type: str | None = None
        reservation = 0.0
        policy: dict[str, Any] = {}
        monthly_limit: float | None = None
        stream_response_started = False
        try:
            budget_cfg = cfg.get("budgets", {})
            policy = {**budget_cfg.get("default", {}), **budget_cfg.get("applications", {}).get(app_name, {}),
                      **tenant_cfg.get("default_policy", {}), **tenant_apps.get(app_name, {})}
            app_rate = policy.get("max_requests_per_minute")
            if app_rate is not None:
                await request.app.state.gateway.acquire_application_quota(ledger_app_name, int(app_rate))
            monthly_limit = policy.get("monthly_limit_usd")
            if monthly_limit is not None:
                reservation = request.app.state.gateway.budget_reservation(route_name, upstream_payload, policy)
                await asyncio.to_thread(request.app.state.store.reserve_budget, request_id, ledger_app_name,
                                        reservation, float(monthly_limit))
            if body.stream:
                async def event_stream():
                    stream_result: dict[str, Any] | None = None
                    stream_error: str | None = None
                    status = "success"
                    text_delta = 0
                    try:
                        yield "data: " + json.dumps({"id": request_id, "object": "chat.completion.chunk",
                            "created": int(time.time()), "model": route_name,
                            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}) + "\n\n"
                        with trace.use_span(span, end_on_exit=False):
                            async for item in request.app.state.gateway.stream_complete(route_name, upstream_payload):
                                if item.get("final"):
                                    stream_result = item
                                    usage = item.get("usage", {})
                                    yield "data: " + json.dumps({"id": request_id, "object": "chat.completion.chunk",
                                        "created": int(time.time()), "model": route_name,
                                        "choices": [{"index": 0, "delta": {}, "finish_reason": item.get("finish_reason")}],
                                        "usage": usage}) + "\n\n"
                                    continue
                                delta = item.get("delta", "")
                                text_delta += len(delta)
                                yield "data: " + json.dumps({"id": request_id, "object": "chat.completion.chunk",
                                    "created": int(time.time()), "model": route_name,
                                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]}) + "\n\n"
                        yield "data: [DONE]\n\n"
                    except ProviderError:
                        status, stream_error = "failure", "upstream_error"
                        yield "event: error\ndata: " + json.dumps({"error": {"message": "All configured providers failed",
                            "type": "upstream_error", "request_id": request_id}}) + "\n\n"
                    except Exception as exc:
                        status, stream_error = "failure", type(exc).__name__
                        logger.exception("stream_request_failed request_id=%s", request_id)
                        yield "event: error\ndata: " + json.dumps({"error": {"message": "Gateway stream failed",
                            "type": "gateway_error", "request_id": request_id}}) + "\n\n"
                    finally:
                        elapsed_ms = int((time.perf_counter() - started) * 1000)
                        usage_data = stream_result.get("usage", {}) if stream_result else {}
                        provider_name = stream_result.get("provider") if stream_result else None
                        model_name = stream_result.get("model") if stream_result else None
                        configured_model = stream_result.get("configured_model", model_name) if stream_result else model_name
                        price = cfg["providers"].get(provider_name, {}).get("models", {}).get(configured_model, {}) if provider_name else {}
                        cost_value = (usage_data.get("prompt_tokens", 0) * float(price.get("input_cost_per_1m", 0)) +
                                      usage_data.get("completion_tokens", 0) * float(price.get("output_cost_per_1m", 0))) / 1_000_000
                        record = {"request_id": request_id, "occurred_at": datetime.now(UTC).isoformat(),
                            "application": ledger_app_name, "route": route_name, "provider": provider_name,
                            "model": model_name, "prompt_tokens": usage_data.get("prompt_tokens", 0),
                            "completion_tokens": usage_data.get("completion_tokens", 0),
                            "total_tokens": usage_data.get("total_tokens", 0), "cost_usd": cost_value,
                            "status": status, "latency_ms": elapsed_ms,
                            "fallback_used": int(bool(stream_result and stream_result.get("fallback_used"))),
                            "cached": int(bool(stream_result and stream_result.get("cached"))), "error_type": stream_error}
                        await asyncio.to_thread(request.app.state.store.record, record)
                        metric_route = route_name if route_name in cfg["routes"] else "unknown"
                        REQUESTS.labels(route=metric_route, status=status).inc()
                        TOKENS.labels(route=metric_route, kind="prompt").inc(record["prompt_tokens"])
                        TOKENS.labels(route=metric_route, kind="completion").inc(record["completion_tokens"])
                        COST.labels(route=metric_route).inc(cost_value)
                        if record["fallback_used"]:
                            FALLBACKS.labels(route=metric_route).inc()
                        if record["cached"]:
                            CACHE_HITS.labels(route=metric_route).inc()
                        LATENCY.labels(route=metric_route, status=status).observe(elapsed_ms / 1000)
                        if monthly_limit is not None:
                            thresholds = [float(value) for value in policy.get("alert_thresholds", [0.8, 1.0])]
                            alerts = await asyncio.to_thread(request.app.state.store.check_budget_alerts,
                                ledger_app_name, float(monthly_limit), thresholds)
                            for alert in alerts:
                                logger.warning("budget_alert application=%s threshold=%s spend_usd=%.8f limit_usd=%.8f",
                                    redact(ledger_app_name), alert["threshold"], alert["spend_usd"], alert["limit_usd"])
                        span.set_attribute("llm.gateway.status", status)
                        span.set_attribute("llm.gateway.application", ledger_app_name)
                        if stream_error:
                            span.set_attribute("error.type", stream_error)
                        span.end()
                        IN_FLIGHT.dec()
                        logger.info("stream_request_id=%s application=%s route=%s status=%s latency_ms=%s tokens=%s cost_usd=%.8f error=%s",
                            request_id, redact(ledger_app_name), route_name, status, elapsed_ms,
                            record["total_tokens"], cost_value, stream_error)

                stream_response_started = True
                return StreamingResponse(event_stream(), media_type="text/event-stream",
                                         headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            with trace.use_span(span, end_on_exit=False):
                result = await request.app.state.gateway.complete(route_name, upstream_payload)
            usage = result["usage"]
            provider = result["provider"]
            model = result["model"]
            price = cfg["providers"][provider].get("models", {}).get(result.get("configured_model", model), {})
            cost = (usage["prompt_tokens"] * float(price.get("input_cost_per_1m", 0)) +
                    usage["completion_tokens"] * float(price.get("output_cost_per_1m", 0))) / 1_000_000
            status = "success"
            span.set_attribute("llm.gateway.provider", provider)
            span.set_attribute("llm.gateway.model", model)
            span.set_attribute("llm.gateway.fallback_used", bool(result["fallback_used"]))
            span.set_attribute("llm.gateway.cached", bool(result["cached"]))
            span.set_attribute("llm.usage.total_tokens", usage["total_tokens"])
            span.set_attribute("llm.cost.usd", cost)
            output = ChatResponse(id=request_id, created=int(time.time()), model=route_name,
                choices=[ChatChoice(message={"role": "assistant", "content": result["text"]}, finish_reason=result.get("finish_reason"))],
                usage=Usage(**usage), gateway={"provider": provider, "upstream_model": model,
                    "fallback_used": result["fallback_used"], "cached": result["cached"], "cost_usd": round(cost, 8)})
            return output
        except KeyError:
            status, error_type = "failure", "unknown_route"
            raise HTTPException(status_code=400, detail=f"Unknown route {route_name!r}")
        except ValueError as exc:
            status, error_type = "rejected", "request_limit_exceeded"
            raise HTTPException(status_code=400, detail=str(exc))
        except BudgetExceeded as exc:
            status, error_type = "rejected", "monthly_budget_exceeded"
            raise HTTPException(status_code=429, detail="Application monthly spend limit reached") from exc
        except RateLimitExceeded as exc:
            status, error_type = "rejected", "application_rate_limit_exceeded"
            raise HTTPException(status_code=429, detail="Application request quota exceeded") from exc
        except RateLimitBackendUnavailable as exc:
            status, error_type = "failure", "shared_rate_limit_store_unavailable"
            raise HTTPException(status_code=503, detail="Shared rate-limit store is unavailable") from exc
        except ProviderError as exc:
            status, error_type = "failure", "upstream_error"
            raise HTTPException(status_code=502, detail="All configured providers failed") from exc
        except Exception as exc:
            status, error_type = "failure", type(exc).__name__
            raise
        finally:
            if not stream_response_started:
                # Persist every attempt outcome; never persist prompt or completion content.
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                usage_data = result.get("usage", {}) if result else {}
                provider_name = result.get("provider") if result else None
                model_name = result.get("model") if result else None
                configured_model = result.get("configured_model", model_name) if result else model_name
                report_price = cfg["providers"].get(provider_name, {}).get("models", {}).get(configured_model, {}) if provider_name else {}
                cost_value = (usage_data.get("prompt_tokens", 0) * float(report_price.get("input_cost_per_1m", 0)) +
                              usage_data.get("completion_tokens", 0) * float(report_price.get("output_cost_per_1m", 0))) / 1_000_000
                record = {"request_id": request_id, "occurred_at": datetime.now(UTC).isoformat(),
                    "application": ledger_app_name, "route": route_name, "provider": provider_name, "model": model_name,
                    "prompt_tokens": usage_data.get("prompt_tokens", 0), "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0), "cost_usd": cost_value,
                    "status": locals().get("status", "failure"), "latency_ms": elapsed_ms,
                    "fallback_used": int(bool(result and result.get("fallback_used"))),
                    "cached": int(bool(result and result.get("cached"))), "error_type": error_type}
                await asyncio.to_thread(request.app.state.store.record, record)
                metric_route = route_name if route_name in cfg["routes"] else "unknown"
                REQUESTS.labels(route=metric_route, status=record["status"]).inc()
                TOKENS.labels(route=metric_route, kind="prompt").inc(record["prompt_tokens"])
                TOKENS.labels(route=metric_route, kind="completion").inc(record["completion_tokens"])
                COST.labels(route=metric_route).inc(cost_value)
                if record["fallback_used"]:
                    FALLBACKS.labels(route=metric_route).inc()
                if record["cached"]:
                    CACHE_HITS.labels(route=metric_route).inc()
                LATENCY.labels(route=metric_route, status=record["status"]).observe(elapsed_ms / 1000)
                if monthly_limit is not None:
                    alert_thresholds = policy.get("alert_thresholds", [0.8, 1.0])
                    alerts = await asyncio.to_thread(request.app.state.store.check_budget_alerts,
                        ledger_app_name, float(monthly_limit), [float(value) for value in alert_thresholds])
                    for alert in alerts:
                        logger.warning("budget_alert application=%s threshold=%s spend_usd=%.8f limit_usd=%.8f",
                            redact(ledger_app_name), alert["threshold"], alert["spend_usd"], alert["limit_usd"])
                span.set_attribute("llm.gateway.status", record["status"])
                span.set_attribute("llm.gateway.application", app_name)
                if error_type:
                    span.set_attribute("error.type", error_type)
                span.end()
                IN_FLIGHT.dec()
                logger.info("request_id=%s application=%s route=%s provider=%s model=%s status=%s latency_ms=%s tokens=%s cost_usd=%.8f fallback=%s cached=%s error=%s",
                    request_id, redact(ledger_app_name), route_name, provider_name, model_name, record["status"], elapsed_ms,
                    record["total_tokens"], cost_value, record["fallback_used"], record["cached"], error_type)
    @app.get("/v1/admin/report", dependencies=[Depends(authenticate_admin)])
    async def report(request: Request, start: str | None = Query(default=None), end: str | None = Query(default=None),
                     application: str | None = Query(default=None)) -> dict[str, Any]:
        """UTC ISO-8601 bounds; end is exclusive. Example: ?start=2026-10-01T00:00:00Z"""
        try:
            return await asyncio.to_thread(request.app.state.store.report, start, end, application)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid report filters") from exc

    @app.get("/v1/admin/budgets", dependencies=[Depends(authenticate_admin)])
    async def budgets(request: Request) -> dict[str, Any]:
        """Return current UTC calendar-month spend and configured application limits."""
        configured = cfg.get("budgets", {})
        applications = configured.get("applications", {})
        defaults = configured.get("default", {})
        month_start = datetime.now(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        budgets_out = []
        for application, override in applications.items():
            policy = {**defaults, **override}
            limit = policy.get("monthly_limit_usd")
            spent = 0.0
            if limit is not None:
                report = await asyncio.to_thread(request.app.state.store.report,
                    month_start.isoformat(), None, application)
                spent = float(report["cost_usd"])
            budgets_out.append({"application": application, "monthly_limit_usd": limit,
                "monthly_spend_usd": round(spent, 8),
                "remaining_usd": round(max(0.0, float(limit) - spent), 8) if limit is not None else None,
                "utilization": spent / float(limit) if limit not in (None, 0) else None,
                "max_requests_per_minute": policy.get("max_requests_per_minute")})
        for tenant_id, tenant in configured_tenants().items():
            for application, override in tenant.get("applications", {}).items():
                policy = {**defaults, **applications.get(application, {}),
                          **tenant.get("default_policy", {}), **override}
                limit = policy.get("monthly_limit_usd")
                scoped_application = f"{tenant_id}:{application}"
                spent = 0.0
                if limit is not None:
                    report = await asyncio.to_thread(request.app.state.store.report,
                        month_start.isoformat(), None, scoped_application)
                    spent = float(report["cost_usd"])
                budgets_out.append({"tenant": tenant_id, "application": application,
                    "monthly_limit_usd": limit, "monthly_spend_usd": round(spent, 8),
                    "remaining_usd": round(max(0.0, float(limit) - spent), 8) if limit is not None else None,
                    "utilization": spent / float(limit) if limit not in (None, 0) else None,
                    "max_requests_per_minute": policy.get("max_requests_per_minute")})
        return {"period_start": month_start.isoformat(), "applications": budgets_out}

    @app.get("/v1/admin/providers", dependencies=[Depends(authenticate_admin)])
    async def provider_health(request: Request) -> dict[str, Any]:
        states = await request.app.state.gateway.breaker.snapshot()
        performance = await request.app.state.gateway.health_snapshot()
        return {"providers": {name: {"kind": provider["kind"], **states.get(name, {"state": "closed", "failures": 0})}
                              | performance.get(name, {})
                              for name, provider in cfg["providers"].items()}}

    @app.exception_handler(HTTPException)
    async def safe_http_errors(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return app


app = create_app()


def run() -> None:
    import uvicorn
    uvicorn.run("llm_gateway.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
