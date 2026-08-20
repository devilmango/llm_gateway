from __future__ import annotations

import asyncio
import csv
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from io import StringIO
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from opentelemetry import trace

from .billing_import import normalize_billing_export
from .capabilities import provider_capabilities
from .config import load_config, validate_tenant_policies
from .dashboard import OPERATIONS_DASHBOARD
from .distributed import RateLimitBackendUnavailable
from .idempotency import IdempotencyRegistry
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
from .schemas import (
    ChatChoice,
    ChatRequest,
    ChatResponse,
    CostReconciliationImportRequest,
    CostReconciliationRequest,
    Usage,
)
from .secrets import SecretUnavailable, resolve_secret
from .service import GatewayService
from .storage import BudgetExceeded, create_usage_store
from .tool_governance import (
    ToolGovernanceError,
    effective_tool_policy,
    generated_tool_names,
    requested_tool_names,
)
from .tool_validation import validate_tool_calls

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("llm_gateway")
TRACER_PROVIDER = initialize_tracing()
TRACER = TRACER_PROVIDER.get_tracer("llm_gateway")


async def _deliver_webhook_batch(app: FastAPI, batch_size: int = 20) -> None:
    store = app.state.store
    if store is None:
        return
    events = await asyncio.to_thread(store.due_webhooks, batch_size)
    for event in events:
        try:
            body = event["body"].encode("utf-8")
            headers = {"Content-Type": "application/json", "Idempotency-Key": event["event_id"]}
            secret = os.getenv(event.get("secret_env") or "", "")
            if secret:
                signature = hmac.new(secret.encode("utf-8"), body, "sha256").hexdigest()
                headers["X-LLM-Gateway-Signature"] = f"sha256={signature}"
            response = await app.state.http.post(event["webhook_url"], content=body,
                                                  headers=headers, timeout=3.0)
            attempts = int(event["attempts"]) + 1
            if not response.is_error:
                await asyncio.to_thread(store.update_webhook, event["event_id"], attempts=attempts,
                    status="delivered", delivered_at=datetime.now(UTC), last_error=None)
            elif response.status_code < 500 and response.status_code != 429:
                await asyncio.to_thread(store.update_webhook, event["event_id"], attempts=attempts,
                    status="dead_letter", last_error=f"HTTP{response.status_code}")
            else:
                status = "dead_letter" if attempts >= 8 else "pending"
                delay = min(300, 2 ** min(attempts, 8))
                await asyncio.to_thread(store.update_webhook, event["event_id"], attempts=attempts,
                    status=status, next_attempt_at=datetime.now(UTC) + timedelta(seconds=delay),
                    last_error=f"HTTP{response.status_code}")
        except (httpx.HTTPError, ValueError) as exc:
            attempts = int(event["attempts"]) + 1
            status = "dead_letter" if attempts >= 8 else "pending"
            delay = min(300, 2 ** min(attempts, 8))
            await asyncio.to_thread(store.update_webhook, event["event_id"], attempts=attempts,
                status=status, next_attempt_at=datetime.now(UTC) + timedelta(seconds=delay),
                last_error=type(exc).__name__)


async def _webhook_delivery_loop(app: FastAPI) -> None:
    while True:
        try:
            await _deliver_webhook_batch(app)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("webhook_delivery_batch_failed")
        await asyncio.sleep(1)


async def _tenant_retention_loop(app: FastAPI, tenants: dict[str, Any]) -> None:
    while True:
        for tenant_id, tenant in tenants.items():
            retention_days = tenant.get("data_policy", {}).get("usage_retention_days")
            if retention_days is None:
                continue
            try:
                deleted = await asyncio.to_thread(app.state.store.prune_tenant_requests,
                                                  tenant_id, int(retention_days))
                if deleted:
                    logger.info("tenant_usage_retention tenant=%s deleted=%s", redact(tenant_id), deleted)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("tenant_usage_retention_failed tenant=%s", redact(tenant_id))
        await asyncio.sleep(24 * 60 * 60)


def create_app(config: dict[str, Any] | None = None,
               http_transport: httpx.AsyncBaseTransport | None = None,
               shared_redis: Any | None = None) -> FastAPI:
    cfg = config or load_config()
    validate_tenant_policies(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        timeout = httpx.Timeout(float(cfg["server"].get("request_timeout_seconds", 45)))
        app.state.http = httpx.AsyncClient(timeout=timeout, transport=http_transport)
        app.state.webhook_task = None
        app.state.retention_task = None
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
            app.state.idempotency = IdempotencyRegistry(
                redis_client, int(cfg.get("server", {}).get("idempotency_ttl_seconds", 86_400)))
            app.state.gateway = GatewayService(cfg, app.state.http, shared_redis=redis_client)
            database = os.getenv("GATEWAY_DATABASE", cfg["server"].get("database", "./gateway.sqlite3"))
            app.state.store = create_usage_store(database)
            app.state.webhook_task = asyncio.create_task(_webhook_delivery_loop(app))
            app.state.retention_task = asyncio.create_task(
                _tenant_retention_loop(app, cfg.get("tenants", cfg.get("security", {}).get("tenants", {}))))
            yield
        finally:
            for task in (app.state.webhook_task, app.state.retention_task):
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
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

    async def configured_secret(env_name: str | None, file_path: str | None) -> str:
        try:
            return await resolve_secret(env_name, file_path)
        except SecretUnavailable as exc:
            raise HTTPException(status_code=503, detail="Configured credential is unavailable") from exc

    async def validate_admin(authorization: str | None, scope: str = "") -> str:
        provided = authorization.removeprefix("Bearer ") if authorization else ""
        oidc = cfg.get("security", {}).get("oidc", {})
        if oidc.get("enabled"):
            if not provided:
                raise HTTPException(status_code=401, detail="Bearer token required")
            try:
                import jwt
                jwks = jwt.PyJWKClient(oidc["jwks_url"], cache_jwk_set=True)
                signing_key = jwks.get_signing_key_from_jwt(provided)
                claims = jwt.decode(provided, signing_key.key,
                    algorithms=oidc.get("algorithms", ["RS256"]),
                    issuer=oidc["issuer"], audience=oidc["audience"],
                    options={"require": ["exp", "sub"]})
            except ImportError as exc:
                raise HTTPException(status_code=503, detail="OIDC support dependency is unavailable") from exc
            except Exception as exc:
                raise HTTPException(status_code=401, detail="Invalid OIDC token") from exc
            role_claim = claims.get("roles", [])
            if isinstance(role_claim, str):
                role_claim = [role_claim]
            scopes = set(str(claims.get("scope", "")).split()) | set(role_claim)
            if scope and not ({scope, f"llm-gateway:{scope}", "admin:*"} & scopes):
                raise HTTPException(status_code=403, detail="OIDC principal lacks required scope")
            return str(claims.get("sub", "oidc-principal"))
        key_records = cfg.get("security", {}).get("admin_api_keys", [])
        if key_records:
            for record in key_records:
                expected = await configured_secret(record.get("api_key_env"), record.get("api_key_file")) \
                    if record.get("enabled", True) else ""
                if expected and hmac.compare_digest(provided, expected):
                    scopes = set(record.get("scopes", []))
                    if "*" not in scopes and scope not in scopes:
                        raise HTTPException(status_code=403, detail="Admin key lacks required scope")
                    return str(record.get("id", "scoped-admin-key"))
            raise HTTPException(status_code=401, detail="Invalid gateway credentials")
        expected = await configured_secret("GATEWAY_API_KEY", cfg.get("security", {}).get("gateway_api_key_file"))
        if configured_tenants() and not expected:
            raise HTTPException(status_code=503, detail="Admin API key is not configured")
        if expected and not hmac.compare_digest(provided, expected):
            raise HTTPException(status_code=401, detail="Invalid gateway credentials")
        return "gateway-api-key" if expected else "anonymous-admin"

    def admin_dependency(scope: str):
        async def dependency(request: Request, authorization: str | None = Header(default=None)) -> None:
            request.state.admin_actor = await validate_admin(authorization, scope)
        return dependency

    async def authenticate_client(authorization: str | None = Header(default=None)) -> dict[str, Any] | None:
        tenants = configured_tenants()
        if tenants:
            provided = authorization.removeprefix("Bearer ") if authorization else ""
            for tenant_id, tenant in tenants.items():
                expected = await configured_secret(tenant.get("api_key_env"),
                                                   tenant.get("secret_file", tenant.get("api_key_file")))
                if expected and hmac.compare_digest(provided, expected):
                    return {"id": tenant_id, "config": tenant}
            raise HTTPException(status_code=401, detail="Invalid tenant credentials")
        await validate_admin(authorization, "client")
        return None

    async def publish_budget_alerts(request: Request, alerts: list[dict[str, Any]], policy: dict[str, Any],
                                    application: str, request_id: str) -> None:
        webhook_url = policy.get("alert_webhook_url")
        secret_env = policy.get("alert_webhook_secret_env")
        for alert in alerts:
            safe_application = redact(application)
            logger.warning("budget_alert application=%s threshold=%s spend_usd=%.8f limit_usd=%.8f",
                safe_application, alert["threshold"], alert["spend_usd"], alert["limit_usd"])
            if not webhook_url:
                continue
            event = {"event": "budget.threshold_crossed", "request_id": request_id,
                "occurred_at": datetime.now(UTC).isoformat(), "application": safe_application,
                "threshold": alert["threshold"], "spend_usd": alert["spend_usd"],
                "limit_usd": alert["limit_usd"], "currency": "USD", "window": "utc_calendar_month"}
            alert_identity = f"{application}:{event['occurred_at'][:7]}:{alert['threshold']}"
            alert_id = hashlib.sha256(alert_identity.encode("utf-8")).hexdigest()
            event["alert_id"] = alert_id
            await asyncio.to_thread(request.app.state.store.enqueue_webhook, event, webhook_url, secret_env)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
    async def operations_dashboard() -> HTMLResponse:
        return HTMLResponse(OPERATIONS_DASHBOARD)

    @app.middleware("http")
    async def audit_admin_requests(request: Request, call_next):
        response = await call_next(request)
        actor = getattr(request.state, "admin_actor", None)
        if actor and request.url.path.startswith("/v1/admin/") and request.app.state.store:
            try:
                await asyncio.to_thread(request.app.state.store.record_audit, actor,
                    f"{request.method} {request.url.path}", response.status_code)
            except Exception:
                logger.exception("admin_audit_record_failed")
        return response

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        body, content_type = metrics_response()
        return Response(content=body, media_type=content_type)

    @app.get("/v1/routes")
    async def routes(request: Request,
                     tenant: dict[str, Any] | None = Depends(authenticate_client)) -> dict[str, Any]:  # noqa: B008
        allowed_routes = set(tenant["config"].get("allowed_routes", [])) if tenant else set(cfg["routes"])
        visible = {}
        for name, route in cfg["routes"].items():
            if name not in allowed_routes:
                continue
            try:
                candidates = request.app.state.gateway.route_candidates(
                    name, request_id="route-catalog", tenant_policy=tenant["config"] if tenant else None,
                    include_canary=False)
            except ValueError:
                continue
            primary, *fallbacks = candidates
            visible[name] = {**primary,
                "capabilities": provider_capabilities(cfg["providers"][primary["provider"]]),
                "fallbacks": [{**candidate,
                    "capabilities": provider_capabilities(cfg["providers"][candidate["provider"]])}
                    for candidate in fallbacks], "cache": route.get("cache", {})}
        return visible

    @app.post("/v1/chat/completions", response_model=None)
    async def chat(body: ChatRequest, request: Request,
                   idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
                   tenant: dict[str, Any] | None = Depends(authenticate_client)) -> Any:  # noqa: B008
        request_id = "gw_" + uuid.uuid4().hex
        started = time.perf_counter()
        route_name = body.model
        messages = [message.model_dump(exclude_none=True) for message in body.messages]
        should_redact = bool(cfg.get("security", {}).get("redact_pii", True))
        if should_redact:
            messages = [{**message, "content": redact(message["content"])}
                        if isinstance(message.get("content"), str) else
                        {**message, "content": [({**part, "text": redact(part["text"])}
                            if part.get("type") == "text" else part) for part in message["content"]]}
                        if isinstance(message.get("content"), list) else message for message in messages]
        upstream_payload = {"messages": messages}
        for source, destination in (("temperature", "temperature"), ("max_tokens", "max_tokens"), ("top_p", "top_p")):
            value = getattr(body, source)
            if value is not None:
                upstream_payload[destination] = value
        for field in ("tools", "tool_choice", "response_format", "modalities", "audio"):
            value = getattr(body, field)
            if value is not None:
                upstream_payload[field] = value
        app_name = body.application.strip() or "default"
        cost_center = body.cost_center.strip() if body.cost_center else None
        if tenant:
            tenant_cfg = {**tenant["config"], "_tenant_id": tenant["id"]}
            if route_name not in tenant_cfg.get("allowed_routes", []):
                raise HTTPException(status_code=403, detail="Tenant is not permitted to use this route")
            tenant_apps = tenant_cfg.get("applications", {})
            if app_name not in tenant_apps:
                raise HTTPException(status_code=403, detail="Tenant is not permitted to use this application")
            ledger_app_name = f"{tenant['id']}:{app_name}"
        else:
            tenant_cfg, tenant_apps, ledger_app_name = {}, {}, app_name
        idempotency_scope = None
        idempotency_fingerprint = None
        idempotency_claimed = False
        idempotency_complete = False
        if idempotency_key is not None:
            if (not idempotency_key.strip() or len(idempotency_key) > 200
                    or any(ord(char) < 33 or ord(char) > 126 for char in idempotency_key)):
                raise HTTPException(status_code=400, detail="Idempotency-Key must be 1-200 visible ASCII characters")
            idempotency_scope = f"{tenant['id'] if tenant else 'default'}:{ledger_app_name}:{idempotency_key}"
            request_fingerprint = json.dumps(body.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
            idempotency_fingerprint = hashlib.sha256(request_fingerprint.encode("utf-8")).hexdigest()
            owner, previous = await request.app.state.idempotency.claim(idempotency_scope, idempotency_fingerprint)
            if not owner:
                if not previous or previous.get("fingerprint") != idempotency_fingerprint:
                    raise HTTPException(status_code=409, detail="Idempotency-Key is already used by another request")
                if previous.get("state") != "complete":
                    raise HTTPException(status_code=409, detail="Request with this Idempotency-Key is in progress or incomplete")
                saved = previous["response"]
                if saved["kind"] == "stream":
                    return StreamingResponse(iter([saved["body"]]), media_type="text/event-stream",
                        headers=saved.get("headers", {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}))
                return JSONResponse(saved["body"], status_code=saved.get("status_code", 200),
                                    headers=saved.get("headers", {}))
            idempotency_claimed = True
        if tenant:
            try:
                request.app.state.gateway.route_candidates(route_name, request_id, tenant_cfg)
            except (KeyError, ValueError) as exc:
                raise HTTPException(status_code=403,
                    detail="Tenant data policy does not permit this provider/model route") from exc
        IN_FLIGHT.inc()
        span = TRACER.start_span("llm.gateway.request")
        span.set_attribute("llm.gateway.request_id", request_id)
        result: dict[str, Any] | None = None
        error_type: str | None = None
        reservation = 0.0
        policy: dict[str, Any] = {}
        monthly_limit: float | None = None
        stream_response_started = False
        tool_audit_records: list[dict[str, Any]] = []
        try:
            budget_cfg = cfg.get("budgets", {})
            policy = {**budget_cfg.get("default", {}), **budget_cfg.get("applications", {}).get(app_name, {}),
                      **tenant_cfg.get("default_policy", {}), **tenant_apps.get(app_name, {})}
            allowed_tools, max_tool_calls, max_tool_calls_per_minute = effective_tool_policy(
                cfg.get("tool_governance", {}), tenant_cfg, policy)

            async def audit_generated_tool_calls(names: list[str]) -> None:
                if not names:
                    return
                denied = []
                reason = None
                if any(not name for name in names):
                    denied = [name for name in names if not name]
                    reason = "missing_tool_name"
                elif max_tool_calls is not None and len(names) > max_tool_calls:
                    reason = "max_tool_calls_per_request"
                elif allowed_tools is not None:
                    denied = [name for name in names if name not in allowed_tools]
                    if denied:
                        reason = "tool_not_allowed"
                if reason:
                    tool_audit_records.extend({"tool_name": name or "<missing>", "outcome": "denied",
                        "reason": reason if reason == "max_tool_calls_per_request" or name in denied else "request_blocked"}
                        for name in names)
                    detail = ("Provider returned more tool calls than the configured per-request limit"
                              if reason == "max_tool_calls_per_request" else
                              "Provider returned a tool not allowed by gateway policy")
                    raise ToolGovernanceError(detail, 502, reason)
                if max_tool_calls_per_minute is not None:
                    try:
                        for _ in names:
                            await request.app.state.gateway.acquire_application_quota(
                                f"{ledger_app_name}:tool-calls", max_tool_calls_per_minute)
                    except RateLimitExceeded as exc:
                        tool_audit_records.extend({"tool_name": name or "<missing>",
                            "outcome": "denied", "reason": "max_tool_calls_per_minute"} for name in names)
                        raise ToolGovernanceError("Tenant tool-call rate limit exceeded", 429,
                                                  "max_tool_calls_per_minute") from exc
                tool_audit_records.extend({"tool_name": name or "<missing>", "outcome": "allowed", "reason": None}
                                          for name in names)

            requested_names = requested_tool_names(body.tools)
            forbidden_requested = ([name for name in requested_names
                                    if allowed_tools is not None and name not in allowed_tools])
            if forbidden_requested:
                tool_audit_records.extend({"tool_name": name, "outcome": "denied",
                    "reason": "tool_not_allowed"} for name in forbidden_requested)
                raise ToolGovernanceError("Request declares a tool not allowed by gateway policy",
                                          403, "tool_not_allowed")
            app_rate = policy.get("max_requests_per_minute")
            if app_rate is not None:
                await request.app.state.gateway.acquire_application_quota(ledger_app_name, int(app_rate))
            monthly_limit = policy.get("monthly_limit_usd")
            if monthly_limit is not None:
                reservation = request.app.state.gateway.budget_reservation(route_name, upstream_payload, policy,
                                                                            tenant_cfg if tenant else None)
                await asyncio.to_thread(request.app.state.store.reserve_budget, request_id, ledger_app_name,
                                        reservation, float(monthly_limit))
            if body.stream:
                async def event_stream():
                    stream_result: dict[str, Any] | None = None
                    stream_error: str | None = None
                    status = "success"
                    stream_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    stream_provider: dict[str, Any] = {}
                    stream_frames: list[str] = []
                    stream_completed = False
                    pending_tool_frames: list[str] = []
                    tool_call_names: dict[int, str] = {}
                    stream_tool_calls: dict[int, dict[str, Any]] = {}
                    tool_calls_validated = False
                    text_delta = 0
                    try:
                        frame = "data: " + json.dumps({"id": request_id, "object": "chat.completion.chunk",
                            "created": int(time.time()), "model": route_name,
                            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}) + "\n\n"
                        stream_frames.append(frame)
                        yield frame
                        with trace.use_span(span, end_on_exit=False):
                            async for item in request.app.state.gateway.stream_complete(route_name, upstream_payload,
                                                                                       request_id, tenant_cfg if tenant else None):
                                if item.get("usage_progress"):
                                    stream_usage = item["usage_progress"]
                                    continue
                                if item.get("provider_progress"):
                                    stream_provider = item["provider_progress"]
                                    continue
                                if item.get("final"):
                                    stream_result = item
                                    if pending_tool_frames:
                                        await audit_generated_tool_calls(
                                            [tool_call_names[index] for index in sorted(tool_call_names)])
                                        try:
                                            validate_tool_calls(
                                                [stream_tool_calls[index] for index in sorted(stream_tool_calls)],
                                                upstream_payload.get("tools"))
                                        except ValueError as exc:
                                            raise ProviderError(f"Provider returned invalid tool calls: {exc}") from exc
                                        tool_calls_validated = True
                                        for tool_frame in pending_tool_frames:
                                            stream_frames.append(tool_frame)
                                            yield tool_frame
                                        pending_tool_frames.clear()
                                    usage = item.get("usage", {})
                                    stream_usage = usage
                                    frame = "data: " + json.dumps({"id": request_id, "object": "chat.completion.chunk",
                                        "created": int(time.time()), "model": route_name,
                                        "choices": [{"index": 0, "delta": {}, "finish_reason": item.get("finish_reason")}],
                                        "usage": usage}) + "\n\n"
                                    stream_frames.append(frame)
                                    yield frame
                                    continue
                                delta = item.get("delta", "")
                                if "tool_call_delta" in item:
                                    tool_call = item["tool_call_delta"]
                                    call_index = int(tool_call.get("index", 0))
                                    function_delta = tool_call.get("function", {})
                                    tool_call_names[call_index] = tool_call_names.get(call_index, "") + function_delta.get("name", "")
                                    assembled = stream_tool_calls.setdefault(call_index, {
                                        "id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                                    if tool_call.get("id"):
                                        assembled["id"] = tool_call["id"]
                                    if function_delta.get("name"):
                                        assembled["function"]["name"] += function_delta["name"]
                                    if function_delta.get("arguments"):
                                        assembled["function"]["arguments"] += function_delta["arguments"]
                                    chunk_delta = {"tool_calls": [tool_call]}
                                    frame = "data: " + json.dumps({"id": request_id, "object": "chat.completion.chunk",
                                        "created": int(time.time()), "model": route_name,
                                        "choices": [{"index": 0, "delta": chunk_delta, "finish_reason": None}]}) + "\n\n"
                                    pending_tool_frames.append(frame)
                                    continue
                                elif "audio_delta" in item:
                                    chunk_delta = {"audio": item["audio_delta"]}
                                else:
                                    chunk_delta = {"content": delta}
                                    text_delta += len(delta)
                                frame = "data: " + json.dumps({"id": request_id, "object": "chat.completion.chunk",
                                    "created": int(time.time()), "model": route_name,
                                    "choices": [{"index": 0, "delta": chunk_delta, "finish_reason": None}]}) + "\n\n"
                                stream_frames.append(frame)
                                yield frame
                        frame = "data: [DONE]\n\n"
                        stream_frames.append(frame)
                        yield frame
                        stream_completed = True
                    except ProviderError:
                        status, stream_error = "failure", "upstream_error"
                        frame = "event: error\ndata: " + json.dumps({"error": {"message": "All configured providers failed",
                            "type": "upstream_error", "request_id": request_id}}) + "\n\n"
                        stream_frames.append(frame)
                        yield frame
                    except ToolGovernanceError as exc:
                        status = "rejected" if exc.status_code == 429 else "failure"
                        stream_error = exc.reason
                        frame = "event: error\ndata: " + json.dumps({"error": {"message": exc.detail,
                            "type": "tool_policy_error", "request_id": request_id}}) + "\n\n"
                        stream_frames.append(frame)
                        yield frame
                    except asyncio.CancelledError:
                        status, stream_error = "cancelled", "client_disconnected"
                        raise
                    except GeneratorExit:
                        status, stream_error = "cancelled", "client_disconnected"
                        raise
                    except Exception as exc:
                        status, stream_error = "failure", type(exc).__name__
                        logger.exception("stream_request_failed request_id=%s", request_id)
                        frame = "event: error\ndata: " + json.dumps({"error": {"message": "Gateway stream failed",
                            "type": "gateway_error", "request_id": request_id}}) + "\n\n"
                        stream_frames.append(frame)
                        yield frame
                    finally:
                        if pending_tool_frames and not tool_calls_validated:
                            tool_audit_records.extend({"tool_name": name or "<missing>",
                                "outcome": "incomplete", "reason": "stream_ended_before_tool_validation"}
                                for name in tool_call_names.values())
                        elapsed_ms = int((time.perf_counter() - started) * 1000)
                        usage_data = stream_result.get("usage", {}) if stream_result else stream_usage
                        provider_name = stream_result.get("provider") if stream_result else stream_provider.get("provider")
                        model_name = stream_result.get("model") if stream_result else stream_provider.get("model")
                        configured_model = (stream_result.get("configured_model", model_name) if stream_result
                                            else stream_provider.get("configured_model", model_name))
                        price = cfg["providers"].get(provider_name, {}).get("models", {}).get(configured_model, {}) if provider_name else {}
                        provider_pricing = cfg["providers"].get(provider_name, {}) if provider_name else {}
                        pricing_catalog = str(provider_pricing.get("pricing_catalog", "default"))
                        pricing_version = str(price.get("pricing_version", provider_pricing.get("pricing_version", "unversioned")))
                        cost_value = (usage_data.get("prompt_tokens", 0) * float(price.get("input_cost_per_1m", 0)) +
                                      usage_data.get("completion_tokens", 0) * float(price.get("output_cost_per_1m", 0))) / 1_000_000
                        record = {"request_id": request_id, "occurred_at": datetime.now(UTC).isoformat(),
                            "application": ledger_app_name, "route": route_name, "provider": provider_name,
                            "cost_center": cost_center,
                            "model": model_name, "pricing_catalog": pricing_catalog,
                            "pricing_version": pricing_version,
                            "prompt_tokens": usage_data.get("prompt_tokens", 0),
                            "completion_tokens": usage_data.get("completion_tokens", 0),
                            "total_tokens": usage_data.get("total_tokens", 0), "cost_usd": cost_value,
                            "status": status, "latency_ms": elapsed_ms,
                            "fallback_used": int(bool(stream_result.get("fallback_used") if stream_result else
                                                      stream_provider.get("fallback_used"))),
                            "cached": int(bool(stream_result and stream_result.get("cached"))),
                            "canary_used": int(bool(stream_result.get("canary_used") if stream_result else
                                                     stream_provider.get("canary_used"))),
                            "error_type": stream_error}
                        await asyncio.to_thread(request.app.state.store.record, record)
                        if tool_audit_records:
                            await asyncio.to_thread(request.app.state.store.record_tool_call_audits,
                                request_id, tenant.get("id") if tenant else None,
                                ledger_app_name, tool_audit_records)
                            tool_audit_records.clear()
                        if idempotency_claimed and idempotency_scope and idempotency_fingerprint:
                            if stream_completed:
                                await request.app.state.idempotency.complete(idempotency_scope,
                                    idempotency_fingerprint, {"kind": "stream", "body": "".join(stream_frames),
                                        "status_code": 200, "headers": {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}})
                            else:
                                await request.app.state.idempotency.abandon(idempotency_scope, idempotency_fingerprint)
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
                            await publish_budget_alerts(request, alerts, policy, ledger_app_name, request_id)
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
                result = await request.app.state.gateway.complete(route_name, upstream_payload, request_id,
                                                                  tenant_cfg if tenant else None)
            await audit_generated_tool_calls(generated_tool_names(result.get("tool_calls")))
            try:
                validate_tool_calls(result.get("tool_calls", []), upstream_payload.get("tools"))
            except ValueError as exc:
                raise ProviderError(f"Provider returned invalid tool calls: {exc}") from exc
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
                choices=[ChatChoice(message={"role": "assistant", "content": result.get("text") or None,
                    **({"tool_calls": result["tool_calls"]} if result.get("tool_calls") else {}),
                    **({"audio": result["audio"]} if result.get("audio") else {})},
                    finish_reason=result.get("finish_reason"))],
                usage=Usage(**usage), gateway={"provider": provider, "upstream_model": model,
                    "fallback_used": result["fallback_used"], "cached": result["cached"],
                    "canary_used": result.get("canary_used", False), "cost_usd": round(cost, 8)})
            if idempotency_claimed and idempotency_scope and idempotency_fingerprint:
                await request.app.state.idempotency.complete(idempotency_scope, idempotency_fingerprint,
                    {"kind": "json", "body": output.model_dump(mode="json"), "status_code": 200})
                idempotency_complete = True
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
        except ToolGovernanceError as exc:
            status, error_type = ("rejected" if exc.status_code in {403, 429} else "failure"), exc.reason
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
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
            if idempotency_claimed and not idempotency_complete and not stream_response_started and idempotency_scope and idempotency_fingerprint:
                await request.app.state.idempotency.abandon(idempotency_scope, idempotency_fingerprint)
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
                    "pricing_catalog": (str(cfg["providers"].get(provider_name, {}).get("pricing_catalog", "default"))
                                        if provider_name else "default"),
                    "pricing_version": (str(cfg["providers"].get(provider_name, {}).get("models", {}).get(
                        configured_model, {}).get("pricing_version", cfg["providers"].get(provider_name, {}).get(
                            "pricing_version", "unversioned"))) if provider_name else "unversioned"),
                    "cost_center": cost_center,
                    "prompt_tokens": usage_data.get("prompt_tokens", 0), "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0), "cost_usd": cost_value,
                    "status": locals().get("status", "failure"), "latency_ms": elapsed_ms,
                    "fallback_used": int(bool(result and result.get("fallback_used"))),
                    "cached": int(bool(result and result.get("cached"))),
                    "canary_used": int(bool(result and result.get("canary_used"))), "error_type": error_type}
                await asyncio.to_thread(request.app.state.store.record, record)
                if tool_audit_records:
                    await asyncio.to_thread(request.app.state.store.record_tool_call_audits,
                        request_id, tenant.get("id") if tenant else None,
                        ledger_app_name, tool_audit_records)
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
                    await publish_budget_alerts(request, alerts, policy, ledger_app_name, request_id)
                span.set_attribute("llm.gateway.status", record["status"])
                span.set_attribute("llm.gateway.application", app_name)
                if error_type:
                    span.set_attribute("error.type", error_type)
                span.end()
                IN_FLIGHT.dec()
                logger.info("request_id=%s application=%s route=%s provider=%s model=%s status=%s latency_ms=%s tokens=%s cost_usd=%.8f fallback=%s cached=%s error=%s",
                    request_id, redact(ledger_app_name), route_name, provider_name, model_name, record["status"], elapsed_ms,
                    record["total_tokens"], cost_value, record["fallback_used"], record["cached"], error_type)
    @app.get("/v1/admin/report", dependencies=[Depends(admin_dependency("report"))])
    async def report(request: Request, start: str | None = Query(default=None), end: str | None = Query(default=None),
                     application: str | None = Query(default=None), cost_center: str | None = Query(default=None),
                     provider: str | None = Query(default=None), route: str | None = Query(default=None)) -> dict[str, Any]:
        """UTC ISO-8601 bounds; end is exclusive. Example: ?start=2026-10-01T00:00:00Z"""
        try:
            return await asyncio.to_thread(request.app.state.store.report, start, end, application,
                                           cost_center, provider, route)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid report filters") from exc

    @app.post("/v1/admin/cost-reconciliation", dependencies=[Depends(admin_dependency("report"))])
    async def reconcile_costs(body: CostReconciliationRequest, request: Request) -> dict[str, Any]:
        """Import actual provider charges keyed by gateway request id; batches are atomic."""
        try:
            records = [item.model_dump() for item in body.records]
            return await asyncio.to_thread(request.app.state.store.reconcile_costs, records, body.source)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/admin/cost-reconciliation/import", dependencies=[Depends(admin_dependency("report"))])
    async def import_billing_export(body: CostReconciliationImportRequest,
                                    request: Request) -> dict[str, Any]:
        """Normalize a CSV/JSON/JSONL billing export and atomically reconcile its rows."""
        try:
            rows = normalize_billing_export(body.content, body.format, body.mapping, body.cost_multiplier)
            batch = CostReconciliationRequest(source=body.source, records=rows)
            result = await asyncio.to_thread(request.app.state.store.reconcile_costs,
                [item.model_dump() for item in batch.records], batch.source)
            return {"format": body.format, "imported_rows": len(rows), **result}
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/admin/report.csv", dependencies=[Depends(admin_dependency("report"))])
    async def report_csv(request: Request, start: str | None = Query(default=None), end: str | None = Query(default=None),
                         application: str | None = Query(default=None), cost_center: str | None = Query(default=None),
                         provider: str | None = Query(default=None), route: str | None = Query(default=None),
                         limit: int = Query(default=10_000, ge=1, le=100_000)) -> Response:
        rows = await asyncio.to_thread(request.app.state.store.export_rows, start, end, application,
                                        cost_center, provider, route, limit)
        output = StringIO()
        fields = ["request_id", "occurred_at", "application", "cost_center", "route", "provider", "model",
                  "pricing_catalog", "pricing_version",
                  "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd", "status", "latency_ms",
                  "fallback_used", "cached", "canary_used", "actual_cost_usd",
                  "cost_reconciliation_id", "cost_reconciled_at", "error_type"]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            safe_row = {}
            for field, value in row.items():
                if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                    value = "'" + value
                safe_row[field] = value
            writer.writerow(safe_row)
        return Response(output.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=llm-gateway-usage.csv"})

    @app.get("/v1/admin/budgets", dependencies=[Depends(admin_dependency("budgets"))])
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

    @app.get("/v1/admin/providers", dependencies=[Depends(admin_dependency("providers"))])
    async def provider_health(request: Request) -> dict[str, Any]:
        states = await request.app.state.gateway.breaker.snapshot()
        performance = await request.app.state.gateway.health_snapshot()
        return {"providers": {name: {"kind": provider["kind"], "capabilities": provider_capabilities(provider),
                                    **states.get(name, {"state": "closed", "failures": 0})}
                              | performance.get(name, {})
                              for name, provider in cfg["providers"].items()}}

    @app.get("/v1/admin/audit", dependencies=[Depends(admin_dependency("audit"))])
    async def audit_events(request: Request, limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
        events = await asyncio.to_thread(request.app.state.store.audit_events, limit)
        return {"events": events}

    @app.get("/v1/admin/tool-audit", dependencies=[Depends(admin_dependency("audit"))])
    async def tool_audit_events(request: Request, tenant_id: str | None = Query(default=None),
                                limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
        events = await asyncio.to_thread(request.app.state.store.tool_call_audit_events, tenant_id, limit)
        return {"events": events}

    @app.get("/v1/admin/webhooks", dependencies=[Depends(admin_dependency("webhooks"))])
    async def webhook_events(request: Request, status: str | None = Query(default=None),
                             limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
        if status not in {None, "pending", "processing", "delivered", "dead_letter"}:
            raise HTTPException(status_code=400, detail="Invalid webhook delivery status")
        events = await asyncio.to_thread(request.app.state.store.webhook_events, status, limit)
        return {"events": events}

    @app.post("/v1/admin/webhooks/{event_id}/redrive", dependencies=[Depends(admin_dependency("webhooks"))])
    async def redrive_webhook(event_id: str, request: Request) -> dict[str, Any]:
        redriven = await asyncio.to_thread(request.app.state.store.redrive_webhook, event_id)
        if not redriven:
            raise HTTPException(status_code=404, detail="Dead-letter webhook event not found")
        return {"event_id": event_id, "status": "pending"}

    @app.exception_handler(HTTPException)
    async def safe_http_errors(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return app


app = create_app()


def run() -> None:
    import uvicorn
    uvicorn.run("llm_gateway.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
