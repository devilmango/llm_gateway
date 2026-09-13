from __future__ import annotations

import asyncio
import csv
import hashlib
import hmac
import json
import logging
import math
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
from .evaluation import compare_saved_runs, retrieval_metrics
from .idempotency import IdempotencyRegistry
from .limiting import RateLimitExceeded
from .observability import (
    CACHE_HITS,
    COST,
    FALLBACKS,
    GEN_AI_INPUT_TOKENS,
    GEN_AI_OPERATION_DURATION,
    GEN_AI_OUTPUT_TOKENS,
    IN_FLIGHT,
    LATENCY,
    REQUESTS,
    SLO_AVAILABILITY_TARGET,
    SLO_LATENCY_OBJECTIVE,
    SLO_LATENCY_REQUESTS,
    SLO_LATENCY_TARGET,
    SLO_REQUESTS,
    TOKENS,
    TOOL_EXECUTIONS_OVERDUE,
    TOOL_EXECUTIONS_PENDING,
    gen_ai_provider_name,
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
    EmbeddingsRequest,
    EvaluationCompareRequest,
    EvaluationReportCreate,
    PromotionEvaluateRequest,
    RerankRequest,
    RetrievalEvaluationCreate,
    ToolApprovalDecisionRequest,
    ToolExecutionOutcomeRequest,
    Usage,
)
from .secrets import SecretUnavailable, resolve_secret
from .service import GatewayService
from .shadow import promotion_gates, quality_passed, should_sample
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


async def _record_shadow_observation(app: FastAPI, *, request_id: str, application: str,
                                     route_name: str, payload: dict[str, Any], baseline: dict[str, Any],
                                     baseline_quality: bool, baseline_cost: float, baseline_latency_ms: int) -> None:
    shadow = app.state.config["routes"][route_name]["shadow"]
    provider_name, model = shadow["provider"], shadow["model"]
    started = time.perf_counter()
    status, failure_type, candidate_quality = "failure", None, False
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    candidate_cost = 0.0
    try:
        gateway = app.state.gateway
        provider_cfg = gateway.providers[provider_name]
        await gateway.limiter.acquire(provider_name, int(provider_cfg.get("requests_per_minute", 600)))
        async with gateway._concurrency():
            result = await gateway.client.complete(provider_name, model, payload,
                request_id=f"shadow_{request_id}"[:120])
        usage = result.get("usage", usage)
        price = provider_cfg.get("models", {}).get(model, {})
        candidate_cost = (usage.get("prompt_tokens", 0) * float(price.get("input_cost_per_1m", 0)) +
            usage.get("completion_tokens", 0) * float(price.get("output_cost_per_1m", 0))) / 1_000_000
        candidate_quality = quality_passed(result, payload)
        status = "success" if candidate_quality else "invalid_output"
        if not candidate_quality:
            failure_type = "quality_check_failed"
    except asyncio.CancelledError:
        raise
    except (ProviderError, RateLimitExceeded, ValueError, TypeError, KeyError) as exc:
        failure_type = type(exc).__name__
    observation = {"request_id": request_id, "occurred_at": datetime.now(UTC).isoformat(),
        "application": application, "route": route_name, "baseline_provider": baseline["provider"],
        "baseline_model": baseline["model"], "candidate_provider": provider_name, "candidate_model": model,
        "baseline_quality": int(baseline_quality), "candidate_quality": int(candidate_quality),
        "baseline_cost_usd": baseline_cost, "candidate_cost_usd": candidate_cost,
        "baseline_latency_ms": baseline_latency_ms,
        "candidate_latency_ms": int((time.perf_counter() - started) * 1000),
        "candidate_status": status, "failure_type": failure_type}
    await asyncio.to_thread(app.state.store.record_shadow_observation, observation)


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
        app.state.config = cfg
        app.state.shadow_tasks = set()
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
            for task in (*app.state.shadow_tasks, app.state.webhook_task, app.state.retention_task):
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

    def schedule_shadow(request: Request, request_id: str, application: str, route_name: str,
                        payload: dict[str, Any], baseline: dict[str, Any], baseline_quality: bool,
                        baseline_cost: float, baseline_latency_ms: int,
                        tenant_policy: dict[str, Any] | None) -> None:
        shadow = cfg["routes"].get(route_name, {}).get("shadow")
        if not shadow or not should_sample(request_id, float(shadow.get("traffic_percent", 0))):
            return
        candidate = {"provider": shadow["provider"], "model": shadow["model"]}
        if tenant_policy and not request.app.state.gateway._tenant_allows(candidate, tenant_policy):
            return
        task = asyncio.create_task(_record_shadow_observation(request.app, request_id=request_id,
            application=application, route_name=route_name, payload=payload, baseline=baseline,
            baseline_quality=baseline_quality, baseline_cost=baseline_cost,
            baseline_latency_ms=baseline_latency_ms))
        request.app.state.shadow_tasks.add(task)
        def shadow_task_done(done: asyncio.Task) -> None:
            request.app.state.shadow_tasks.discard(done)
            if not done.cancelled() and (error := done.exception()) is not None:
                logger.error("shadow_observation_failed request_id=%s error=%s",
                    request_id, type(error).__name__)
        task.add_done_callback(shadow_task_done)

    def configured_tenants() -> dict[str, Any]:
        return cfg.get("tenants", cfg.get("security", {}).get("tenants", {}))

    def slo_targets(application: str, route_name: str) -> dict[str, float]:
        slo_cfg = cfg.get("slos", {})
        targets = {"availability_target": 0.99, "latency_target_ms": 1000.0,
                   "latency_objective": 0.95, **slo_cfg.get("default", {})}
        targets.update(slo_cfg.get("routes", {}).get(route_name, {}))
        targets.update(slo_cfg.get("applications", {}).get(application, {}))
        targets["availability_target"] = min(1.0, max(0.0, float(targets["availability_target"])))
        targets["latency_objective"] = min(1.0, max(0.0, float(targets["latency_objective"])))
        targets["latency_target_ms"] = max(0.0, float(targets["latency_target_ms"]))
        return targets

    def record_request_telemetry(application: str, route_name: str, status: str, elapsed_ms: int,
                                 provider_name: str | None, configured_model: str | None,
                                 usage: dict[str, Any], operation: str = "chat") -> None:
        targets = slo_targets(application, route_name)
        SLO_AVAILABILITY_TARGET.labels(application=application, route=route_name).set(
            targets["availability_target"])
        SLO_LATENCY_TARGET.labels(application=application, route=route_name).set(
            targets["latency_target_ms"] / 1000)
        SLO_LATENCY_OBJECTIVE.labels(application=application, route=route_name).set(
            targets["latency_objective"])
        outcome = "success" if status == "success" else "failure"
        SLO_REQUESTS.labels(application=application, route=route_name, outcome=outcome).inc()
        latency_outcome = "good" if elapsed_ms <= targets["latency_target_ms"] else "bad"
        SLO_LATENCY_REQUESTS.labels(application=application, route=route_name, outcome=latency_outcome).inc()
        semantic_provider = (gen_ai_provider_name(cfg["providers"].get(provider_name, {}))
                             if provider_name else "unknown")
        labels = {"gen_ai_operation_name": operation, "gen_ai_provider_name": semantic_provider,
                  "gen_ai_request_model": configured_model or route_name}
        GEN_AI_OPERATION_DURATION.labels(**labels).observe(elapsed_ms / 1000)
        GEN_AI_INPUT_TOKENS.labels(**labels).inc(max(0, int(usage.get("prompt_tokens", 0))))
        GEN_AI_OUTPUT_TOKENS.labels(**labels).inc(max(0, int(usage.get("completion_tokens", 0))))

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
    async def metrics(request: Request) -> Response:
        timeout_seconds = int(cfg.get("tool_governance", {}).get("execution_timeout_seconds", 300))
        pending = await asyncio.to_thread(request.app.state.store.pending_tool_execution_calls,
                                          timeout_seconds, None, 1)
        TOOL_EXECUTIONS_PENDING.set(pending["pending_count"])
        TOOL_EXECUTIONS_OVERDUE.set(pending["overdue_count"])
        body, content_type = metrics_response()
        return Response(content=body, media_type=content_type)

    @app.get("/v1/routes")
    async def routes(request: Request,
                     tenant: dict[str, Any] | None = Depends(authenticate_client)) -> dict[str, Any]:  # noqa: B008
        tenant_allowed_routes = tenant["config"].get("allowed_routes") if tenant else None
        allowed_routes = (set(tenant_allowed_routes) if tenant_allowed_routes is not None else
                          set(cfg["routes"]) | set(cfg.get("embedding_routes", {})))
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
        visible_embedding_routes = {}
        for name, route in cfg.get("embedding_routes", {}).items():
            if tenant and tenant["config"].get("allowed_routes") is not None and name not in allowed_routes:
                continue
            candidates = [route, *route.get("fallbacks", [])]
            policy = tenant["config"] if tenant else {}
            policy_providers = policy.get("allowed_providers")
            policy_models = policy.get("allowed_models")
            policy_regions = policy.get("allowed_regions")
            allowed_provider_names = set(policy_providers) if policy_providers is not None else set(cfg["providers"])
            candidates = [candidate for candidate in candidates
                if candidate["provider"] in allowed_provider_names
                and (policy_models is None or candidate["model"] in policy_models)
                and (policy_regions is None or cfg["providers"][candidate["provider"]].get(
                    "data_region", "global") in policy_regions)]
            if not candidates:
                continue
            primary, *fallbacks = candidates
            visible_embedding_routes[name] = {**primary,
                "capabilities": provider_capabilities(cfg["providers"][primary["provider"]]),
                "fallbacks": [{**candidate,
                    "capabilities": provider_capabilities(cfg["providers"][candidate["provider"]])}
                    for candidate in fallbacks]}
        if visible_embedding_routes:
            visible["embedding_routes"] = visible_embedding_routes
        return visible

    async def run_retrieval(request: Request, model_name: str, inputs: list[str], application: str,
                            cost_center: str | None, operation: str,
                            tenant: dict[str, Any] | None) -> tuple[list[list[float]], dict[str, Any]]:
        request_id = str(uuid.uuid4())
        tenant_id = tenant["id"] if tenant else None
        tenant_cfg = tenant["config"] if tenant else {}
        ledger_application = f"{tenant_id}:{application}" if tenant_id else application
        retrieval_routes = cfg.get("embedding_routes", {})
        route = retrieval_routes.get(model_name)
        if route is None:
            raise HTTPException(status_code=400, detail=f"Unknown embedding route {model_name!r}")
        if tenant and tenant_cfg.get("allowed_routes") is not None and model_name not in tenant_cfg["allowed_routes"]:
            raise HTTPException(status_code=403, detail="Tenant policy does not allow this embedding route")
        configured_candidates = [route, *route.get("fallbacks", [])]
        provider_policy = tenant_cfg.get("allowed_providers") if tenant else None
        model_policy = tenant_cfg.get("allowed_models") if tenant else None
        region_policy = tenant_cfg.get("allowed_regions") if tenant else None
        allowed_providers = set(provider_policy) if provider_policy is not None else set(cfg["providers"])
        allowed_models = set(model_policy) if model_policy is not None else None
        allowed_regions = set(region_policy) if region_policy is not None else None
        candidates = [item for item in configured_candidates if item["provider"] in allowed_providers
            and (allowed_models is None or item["model"] in allowed_models)
            and (allowed_regions is None or cfg["providers"][item["provider"]].get("data_region", "global") in allowed_regions)]
        routing_decisions = [{"provider": item["provider"], "model": item["model"],
            "decision": "skipped", "reason": "tenant_policy", "attempt_count": 0}
            for item in configured_candidates if item not in candidates]
        if not candidates:
            raise HTTPException(status_code=403, detail="Tenant policy does not allow an embedding provider/model")
        policy = {**cfg.get("budgets", {}).get("default", {}),
            **cfg.get("budgets", {}).get("applications", {}).get(application, {}),
            **tenant_cfg.get("default_policy", {}),
            **tenant_cfg.get("applications", {}).get(application, {})}
        encoded_size = sum(len(value.encode("utf-8")) for value in inputs)
        max_bytes = int(policy.get("max_input_bytes", cfg.get("retrieval", {}).get("max_input_bytes", 2_000_000)))
        rate = policy.get("max_requests_per_minute")
        prices = [cfg["providers"][item["provider"]].get("models", {}).get(item["model"], {})
                  for item in candidates]
        input_price = max((float(price.get("input_cost_per_1m", 0)) for price in prices), default=0.0)
        reservation = encoded_size * input_price / 1_000_000
        monthly_limit = policy.get("monthly_limit_usd")
        started = time.perf_counter()
        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        vectors: list[list[float]] = []
        provider_name: str | None = None
        configured_model: str | None = None
        error_type: str | None = None
        fallback_used = False
        cached = False
        status = "failure"
        cost = 0.0
        try:
            if encoded_size > max_bytes:
                status, error_type = "rejected", "request_limit_exceeded"
                raise HTTPException(status_code=400,
                    detail=f"Request exceeds application input limit of {max_bytes} bytes")
            if rate is not None:
                try:
                    await request.app.state.gateway.acquire_application_quota(ledger_application, int(rate))
                except RateLimitExceeded as exc:
                    status, error_type = "rejected", "application_rate_limit_exceeded"
                    raise HTTPException(status_code=429, detail="Application request quota exceeded") from exc
            if monthly_limit is not None:
                try:
                    await asyncio.to_thread(request.app.state.store.reserve_budget, request_id,
                        ledger_application, reservation, float(monthly_limit))
                except BudgetExceeded as exc:
                    status, error_type = "rejected", "monthly_budget_exceeded"
                    raise HTTPException(status_code=429, detail="Application monthly spend limit reached") from exc
            for index, candidate in enumerate(candidates):
                provider_name, configured_model = candidate["provider"], candidate["model"]
                provider_cfg = cfg["providers"][provider_name]
                if not provider_capabilities(provider_cfg).get("embeddings", False):
                    routing_decisions.append({"provider": provider_name, "model": configured_model,
                        "decision": "skipped", "reason": "embeddings_capability_unavailable", "attempt_count": 0})
                    continue
                try:
                    await request.app.state.gateway.limiter.acquire(provider_name,
                        int(provider_cfg.get("requests_per_minute", 600)))
                    cache_cfg = cfg.get("retrieval", {}).get("embedding_cache", {})
                    cache_secret = os.getenv(cache_cfg.get("hmac_key_env", ""), "") if cache_cfg.get("enabled") else ""
                    cache_ttl = int(cache_cfg.get("ttl_seconds", 300))
                    pricing_version = provider_cfg.get("models", {}).get(configured_model, {}).get(
                        "pricing_version", "unversioned")
                    cache_keys = [request.app.state.gateway.embedding_cache.key(value,
                        tenant=tenant_id or "public", provider=provider_name, model=configured_model,
                        pricing_version=pricing_version, secret=cache_secret)
                        for value in inputs] if cache_secret and cache_ttl > 0 else []
                    cached_vectors: dict[int, list[float]] = {}
                    if cache_keys:
                        for input_index, cache_key in enumerate(cache_keys):
                            vector = await request.app.state.gateway.embedding_cache.get(cache_key)
                            if vector is not None:
                                cached_vectors[input_index] = vector
                    missing_indexes = [input_index for input_index in range(len(inputs))
                                       if input_index not in cached_vectors]
                    if missing_indexes:
                        async with request.app.state.gateway._concurrency():
                            result = await request.app.state.gateway.client.embed(provider_name,
                                configured_model, [inputs[input_index] for input_index in missing_indexes], request_id)
                        usage = result["usage"]
                        for result_index, input_index in enumerate(missing_indexes):
                            vector = result["data"][result_index]
                            cached_vectors[input_index] = vector
                            if cache_keys:
                                await request.app.state.gateway.embedding_cache.put(
                                    cache_keys[input_index], vector, cache_ttl)
                    else:
                        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    vectors = [cached_vectors[input_index] for input_index in range(len(inputs))]
                    cached = bool(cache_keys and cached_vectors)
                    if cached:
                        CACHE_HITS.labels(route=model_name).inc()
                    fallback_used = (provider_name, configured_model) != (
                        configured_candidates[0]["provider"], configured_candidates[0]["model"])
                    routing_decisions.append({"provider": provider_name, "model": configured_model,
                        "decision": "fallback" if fallback_used else "selected",
                        "reason": "provider_error" if fallback_used else "primary_candidate", "attempt_count": 1})
                    for remaining in candidates[index + 1:]:
                        routing_decisions.append({"provider": remaining["provider"], "model": remaining["model"],
                            "decision": "skipped", "reason": "not_needed_after_prior_success", "attempt_count": 0})
                    price = provider_cfg.get("models", {}).get(configured_model, {})
                    cost = usage["prompt_tokens"] * float(price.get("input_cost_per_1m", 0)) / 1_000_000
                    status = "success"
                    break
                except RateLimitExceeded as exc:
                    status, error_type = "rejected", "provider_rate_limit_exceeded"
                    raise HTTPException(status_code=429, detail="Provider request quota exceeded") from exc
                except ProviderError as exc:
                    error_type = type(exc).__name__
                    routing_decisions.append({"provider": provider_name, "model": configured_model,
                        "decision": "skipped", "reason": "provider_error", "attempt_count": 1})
                    if not exc.retryable and index == len(candidates) - 1:
                        break
            if status != "success":
                raise HTTPException(status_code=502, detail="All configured embedding providers failed")
            return vectors, {"request_id": request_id, "usage": usage, "provider": provider_name,
                "model": configured_model, "cost_usd": cost, "application": ledger_application,
                "cost_center": cost_center, "fallback_used": fallback_used}
        finally:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            record = {"request_id": request_id, "occurred_at": datetime.now(UTC).isoformat(),
                "application": ledger_application, "route": model_name, "provider": provider_name,
                "model": configured_model, "pricing_catalog": cfg["providers"].get(provider_name, {}).get("pricing_catalog", "default") if provider_name else "default",
                "pricing_version": cfg["providers"].get(provider_name, {}).get("models", {}).get(configured_model, {}).get("pricing_version", "unversioned") if provider_name else "unversioned",
                "prompt_tokens": usage["prompt_tokens"], "completion_tokens": 0,
                "total_tokens": usage["total_tokens"], "cost_usd": cost,
                "status": status, "latency_ms": elapsed_ms, "fallback_used": int(fallback_used),
                "cached": int(cached), "error_type": error_type, "cost_center": cost_center,
                "canary_used": 0, "operation": operation}
            await asyncio.to_thread(request.app.state.store.record, record)
            if routing_decisions:
                await asyncio.to_thread(request.app.state.store.record_routing_decisions,
                    request_id, ledger_application, model_name, routing_decisions)
            metric_route = model_name
            REQUESTS.labels(route=metric_route, status=status).inc()
            TOKENS.labels(route=metric_route, kind="prompt").inc(usage["prompt_tokens"])
            COST.labels(route=metric_route).inc(cost)
            if fallback_used:
                FALLBACKS.labels(route=metric_route).inc()
            LATENCY.labels(route=metric_route, status=status).observe(elapsed_ms / 1000)
            record_request_telemetry(ledger_application, metric_route, status, elapsed_ms,
                provider_name, configured_model, usage, operation=operation)
            if monthly_limit is not None and status == "success":
                thresholds = [float(value) for value in policy.get("alert_thresholds", [0.8, 1.0])]
                alerts = await asyncio.to_thread(request.app.state.store.check_budget_alerts,
                    ledger_application, float(monthly_limit), thresholds)
                await publish_budget_alerts(request, alerts, policy, ledger_application, request_id)

    @app.post("/v1/embeddings")
    async def embeddings(body: EmbeddingsRequest, request: Request,
                         tenant: dict[str, Any] | None = Depends(authenticate_client)) -> dict[str, Any]:  # noqa: B008
        inputs = [body.input] if isinstance(body.input, str) else body.input
        vectors, info = await run_retrieval(request, body.model, inputs,
            body.application, body.cost_center, "embeddings", tenant)
        return {"object": "list", "data": [{"object": "embedding", "index": index,
            "embedding": vector} for index, vector in enumerate(vectors)], "model": body.model,
            "usage": {"prompt_tokens": info["usage"]["prompt_tokens"],
                "total_tokens": info["usage"]["total_tokens"]},
            "gateway": {"request_id": info["request_id"], "provider": info["provider"],
                "upstream_model": info["model"], "fallback_used": info["fallback_used"],
                "cost_usd": round(info["cost_usd"], 8)}}

    @app.post("/v1/rerank")
    async def rerank(body: RerankRequest, request: Request,
                     tenant: dict[str, Any] | None = Depends(authenticate_client)) -> dict[str, Any]:  # noqa: B008
        route = cfg.get("embedding_routes", {}).get(body.model)
        if route is None:
            raise HTTPException(status_code=400, detail=f"Unknown embedding route {body.model!r}")
        if not provider_capabilities(cfg["providers"][route["provider"]]).get("reranking", False):
            raise HTTPException(status_code=400, detail="Embedding route does not support reranking")
        tenant_cfg = tenant["config"] if tenant else {}
        if tenant and tenant_cfg.get("allowed_routes") is not None and body.model not in tenant_cfg["allowed_routes"]:
            raise HTTPException(status_code=403, detail="Tenant policy does not allow this retrieval route")
        reranker = route.get("reranker", {})
        native_candidates = [reranker, *reranker.get("fallbacks", [])] if reranker else []
        allowed_providers = set(tenant_cfg["allowed_providers"]) if tenant_cfg.get("allowed_providers") is not None else set(cfg["providers"])
        allowed_models = set(tenant_cfg["allowed_models"]) if tenant_cfg.get("allowed_models") is not None else None
        allowed_regions = set(tenant_cfg["allowed_regions"]) if tenant_cfg.get("allowed_regions") is not None else None
        native_candidates = [candidate for candidate in native_candidates
            if candidate["provider"] in allowed_providers
            and (allowed_models is None or candidate["model"] in allowed_models)
            and (allowed_regions is None or cfg["providers"][candidate["provider"]].get("data_region", "global") in allowed_regions)]
        native_error: ProviderError | None = None
        if native_candidates:
            ledger_application = f"{tenant['id']}:{body.application}" if tenant else body.application
            budgets = cfg.get("budgets", {})
            policy = {**budgets.get("default", {}), **budgets.get("applications", {}).get(body.application, {}),
                      **tenant_cfg.get("default_policy", {}), **tenant_cfg.get("applications", {}).get(body.application, {})}
            max_bytes = int(policy.get("max_input_bytes", cfg.get("retrieval", {}).get("max_input_bytes", 2_000_000)))
            if sum(len(text.encode("utf-8")) for text in [body.query, *body.documents]) > max_bytes:
                raise HTTPException(status_code=400, detail=f"Request exceeds application input limit of {max_bytes} bytes")
            for candidate_index, candidate in enumerate(native_candidates):
                provider_name, upstream_model = candidate["provider"], candidate["model"]
                provider_cfg = cfg["providers"][provider_name]
                price = provider_cfg.get("models", {}).get(upstream_model, {})
                estimated_cost = (len(body.documents) * float(price.get("rerank_cost_per_1k_documents", 0)) / 1000
                    if "rerank_cost_per_1k_documents" in price else
                    (len(body.query) + sum(map(len, body.documents))) / 4
                    * float(price.get("input_cost_per_1m", 0)) / 1_000_000)
                request_id = str(uuid.uuid4())
                native_started = time.perf_counter()
                try:
                    request_rate = policy.get("max_requests_per_minute")
                    if request_rate is not None:
                        await request.app.state.gateway.acquire_application_quota(ledger_application, int(request_rate))
                    if policy.get("monthly_limit_usd") is not None:
                        await asyncio.to_thread(request.app.state.store.reserve_budget, request_id,
                            ledger_application, estimated_cost, float(policy["monthly_limit_usd"]))
                    await request.app.state.gateway.limiter.acquire(provider_name,
                        int(provider_cfg.get("requests_per_minute", 600)))
                    async with request.app.state.gateway._concurrency():
                        result = await request.app.state.gateway.client.rerank(provider_name, upstream_model,
                            body.query, body.documents, body.top_n or len(body.documents), request_id)
                    usage = result["usage"]
                    if "rerank_cost_per_1k_documents" in price:
                        cost = estimated_cost
                    else:
                        cost = usage["total_tokens"] * float(price.get("input_cost_per_1m", 0)) / 1_000_000
                    elapsed_ms = int((time.perf_counter() - native_started) * 1000)
                    native_fallback_used = candidate_index > 0
                    record = {"request_id": request_id, "occurred_at": datetime.now(UTC).isoformat(),
                        "application": ledger_application, "route": body.model, "provider": provider_name,
                        "model": upstream_model, "pricing_catalog": provider_cfg.get("pricing_catalog", "default"),
                        "pricing_version": price.get("pricing_version", "unversioned"),
                        "prompt_tokens": usage["prompt_tokens"], "completion_tokens": 0,
                        "total_tokens": usage["total_tokens"], "cost_usd": cost, "status": "success",
                        "latency_ms": elapsed_ms, "fallback_used": int(native_fallback_used), "cached": 0,
                        "cost_center": body.cost_center, "canary_used": 0, "operation": "rerank"}
                    await asyncio.to_thread(request.app.state.store.record, record)
                    await asyncio.to_thread(request.app.state.store.record_routing_decisions,
                        request_id, ledger_application, body.model, [{"provider": provider_name,
                            "model": upstream_model,
                            "decision": "fallback" if native_fallback_used else "selected",
                            "reason": "prior_native_provider_error" if native_fallback_used else "native_reranker",
                            "attempt_count": 1}])
                    REQUESTS.labels(route=body.model, status="success").inc()
                    TOKENS.labels(route=body.model, kind="prompt").inc(usage["prompt_tokens"])
                    COST.labels(route=body.model).inc(cost)
                    if native_fallback_used:
                        FALLBACKS.labels(route=body.model).inc()
                    record_request_telemetry(ledger_application, body.model, "success", elapsed_ms,
                        provider_name, upstream_model, usage, operation="rerank")
                    ranked = [{**item, "document": body.documents[item["index"]]} for item in result["data"]]
                    return {"model": body.model, "results": ranked, "usage": usage,
                        "gateway": {"request_id": request_id, "provider": provider_name,
                            "upstream_model": upstream_model, "cost_usd": round(cost, 8),
                            "fallback_used": native_fallback_used, "reranker": provider_cfg["kind"],
                            "billed_units": result.get("billed_units")}}
                except RateLimitExceeded as exc:
                    raise HTTPException(status_code=429, detail="Provider request quota exceeded") from exc
                except BudgetExceeded as exc:
                    raise HTTPException(status_code=429, detail="Application monthly spend limit reached") from exc
                except ProviderError as exc:
                    native_error = exc
                    elapsed_ms = int((time.perf_counter() - native_started) * 1000)
                    await asyncio.to_thread(request.app.state.store.record, {
                        "request_id": request_id, "occurred_at": datetime.now(UTC).isoformat(),
                        "application": ledger_application, "route": body.model, "provider": provider_name,
                        "model": upstream_model, "pricing_catalog": provider_cfg.get("pricing_catalog", "default"),
                        "pricing_version": price.get("pricing_version", "unversioned"),
                        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                        "cost_usd": 0, "status": "failure", "latency_ms": elapsed_ms,
                        "fallback_used": 0, "cached": 0, "error_type": type(exc).__name__,
                        "cost_center": body.cost_center, "canary_used": 0, "operation": "rerank"})
                    await asyncio.to_thread(request.app.state.store.record_routing_decisions,
                        request_id, ledger_application, body.model, [{"provider": provider_name,
                            "model": upstream_model, "decision": "skipped", "reason": "provider_error", "attempt_count": 1}])
                    REQUESTS.labels(route=body.model, status="failure").inc()
                    LATENCY.labels(route=body.model, status="failure").observe(elapsed_ms / 1000)
            if reranker.get("fallback", "cosine") == "error":
                raise HTTPException(status_code=502, detail="Configured native reranker failed") from native_error
        vectors, info = await run_retrieval(request, body.model, [body.query, *body.documents],
            body.application, body.cost_center, "rerank", tenant)
        query_vector, document_vectors = vectors[0], vectors[1:]
        query_norm = math.sqrt(sum(value * value for value in query_vector)) or 1.0
        ranked = []
        for index, vector in enumerate(document_vectors):
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            score = sum(left * right for left, right in zip(query_vector, vector)) / (query_norm * norm)
            ranked.append({"index": index, "relevance_score": score, "document": body.documents[index]})
        ranked.sort(key=lambda item: (-item["relevance_score"], item["index"]))
        return {"model": body.model, "results": ranked[:body.top_n], "usage": info["usage"],
            "gateway": {"request_id": info["request_id"], "provider": info["provider"],
                "upstream_model": info["model"], "cost_usd": round(info["cost_usd"], 8),
                "fallback_used": info["fallback_used"], "reranker": "cosine_similarity",
                "native_fallback": bool(native_candidates)}}

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
                policy_exclusions = request.app.state.gateway.routing_policy_exclusions(
                    route_name, request_id, tenant_cfg)
                if policy_exclusions:
                    await asyncio.to_thread(request.app.state.store.record_routing_decisions,
                        request_id, ledger_app_name, route_name, policy_exclusions)
                raise HTTPException(status_code=403,
                    detail="Tenant data policy does not permit this provider/model route") from exc
        IN_FLIGHT.inc()
        span = TRACER.start_span("llm.gateway.request")
        span.set_attribute("llm.gateway.request_id", request_id)
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.request.model", route_name)
        result: dict[str, Any] | None = None
        routing_decisions: list[dict[str, Any]] = []
        error_type: str | None = None
        reservation = 0.0
        policy: dict[str, Any] = {}
        monthly_limit: float | None = None
        stream_response_started = False
        tool_audit_records: list[dict[str, Any]] = []
        approval_required_tool_calls: list[str] = []
        try:
            budget_cfg = cfg.get("budgets", {})
            policy = {**budget_cfg.get("default", {}), **budget_cfg.get("applications", {}).get(app_name, {}),
                      **tenant_cfg.get("default_policy", {}), **tenant_apps.get(app_name, {})}
            allowed_tools, max_tool_calls, max_tool_calls_per_minute, require_approval_tools = effective_tool_policy(
                cfg.get("tool_governance", {}), tenant_cfg, policy)

            async def audit_generated_tool_calls(calls: list[dict[str, Any]]) -> None:
                names = generated_tool_names(calls)
                if not names:
                    return

                def event(index: int, outcome: str, reason: str | None = None) -> dict[str, Any]:
                    call_id = calls[index].get("id")
                    return {"tool_call_id": call_id, "tool_name": names[index] or "<missing>",
                            "outcome": outcome, "reason": reason}

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
                    tool_audit_records.extend(event(index, "denied",
                        reason if reason == "max_tool_calls_per_request" or name in denied else "request_blocked")
                        for index, name in enumerate(names))
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
                        tool_audit_records.extend(event(index, "denied", "max_tool_calls_per_minute")
                                                  for index, _ in enumerate(names))
                        raise ToolGovernanceError("Tenant tool-call rate limit exceeded", 429,
                                                  "max_tool_calls_per_minute") from exc
                for index, name in enumerate(names):
                    if name in require_approval_tools:
                        approval_required_tool_calls.append(str(calls[index].get("id") or ""))
                        tool_audit_records.append(event(index, "approval_required", "external_approval_pending"))
                    else:
                        tool_audit_records.append(event(index, "allowed"))

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
                    stream_routing_decisions: list[dict[str, Any]] = []
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
                                                                                       request_id, tenant_cfg if tenant else None,
                                {"application": app_name, "tenant_id": tenant.get("id") if tenant else None}):
                                if item.get("usage_progress"):
                                    stream_usage = item["usage_progress"]
                                    continue
                                if item.get("provider_progress"):
                                    stream_provider = item["provider_progress"]
                                    continue
                                if item.get("final"):
                                    stream_result = item
                                    stream_routing_decisions = item.get("routing_decisions", [])
                                    if pending_tool_frames:
                                        await audit_generated_tool_calls(
                                            [stream_tool_calls[index] for index in sorted(stream_tool_calls)])
                                        try:
                                            validate_tool_calls(
                                                [stream_tool_calls[index] for index in sorted(stream_tool_calls)],
                                                upstream_payload.get("tools"))
                                        except ValueError as exc:
                                            raise ProviderError(f"Provider returned invalid tool calls: {exc}") from exc
                                        tool_calls_validated = True
                                        if approval_required_tool_calls:
                                            approval_event = "event: tool_approval\ndata: " + json.dumps({
                                                "request_id": request_id,
                                                "required_tool_call_ids": approval_required_tool_calls,
                                                "status": "pending"}) + "\n\n"
                                            stream_frames.append(approval_event)
                                            yield approval_event
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
                    except ProviderError as exc:
                        stream_routing_decisions = getattr(exc, "routing_decisions", stream_routing_decisions)
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
                        stream_routing_decisions = getattr(exc, "routing_decisions", stream_routing_decisions)
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
                        if (stream_completed and stream_result and provider_name and configured_model
                                and not stream_result.get("cached")):
                            baseline_output = {"text": stream_result.get("text", ""),
                                "tool_calls": list(stream_tool_calls.values())}
                            schedule_shadow(request, request_id, ledger_app_name, route_name, upstream_payload,
                                {"provider": provider_name, "model": configured_model},
                                quality_passed(baseline_output, upstream_payload), cost_value, elapsed_ms,
                                tenant_cfg if tenant else None)
                        if stream_routing_decisions:
                            await asyncio.to_thread(request.app.state.store.record_routing_decisions,
                                request_id, ledger_app_name, route_name, stream_routing_decisions)
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
                        record_request_telemetry(ledger_app_name, metric_route, status, elapsed_ms,
                            provider_name, configured_model, usage_data)
                        span.set_attribute("gen_ai.usage.input_tokens", int(record["prompt_tokens"]))
                        span.set_attribute("gen_ai.usage.output_tokens", int(record["completion_tokens"]))
                        if provider_name:
                            span.set_attribute("gen_ai.provider.name",
                                gen_ai_provider_name(cfg["providers"].get(provider_name, {})))
                        if configured_model:
                            span.set_attribute("gen_ai.request.model", str(configured_model))
                        if model_name:
                            span.set_attribute("gen_ai.response.model", str(model_name))
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
            provider_started = time.perf_counter()
            with trace.use_span(span, end_on_exit=False):
                result = await request.app.state.gateway.complete(route_name, upstream_payload, request_id,
                    tenant_cfg if tenant else None,
                    {"application": app_name, "tenant_id": tenant.get("id") if tenant else None})
            provider_latency_ms = int((time.perf_counter() - provider_started) * 1000)
            routing_decisions = result.get("routing_decisions", [])
            await audit_generated_tool_calls(result.get("tool_calls", []))
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
            if not result.get("cached"):
                schedule_shadow(request, request_id, ledger_app_name, route_name, upstream_payload,
                    {"provider": provider, "model": result.get("configured_model", model)},
                    quality_passed(result, upstream_payload), cost, provider_latency_ms,
                    tenant_cfg if tenant else None)
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
                    "canary_used": result.get("canary_used", False), "cost_usd": round(cost, 8),
                    "approval_required_tool_call_ids": approval_required_tool_calls})
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
            routing_decisions = getattr(exc, "routing_decisions", routing_decisions)
            status, error_type = "failure", "shared_rate_limit_store_unavailable"
            raise HTTPException(status_code=503, detail="Shared rate-limit store is unavailable") from exc
        except ProviderError as exc:
            routing_decisions = getattr(exc, "routing_decisions", routing_decisions)
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
                if routing_decisions:
                    await asyncio.to_thread(request.app.state.store.record_routing_decisions,
                        request_id, ledger_app_name, route_name, routing_decisions)
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
                record_request_telemetry(ledger_app_name, metric_route, record["status"], elapsed_ms,
                    provider_name, configured_model, usage_data)
                span.set_attribute("gen_ai.usage.input_tokens", int(record["prompt_tokens"]))
                span.set_attribute("gen_ai.usage.output_tokens", int(record["completion_tokens"]))
                if provider_name:
                    span.set_attribute("gen_ai.provider.name",
                        gen_ai_provider_name(cfg["providers"].get(provider_name, {})))
                if configured_model:
                    span.set_attribute("gen_ai.request.model", str(configured_model))
                if model_name:
                    span.set_attribute("gen_ai.response.model", str(model_name))
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

    @app.get("/v1/admin/tenants/{tenant_id}/data-export",
             dependencies=[Depends(admin_dependency("tenant_data"))])
    async def export_tenant_data(tenant_id: str, request: Request) -> dict[str, Any]:
        tenant_cfg = configured_tenants().get(tenant_id)
        if tenant_cfg is None:
            raise HTTPException(status_code=404, detail="Unknown tenant")
        retention_days = tenant_cfg.get("data_policy", {}).get("usage_retention_days")
        if retention_days is None:
            raise HTTPException(status_code=409, detail="Tenant data export requires a configured usage retention policy")
        actor = getattr(request.state, "admin_actor", "unknown-admin")
        exported = await asyncio.to_thread(request.app.state.store.export_tenant_data, tenant_id, actor)
        counts = {name: len(rows) for name, rows in exported.items() if name != "audit_history"}
        return {"tenant_id": tenant_id, "retention_days": int(retention_days),
            "exported_at": datetime.now(UTC).isoformat(), "record_counts": counts, "data": exported}

    @app.delete("/v1/admin/tenants/{tenant_id}/data",
                dependencies=[Depends(admin_dependency("tenant_data"))])
    async def delete_tenant_data(tenant_id: str, request: Request,
                                 confirm_tenant_id: str = Query(..., min_length=1, max_length=120)) -> dict[str, Any]:
        tenant_cfg = configured_tenants().get(tenant_id)
        if tenant_cfg is None:
            raise HTTPException(status_code=404, detail="Unknown tenant")
        retention_days = tenant_cfg.get("data_policy", {}).get("usage_retention_days")
        if retention_days is None:
            raise HTTPException(status_code=409, detail="Tenant data deletion requires a configured usage retention policy")
        if not hmac.compare_digest(confirm_tenant_id, tenant_id):
            raise HTTPException(status_code=400, detail="confirm_tenant_id must match the requested tenant")
        actor = getattr(request.state, "admin_actor", "unknown-admin")
        counts = await asyncio.to_thread(request.app.state.store.delete_tenant_data, tenant_id, actor)
        return {"tenant_id": tenant_id, "deleted_at": datetime.now(UTC).isoformat(),
            "retention_days": int(retention_days), "deleted_record_counts": counts,
            "audit_record_retained": True}

    @app.get("/v1/admin/slos", dependencies=[Depends(admin_dependency("report"))])
    async def slo_report(request: Request,
                         window_hours: int = Query(default=24, ge=1, le=720),
                         application: str | None = Query(default=None),
                         route: str | None = Query(default=None)) -> dict[str, Any]:
        """Report availability and latency objective compliance for the requested window."""
        end = datetime.now(UTC)
        start = end - timedelta(hours=window_hours)
        observations = await asyncio.to_thread(request.app.state.store.slo_observations,
            start.isoformat(), end.isoformat())
        groups: dict[tuple[str, str], dict[str, int]] = {}
        for item in observations:
            app_key, route_key = str(item["application"]), str(item["route"])
            if application and app_key != application:
                continue
            if route and route_key != route:
                continue
            bucket = groups.setdefault((app_key, route_key), {"requests": 0, "failures": 0,
                "latency_good": 0, "latency_bad": 0})
            count = int(item["requests"])
            bucket["requests"] += count
            if item["status"] != "success":
                bucket["failures"] += count
            if int(item["latency_ms"]) <= slo_targets(app_key, route_key)["latency_target_ms"]:
                bucket["latency_good"] += count
            else:
                bucket["latency_bad"] += count
        reports = []
        for (app_key, route_key), counts in sorted(groups.items()):
            total = counts["requests"]
            targets = slo_targets(app_key, route_key)
            availability = (total - counts["failures"]) / total if total else 1.0
            latency_compliance = counts["latency_good"] / total if total else 1.0
            availability_budget = 1.0 - targets["availability_target"]
            latency_budget = 1.0 - targets["latency_objective"]
            availability_burn = (counts["failures"] / total / availability_budget
                                 if availability_budget > 0 and total else 0.0)
            latency_burn = (counts["latency_bad"] / total / latency_budget
                            if latency_budget > 0 and total else 0.0)
            reports.append({"application": app_key, "route": route_key, **counts,
                "availability_target": targets["availability_target"],
                "availability": availability,
                "availability_error_budget_burn_rate": availability_burn,
                "availability_error_budget_remaining_percent": (1.0 - availability_burn) * 100,
                "latency_target_ms": targets["latency_target_ms"],
                "latency_objective": targets["latency_objective"],
                "latency_compliance": latency_compliance,
                "latency_error_budget_burn_rate": latency_burn,
                "latency_error_budget_remaining_percent": (1.0 - latency_burn) * 100})
        return {"window_hours": window_hours, "start": start.isoformat(), "end": end.isoformat(),
                "generated_at": end.isoformat(), "slos": reports}

    @app.get("/v1/admin/cost-anomalies", dependencies=[Depends(admin_dependency("report"))])
    async def cost_anomalies(request: Request,
                             window_hours: int = Query(default=24, ge=1, le=720),
                             threshold_percent: float = Query(default=50, gt=0, le=1000),
                             min_requests: int = Query(default=5, ge=2, le=100_000)) -> dict[str, Any]:
        return await asyncio.to_thread(request.app.state.store.cost_anomalies,
            window_hours, threshold_percent, min_requests)

    @app.get("/v1/admin/routing-decisions", dependencies=[Depends(admin_dependency("audit"))])
    async def routing_decisions(request: Request, request_id: str | None = Query(default=None, max_length=120),
                                application: str | None = Query(default=None, max_length=120),
                                route: str | None = Query(default=None, max_length=120),
                                provider: str | None = Query(default=None, max_length=120),
                                decision: str | None = Query(default=None),
                                limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
        if decision not in {None, "selected", "fallback", "skipped"}:
            raise HTTPException(status_code=400, detail="decision must be selected, fallback, or skipped")
        events = await asyncio.to_thread(request.app.state.store.routing_decision_events,
            request_id, application, route, provider, decision, limit)
        return {"events": events}

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
                  "operation", "pricing_catalog", "pricing_version",
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

    @app.post("/v1/admin/tool-approvals", dependencies=[Depends(admin_dependency("tool_approvals"))])
    async def decide_tool_approval(body: ToolApprovalDecisionRequest, request: Request) -> dict[str, Any]:
        """Record an external approval decision; the authenticated admin identity is the approver."""
        actor = request.state.admin_actor
        if body.decision != "approved" and body.expires_in_seconds is not None:
            raise HTTPException(status_code=422, detail="expires_in_seconds only applies to approvals")
        ttl_seconds = body.expires_in_seconds or int(cfg.get("tool_governance", {}).get("approval_ttl_seconds", 3600))
        expires_at = (datetime.now(UTC) + timedelta(seconds=ttl_seconds)).isoformat() \
            if body.decision == "approved" else None
        try:
            result = await asyncio.to_thread(request.app.state.store.record_tool_approval,
                body.request_id, body.tool_call_id, body.decision, actor, expires_at)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raw_application = result["application"].split(":", 1)[-1]
        webhooks = cfg.get("tool_governance", {}).get("approval_webhooks", {})
        webhook = webhooks.get(result["application"], webhooks.get(raw_application))
        if webhook:
            occurred_at = datetime.now(UTC).isoformat()
            event_id = hashlib.sha256((f"{body.request_id}:{body.tool_call_id}:{body.decision}:"
                f"{result.get('expires_at') or ''}").encode()).hexdigest()
            event = {"alert_id": event_id, "type": "tool_approval.updated", "occurred_at": occurred_at,
                "application": result["application"], "request_id": body.request_id,
                "tool_call_id": body.tool_call_id, "decision": body.decision,
                "actor": result.get("actor", actor), "expires_at": result.get("expires_at")}
            await asyncio.to_thread(request.app.state.store.enqueue_webhook,
                event, webhook["url"], webhook.get("secret_env"))
        return result

    @app.get("/v1/tool-approvals/{request_id}/{tool_call_id}")
    async def tool_approval_status(request_id: str, tool_call_id: str, request: Request,
                                   application: str = Query(min_length=1, max_length=120),
                                   tenant: dict[str, Any] | None = Depends(authenticate_client)) -> dict[str, Any]:  # noqa: B008
        tenant_id = tenant["id"] if tenant else None
        scoped_application = application
        if tenant:
            if application not in tenant["config"].get("applications", {}):
                raise HTTPException(status_code=403, detail="Tenant is not permitted to use this application")
            scoped_application = f"{tenant_id}:{application}"
        try:
            return await asyncio.to_thread(request.app.state.store.tool_approval_status,
                request_id, tool_call_id, tenant_id, scoped_application)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/tool-executions")
    async def report_tool_execution(body: ToolExecutionOutcomeRequest, request: Request,
                                    tenant: dict[str, Any] | None = Depends(authenticate_client)) -> dict[str, Any]:  # noqa: B008
        tenant_id = tenant["id"] if tenant else None
        application = body.application
        if tenant:
            applications = tenant["config"].get("applications", {})
            if application not in applications:
                raise HTTPException(status_code=403, detail="Tenant is not permitted to use this application")
            application = f"{tenant_id}:{application}"
        try:
            return await asyncio.to_thread(request.app.state.store.record_tool_execution_outcome,
                body.request_id, body.tool_call_id, tenant_id, application, body.outcome, body.error_code)
        except ValueError as exc:
            if "requires recorded external approval" in str(exc):
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/admin/tool-executions", dependencies=[Depends(admin_dependency("audit"))])
    async def tool_execution_events(request: Request, tenant_id: str | None = Query(default=None),
                                    limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
        events = await asyncio.to_thread(request.app.state.store.tool_execution_outcomes, tenant_id, limit)
        return {"events": events}

    @app.get("/v1/admin/tool-executions/pending", dependencies=[Depends(admin_dependency("audit"))])
    async def pending_tool_execution_events(request: Request, tenant_id: str | None = Query(default=None),
                                            limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
        timeout_seconds = int(cfg.get("tool_governance", {}).get("execution_timeout_seconds", 300))
        pending = await asyncio.to_thread(request.app.state.store.pending_tool_execution_calls,
                                          timeout_seconds, tenant_id, limit)
        return {**pending, "timeout_seconds": timeout_seconds}

    @app.post("/v1/admin/evaluations", dependencies=[Depends(admin_dependency("report"))])
    async def persist_evaluation(body: EvaluationReportCreate, request: Request) -> dict[str, Any]:
        try:
            result = await asyncio.to_thread(request.app.state.store.save_evaluation_report,
                body.dataset_name, body.dataset_version, body.dataset_sha256, body.case_count, body.report)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"saved": True, **result}

    @app.post("/v1/admin/retrieval/evaluations", dependencies=[Depends(admin_dependency("report"))])
    async def persist_retrieval_evaluation(body: RetrievalEvaluationCreate, request: Request) -> dict[str, Any]:
        if body.route not in (set(cfg.get("embedding_routes", {})) | set(cfg.get("routes", {}))):
            raise HTTPException(status_code=400, detail=f"Unknown retrieval route {body.route!r}")
        cases = [item.model_dump() for item in body.cases]
        per_case_metrics = [retrieval_metrics(item["ranked_document_ids"], item["relevant_document_ids"])
                            for item in cases]
        metric_names = per_case_metrics[0].keys()
        quality = {name: round(sum(metrics[name] for metrics in per_case_metrics) / len(per_case_metrics), 8)
                   for name in metric_names}
        summary = {"requests": len(cases), "failures": sum(bool(item["failed"]) for item in cases),
            "estimated_cost_usd": round(sum(item["cost_usd"] for item in cases), 8),
            "average_latency_ms": round(sum(item["latency_ms"] for item in cases) / len(cases), 2),
            "retrieval_metrics": quality}
        report = {"routes": {body.route: summary}, "results": [
            {"case": index, "route": body.route, "passed": not item["failed"],
             "latency_ms": item["latency_ms"], "cost_usd": item["cost_usd"]}
            for index, item in enumerate(cases)]}
        try:
            saved = await asyncio.to_thread(request.app.state.store.save_evaluation_report,
                body.dataset_name, body.dataset_version, body.dataset_sha256, len(cases), report)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"saved": True, **saved, "route": body.route, "summary": summary}

    @app.get("/v1/admin/evaluations", dependencies=[Depends(admin_dependency("report"))])
    async def evaluation_reports(request: Request,
                                 dataset_name: str | None = Query(default=None, max_length=200),
                                 dataset_version: str | None = Query(default=None, max_length=120),
                                 limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
        reports = await asyncio.to_thread(request.app.state.store.evaluation_reports,
            dataset_name, dataset_version, limit)
        return {"reports": reports}

    @app.post("/v1/admin/evaluations/compare", dependencies=[Depends(admin_dependency("report"))])
    async def compare_evaluation_reports(body: EvaluationCompareRequest, request: Request) -> dict[str, Any]:
        if body.baseline_run_id == body.candidate_run_id:
            raise HTTPException(status_code=400, detail="Baseline and candidate runs must be different")
        baseline, candidate = await asyncio.gather(
            asyncio.to_thread(request.app.state.store.evaluation_report, body.baseline_run_id),
            asyncio.to_thread(request.app.state.store.evaluation_report, body.candidate_run_id))
        if baseline is None or candidate is None:
            missing = body.baseline_run_id if baseline is None else body.candidate_run_id
            raise HTTPException(status_code=404, detail=f"Evaluation run {missing!r} was not found")
        report = compare_saved_runs(baseline, candidate,
            min_pass_rate=body.min_pass_rate,
            max_cost_regression_percent=body.max_cost_regression_percent,
            max_latency_regression_percent=body.max_latency_regression_percent,
            max_failure_rate=body.max_failure_rate,
            min_recall_at_5=body.min_recall_at_5,
            min_mrr=body.min_mrr,
            min_ndcg_at_5=body.min_ndcg_at_5)
        saved = await asyncio.to_thread(request.app.state.store.save_evaluation_comparison,
            body.baseline_run_id, body.candidate_run_id, report)
        return {"saved": True, **saved, "report": report}

    @app.get("/v1/admin/evaluations/comparisons", dependencies=[Depends(admin_dependency("report"))])
    async def evaluation_comparisons(request: Request,
                                     limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
        comparisons = await asyncio.to_thread(request.app.state.store.evaluation_comparisons, limit)
        return {"comparisons": comparisons}

    @app.get("/v1/admin/shadow-reports", dependencies=[Depends(admin_dependency("report"))])
    async def shadow_reports(request: Request, application: str = Query(min_length=1, max_length=120),
                             route: str = Query(min_length=1, max_length=120),
                             window_hours: int = Query(default=24, ge=1, le=720)) -> dict[str, Any]:
        shadow = cfg["routes"].get(route, {}).get("shadow")
        if not shadow:
            raise HTTPException(status_code=404, detail="No shadow candidate is configured for this route")
        end = datetime.now(UTC)
        start = end - timedelta(hours=window_hours)
        result = await asyncio.to_thread(request.app.state.store.shadow_report,
            application, route, start.isoformat(), end.isoformat())
        result["candidate_provider"] = shadow["provider"]
        result["candidate_model"] = shadow["model"]
        result["window_hours"] = window_hours
        result["promotion"] = promotion_gates(result, shadow.get("promotion", {}))
        return result

    @app.post("/v1/admin/promotions/evaluate", dependencies=[Depends(admin_dependency("report"))])
    async def evaluate_promotion(body: PromotionEvaluateRequest, request: Request) -> dict[str, Any]:
        route_config = cfg["routes"].get(body.route)
        if not route_config or not route_config.get("shadow"):
            raise HTTPException(status_code=404, detail="No shadow candidate is configured for this route")
        comparison = await asyncio.to_thread(request.app.state.store.evaluation_comparison, body.comparison_id)
        if comparison is None:
            raise HTTPException(status_code=404, detail="Evaluation comparison was not found")
        evaluation_report = comparison["report"]
        evaluation_route = evaluation_report.get("routes", {}).get(body.route)
        evaluation_passed = (evaluation_report.get("promotion_passed") is True and evaluation_route is not None
                             and evaluation_route.get("change") == "compared")
        end = datetime.now(UTC)
        start = end - timedelta(hours=body.window_hours)
        shadow_report = await asyncio.to_thread(request.app.state.store.shadow_report,
            body.application, body.route, start.isoformat(), end.isoformat())
        shadow_result = promotion_gates(shadow_report, route_config["shadow"].get("promotion", {}))
        targets = slo_targets(body.application, body.route)
        observations = await asyncio.to_thread(request.app.state.store.slo_observations,
            start.isoformat(), end.isoformat())
        matching = [item for item in observations
                    if item["application"] == body.application and item["route"] == body.route]
        total = sum(int(item["requests"]) for item in matching)
        failures = sum(int(item["requests"]) for item in matching if item["status"] != "success")
        within_latency = sum(int(item["requests"]) for item in matching
            if int(item["latency_ms"]) <= targets["latency_target_ms"])
        availability = (total - failures) / total if total else 0.0
        latency_compliance = within_latency / total if total else 0.0
        slo = {"requests": total, "availability": availability,
            "availability_target": targets["availability_target"],
            "latency_compliance": latency_compliance,
            "latency_objective": targets["latency_objective"],
            "passed": bool(total and availability >= targets["availability_target"]
                            and latency_compliance >= targets["latency_objective"])}
        gates = {"evaluation": {"passed": evaluation_passed, "comparison_id": body.comparison_id},
            "shadow": shadow_result, "slo": slo}
        return {"application": body.application, "route": body.route,
            "candidate_provider": route_config["shadow"]["provider"],
            "candidate_model": route_config["shadow"]["model"],
            "window_hours": body.window_hours, "promotion_passed": (
                gates["evaluation"]["passed"] and gates["shadow"]["passed"] and gates["slo"]["passed"]),
            "gates": gates}

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
