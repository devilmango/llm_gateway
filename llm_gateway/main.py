from __future__ import annotations

import asyncio
import hmac
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from .config import load_config
from .providers import ProviderError
from .redaction import redact
from .schemas import ChatChoice, ChatRequest, ChatResponse, Usage
from .service import GatewayService
from .storage import UsageStore

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("llm_gateway")


def create_app(config: dict[str, Any] | None = None) -> FastAPI:
    cfg = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        timeout = httpx.Timeout(float(cfg["server"].get("request_timeout_seconds", 45)))
        app.state.http = httpx.AsyncClient(timeout=timeout)
        app.state.gateway = GatewayService(cfg, app.state.http)
        database = os.getenv("GATEWAY_DATABASE", cfg["server"].get("database", "./gateway.sqlite3"))
        app.state.store = UsageStore(database)
        yield
        await app.state.http.aclose()

    app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)

    async def authenticate(authorization: str | None = Header(default=None)) -> None:
        expected = os.getenv("GATEWAY_API_KEY")
        if expected:
            provided = authorization.removeprefix("Bearer ") if authorization else ""
            if not hmac.compare_digest(provided, expected):
                raise HTTPException(status_code=401, detail="Invalid gateway credentials")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/routes", dependencies=[Depends(authenticate)])
    async def routes() -> dict[str, Any]:
        return {name: {"provider": route["provider"], "model": route["model"],
                       "fallbacks": route.get("fallbacks", []), "cache": route.get("cache", {})}
                for name, route in cfg["routes"].items()}

    @app.post("/v1/chat/completions", response_model=ChatResponse, dependencies=[Depends(authenticate)])
    async def chat(body: ChatRequest, request: Request) -> ChatResponse:
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
        result: dict[str, Any] | None = None
        error_type: str | None = None
        try:
            result = await request.app.state.gateway.complete(route_name, upstream_payload)
            usage = result["usage"]
            provider = result["provider"]
            model = result["model"]
            price = cfg["providers"][provider].get("models", {}).get(result.get("configured_model", model), {})
            cost = (usage["prompt_tokens"] * float(price.get("input_cost_per_1m", 0)) +
                    usage["completion_tokens"] * float(price.get("output_cost_per_1m", 0))) / 1_000_000
            status = "success"
            output = ChatResponse(id=request_id, created=int(time.time()), model=route_name,
                choices=[ChatChoice(message={"role": "assistant", "content": result["text"]}, finish_reason=result.get("finish_reason"))],
                usage=Usage(**usage), gateway={"provider": provider, "upstream_model": model,
                    "fallback_used": result["fallback_used"], "cached": result["cached"], "cost_usd": round(cost, 8)})
            return output
        except KeyError:
            status, error_type = "failure", "unknown_route"
            raise HTTPException(status_code=400, detail=f"Unknown route {route_name!r}")
        except ValueError as exc:
            status, error_type = "failure", "configuration_error"
            raise HTTPException(status_code=500, detail=str(exc))
        except ProviderError as exc:
            status, error_type = "failure", "upstream_error"
            raise HTTPException(status_code=502, detail="All configured providers failed") from exc
        except Exception as exc:
            status, error_type = "failure", type(exc).__name__
            raise
        finally:
            # Persist every attempt outcome; never persist prompt or completion content.
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            usage_data = result.get("usage", {}) if result else {}
            provider_name = result.get("provider") if result else None
            model_name = result.get("model") if result else None
            configured_model = result.get("configured_model", model_name) if result else model_name
            report_price = cfg["providers"].get(provider_name, {}).get("models", {}).get(configured_model, {}) if provider_name else {}
            cost_value = (usage_data.get("prompt_tokens", 0) * float(report_price.get("input_cost_per_1m", 0)) +
                          usage_data.get("completion_tokens", 0) * float(report_price.get("output_cost_per_1m", 0))) / 1_000_000
            record = {"request_id": request_id, "occurred_at": datetime.now(timezone.utc).isoformat(),
                "application": app_name, "route": route_name, "provider": provider_name, "model": model_name,
                "prompt_tokens": usage_data.get("prompt_tokens", 0), "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0), "cost_usd": cost_value,
                "status": locals().get("status", "failure"), "latency_ms": elapsed_ms,
                "fallback_used": int(bool(result and result.get("fallback_used"))),
                "cached": int(bool(result and result.get("cached"))), "error_type": error_type}
            await asyncio.to_thread(request.app.state.store.record, record)
            logger.info("request_id=%s application=%s route=%s provider=%s model=%s status=%s latency_ms=%s tokens=%s cost_usd=%.8f fallback=%s cached=%s error=%s",
                request_id, redact(app_name), route_name, provider_name, model_name, record["status"], elapsed_ms,
                record["total_tokens"], cost_value, record["fallback_used"], record["cached"], error_type)

    @app.get("/v1/admin/report", dependencies=[Depends(authenticate)])
    async def report(request: Request, start: str | None = Query(default=None), end: str | None = Query(default=None),
                     application: str | None = Query(default=None)) -> dict[str, Any]:
        """UTC ISO-8601 bounds; end is exclusive. Example: ?start=2026-10-01T00:00:00Z"""
        try:
            return await asyncio.to_thread(request.app.state.store.report, start, end, application)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid report filters") from exc

    @app.exception_handler(HTTPException)
    async def safe_http_errors(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return app


app = create_app()


def run() -> None:
    import uvicorn
    uvicorn.run("llm_gateway.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
