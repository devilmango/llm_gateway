from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import fakeredis.aioredis
import httpx
import pytest

from llm_gateway.main import create_app


def config(database: str) -> dict:
    return {
        "server": {"database": database, "request_timeout_seconds": 2, "max_retries": 1,
                   "retry_base_delay_seconds": 0, "max_concurrent_requests": 10},
        "security": {"redact_pii": True},
        "budgets": {"default": {}, "applications": {}},
        "providers": {
            "primary": {"kind": "openai", "base_url": "https://primary.test/v1",
                        "api_key_env": "TEST_PRIMARY_KEY", "requests_per_minute": 100,
                        "models": {"primary-model": {"input_cost_per_1m": 1, "output_cost_per_1m": 2}}},
            "backup": {"kind": "openai", "base_url": "https://backup.test/v1",
                       "api_key_env": "TEST_BACKUP_KEY", "requests_per_minute": 100,
                       "models": {"backup-model": {"input_cost_per_1m": 1, "output_cost_per_1m": 2}}},
        },
        "routes": {"simple": {"provider": "primary", "model": "primary-model",
                    "fallbacks": [{"provider": "backup", "model": "backup-model"}],
                    "cache": {"enabled": False}}},
    }


@asynccontextmanager
async def gateway_client(config_data: dict, transport: httpx.AsyncBaseTransport, shared_redis=None):
    app = create_app(config_data, http_transport=transport, shared_redis=shared_redis)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway"
    ) as client:
        yield client


def success_response() -> httpx.Response:
    return httpx.Response(200, json={
        "id": "upstream-id", "model": "primary-model",
        "choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })


@pytest.mark.asyncio
async def test_retries_then_falls_back_redacts_and_reports_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    sent: list[tuple[str, dict]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        sent.append((str(request.url), json.loads(request.content)))
        if request.url.host == "primary.test":
            return httpx.Response(503)
        body = success_response().json()
        body["model"] = "backup-model"
        return httpx.Response(200, json=body)

    async with gateway_client(config(str(tmp_path / "usage.sqlite3")), httpx.MockTransport(upstream)) as client:
        response = await client.post("/v1/chat/completions", json={
            "model": "simple", "application": "support-router",
            "messages": [{"role": "user", "content": "Email me at person@example.com"}],
        })
        assert response.status_code == 200
        body = response.json()
        assert body["gateway"]["provider"] == "backup"
        assert body["gateway"]["fallback_used"] is True
        assert body["usage"]["total_tokens"] == 15
        assert body["gateway"]["cost_usd"] == 0.00002
        assert len(sent) == 3  # two primary attempts, then the fallback
        assert "person@example.com" not in sent[-1][1]["messages"][0]["content"]
        assert "[REDACTED_EMAIL]" in sent[-1][1]["messages"][0]["content"]

        report = (await client.get("/v1/admin/report", params={"application": "support-router"})).json()
        assert report["requests"] == 1
        assert report["total_tokens"] == 15
        assert report["cost_usd"] == 0.00002
        assert report["fallbacks"] == 1
        metrics = (await client.get("/metrics")).text
        assert 'llm_gateway_provider_requests_total{outcome="upstream_error",provider="primary"} 2.0' in metrics
        assert 'llm_gateway_provider_retries_total{provider="primary"} 1.0' in metrics


@pytest.mark.asyncio
async def test_monthly_budget_reserves_estimated_spend_before_upstream(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    calls = 0

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return success_response()

    cfg = config(str(tmp_path / "budget.sqlite3"))
    cfg["budgets"]["applications"]["costly-app"] = {
        "monthly_limit_usd": 0.000001,
        "max_input_bytes": 100,
        "max_completion_tokens": 10,
    }
    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        response = await client.post("/v1/chat/completions", json={
            "model": "simple", "application": "costly-app",
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 429
        assert response.json()["detail"] == "Application monthly spend limit reached"
        assert calls == 0
        budget = (await client.get("/v1/admin/budgets")).json()["applications"][0]
        assert budget["monthly_limit_usd"] == 0.000001


@pytest.mark.asyncio
async def test_budget_alerts_once_at_configured_threshold(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "model": "primary-model",
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        })

    cfg = config(str(tmp_path / "alerts.sqlite3"))
    cfg["budgets"]["applications"]["alert-app"] = {
        "monthly_limit_usd": 0.000008,
        "max_input_bytes": 5,
        "max_completion_tokens": 1,
        "alert_thresholds": [0.8],
    }
    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        request = {"model": "simple", "application": "alert-app",
                   "messages": [{"role": "user", "content": "hello"}]}
        assert (await client.post("/v1/chat/completions", json=request)).status_code == 200
        assert "budget_alert application=alert-app threshold=0.8" in caplog.text
        budget = (await client.get("/v1/admin/budgets")).json()["applications"][0]
        assert budget["utilization"] == 0.875
        assert (await client.post("/v1/chat/completions", json=request)).status_code == 429
        assert caplog.text.count("budget_alert application=alert-app threshold=0.8") == 1


@pytest.mark.asyncio
async def test_concurrent_requests_cannot_reserve_above_monthly_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    calls = 0

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={
            "model": "primary-model",
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        })

    cfg = config(str(tmp_path / "concurrent-budget.sqlite3"))
    cfg["budgets"]["applications"]["concurrent-app"] = {
        "monthly_limit_usd": 0.000007,
        "max_input_bytes": 5,
        "max_completion_tokens": 1,
    }
    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        request = {"model": "simple", "application": "concurrent-app",
                   "messages": [{"role": "user", "content": "hello"}]}
        first, second = await asyncio.gather(
            client.post("/v1/chat/completions", json=request),
            client.post("/v1/chat/completions", json=request),
        )
        assert sorted([first.status_code, second.status_code]) == [200, 429]
        assert calls == 1


@pytest.mark.asyncio
async def test_circuit_breaker_skips_unhealthy_provider_then_recovers(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    primary_calls = 0

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal primary_calls
        if request.url.host == "primary.test":
            primary_calls += 1
            if primary_calls == 1:
                return httpx.Response(503)
            return success_response()
        body = success_response().json()
        body["model"] = "backup-model"
        return httpx.Response(200, json=body)

    cfg = config(str(tmp_path / "breaker.sqlite3"))
    cfg["circuit_breaker"] = {"failure_threshold": 1, "recovery_seconds": 0.05}
    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        request = {"model": "simple", "messages": [{"role": "user", "content": "hello"}]}
        first = (await client.post("/v1/chat/completions", json=request)).json()
        providers = (await client.get("/v1/admin/providers")).json()["providers"]
        assert first["gateway"]["fallback_used"] is True
        assert providers["primary"]["state"] == "open"
        await asyncio.sleep(0.12)
        second = (await client.post("/v1/chat/completions", json=request)).json()
        assert second["gateway"]["provider"] == "primary"
        assert second["gateway"]["fallback_used"] is False
        assert (await client.get("/v1/admin/providers")).json()["providers"]["primary"]["state"] == "closed"


@pytest.mark.asyncio
async def test_streaming_endpoint_emits_sse_and_records_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    stream_body = (
        'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}\n\n'
        'data: [DONE]\n\n'
    )

    def upstream(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=stream_body, headers={"content-type": "text/event-stream"})

    async with gateway_client(config(str(tmp_path / "stream.sqlite3")), httpx.MockTransport(upstream)) as client:
        response = await client.post("/v1/chat/completions", json={
            "model": "simple", "application": "stream-app", "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert '"content": "hello"' in response.text
        assert '"content": " world"' in response.text
        assert '"total_tokens": 4' in response.text
        assert "data: [DONE]" in response.text
        report = (await client.get("/v1/admin/report", params={"application": "stream-app"})).json()
        assert report["requests"] == 1
        assert report["total_tokens"] == 4


@pytest.mark.asyncio
async def test_streaming_uses_compatible_fallback_before_emitting_text(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    stream_body = ('data: {"choices":[{"delta":{"content":"backup"}}]}\n\n'
                   'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"total_tokens":1}}\n\n'
                   'data: [DONE]\n\n')

    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "backup.test"
        return httpx.Response(200, content=stream_body, headers={"content-type": "text/event-stream"})

    cfg = config(str(tmp_path / "stream-fallback.sqlite3"))
    cfg["providers"]["primary"]["kind"] = "anthropic"
    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        response = await client.post("/v1/chat/completions", json={
            "model": "simple", "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200
        assert '"content": "backup"' in response.text
        report = (await client.get("/v1/admin/report", params={"application": "default"})).json()
        assert report["fallbacks"] == 1


@pytest.mark.asyncio
async def test_tenant_credentials_scope_routes_applications_and_admin_access(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    monkeypatch.setenv("SUPPORT_TENANT_KEY", "tenant-secret")
    monkeypatch.setenv("GATEWAY_API_KEY", "admin-secret")

    def upstream(request: httpx.Request) -> httpx.Response:
        return success_response()

    cfg = config(str(tmp_path / "tenant.sqlite3"))
    cfg["routes"]["reasoning"] = {"provider": "primary", "model": "primary-model", "fallbacks": []}
    cfg["tenants"] = {"support": {"api_key_env": "SUPPORT_TENANT_KEY", "allowed_routes": ["simple"],
        "applications": {"support-router": {"max_requests_per_minute": 50, "max_completion_tokens": 100}}}}
    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        request = {"model": "simple", "application": "support-router",
                   "messages": [{"role": "user", "content": "hello"}]}
        assert (await client.post("/v1/chat/completions", json=request)).status_code == 401
        headers = {"Authorization": "Bearer tenant-secret"}
        assert (await client.get("/v1/routes", headers=headers)).json().keys() == {"simple"}
        assert (await client.get("/v1/admin/report", headers=headers)).status_code == 401
        denied_app = {**request, "application": "other-app"}
        assert (await client.post("/v1/chat/completions", json=denied_app, headers=headers)).status_code == 403
        denied_route = {**request, "model": "reasoning"}
        assert (await client.post("/v1/chat/completions", json=denied_route, headers=headers)).status_code == 403
        assert (await client.post("/v1/chat/completions", json=request, headers=headers)).status_code == 200
        report = (await client.get("/v1/admin/report", params={"application": "support:support-router"},
                                   headers={"Authorization": "Bearer admin-secret"})).json()
        assert report["requests"] == 1


@pytest.mark.asyncio
async def test_latency_aware_routing_uses_shared_provider_health_signal(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    calls: list[str] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        body = success_response().json()
        if request.url.host == "backup.test":
            body["model"] = "backup-model"
        return httpx.Response(200, json=body)

    cfg = config(str(tmp_path / "latency.sqlite3"))
    cfg["routing"] = {"strategy": "latency_aware"}
    app = create_app(cfg, http_transport=httpx.MockTransport(upstream))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway"
    ) as client:
        health = app.state.gateway.provider_health
        await health.record("primary", 500, True)
        await health.record("backup", 50, True)
        response = await client.post("/v1/chat/completions", json={
            "model": "simple", "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200
        assert calls == ["backup.test"]
        assert response.json()["gateway"]["fallback_used"] is True
        providers = (await client.get("/v1/admin/providers")).json()["providers"]
        assert providers["primary"]["ewma_latency_ms"] == 500
        assert providers["backup"]["ewma_latency_ms"] < 50


@pytest.mark.asyncio
async def test_redis_cache_and_quotas_are_shared_across_gateway_instances(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    redis_url = os.getenv("TEST_REDIS_URL")
    if redis_url:
        from redis.asyncio import Redis

        redis = Redis.from_url(redis_url, decode_responses=True)
        await redis.flushdb()
    else:
        redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    first_calls = 0

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal first_calls
        first_calls += 1
        return success_response()

    cfg1 = config(str(tmp_path / "instance1.sqlite3"))
    cfg2 = config(str(tmp_path / "instance2.sqlite3"))
    cfg1["routes"]["simple"]["cache"] = {"enabled": True, "ttl_seconds": 60}
    cfg2["routes"]["simple"]["cache"] = {"enabled": True, "ttl_seconds": 60}
    for cfg in (cfg1, cfg2):
        cfg["budgets"]["applications"]["limited"] = {"max_requests_per_minute": 1}
    app1 = create_app(cfg1, http_transport=httpx.MockTransport(upstream), shared_redis=redis)
    app2 = create_app(cfg2, http_transport=httpx.MockTransport(upstream), shared_redis=redis)
    async with app1.router.lifespan_context(app1), app2.router.lifespan_context(app2), \
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app1), base_url="http://one") as client1, \
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app2), base_url="http://two") as client2:
        cache_request = {"model": "simple", "application": "cache-shared",
                         "messages": [{"role": "user", "content": "same prompt"}]}
        await client1.post("/v1/chat/completions", json=cache_request)
        cached = await client2.post("/v1/chat/completions", json=cache_request)
        assert cached.json()["gateway"]["cached"] is True
        assert first_calls == 1
        shared_health = await app2.state.gateway.health_snapshot()
        assert shared_health["primary"]["successes"] == 1

        quota_request = {"model": "simple", "application": "limited",
                         "messages": [{"role": "user", "content": "different prompt"}]}
        assert (await client1.post("/v1/chat/completions", json=quota_request)).status_code == 200
        assert (await client2.post("/v1/chat/completions", json=quota_request)).status_code == 429
    await redis.aclose()


@pytest.mark.asyncio
async def test_postgres_usage_store_when_test_database_is_configured(monkeypatch):
    import uuid

    dsn = os.getenv("TEST_POSTGRES_URL")
    if not dsn:
        pytest.skip("TEST_POSTGRES_URL is not configured; CI provides a PostgreSQL service")
    from llm_gateway.postgres_storage import PostgresUsageStore

    store = PostgresUsageStore(dsn)
    request_id = "pg_test_" + uuid.uuid4().hex
    application = "pg-test-" + uuid.uuid4().hex
    store.record({"request_id": request_id, "occurred_at": datetime.now(UTC).isoformat(),
        "application": application, "route": "simple", "provider": "test", "model": "test-model",
        "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10, "cost_usd": 0.00002,
        "status": "success", "latency_ms": 25, "fallback_used": 0, "cached": 0, "error_type": None})
    try:
        report = store.report(None, None, application)
        assert report["requests"] == 1
        assert report["total_tokens"] == 10
        assert report["cost_usd"] == pytest.approx(0.00002)
        store.reserve_budget("pg_res_" + request_id, application, 0.00001, 0.0001)
        alerts = store.check_budget_alerts(application, 0.00002, [0.8])
        assert len(alerts) == 1
        assert alerts[0]["threshold"] == 0.8
        assert alerts[0]["spend_usd"] == pytest.approx(0.00002)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_application_request_quota_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    cfg = config(str(tmp_path / "quota.sqlite3"))
    cfg["budgets"]["applications"]["single-use"] = {"max_requests_per_minute": 1}
    async with gateway_client(cfg, httpx.MockTransport(lambda request: success_response())) as client:
        request = {"model": "simple", "application": "single-use",
                   "messages": [{"role": "user", "content": "hello"}]}
        first = await client.post("/v1/chat/completions", json=request)
        second = await client.post("/v1/chat/completions", json=request)
        assert first.status_code == 200
        assert second.status_code == 429
        assert second.json()["detail"] == "Application request quota exceeded"
        metrics = (await client.get("/metrics")).text
        assert "llm_gateway_requests_total" in metrics
        assert 'route="simple",status="success"' in metrics


@pytest.mark.asyncio
async def test_cache_hits_return_zero_billable_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    calls = 0

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return success_response()

    cfg = config(str(tmp_path / "cache.sqlite3"))
    cfg["routes"]["simple"]["cache"] = {"enabled": True, "ttl_seconds": 30}
    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        request = {"model": "simple", "application": "cache-app",
                   "messages": [{"role": "user", "content": "same request"}]}
        first = (await client.post("/v1/chat/completions", json=request)).json()
        second = (await client.post("/v1/chat/completions", json=request)).json()
        assert calls == 1
        assert first["usage"]["total_tokens"] == 15
        assert second["gateway"]["cached"] is True
        assert second["usage"]["total_tokens"] == 0
        assert second["gateway"]["cost_usd"] == 0
        report = (await client.get("/v1/admin/report")).json()
        assert report["cache_hits"] == 1
        assert report["cost_usd"] == first["gateway"]["cost_usd"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,host,api_key_env,response_body,expected_path", [
    ("anthropic", "anthropic.test", "TEST_ANTHROPIC_KEY",
     {"id": "msg_1", "model": "claude-test", "content": [{"type": "text", "text": "from anthropic"}],
      "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 4}}, "/v1/messages"),
    ("gemini", "gemini.test", "TEST_GEMINI_KEY",
     {"candidates": [{"content": {"parts": [{"text": "from gemini"}]}, "finishReason": "STOP"}],
      "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4, "totalTokenCount": 7}},
     "/v1beta/models/gemini-test:generateContent"),
])
async def test_provider_adapters_normalize_responses(tmp_path, monkeypatch, kind, host, api_key_env,
                                                     response_body, expected_path):
    monkeypatch.setenv(api_key_env, "provider-secret")
    requests = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=response_body)

    cfg = config(str(tmp_path / f"{kind}.sqlite3"))
    cfg["providers"] = {"selected": {"kind": kind, "base_url": f"https://{host}" if kind == "anthropic" else f"https://{host}/v1beta",
        "api_key_env": api_key_env, "models": {"claude-test" if kind == "anthropic" else "gemini-test": {
            "input_cost_per_1m": 1, "output_cost_per_1m": 1}}}}
    cfg["routes"] = {"simple": {"provider": "selected", "model": "claude-test" if kind == "anthropic" else "gemini-test",
                                  "fallbacks": [], "cache": {"enabled": False}}}
    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        response = await client.post("/v1/chat/completions", json={
            "model": "simple", "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200
        result = response.json()
        assert result["choices"][0]["message"]["content"] == f"from {kind}"
        assert result["usage"]["total_tokens"] == 7
        assert requests[0].url.path == expected_path


@pytest.mark.asyncio
async def test_exhausted_upstream_failures_are_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    async with gateway_client(config(str(tmp_path / "failure.sqlite3")),
                              httpx.MockTransport(lambda request: httpx.Response(500))) as client:
        response = await client.post("/v1/chat/completions", json={
            "model": "simple", "application": "broken-app",
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 502
        report = (await client.get("/v1/admin/report", params={"application": "broken-app"})).json()
        assert report["requests"] == 1
        assert report["failures"] == 1
        assert report["total_tokens"] == 0
