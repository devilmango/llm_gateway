from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import sqlite3
import struct
import sys
import zlib
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import ModuleType, SimpleNamespace

import fakeredis.aioredis
import httpx
import pytest
from pydantic import ValidationError

from llm_gateway.client import ToolApprovalError, execute_tool_with_approval
from llm_gateway.evaluation import compare_routes, evaluate_cases
from llm_gateway.main import _deliver_webhook_batch, _emit_async_job_event, create_app
from llm_gateway.observability import metrics_response
from llm_gateway.providers import ProviderClient
from llm_gateway.schemas import ChatRequest
from llm_gateway.storage import create_usage_store
from llm_gateway.webhook_security import UnsafeWebhookTarget, validate_webhook_url


def config(database: str) -> dict:
    return {
        "server": {"database": database, "request_timeout_seconds": 2, "max_retries": 1,
                   "retry_base_delay_seconds": 0, "max_concurrent_requests": 10,
                   "webhooks_allow_private_targets": True},
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


def bedrock_stream_frame(event_type: str, payload: dict) -> bytes:
    event_name = b":event-type"
    event_value = event_type.encode()
    headers = bytes([len(event_name)]) + event_name + b"\x07" + struct.pack(">H", len(event_value)) + event_value
    message_name = b":message-type"
    headers += bytes([len(message_name)]) + message_name + b"\x07" + struct.pack(">H", 5) + b"event"
    body = json.dumps(payload, separators=(",", ":")).encode()
    total_length = 12 + len(headers) + len(body) + 4
    prelude = struct.pack(">II", total_length, len(headers))
    frame = prelude + struct.pack(">I", zlib.crc32(prelude)) + headers + body
    return frame + struct.pack(">I", zlib.crc32(frame))


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
        route_history = (await client.get("/v1/admin/routing-decisions",
            params={"request_id": body["id"]})).json()["events"]
        assert [(event["provider"], event["decision"], event["reason"], event["attempt_count"])
                for event in route_history] == [
                    ("primary", "skipped", "upstream_server_error", 2),
                    ("backup", "fallback", "prior_candidate_failed:upstream_server_error", 1),
                ]
        assert any(row["reason"] == "prior_candidate_failed:upstream_server_error"
                   for row in report["by_routing_decision"])
        metrics = (await client.get("/metrics")).text
        assert 'llm_gateway_provider_requests_total{outcome="upstream_error",provider="primary"} 2.0' in metrics
        assert 'llm_gateway_provider_retries_total{provider="primary"} 1.0' in metrics


@pytest.mark.asyncio
async def test_idempotency_replays_success_and_rejects_key_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    calls = 0

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return success_response()

    async with gateway_client(config(str(tmp_path / "idempotency.sqlite3")), httpx.MockTransport(upstream)) as client:
        request = {"model": "simple", "application": "support-router",
                   "messages": [{"role": "user", "content": "hello"}]}
        first = await client.post("/v1/chat/completions", json=request,
                                  headers={"Idempotency-Key": "checkout-42"})
        replay = await client.post("/v1/chat/completions", json=request,
                                   headers={"Idempotency-Key": "checkout-42"})
        changed = {**request, "messages": [{"role": "user", "content": "different"}]}
        conflict = await client.post("/v1/chat/completions", json=changed,
                                     headers={"Idempotency-Key": "checkout-42"})
        assert first.status_code == replay.status_code == 200
        assert first.json() == replay.json()
        assert calls == 1
        assert conflict.status_code == 409


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
async def test_cost_anomalies_compare_request_and_token_cost_by_pricing_version(tmp_path):
    now = datetime.now(UTC)
    async with gateway_client(config(str(tmp_path / "cost-anomalies.sqlite3")),
                              httpx.MockTransport(lambda _: success_response())) as client:
        store = client._transport.app.state.store
        for index in range(5):
            for occurred_at, cost in ((now - timedelta(hours=30, minutes=index), 0.001),
                                      (now - timedelta(hours=2, minutes=index), 0.003)):
                store.record({"request_id": f"anomaly-{index}-{cost}",
                    "occurred_at": occurred_at.isoformat(), "application": "support-router",
                    "route": "simple", "provider": "primary", "model": "primary-model",
                    "pricing_catalog": "test-catalog", "pricing_version": "v7",
                    "prompt_tokens": 400, "completion_tokens": 600, "total_tokens": 1000,
                    "cost_usd": cost, "status": "success", "latency_ms": 20,
                    "fallback_used": 0, "cached": 0, "canary_used": 0})

        detected = await client.get("/v1/admin/cost-anomalies", params={
            "window_hours": 24, "threshold_percent": 50, "min_requests": 3})
        assert detected.status_code == 200
        report = detected.json()
        assert len(report["anomalies"]) == 1
        anomaly = report["anomalies"][0]
        assert (anomaly["application"], anomaly["provider"], anomaly["model"],
                anomaly["pricing_version"]) == ("support-router", "primary", "primary-model", "v7")
        assert anomaly["change_percent"] == {"per_request": 200.0, "per_1k_tokens": 200.0}
        quiet = await client.get("/v1/admin/cost-anomalies", params={
            "window_hours": 24, "threshold_percent": 250, "min_requests": 3})
        assert quiet.json()["anomalies"] == []


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
async def test_budget_alert_webhook_is_persisted_for_durable_delivery(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    monkeypatch.setenv("ALERT_SIGNING_SECRET", "webhook-secret")
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "model": "primary-model",
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        })

    cfg = config(str(tmp_path / "webhook.sqlite3"))
    cfg["budgets"]["applications"]["alert-app"] = {
        "monthly_limit_usd": 0.000008, "max_input_bytes": 5, "max_completion_tokens": 1,
        "alert_thresholds": [0.8], "alert_webhook_url": "https://alerts.test/hook",
        "alert_webhook_secret_env": "ALERT_SIGNING_SECRET", "alert_webhook_max_attempts": 2,
    }
    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        request = {"model": "simple", "application": "alert-app",
                   "messages": [{"role": "user", "content": "hello"}]}
        assert (await client.post("/v1/chat/completions", json=request)).status_code == 200
        assert (await client.post("/v1/chat/completions", json=request)).status_code == 429
        events = (await client.get("/v1/admin/webhooks")).json()["events"]
        assert len(events) == 1
        assert events[0]["status"] == "pending"
    assert events[0]["event_id"]


@pytest.mark.asyncio
async def test_budget_webhook_delivery_failure_does_not_fail_model_request(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "model": "primary-model",
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        })

    cfg = config(str(tmp_path / "webhook-failure.sqlite3"))
    cfg["budgets"]["applications"]["alert-app"] = {
        "monthly_limit_usd": 0.000008, "max_input_bytes": 5, "max_completion_tokens": 1,
        "alert_thresholds": [0.8], "alert_webhook_url": "https://alerts.test/hook",
    }
    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        response = await client.post("/v1/chat/completions", json={
            "model": "simple", "application": "alert-app",
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200
        events = (await client.get("/v1/admin/webhooks")).json()["events"]
        assert events[0]["status"] == "pending"


@pytest.mark.asyncio
async def test_webhook_delivery_signs_event_and_records_result(tmp_path, monkeypatch):
    monkeypatch.setenv("ALERT_SIGNING_SECRET", "webhook-secret")
    store = create_usage_store(str(tmp_path / "outbox.sqlite3"))
    event = {"alert_id": "stable-alert-id", "application": "app", "event": "budget.threshold_crossed"}
    store.enqueue_webhook(event, "https://alerts.test/hook", "ALERT_SIGNING_SECRET")
    received = []

    def delivery(request: httpx.Request) -> httpx.Response:
        fields = dict(part.split("=", 1) for part in request.headers["X-LLM-Gateway-Signature"].split(","))
        signed = fields["t"].encode() + b"." + request.content
        expected = hmac.new(b"webhook-secret", signed, hashlib.sha256).hexdigest()
        assert fields["kid"] == request.headers["X-LLM-Gateway-Key-Id"] == "default"
        assert hmac.compare_digest(fields["v1"], expected)
        assert request.headers["Idempotency-Key"] == "stable-alert-id"
        received.append(request.content)
        return httpx.Response(204)

    client = httpx.AsyncClient(transport=httpx.MockTransport(delivery))
    app = SimpleNamespace(state=SimpleNamespace(store=store, http=client,
        config={"server": {"webhooks_allow_private_targets": True,
            "webhook_circuit_failure_threshold": 100}}, webhook_circuits={}))
    await _deliver_webhook_batch(app)
    assert len(received) == 1
    assert store.webhook_events()[0]["status"] == "delivered"
    await client.aclose()
    if hasattr(store, "close"):
        store.close()


@pytest.mark.asyncio
async def test_webhook_transient_failures_dead_letter_and_redrive(tmp_path):
    store = create_usage_store(str(tmp_path / "outbox-retries.sqlite3"))
    store.enqueue_webhook({"alert_id": "retry-alert", "application": "app"}, "https://alerts.test/hook")
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(503)))
    app = SimpleNamespace(state=SimpleNamespace(store=store, http=client,
        config={"server": {"webhooks_allow_private_targets": True,
            "webhook_circuit_failure_threshold": 100}}, webhook_circuits={}))
    for attempt in range(8):
        if attempt:
            store.update_webhook("retry-alert", attempts=attempt, status="pending", next_attempt_at=datetime.now(UTC))
        await _deliver_webhook_batch(app)
    event = store.webhook_events()[0]
    assert event["status"] == "dead_letter"
    assert event["attempts"] == 8
    assert store.redrive_webhook("retry-alert") is True
    assert store.webhook_events("pending")[0]["event_id"] == "retry-alert"
    await client.aclose()
    if hasattr(store, "close"):
        store.close()


@pytest.mark.asyncio
async def test_openai_tool_calls_and_structured_output_are_normalized(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    received: dict = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        received.update(json.loads(request.content))
        return httpx.Response(200, json={
            "model": "primary-model",
            "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_123", "type": "function",
                "function": {"name": "lookup_order", "arguments": "{\"order_id\":\"A-17\"}"},
            }]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        })

    cfg = config(str(tmp_path / "tools.sqlite3"))
    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        response = await client.post("/v1/chat/completions", json={
            "model": "simple", "messages": [{"role": "user", "content": "Find order A-17"}],
            "tools": [{"type": "function", "function": {"name": "lookup_order",
                "description": "Look up an order", "parameters": {"type": "object",
                    "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}}],
            "tool_choice": {"type": "function", "function": {"name": "lookup_order"}},
            "response_format": {"type": "json_schema", "json_schema": {"name": "order_result",
                "strict": True, "schema": {"type": "object", "properties": {"status": {"type": "string"}},
                    "required": ["status"], "additionalProperties": False}}},
        })
        assert response.status_code == 200
        body = response.json()
        assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "lookup_order"
        assert json.loads(body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == {"order_id": "A-17"}
        assert received["tool_choice"]["function"]["name"] == "lookup_order"
        assert received["response_format"]["json_schema"]["name"] == "order_result"


@pytest.mark.asyncio
async def test_tool_arguments_are_validated_and_gateway_request_id_is_correlated(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    cfg = config(str(tmp_path / "tool-validation.sqlite3"))
    cfg["providers"]["backup"]["pricing_catalog"] = "test-rates"
    cfg["providers"]["backup"]["models"]["backup-model"]["pricing_version"] = "2026-10-test"
    sent: list[tuple[str, str | None]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        sent.append((request.url.host, request.headers.get("X-Gateway-Request-ID")))
        if request.url.host == "primary.test":
            body = success_response().json()
            body["choices"][0] = {"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-bad", "type": "function", "function": {"name": "lookup_order", "arguments": "not-json"}}
            ]}, "finish_reason": "tool_calls"}
            return httpx.Response(200, json=body)
        body = success_response().json()
        body["choices"][0] = {"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call-ok", "type": "function", "function": {"name": "lookup_order", "arguments": '{"order_id":"A-17"}'}}
        ]}, "finish_reason": "tool_calls"}
        body["model"] = "backup-model"
        return httpx.Response(200, json=body)

    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        response = await client.post("/v1/chat/completions", json={
            "model": "simple", "application": "support-router",
            "messages": [{"role": "user", "content": "look up A-17"}],
            "tools": [{"type": "function", "function": {"name": "lookup_order",
                "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                               "required": ["order_id"], "additionalProperties": False}}}],
        })
        assert response.status_code == 200
        request_id = response.json()["id"]
        assert response.json()["gateway"]["provider"] == "backup"
        assert json.loads(response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == {
            "order_id": "A-17"}
        assert len(sent) == 3  # invalid primary is retried, then fallback succeeds
        assert {correlation_id for _, correlation_id in sent} == {request_id}

        reconciled = await client.post("/v1/admin/cost-reconciliation", json={
            "source": "test-invoice", "records": [{"request_id": request_id, "actual_cost_usd": 0.00005}],
        })
        assert reconciled.status_code == 200
        report = (await client.get("/v1/admin/report", params={"application": "support-router"})).json()
        catalog = report["by_pricing_catalog"][0]
        assert catalog["provider"] == "backup"
        assert catalog["model"] == "backup-model"
        assert catalog["pricing_catalog"] == "test-rates"
        assert catalog["pricing_version"] == "2026-10-test"
        assert catalog["actual_cost_usd"] == 0.00005
        assert catalog["cost_variance_usd"] == pytest.approx(0.00003)


def test_request_rejects_invalid_or_remote_referenced_tool_schemas():
    with pytest.raises(ValidationError, match="Invalid tool schema"):
        ChatRequest(messages=[{"role": "user", "content": "hello"}], tools=[{
            "type": "function", "function": {"name": "f", "parameters": {"type": "object",
                "properties": {"x": {"type": "not-a-schema-type"}}}}
        }])
    with pytest.raises(ValidationError, match="local JSON pointer"):
        ChatRequest(messages=[{"role": "user", "content": "hello"}], tools=[{
            "type": "function", "function": {"name": "f", "parameters": {
                "type": "object", "properties": {"x": {"$ref": "https://example.test/schema.json"}}
            }}
        }])


@pytest.mark.asyncio
async def test_invalid_streamed_tool_arguments_are_withheld(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    stream_body = (
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",'
        '"function":{"name":"lookup_order","arguments":"{\\"order_id\\":17}"}}]}}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n\n'
        'data: [DONE]\n\n')
    seen_ids: list[str | None] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen_ids.append(request.headers.get("X-Gateway-Request-ID"))
        return httpx.Response(200, content=stream_body, headers={"content-type": "text/event-stream"})

    async with gateway_client(config(str(tmp_path / "invalid-stream-tool.sqlite3")),
                              httpx.MockTransport(upstream)) as client:
        response = await client.post("/v1/chat/completions", json={
            "model": "simple", "stream": True,
            "messages": [{"role": "user", "content": "find order"}],
            "tools": [{"type": "function", "function": {"name": "lookup_order",
                "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                               "required": ["order_id"]}}}],
        })
        assert response.status_code == 200
        assert "event: error" in response.text
        assert '"tool_calls"' not in response.text
        assert seen_ids and seen_ids[0]


@pytest.mark.asyncio
async def test_tool_execution_outcomes_are_idempotent_and_admin_visible(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("SUPPORT_TENANT_KEY", "support-secret")
    monkeypatch.setenv("GATEWAY_API_KEY", "admin-secret")
    tool_response = success_response().json()
    tool_response["choices"][0] = {"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call-execute-1", "type": "function", "function": {
            "name": "lookup_order", "arguments": '{"order_id":"A-17"}'}}
    ]}, "finish_reason": "tool_calls"}
    cfg = config(str(tmp_path / "tool-outcomes.sqlite3"))
    cfg["tenants"] = {"support": {"api_key_env": "SUPPORT_TENANT_KEY", "allowed_routes": ["simple"],
        "allowed_tools": ["lookup_order"], "applications": {"support-router": {}}}}
    async with gateway_client(cfg,
                              httpx.MockTransport(lambda _: httpx.Response(200, json=tool_response))) as client:
        completion = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer support-secret"}, json={
            "model": "simple", "application": "support-router",
            "messages": [{"role": "user", "content": "find A-17"}],
            "tools": [{"type": "function", "function": {"name": "lookup_order",
                "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}}}],
        })
        assert completion.status_code == 200
        request_id = completion.json()["id"]
        outcome = {"request_id": request_id, "tool_call_id": "call-execute-1",
                   "application": "support-router", "outcome": "succeeded"}
        first = await client.post("/v1/tool-executions", headers={"Authorization": "Bearer support-secret"}, json=outcome)
        retry = await client.post("/v1/tool-executions", headers={"Authorization": "Bearer support-secret"}, json=outcome)
        assert first.status_code == retry.status_code == 200
        assert first.json()["duplicate"] is False
        assert retry.json()["duplicate"] is True
        conflict = await client.post("/v1/tool-executions", headers={"Authorization": "Bearer support-secret"},
                                     json={**outcome, "outcome": "failed"})
        assert conflict.status_code == 404
        events = (await client.get("/v1/admin/tool-executions", headers={
            "Authorization": "Bearer admin-secret"})).json()["events"]
        assert len(events) == 1
        assert events[0]["tool_name"] == "lookup_order"
        assert events[0]["outcome"] == "succeeded"


@pytest.mark.asyncio
async def test_high_impact_tool_requires_recorded_approval_before_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("SUPPORT_TENANT_KEY", "support-secret")
    monkeypatch.setenv("GATEWAY_API_KEY", "admin-secret")
    tool_response = success_response().json()
    tool_response["choices"][0] = {"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call-delete-1", "type": "function", "function": {
            "name": "delete_account", "arguments": '{"account_id":"A-17"}'}}
    ]}, "finish_reason": "tool_calls"}
    cfg = config(str(tmp_path / "tool-approval.sqlite3"))
    cfg["tool_governance"] = {"require_approval_tools": ["delete_account"]}
    cfg["tool_governance"]["approval_webhooks"] = {"support-router": {
        "url": "https://app.test/tool-approval-events"}}
    cfg["tenants"] = {"support": {"api_key_env": "SUPPORT_TENANT_KEY", "allowed_routes": ["simple"],
        "allowed_tools": ["delete_account"], "applications": {"support-router": {}}}}
    async with gateway_client(cfg,
            httpx.MockTransport(lambda _: httpx.Response(200, json=tool_response))) as client:
        completion = await client.post("/v1/chat/completions",
            headers={"Authorization": "Bearer support-secret"}, json={
                "model": "simple", "application": "support-router",
                "messages": [{"role": "user", "content": "delete account"}],
                "tools": [{"type": "function", "function": {"name": "delete_account",
                    "parameters": {"type": "object"}}}],
            })
        assert completion.status_code == 200
        request_id = completion.json()["id"]
        assert completion.json()["gateway"]["approval_required_tool_call_ids"] == ["call-delete-1"]
        approval_url = f"/v1/tool-approvals/{request_id}/call-delete-1?application=support-router"
        headers = {"Authorization": "Bearer support-secret"}
        client.headers.update(headers)
        status = await client.get(approval_url, headers=headers)
        assert status.json()["status"] == "pending"
        outcome = {"request_id": request_id, "tool_call_id": "call-delete-1",
            "application": "support-router", "outcome": "succeeded"}
        blocked = await client.post("/v1/tool-executions", headers=headers, json=outcome)
        assert blocked.status_code == 409
        invoked = False

        async def invoke_delete():
            nonlocal invoked
            invoked = True

        with pytest.raises(ToolApprovalError, match="pending"):
            await execute_tool_with_approval(client, request_id=request_id,
                tool_call_id="call-delete-1", application="support-router", execute=invoke_delete)
        assert not invoked
        approved = await client.post("/v1/admin/tool-approvals", headers={
            "Authorization": "Bearer admin-secret"}, json={"request_id": request_id,
                "tool_call_id": "call-delete-1", "decision": "approved", "expires_in_seconds": 3600})
        assert approved.status_code == 200
        assert approved.json()["actor"] == "gateway-api-key"
        assert approved.json()["expires_at"]
        duplicate = await client.post("/v1/admin/tool-approvals", headers={
            "Authorization": "Bearer admin-secret"}, json={"request_id": request_id,
                "tool_call_id": "call-delete-1", "decision": "approved"})
        assert duplicate.json()["duplicate"] is True
        conflicting = await client.post("/v1/admin/tool-approvals", headers={
            "Authorization": "Bearer admin-secret"}, json={"request_id": request_id,
                "tool_call_id": "call-delete-1", "decision": "denied"})
        assert conflicting.status_code == 409
        status = await client.get(approval_url, headers=headers)
        assert status.json()["status"] == "approved"
        await execute_tool_with_approval(client, request_id=request_id,
            tool_call_id="call-delete-1", application="support-router", execute=invoke_delete)
        assert invoked
        executed = await client.post("/v1/tool-executions", headers=headers, json=outcome)
        assert executed.status_code == 200 and executed.json()["duplicate"] is True
        app = client._transport.app
        await _deliver_webhook_batch(app)
        webhook_events = await asyncio.to_thread(app.state.store.webhook_events, "delivered")
        assert any(event["application"] == "support:support-router" for event in webhook_events)
        audit = await client.get("/v1/admin/tool-audit?tenant_id=support", headers={
            "Authorization": "Bearer admin-secret"})
        events = audit.json()["events"]
        assert any(event["outcome"] == "approval_required" for event in events)
        assert any(event["outcome"] == "approved" and event["actor"] == "gateway-api-key"
                   for event in events)
        assert all("arguments" not in event for event in events)


def test_tool_approvals_expire_and_can_be_revoked(tmp_path):
    store = create_usage_store(str(tmp_path / "tool-approval-expiry.sqlite3"))
    now = datetime.now(UTC)
    for tool_call_id in ("expired-call", "revoked-call"):
        store.record_tool_call_audits("approval-request", None, "app", [{
            "tool_call_id": tool_call_id, "tool_name": "delete_account",
            "outcome": "approval_required", "reason": "external_approval_pending"}])
    store.record_tool_approval("approval-request", "expired-call", "approved", "approver",
                                (now - timedelta(seconds=1)).isoformat())
    expired = store.tool_approval_status("approval-request", "expired-call", None, "app")
    assert expired["status"] == "expired"
    with pytest.raises(ValueError, match="requires recorded external approval"):
        store.record_tool_execution_outcome("approval-request", "expired-call", None, "app", "succeeded")

    store.record_tool_approval("approval-request", "revoked-call", "approved", "approver",
                                (now + timedelta(hours=1)).isoformat())
    store.record_tool_approval("approval-request", "revoked-call", "revoked", "security-admin")
    revoked = store.tool_approval_status("approval-request", "revoked-call", None, "app")
    assert revoked["status"] == "revoked"
    with pytest.raises(ValueError, match="requires recorded external approval"):
        store.record_tool_execution_outcome("approval-request", "revoked-call", None, "app", "succeeded")


@pytest.mark.asyncio
async def test_pending_tool_calls_show_overdue_and_export_alert_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("GATEWAY_API_KEY", "admin-secret")
    tool_response = success_response().json()
    tool_response["choices"][0] = {"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call-pending-1", "type": "function", "function": {
            "name": "lookup_order", "arguments": '{"order_id":"A-17"}'}}
    ]}, "finish_reason": "tool_calls"}
    cfg = config(str(tmp_path / "pending-tools.sqlite3"))
    cfg["tool_governance"] = {"execution_timeout_seconds": 1}
    async with gateway_client(cfg, httpx.MockTransport(lambda _: httpx.Response(200, json=tool_response))) as client:
        completion = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer admin-secret"}, json={
            "model": "simple", "messages": [{"role": "user", "content": "find A-17"}],
            "tools": [{"type": "function", "function": {"name": "lookup_order",
                "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}}}],
        })
        assert completion.status_code == 200
        pending = await client.get("/v1/admin/tool-executions/pending",
                                   headers={"Authorization": "Bearer admin-secret"})
        assert pending.status_code == 200
        assert pending.json()["pending_count"] == 1
        assert pending.json()["overdue_count"] == 0
        assert pending.json()["calls"][0]["tool_call_id"] == "call-pending-1"

        # Deterministically age the recorded call without sleeping in the test.
        with client._transport.app.state.store._connect() as db:
            db.execute("UPDATE tool_call_audits SET occurred_at=? WHERE tool_call_id=?",
                       ((datetime.now(UTC) - timedelta(seconds=5)).isoformat(), "call-pending-1"))
        overdue = (await client.get("/v1/admin/tool-executions/pending",
                                    headers={"Authorization": "Bearer admin-secret"})).json()
        assert overdue["overdue_count"] == 1
        assert overdue["calls"][0]["overdue"] is True
        metrics = (await client.get("/metrics")).text
        assert "llm_gateway_tool_executions_pending 1.0" in metrics
        assert "llm_gateway_tool_executions_overdue 1.0" in metrics


@pytest.mark.asyncio
async def test_evaluation_dataset_versions_are_immutable_and_reports_persist(tmp_path):
    async with gateway_client(config(str(tmp_path / "evaluation-runs.sqlite3")),
                              httpx.MockTransport(lambda _: success_response())) as client:
        metadata = {"dataset_name": "support", "dataset_version": "v1",
                    "dataset_sha256": "a" * 64, "case_count": 2}
        saved = await client.post("/v1/admin/evaluations", json={**metadata, "report": {
            "comparison": {"promotion_passed": True}, "baseline": {}, "candidate": {}}})
        assert saved.status_code == 200
        listing = await client.get("/v1/admin/evaluations", params={"dataset_name": "support", "dataset_version": "v1"})
        assert listing.status_code == 200
        report = listing.json()["reports"][0]
        assert report["dataset_sha256"] == "a" * 64
        assert report["report"]["comparison"]["promotion_passed"] is True
        conflict = await client.post("/v1/admin/evaluations", json={**metadata,
            "dataset_sha256": "b" * 64, "report": {}})
        assert conflict.status_code == 409

        baseline = await client.post("/v1/admin/evaluations", json={
            "dataset_name": "support", "dataset_version": "v2", "dataset_sha256": "c" * 64,
            "case_count": 2, "report": {"routes": {"simple": {
                "requests": 2, "graded_cases": 2, "pass_rate": 0.5, "average_latency_ms": 100,
                "estimated_cost_usd": 0.01, "failures": 1}},
                "results": [{"route": "simple", "passed": True}, {"route": "simple", "passed": False}]}})
        candidate = await client.post("/v1/admin/evaluations", json={
            "dataset_name": "support", "dataset_version": "v3", "dataset_sha256": "d" * 64,
            "case_count": 2, "report": {"routes": {"simple": {
                "requests": 2, "graded_cases": 2, "pass_rate": 1.0, "average_latency_ms": 80,
                "estimated_cost_usd": 0.008, "failures": 0}, "reasoning": {
                "requests": 2, "graded_cases": 2, "pass_rate": 0.5, "average_latency_ms": 150,
                "estimated_cost_usd": 0.02, "failures": 1}},
                "results": [{"route": "simple", "passed": True}, {"route": "simple", "passed": True}]}})
        compared = await client.post("/v1/admin/evaluations/compare", json={
            "baseline_run_id": baseline.json()["run_id"], "candidate_run_id": candidate.json()["run_id"],
            "min_pass_rate": 0.75, "max_cost_regression_percent": 0, "max_failure_rate": 0})
        assert compared.status_code == 200
        comparison = compared.json()
        assert comparison["report"]["routes"]["simple"]["deltas"]["pass_rate"] == 0.5
        assert comparison["report"]["routes"]["reasoning"]["change"] == "added"
        assert comparison["report"]["promotion_passed"] is True
        saved_comparisons = await client.get("/v1/admin/evaluations/comparisons")
        assert saved_comparisons.json()["comparisons"][0]["comparison_id"] == comparison["comparison_id"]


def test_anthropic_and_gemini_tool_and_schema_translation():
    payload = {"messages": [
        {"role": "user", "content": "Find order A-17"},
        {"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function",
            "function": {"name": "lookup_order", "arguments": "{\"order_id\":\"A-17\"}"}}]},
        {"role": "tool", "tool_call_id": "call_123", "name": "lookup_order", "content": "{\"status\":\"shipped\"}"},
    ], "tools": [{"type": "function", "function": {"name": "lookup_order",
        "description": "Look up an order", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}}}],
        "tool_choice": "required", "response_format": {"type": "json_schema", "json_schema": {
            "name": "order_result", "schema": {"type": "object", "properties": {"status": {"type": "string"}}}}}}
    anthropic = ProviderClient._anthropic_payload("claude-test", payload)
    assert anthropic["tools"][0]["name"] == "lookup_order"
    assert anthropic["tool_choice"] == {"type": "any"}
    assert anthropic["messages"][1]["content"][0]["type"] == "tool_use"
    assert anthropic["messages"][2]["content"][0]["type"] == "tool_result"
    assert anthropic["output_config"]["format"]["schema"]["properties"]["status"]["type"] == "string"

    gemini = ProviderClient._gemini_payload(payload)
    assert gemini["tools"][0]["functionDeclarations"][0]["name"] == "lookup_order"
    assert gemini["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
    assert gemini["contents"][1]["parts"][0]["functionCall"]["args"] == {"order_id": "A-17"}
    assert gemini["contents"][2]["parts"][0]["functionResponse"]["response"] == {"status": "shipped"}
    assert gemini["generationConfig"]["responseSchema"]["properties"]["status"]["type"] == "string"


def test_image_content_is_translated_for_anthropic_and_gemini():
    payload = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
    ]}]}
    anthropic = ProviderClient._anthropic_payload("claude-test", payload)
    assert anthropic["messages"][0]["content"][0] == {"type": "text", "text": "Describe this"}
    assert anthropic["messages"][0]["content"][1]["source"] == {
        "type": "base64", "media_type": "image/png", "data": "aGVsbG8="}
    gemini = ProviderClient._gemini_payload(payload)
    assert gemini["contents"][0]["parts"][1]["inlineData"] == {
        "mimeType": "image/png", "data": "aGVsbG8="}


def test_gemini_audio_input_translation_and_capability_registry():
    from llm_gateway.capabilities import provider_capabilities

    payload = {"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}},
    ]}]}
    converted = ProviderClient._gemini_payload(payload)
    assert converted["contents"][0]["parts"][0]["inlineData"] == {
        "mimeType": "audio/wav", "data": "aGVsbG8="}
    assert provider_capabilities({"kind": "azure_openai"})["audio_output"] is True
    assert provider_capabilities({"kind": "bedrock"})["streaming"] is True
    assert provider_capabilities({"kind": "vllm"})["tools"] is True


@pytest.mark.asyncio
async def test_gemini_native_streaming_and_image_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    upstream_calls = 0
    stream_body = ('data: {"candidates":[{"content":{"parts":[{"text":"gemini"}]}}],'
                   '"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":2,"totalTokenCount":4}}\n\n')

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal upstream_calls
        upstream_calls += 1
        assert request.url.path.endswith(":streamGenerateContent")
        assert request.url.params["alt"] == "sse"
        body = json.loads(request.content)
        assert body["contents"][0]["parts"][1]["inlineData"]["mimeType"] == "image/png"
        return httpx.Response(200, content=stream_body, headers={"content-type": "text/event-stream"})

    cfg = config(str(tmp_path / "gemini-stream.sqlite3"))
    cfg["providers"]["primary"].update({"kind": "gemini", "base_url": "https://primary.test/v1beta"})
    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        payload = {
            "model": "simple", "stream": True,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
            ]}],
        }
        response = await client.post("/v1/chat/completions", json=payload,
                                     headers={"Idempotency-Key": "image-request-7"})
        replay = await client.post("/v1/chat/completions", json=payload,
                                   headers={"Idempotency-Key": "image-request-7"})
        assert response.status_code == 200
        assert '"content": "gemini"' in response.text
        assert replay.status_code == 200 and replay.text == response.text
        assert upstream_calls == 1
        report = (await client.get("/v1/admin/report")).json()
        assert report["total_tokens"] == 4


@pytest.mark.asyncio
async def test_cancelled_stream_records_usage_received_before_disconnect(tmp_path):
    from starlette.requests import Request

    app = create_app(config(str(tmp_path / "partial-stream.sqlite3")), http_transport=httpx.MockTransport(
        lambda request: httpx.Response(500)))

    async def interrupted_stream(route_name, payload, request_id=None, tenant_policy=None,
                                evaluation_context=None):
        yield {"provider_progress": {"provider": "primary", "model": "primary-model",
            "configured_model": "primary-model", "fallback_used": False}}
        yield {"usage_progress": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}}
        yield {"delta": "partial"}
        await asyncio.Event().wait()

    async with app.router.lifespan_context(app):
        app.state.gateway.stream_complete = interrupted_stream
        route = next(route for route in app.routes if route.path == "/v1/chat/completions")
        request = Request({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions", "query_string": b"", "headers": [],
            "server": ("gateway", 80), "client": ("test", 1), "root_path": "", "app": app})
        response = await route.endpoint(ChatRequest(model="simple", stream=True,
            messages=[{"role": "user", "content": "hello"}]), request, None, None)
        frames = response.body_iterator
        await frames.__anext__()  # assistant role preamble
        partial = await frames.__anext__()
        assert '"content": "partial"' in partial
        await frames.aclose()
        report = app.state.store.report(None, None, None)
        assert report["total_tokens"] == 10
        assert report["failures"] == 1
        assert report["cost_usd"] == 0.000012


@pytest.mark.asyncio
async def test_comparison_evaluator_enforces_candidate_regression_gates():
    calls: list[str] = []

    def gateway(request: httpx.Request) -> httpx.Response:
        route = json.loads(request.content)["model"]
        calls.append(route)
        content = "approved within 30 days" if route == "baseline" else "not sure"
        cost = 0.001 if route == "baseline" else 0.002
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
            "gateway": {"cost_usd": cost, "provider": "mock", "upstream_model": route}})

    cases = [{"id": "refund", "messages": [{"role": "user", "content": "refund?"}],
              "expected_contains": ["30 days"]}]
    report = await compare_routes(cases, "http://gateway", "baseline", "candidate",
        transport=httpx.MockTransport(gateway), min_pass_rate=1.0,
        max_cost_regression_percent=50, max_latency_regression_percent=1000)
    assert sorted(calls) == ["baseline", "candidate"]
    assert report["comparison"]["pass_rate_delta"] == -1.0
    assert report["comparison"]["cost_regression_percent"] == 100.0
    assert report["comparison"]["promotion_passed"] is False


@pytest.mark.asyncio
async def test_streaming_tool_call_deltas_are_normalized(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    stream_body = (
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",'
        '"function":{"name":"lookup_order","arguments":""}}]}}]}\n\n'
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"id\\":\\"A-7\\"}"}}]}}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}}\n\n'
        'data: [DONE]\n\n')

    def upstream(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=stream_body, headers={"content-type": "text/event-stream"})

    async with gateway_client(config(str(tmp_path / "stream-tools.sqlite3")), httpx.MockTransport(upstream)) as client:
        response = await client.post("/v1/chat/completions", json={
            "model": "simple", "stream": True,
            "messages": [{"role": "user", "content": "Look up A-7"}],
            "tools": [{"type": "function", "function": {"name": "lookup_order",
                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}}}}],
        })
        assert response.status_code == 200
        assert '"tool_calls"' in response.text
        assert '"name": "lookup_order"' in response.text
        assert '"arguments": "{\\"id\\":\\"A-7\\"}"' in response.text


@pytest.mark.asyncio
async def test_audio_input_and_output_api_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    received = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        received.update(json.loads(request.content))
        return httpx.Response(200, json={"model": "primary-model", "choices": [{
            "message": {"role": "assistant", "content": "spoken", "audio": {
                "id": "audio-1", "data": "YXVkaW8=", "transcript": "spoken"}},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}})

    async with gateway_client(config(str(tmp_path / "audio.sqlite3")), httpx.MockTransport(upstream)) as client:
        response = await client.post("/v1/chat/completions", json={
            "model": "simple", "modalities": ["text", "audio"],
            "audio": {"voice": "alloy", "format": "wav"},
            "messages": [{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}},
            ]}],
        })
        assert response.status_code == 200
        assert received["modalities"] == ["text", "audio"]
        assert received["audio"]["voice"] == "alloy"
        assert response.json()["choices"][0]["message"]["audio"]["id"] == "audio-1"


@pytest.mark.asyncio
async def test_capability_registry_skips_provider_without_audio_support(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    requests = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return success_response()

    cfg = config(str(tmp_path / "audio-fallback.sqlite3"))
    cfg["providers"]["primary"]["kind"] = "anthropic"
    cfg["providers"]["primary"]["base_url"] = "https://primary.test"
    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        response = await client.post("/v1/chat/completions", json={
            "model": "simple", "messages": [{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}},
            ]}],
        })
        assert response.status_code == 200
        assert response.json()["gateway"]["provider"] == "backup"
        assert len(requests) == 1
        route = (await client.get("/v1/routes")).json()["simple"]
        assert route["capabilities"]["audio_input"] is False
        assert route["fallbacks"][0]["capabilities"]["audio_input"] is True


@pytest.mark.asyncio
async def test_azure_bedrock_vertex_and_vllm_provider_adapters(monkeypatch):
    monkeypatch.setenv("AZURE_TEST_KEY", "azure-secret")
    monkeypatch.setenv("GOOGLE_CLOUD_ACCESS_TOKEN", "vertex-token")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-key")
    requests = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "bedrock.test":
            return httpx.Response(200, json={"output": {"message": {"content": [{"text": "bedrock"}]}},
                "stopReason": "end_turn", "usage": {"inputTokens": 2, "outputTokens": 1}})
        if request.url.host == "vertex.test":
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "vertex"}]}}],
                "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3}})
        return httpx.Response(200, json={"model": "served", "choices": [{"message": {
            "role": "assistant", "content": request.url.host}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}})

    providers = {
        "azure": {"kind": "azure_openai", "base_url": "https://azure.test", "api_key_env": "AZURE_TEST_KEY",
                  "api_version": "2025-01-01", "models": {}},
        "bedrock": {"kind": "bedrock", "base_url": "https://bedrock.test", "region": "us-east-1", "models": {}},
        "vertex": {"kind": "vertex", "base_url": "https://vertex.test/v1", "region": "us-central1",
                   "project_id": "demo-project", "models": {}},
        "vllm": {"kind": "vllm", "base_url": "https://vllm.test/v1", "api_key_optional": True, "models": {}},
    }
    http = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    provider = ProviderClient(http, providers, 2)
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    try:
        for name, model in (("azure", "deployment-a"), ("bedrock", "anthropic.claude-v3"),
                            ("vertex", "gemini-pro"), ("vllm", "local-model")):
            result = await provider.complete(name, model, payload)
            assert result["usage"]["total_tokens"] == 3
        azure_request, bedrock_request, vertex_request, vllm_request = requests
        assert azure_request.url.path == "/openai/deployments/deployment-a/chat/completions"
        assert azure_request.url.params["api-version"] == "2025-01-01"
        assert azure_request.headers["api-key"] == "azure-secret"
        assert bedrock_request.url.path == "/model/anthropic.claude-v3/converse"
        assert bedrock_request.headers["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert vertex_request.url.path.endswith("/publishers/google/models/gemini-pro:generateContent")
        assert vertex_request.headers["Authorization"] == "Bearer vertex-token"
        assert vllm_request.url.path == "/v1/chat/completions"
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_oidc_scopes_audit_log_and_dashboard(tmp_path, monkeypatch):
    fake_jwt = ModuleType("jwt")

    class FakeJwkClient:
        def __init__(self, jwks_url, cache_jwk_set=True):
            assert jwks_url == "https://identity.test/jwks"

        def get_signing_key_from_jwt(self, token):
            assert token in {"signed-token", "audit-token"}
            return SimpleNamespace(key="public-key")

    def decode(token, key, algorithms, issuer, audience, options):
        assert key == "public-key"
        assert algorithms == ["RS256"]
        assert issuer == "https://identity.test/"
        assert audience == "llm-gateway"
        assert options["require"] == ["exp", "sub"]
        if token == "audit-token":
            return {"sub": "auditor", "scope": "llm-gateway:audit"}
        return {"sub": "finance-user", "scope": "llm-gateway:report"}

    fake_jwt.PyJWKClient = FakeJwkClient
    fake_jwt.decode = decode
    monkeypatch.setitem(sys.modules, "jwt", fake_jwt)
    cfg = config(str(tmp_path / "oidc.sqlite3"))
    cfg["security"]["oidc"] = {"enabled": True, "issuer": "https://identity.test/",
        "audience": "llm-gateway", "jwks_url": "https://identity.test/jwks"}
    async with gateway_client(cfg, httpx.MockTransport(lambda request: success_response())) as client:
        headers = {"Authorization": "Bearer signed-token"}
        assert (await client.get("/v1/admin/report", headers=headers)).status_code == 200
        assert (await client.get("/v1/admin/budgets", headers=headers)).status_code == 403
        dashboard = await client.get("/admin")
        assert dashboard.status_code == 200
        assert "Spend by cost center" in dashboard.text
        audit = await client.get("/v1/admin/audit", headers={"Authorization": "Bearer audit-token"})
        assert audit.status_code == 200
        assert audit.json()["events"][0]["actor"] == "finance-user"


@pytest.mark.asyncio
async def test_cost_center_reporting_and_csv_export(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")

    def upstream(request: httpx.Request) -> httpx.Response:
        return success_response()

    async with gateway_client(config(str(tmp_path / "exports.sqlite3")), httpx.MockTransport(upstream)) as client:
        response = await client.post("/v1/chat/completions", json={
            "model": "simple", "application": "billing", "cost_center": "=HYPERLINK(1)",
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200
        report = (await client.get("/v1/admin/report", params={"cost_center": "=HYPERLINK(1)"})).json()
        assert report["requests"] == 1
        assert report["by_cost_center"][0]["cost_center"] == "=HYPERLINK(1)"
        assert report["by_provider"][0]["provider"] == "primary"
        exported = await client.get("/v1/admin/report.csv", params={"cost_center": "=HYPERLINK(1)"})
        assert exported.status_code == 200
        assert "'=HYPERLINK(1)" in exported.text
        assert "cost_usd" in exported.text.splitlines()[0]


@pytest.mark.asyncio
async def test_admin_keys_support_scopes_rotation_and_revocation(tmp_path, monkeypatch):
    monkeypatch.setenv("REPORT_KEY", "report-key")
    monkeypatch.setenv("OLD_REPORT_KEY", "old-report-key")
    monkeypatch.delenv("GATEWAY_API_KEY", raising=False)
    cfg = config(str(tmp_path / "admin-keys.sqlite3"))
    cfg["security"]["admin_api_keys"] = [
        {"id": "report-current", "api_key_env": "REPORT_KEY", "scopes": ["report"]},
        {"id": "report-old", "api_key_env": "OLD_REPORT_KEY", "scopes": ["report"], "enabled": False},
    ]
    async with gateway_client(cfg, httpx.MockTransport(lambda request: success_response())) as client:
        headers = {"Authorization": "Bearer report-key"}
        assert (await client.get("/v1/admin/report", headers=headers)).status_code == 200
        assert (await client.get("/v1/admin/budgets", headers=headers)).status_code == 403
        old_headers = {"Authorization": "Bearer old-report-key"}
        assert (await client.get("/v1/admin/report", headers=old_headers)).status_code == 401


@pytest.mark.asyncio
async def test_async_job_admin_scopes_separate_read_cancel_and_retry(tmp_path, monkeypatch):
    for name, value in (("REPORT_KEY", "report-key"), ("JOB_READ_KEY", "job-read-key"),
                        ("JOB_CANCEL_KEY", "job-cancel-key"), ("JOB_RETRY_KEY", "job-retry-key")):
        monkeypatch.setenv(name, value)
    cfg = config(str(tmp_path / "job-scopes.sqlite3"))
    cfg["security"]["admin_api_keys"] = [
        {"id": "report", "api_key_env": "REPORT_KEY", "scopes": ["report"]},
        {"id": "job-reader", "api_key_env": "JOB_READ_KEY", "scopes": ["async_jobs:read"]},
        {"id": "job-canceller", "api_key_env": "JOB_CANCEL_KEY", "scopes": ["async_jobs:cancel"]},
        {"id": "job-retry", "api_key_env": "JOB_RETRY_KEY", "scopes": ["async_jobs:retry"]},
    ]
    async with gateway_client(cfg, httpx.MockTransport(lambda request: success_response())) as client:
        report = {"Authorization": "Bearer report-key"}
        reader = {"Authorization": "Bearer job-read-key"}
        canceller = {"Authorization": "Bearer job-cancel-key"}
        retryer = {"Authorization": "Bearer job-retry-key"}
        missing_retry = "/v1/admin/async-jobs/missing/retry?confirm_job_id=missing"
        assert (await client.get("/v1/admin/async-jobs", headers=report)).status_code == 403
        assert (await client.post(missing_retry, headers=report)).status_code == 403
        assert (await client.get("/v1/admin/async-jobs", headers=reader)).status_code == 200
        assert (await client.delete("/v1/admin/async-jobs/missing", headers=reader)).status_code == 403
        assert (await client.delete("/v1/admin/async-jobs/missing", headers=canceller)).status_code == 404
        assert (await client.post(missing_retry, headers=retryer)).status_code == 404


def test_provider_normalization_preserves_tool_calls_without_text():
    anthropic = ProviderClient._normalize({"id": "msg_1", "content": [{"type": "tool_use", "id": "toolu_1",
        "name": "lookup_order", "input": {"order_id": "A-17"}}], "usage": {"input_tokens": 3, "output_tokens": 4}},
        "anthropic", "claude-test")
    assert anthropic["text"] == ""
    assert anthropic["tool_calls"][0]["id"] == "toolu_1"
    gemini = ProviderClient._normalize({"candidates": [{"content": {"parts": [{"functionCall": {
        "name": "lookup_order", "args": {"order_id": "A-17"}}}]}}], "usageMetadata": {"promptTokenCount": 2}},
        "gemini", "gemini-test")
    assert gemini["tool_calls"][0]["function"]["name"] == "lookup_order"


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
async def test_anthropic_native_streaming_is_normalized(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    stream_body = ('event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":2}}}\n\n'
                   'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"native "}}\n\n'
                   'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"stream"}}\n\n'
                   'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}\n\n'
                   'event: message_stop\ndata: {"type":"message_stop"}\n\n')

    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "primary.test"
        assert request.url.path == "/v1/messages"
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=stream_body, headers={"content-type": "text/event-stream"})

    cfg = config(str(tmp_path / "stream-fallback.sqlite3"))
    cfg["providers"]["primary"]["kind"] = "anthropic"
    cfg["providers"]["primary"]["base_url"] = "https://primary.test"
    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        response = await client.post("/v1/chat/completions", json={
            "model": "simple", "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200
        assert '"content": "native "' in response.text
        assert '"content": "stream"' in response.text
        report = (await client.get("/v1/admin/report", params={"application": "default"})).json()
        assert report["fallbacks"] == 0
        assert report["total_tokens"] == 5


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


@pytest.mark.asyncio
async def test_canary_route_is_sampled_and_reported_without_fallback_accounting(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    cfg = config(str(tmp_path / "canary.sqlite3"))
    cfg["routes"]["simple"]["canary"] = {"provider": "backup", "model": "backup-model", "traffic_percent": 100}
    cfg["routes"]["simple"]["cache"] = {"enabled": True, "ttl_seconds": 30}
    hosts = []

    def upstream(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        body = success_response().json()
        body["model"] = "backup-model" if request.url.host == "backup.test" else "primary-model"
        return httpx.Response(200, json=body)

    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        response = await client.post("/v1/chat/completions", json={"model": "simple", "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 200
        assert response.json()["gateway"]["canary_used"] is True
        assert response.json()["gateway"]["fallback_used"] is False
        assert hosts == ["backup.test"]
        report = (await client.get("/v1/admin/report")).json()
        assert report["canary_requests"] == 1


@pytest.mark.asyncio
async def test_ofrep_canary_selection_uses_application_context(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    cfg = config(str(tmp_path / "flag-canary.sqlite3"))
    cfg["routes"]["simple"]["canary"] = {
        "provider": "backup", "model": "backup-model", "traffic_percent": 0}
    cfg["routing"] = {"feature_flags": {"enabled": True, "base_url": "https://flags.test",
        "flag_key": "canary", "timeout_seconds": 0.5}}
    seen_context = []
    provider_hosts = []

    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.host == "flags.test":
            seen_context.append(json.loads(request.content)["context"])
            return httpx.Response(200, json={"key": "canary", "value": True,
                "reason": "TARGETING_MATCH"})
        provider_hosts.append(request.url.host)
        return success_response()

    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        response = await client.post("/v1/chat/completions", json={"model": "simple",
            "application": "support-router", "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 200
        assert response.json()["gateway"]["canary_used"] is True
        assert provider_hosts == ["backup.test"]
        assert seen_context == [{"targetingKey": "support-router", "application": "support-router"}]
        decisions = (await client.get("/v1/admin/routing-decisions")).json()["events"]
        assert any(item["reason"] == "TARGETING_MATCH" and item["decision"] == "selected"
                   for item in decisions)


@pytest.mark.asyncio
async def test_ofrep_failure_fails_closed_and_slo_report_tracks_burn(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    cfg = config(str(tmp_path / "slo.sqlite3"))
    cfg["routes"]["simple"]["canary"] = {
        "provider": "backup", "model": "backup-model", "traffic_percent": 100}
    cfg["routing"] = {"feature_flags": {"enabled": True, "base_url": "https://flags.test",
        "flag_key": "canary", "timeout_seconds": 0.5}}
    cfg["slos"] = {"default": {"availability_target": 0.9,
        "latency_target_ms": 0, "latency_objective": 0.9}}
    provider_hosts = []

    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.host == "flags.test":
            return httpx.Response(503)
        provider_hosts.append(request.url.host)
        return httpx.Response(500)

    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        response = await client.post("/v1/chat/completions", json={"model": "simple",
            "application": "billing", "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 502
        assert provider_hosts == ["primary.test", "primary.test", "backup.test", "backup.test"]
        slo = (await client.get("/v1/admin/slos", params={"application": "billing"})).json()
        row = slo["slos"][0]
        assert row["requests"] == 1
        assert row["availability"] == 0
        assert row["availability_error_budget_burn_rate"] == pytest.approx(10)
        assert row["latency_compliance"] == 0
        metrics = (await client.get("/metrics")).text
        assert "gen_ai_client_inference_duration_seconds" in metrics
        assert 'llm_gateway_slo_requests_total{application="billing",outcome="failure",route="simple"} 1.0' in metrics


@pytest.mark.asyncio
async def test_cost_reconciliation_preserves_estimates_and_reports_actual_cost(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    async with gateway_client(config(str(tmp_path / "reconcile.sqlite3")),
                              httpx.MockTransport(lambda request: success_response())) as client:
        created = await client.post("/v1/chat/completions", json={"model": "simple", "messages": [{"role": "user", "content": "hello"}]})
        request_id = created.json()["id"]
        estimate = created.json()["gateway"]["cost_usd"]
        imported = await client.post("/v1/admin/cost-reconciliation", json={"source": "invoice-2026-10", "records": [
            {"request_id": request_id, "actual_cost_usd": 0.0005}]})
        assert imported.status_code == 200
        report = (await client.get("/v1/admin/report")).json()
        assert report["cost_usd"] == estimate
        assert report["actual_cost_usd"] == 0.0005
        assert report["cost_variance_usd"] == pytest.approx(0.0005 - estimate)
        assert report["reconciled_requests"] == 1
        unknown = await client.post("/v1/admin/cost-reconciliation", json={"source": "bad-batch", "records": [
            {"request_id": request_id, "actual_cost_usd": 0.0003},
            {"request_id": "missing", "actual_cost_usd": 0.0002}]})
        assert unknown.status_code == 400
        report = (await client.get("/v1/admin/report")).json()
        assert report["actual_cost_usd"] == 0.0005


@pytest.mark.asyncio
async def test_billing_export_import_maps_columns_and_scales_micro_usd(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    async with gateway_client(config(str(tmp_path / "billing-import.sqlite3")),
                              httpx.MockTransport(lambda request: success_response())) as client:
        created = await client.post("/v1/chat/completions", json={
            "model": "simple", "messages": [{"role": "user", "content": "hello"}]})
        request_id = created.json()["id"]
        imported = await client.post("/v1/admin/cost-reconciliation/import", json={
            "source": "provider-export", "format": "csv",
            "mapping": {"request_id": "gateway_request_id", "actual_cost_usd": "billed_micros",
                        "currency": "currency"},
            "cost_multiplier": 0.000001,
            "content": f"gateway_request_id,billed_micros,currency\n{request_id},1250,USD",
        })
        assert imported.status_code == 200
        assert imported.json()["imported_rows"] == 1
        report = (await client.get("/v1/admin/report")).json()
        assert report["actual_cost_usd"] == pytest.approx(0.00125)
        foreign_currency = await client.post("/v1/admin/cost-reconciliation/import", json={
            "source": "bad-export", "format": "json",
            "content": json.dumps([{"request_id": request_id, "cost_usd": 1, "currency": "EUR"}]),
        })
        assert foreign_currency.status_code == 400


@pytest.mark.asyncio
async def test_tenant_tool_allowlist_and_call_limits_are_audited(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPPORT_TENANT_KEY", "support-secret")
    monkeypatch.setenv("GATEWAY_API_KEY", "admin-secret")
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    provider_calls = 0

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal provider_calls
        provider_calls += 1
        calls = [{"id": f"call-{index}", "type": "function", "function": {
            "name": "lookup_order", "arguments": "{}"}} for index in range(2 if provider_calls == 2 else 1)]
        return httpx.Response(200, json={"model": "primary-model", "choices": [{"message": {
            "role": "assistant", "content": None, "tool_calls": calls}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}})

    cfg = config(str(tmp_path / "tool-policy.sqlite3"))
    cfg["tenants"] = {"support": {"api_key_env": "SUPPORT_TENANT_KEY", "allowed_routes": ["simple"],
        "allowed_tools": ["lookup_order"], "applications": {"support-router": {
            "max_tool_calls_per_request": 1}}}}
    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        request_body = {"model": "simple", "application": "support-router",
            "messages": [{"role": "user", "content": "find order"}],
            "tools": [{"type": "function", "function": {"name": "lookup_order",
                "parameters": {"type": "object"}}}]}
        headers = {"Authorization": "Bearer support-secret"}
        allowed = await client.post("/v1/chat/completions", json=request_body, headers=headers)
        assert allowed.status_code == 200, allowed.text
        excessive = await client.post("/v1/chat/completions", json=request_body, headers=headers)
        assert excessive.status_code == 502
        forbidden_body = {**request_body, "tools": [{"type": "function", "function": {
            "name": "delete_account", "parameters": {"type": "object"}}}]}
        forbidden = await client.post("/v1/chat/completions", json=forbidden_body, headers=headers)
        assert forbidden.status_code == 403
        assert provider_calls == 2
        audit = await client.get("/v1/admin/tool-audit?tenant_id=support",
                                 headers={"Authorization": "Bearer admin-secret"})
        assert audit.status_code == 200
        events = audit.json()["events"]
        assert any(event["tool_name"] == "lookup_order" and event["outcome"] == "allowed" for event in events)
        assert any(event["tool_name"] == "lookup_order" and event["reason"] == "max_tool_calls_per_request"
                   for event in events)
        assert any(event["tool_name"] == "delete_account" and event["outcome"] == "denied" for event in events)
        assert all("arguments" not in event for event in events)


@pytest.mark.asyncio
async def test_streamed_disallowed_tool_call_is_withheld_and_audited(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPPORT_TENANT_KEY", "support-secret")
    monkeypatch.setenv("GATEWAY_API_KEY", "admin-secret")
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    stream_body = (
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",'
        '"function":{"name":"lookup_order","arguments":"{}"}}]}}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n\n'
        'data: [DONE]\n\n')
    cfg = config(str(tmp_path / "stream-tool-policy.sqlite3"))
    cfg["tenants"] = {"support": {"api_key_env": "SUPPORT_TENANT_KEY", "allowed_routes": ["simple"],
        "allowed_tools": ["safe_lookup"], "applications": {"support-router": {}}}}
    async with gateway_client(cfg, httpx.MockTransport(lambda request: httpx.Response(200,
            content=stream_body, headers={"content-type": "text/event-stream"}))) as client:
        response = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer support-secret"},
            json={"model": "simple", "application": "support-router", "stream": True,
                "messages": [{"role": "user", "content": "find order"}],
                "tools": [{"type": "function", "function": {"name": "safe_lookup",
                    "parameters": {"type": "object"}}}]})
        assert response.status_code == 200
        assert "event: error" in response.text
        assert '"tool_calls"' not in response.text
        audit = await client.get("/v1/admin/tool-audit?tenant_id=support",
                                 headers={"Authorization": "Bearer admin-secret"})
        assert any(event["tool_name"] == "lookup_order" and event["outcome"] == "denied"
                   for event in audit.json()["events"])


@pytest.mark.asyncio
async def test_tool_call_minute_quota_rejects_excess_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPPORT_TENANT_KEY", "support-secret")
    monkeypatch.setenv("GATEWAY_API_KEY", "admin-secret")
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    cfg = config(str(tmp_path / "tool-minute-quota.sqlite3"))
    cfg["tenants"] = {"support": {"api_key_env": "SUPPORT_TENANT_KEY", "allowed_routes": ["simple"],
        "allowed_tools": ["lookup_order"], "applications": {"support-router": {
            "max_tool_calls_per_minute": 1}}}}
    async with gateway_client(cfg, httpx.MockTransport(lambda request: httpx.Response(200, json={
            "model": "primary-model", "choices": [{"message": {"role": "assistant", "content": None,
                "tool_calls": [{"id": "call-1", "type": "function", "function": {
                    "name": "lookup_order", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}))) as client:
        body = {"model": "simple", "application": "support-router",
            "messages": [{"role": "user", "content": "find order"}],
            "tools": [{"type": "function", "function": {"name": "lookup_order",
                "parameters": {"type": "object"}}}]}
        headers = {"Authorization": "Bearer support-secret"}
        assert (await client.post("/v1/chat/completions", json=body, headers=headers)).status_code == 200
        limited = await client.post("/v1/chat/completions", json=body, headers=headers)
        assert limited.status_code == 429
        audit = await client.get("/v1/admin/tool-audit?tenant_id=support",
                                 headers={"Authorization": "Bearer admin-secret"})
        assert any(event["reason"] == "max_tool_calls_per_minute" and event["outcome"] == "denied"
                   for event in audit.json()["events"])


@pytest.mark.asyncio
async def test_eval_runner_reports_quality_cost_and_latency():
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "eval-id", "model": "test-model",
            "choices": [{"message": {"role": "assistant", "content": "The result is 42."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            "gateway": {"provider": "primary", "upstream_model": "test-model", "cost_usd": 0.0001}})

    report = await evaluate_cases([{"id": "math", "route": "simple", "messages": [{"role": "user", "content": "compute"}],
                                   "expected_contains": "42"}], "http://gateway", transport=httpx.MockTransport(upstream))
    summary = report["routes"]["simple"]
    assert summary["pass_rate"] == 1
    assert summary["estimated_cost_usd"] == 0.0001
    assert summary["graded_cases"] == 1


@pytest.mark.asyncio
async def test_redis_concurrency_leases_are_shared_across_instances():
    from llm_gateway.distributed import RedisConcurrencyLimiter

    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    first = RedisConcurrencyLimiter(redis, 1)
    second = RedisConcurrencyLimiter(redis, 1)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold_slot():
        async with first:
            entered.set()
            await release.wait()

    async def wait_for_slot():
        await entered.wait()
        async with second:
            return True

    holder = asyncio.create_task(hold_slot())
    waiter = asyncio.create_task(wait_for_slot())
    await entered.wait()
    await asyncio.sleep(0.05)
    assert not waiter.done()
    release.set()
    assert await waiter is True
    await holder
    await redis.aclose()


@pytest.mark.asyncio
async def test_circuit_breaker_state_is_shared_in_redis():
    from llm_gateway.circuit import ProviderCircuitBreaker

    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    first = ProviderCircuitBreaker(["provider"], failure_threshold=1, recovery_seconds=10, redis=redis)
    second = ProviderCircuitBreaker(["provider"], failure_threshold=1, recovery_seconds=10, redis=redis)
    assert await first.allow("provider")
    await first.failure("provider")
    assert not await second.allow("provider")
    assert (await second.snapshot())["provider"]["state"] == "open"
    await redis.aclose()


@pytest.mark.asyncio
async def test_provider_credential_file_rotation_takes_effect_without_recreating_client(tmp_path):
    secret_file = tmp_path / "openai-key"
    secret_file.write_text("key-v1\n", encoding="utf-8")
    seen = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return success_response()

    provider = {"primary": {"kind": "openai", "base_url": "https://primary.test/v1",
        "api_key_file": str(secret_file), "models": {"primary-model": {}}}}
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http:
        client = ProviderClient(http, provider, 2)
        await client.complete("primary", "primary-model", {"messages": [{"role": "user", "content": "hello"}]})
        secret_file.write_text("key-v2\n", encoding="utf-8")
        await client.complete("primary", "primary-model", {"messages": [{"role": "user", "content": "hello"}]})
    assert seen == ["Bearer key-v1", "Bearer key-v2"]


@pytest.mark.asyncio
async def test_provider_request_correlation_supports_custom_header_and_body_path():
    seen = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append((request.headers.get("X-Invoice-Request"), json.loads(request.content)))
        return success_response()

    provider = {"primary": {"kind": "openai", "base_url": "https://primary.test/v1",
        "api_key_optional": True, "request_id_header": "X-Invoice-Request",
        "request_id_body_field": "metadata.gateway.request_id", "models": {"primary-model": {}}}}
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http:
        client = ProviderClient(http, provider, 2)
        await client.complete("primary", "primary-model", {"messages": [
            {"role": "user", "content": "hello"}]}, request_id="gw_invoice_123")
    assert seen[0][0] == "gw_invoice_123"
    assert seen[0][1]["metadata"]["gateway"]["request_id"] == "gw_invoice_123"


@pytest.mark.asyncio
async def test_tenant_secret_file_rotation_rejects_old_key_immediately(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch_path = tmp_path / "tenant-key"
    monkeypatch_path.write_text("tenant-v1\n", encoding="utf-8")
    cfg = config(str(tmp_path / "tenant-rotation.sqlite3"))
    cfg["tenants"] = {"support": {"secret_file": str(monkeypatch_path), "allowed_routes": ["simple"],
        "applications": {"support-router": {}}}}
    async with gateway_client(cfg, httpx.MockTransport(lambda request: success_response())) as client:
        payload = {"model": "simple", "application": "support-router",
                   "messages": [{"role": "user", "content": "hello"}]}
        assert (await client.post("/v1/chat/completions", json=payload,
                                  headers={"Authorization": "Bearer tenant-v1"})).status_code == 200
        monkeypatch_path.write_text("tenant-v2\n", encoding="utf-8")
        assert (await client.post("/v1/chat/completions", json=payload,
                                  headers={"Authorization": "Bearer tenant-v1"})).status_code == 401
        assert (await client.post("/v1/chat/completions", json=payload,
                                  headers={"Authorization": "Bearer tenant-v2"})).status_code == 200


@pytest.mark.asyncio
async def test_tenant_provider_model_and_region_policies_filter_routes(tmp_path, monkeypatch):
    monkeypatch.setenv("EU_CUSTOMER_KEY", "eu-key")
    monkeypatch.setenv("OPEN_CUSTOMER_KEY", "open-key")
    monkeypatch.setenv("APAC_CUSTOMER_KEY", "apac-key")
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    cfg = config(str(tmp_path / "tenant-data-policy.sqlite3"))
    cfg["routes"]["simple"]["cache"] = {"enabled": True, "ttl_seconds": 30}
    cfg["providers"]["primary"]["data_region"] = "us"
    cfg["providers"]["backup"]["data_region"] = "eu"
    cfg["tenants"] = {
        "open-customer": {"api_key_env": "OPEN_CUSTOMER_KEY", "allowed_routes": ["simple"],
            "applications": {"support-router": {}}},
        "eu-customer": {"api_key_env": "EU_CUSTOMER_KEY", "allowed_routes": ["simple"],
            "allowed_providers": ["backup"], "allowed_models": ["backup-model"], "allowed_regions": ["eu"],
            "applications": {"support-router": {}}},
        "apac-customer": {"api_key_env": "APAC_CUSTOMER_KEY", "allowed_routes": ["simple"],
            "allowed_regions": ["apac"], "applications": {"support-router": {}}},
    }
    hosts = []

    def upstream(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return success_response()

    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        open_headers = {"Authorization": "Bearer open-key"}
        eu_headers = {"Authorization": "Bearer eu-key"}
        routes = (await client.get("/v1/routes", headers=eu_headers)).json()
        assert routes["simple"]["provider"] == "backup"
        assert routes["simple"]["model"] == "backup-model"
        assert routes["simple"]["fallbacks"] == []
        payload = {
            "model": "simple", "application": "support-router",
            "messages": [{"role": "user", "content": "hello"}]}
        assert (await client.post("/v1/chat/completions", headers=open_headers, json=payload)).status_code == 200
        response = await client.post("/v1/chat/completions", headers=eu_headers, json=payload)
        assert response.status_code == 200
        assert hosts == ["primary.test", "backup.test"]
        apac_headers = {"Authorization": "Bearer apac-key"}
        assert (await client.get("/v1/routes", headers=apac_headers)).json() == {}
        denied = await client.post("/v1/chat/completions", headers=apac_headers, json=payload)
        assert denied.status_code == 403
        assert hosts == ["primary.test", "backup.test"]


def test_tenant_usage_retention_prunes_only_the_tenant_prefix(tmp_path):
    store = create_usage_store(str(tmp_path / "retention.sqlite3"))
    old_time = datetime(2026, 1, 1, tzinfo=UTC)
    base = {"occurred_at": old_time.isoformat(), "route": "simple", "provider": "primary",
        "model": "primary-model", "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
        "cost_usd": 0.0001, "status": "success", "latency_ms": 1,
        "fallback_used": 0, "cached": 0, "canary_used": 0}
    store.record({**base, "request_id": "tenant-request", "application": "support:app"})
    store.record({**base, "request_id": "other-request", "application": "other:app"})
    deleted = store.prune_tenant_requests("support", 35, datetime(2026, 3, 1, tzinfo=UTC))
    assert deleted == 1
    assert store.report(None, None, "support:app")["requests"] == 0
    assert store.report(None, None, "other:app")["requests"] == 1


def test_tenant_usage_retention_must_preserve_current_month_budget_window(tmp_path):
    cfg = config(str(tmp_path / "invalid-retention.sqlite3"))
    cfg["tenants"] = {"short": {"data_policy": {"usage_retention_days": 7}}}
    with pytest.raises(ValueError, match="at least 35"):
        create_app(cfg, http_transport=httpx.MockTransport(lambda request: success_response()))


@pytest.mark.asyncio
async def test_embeddings_and_rerank_are_billed_and_reported_by_operation(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    cfg = config(str(tmp_path / "retrieval.sqlite3"))
    cfg["providers"]["primary"]["models"]["embed-model"] = {
        "input_cost_per_1m": 2.0, "pricing_version": "prices-v1"}
    cfg["providers"]["backup"]["models"]["embed-model"] = {"input_cost_per_1m": 2.0}
    cfg["embedding_routes"] = {"retrieval": {"provider": "primary", "model": "embed-model",
        "fallbacks": [{"provider": "backup", "model": "embed-model"}]}}

    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/embeddings")
        if request.url.host == "primary.test":
            return httpx.Response(503)
        items = json.loads(request.content)["input"]
        vectors = [[1.0, 0.0] if value in {"query", "best"} else [0.0, 1.0] for value in items]
        return httpx.Response(200, json={"model": "embed-model", "data": [
            {"index": index, "embedding": vector} for index, vector in enumerate(vectors)],
            "usage": {"prompt_tokens": len(items) * 3, "total_tokens": len(items) * 3}})

    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        catalog = (await client.get("/v1/routes")).json()
        assert catalog["embedding_routes"]["retrieval"]["capabilities"]["embeddings"] is True
        embedding = await client.post("/v1/embeddings", json={"model": "retrieval",
            "input": ["query", "best"], "application": "search"})
        assert embedding.status_code == 200
        assert embedding.json()["data"][1]["embedding"] == [1.0, 0.0]
        assert embedding.json()["gateway"]["fallback_used"] is True
        ranked = await client.post("/v1/rerank", json={"model": "retrieval", "query": "query",
            "documents": ["other", "best"], "application": "search"})
        assert ranked.status_code == 200
        assert [item["index"] for item in ranked.json()["results"]] == [1, 0]
        assert ranked.json()["usage"]["prompt_tokens"] == 9
        report = (await client.get("/v1/admin/report", params={"application": "search"})).json()
        by_operation = {item["operation"]: item for item in report["by_operation"]}
        assert by_operation["embeddings"]["requests"] == 1
        assert by_operation["rerank"]["requests"] == 1
        assert report["total_tokens"] == 15
        assert report["cost_usd"] == pytest.approx(0.00003)
        assert report["fallbacks"] == 2
        assert any(item["decision"] == "fallback" for item in report["by_routing_decision"])


@pytest.mark.asyncio
async def test_async_embedding_job_batches_reports_progress_usage_and_item_results(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    cfg = config(str(tmp_path / "embedding-jobs.sqlite3"))
    cfg["providers"]["primary"]["models"]["embed-model"] = {"input_cost_per_1m": 1}
    cfg["providers"]["primary"]["max_embedding_batch_size"] = 2
    cfg["embedding_routes"] = {"retrieval": {"provider": "primary", "model": "embed-model"}}
    batches = []

    def upstream(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        batches.append(inputs)
        if "bad-item" in inputs:
            return httpx.Response(400)
        return httpx.Response(200, json={"model": "embed-model", "data": [
            {"index": index, "embedding": [float(len(value)), 1.0]} for index, value in enumerate(inputs)],
            "usage": {"prompt_tokens": len(inputs) * 3, "total_tokens": len(inputs) * 3}})

    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        created = await client.post("/v1/embeddings/jobs", json={"model": "retrieval",
            "input": ["one", "bad-item", "three", "four", "five"], "application": "batch-search"})
        assert created.status_code == 202
        job_id = created.json()["job_id"]
        for _ in range(100):
            status = await client.get(f"/v1/embeddings/jobs/{job_id}")
            if status.json()["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.01)
        assert status.json()["status"] == "completed"
        assert status.json()["progress"] == {"completed": 5, "total": 5}
        assert status.json()["usage"]["total_tokens"] == 12
        results = await client.get(f"/v1/embeddings/jobs/{job_id}/results")
        assert results.status_code == 200
        assert [item["status"] for item in results.json()["items"]] == [
            "succeeded", "failed", "succeeded", "succeeded", "succeeded"]
        assert results.json()["items"][0]["embedding"] == [3.0, 1.0]
        assert batches[0] == ["one", "bad-item"]


@pytest.mark.asyncio
async def test_embedding_job_submission_is_idempotent_and_rejects_key_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    cfg = config(str(tmp_path / "job-idempotency-api.sqlite3"))
    cfg["providers"]["primary"]["models"]["embed-model"] = {"input_cost_per_1m": 1}
    cfg["embedding_routes"] = {"retrieval": {"provider": "primary", "model": "embed-model"}}
    calls = 0

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        inputs = json.loads(request.content)["input"]
        return httpx.Response(200, json={"model": "embed-model", "data": [
            {"index": index, "embedding": [1.0]} for index, _ in enumerate(inputs)],
            "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)}})

    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        headers = {"Idempotency-Key": "batch-77"}
        request = {"model": "retrieval", "input": ["one", "two"], "application": "search"}
        first = await client.post("/v1/embeddings/jobs", json=request, headers=headers)
        second = await client.post("/v1/embeddings/jobs", json=request, headers=headers)
        changed = await client.post("/v1/embeddings/jobs",
            json={**request, "input": ["different"]}, headers=headers)
        assert first.status_code == second.status_code == 202
        assert first.json()["job_id"] == second.json()["job_id"]
        assert changed.status_code == 409
        for _ in range(100):
            status = await client.get(f"/v1/embeddings/jobs/{first.json()['job_id']}")
            if status.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)
        assert calls == 1


@pytest.mark.asyncio
async def test_async_embedding_results_are_artifact_backed_and_paginated(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("GATEWAY_API_KEY", "admin-secret")
    cfg = config(str(tmp_path / "artifact-jobs.sqlite3"))
    cfg["providers"]["primary"]["models"]["embed-model"] = {"input_cost_per_1m": 1}
    cfg["embedding_routes"] = {"retrieval": {"provider": "primary", "model": "embed-model"}}
    cfg["object_storage"] = {"backend": "local", "directory": str(tmp_path / "objects")}

    def upstream(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        return httpx.Response(200, json={"model": "embed-model", "data": [
            {"index": index, "embedding": [float(index)]} for index, _ in enumerate(inputs)],
            "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)}})

    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        headers = {"Authorization": "Bearer admin-secret"}
        created = await client.post("/v1/embeddings/jobs", json={"model": "retrieval",
            "input": ["one", "two", "three"], "application": "artifact-search"}, headers=headers)
        job_id = created.json()["job_id"]
        for _ in range(100):
            status = await client.get(f"/v1/embeddings/jobs/{job_id}", headers=headers)
            if status.json()["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.01)
        assert status.json()["status"] == "completed"
        assert status.json()["manifest"]["artifact_key"]
        assert status.json()["duration_ms"] is not None
        persisted = create_usage_store(cfg["server"]["database"]).async_job(job_id, include_results=True)
        assert persisted["result"]["artifact_key"] == status.json()["manifest"]["artifact_key"]
        assert "items" not in persisted["result"]
        first = await client.get(f"/v1/embeddings/jobs/{job_id}/results",
                                 params={"offset": 0, "limit": 2}, headers=headers)
        second = await client.get(f"/v1/embeddings/jobs/{job_id}/results",
                                  params={"offset": 2, "limit": 2}, headers=headers)
        assert [item["index"] for item in first.json()["items"]] == [0, 1]
        assert [item["index"] for item in second.json()["items"]] == [2]
        assert first.json()["pagination"] == {"offset": 0, "limit": 2, "total": 3, "has_more": True}
        queue = await client.get("/v1/admin/async-jobs", headers=headers)
        assert queue.json()["queue_depth"] == 0
        assert queue.json()["counts_by_status"]["completed"] == 1


def test_async_jobs_support_retry_cancellation_and_queue_statistics(tmp_path):
    store = create_usage_store(str(tmp_path / "job-ops.sqlite3"))
    queued = store.create_async_job("embedding_batch", None, "job-ops",
                                    {"input": ["one"]}, progress_total=1)
    assert store.cancel_async_job(queued["job_id"])["status"] == "cancelled"
    assert store.async_job(queued["job_id"], include_payload=True).get("payload") is None

    retrying = store.create_async_job("embedding_batch", None, "job-ops",
                                      {"input": ["two"]}, progress_total=1)
    assert store.claim_async_job(retrying["job_id"], lease_seconds=60)
    assert store.renew_async_job_lease(retrying["job_id"], lease_seconds=60)
    assert store.fail_async_job(retrying["job_id"], "TimeoutError", max_attempts=3,
                                base_delay_seconds=10, max_delay_seconds=30) == "queued"
    assert store.async_job(retrying["job_id"])["attempt_count"] == 1
    assert retrying["job_id"] not in store.pending_async_jobs()
    assert store.async_job_stats()["queue_depth"] == 1

    terminal = store.create_async_job("embedding_batch", None, "job-ops",
                                      {"input": ["three"]}, progress_total=1, max_attempts=1)
    assert store.claim_async_job(terminal["job_id"], lease_seconds=60)
    assert store.fail_async_job(terminal["job_id"], "RuntimeError", max_attempts=1,
                                base_delay_seconds=0, max_delay_seconds=0) == "dead_letter"

    stalled = store.create_async_job("embedding_batch", None, "job-ops",
                                     {"input": ["four"]}, progress_total=1)
    assert store.claim_async_job(stalled["job_id"], lease_seconds=0)
    assert store.recover_stalled_async_jobs(max_attempts=5, base_delay_seconds=0,
                                            max_delay_seconds=0) >= 1
    assert store.async_job(stalled["job_id"])["status"] == "queued"
    assert store.async_job_stats()["dead_letter"] == 1


def test_dead_letter_job_can_be_requeued_with_payload_and_operator_audit(tmp_path):
    store = create_usage_store(str(tmp_path / "dead-letter-requeue.sqlite3"))
    job = store.create_async_job("embedding_batch", "tenant-a", "operator",
        {"application": "search", "input": ["keep me"]}, progress_total=1, max_attempts=1)
    assert store.claim_async_job(job["job_id"])
    assert store.fail_async_job(job["job_id"], "TimeoutError", max_attempts=1,
        base_delay_seconds=0, max_delay_seconds=0) == "dead_letter"
    assert store.async_job(job["job_id"], include_payload=True)["payload"]["input"] == ["keep me"]

    assert store.requeue_dead_letter_async_job(job["job_id"], "ops-user") is True
    recovered = store.async_job(job["job_id"], include_payload=True)
    assert recovered["status"] == "queued"
    assert recovered["attempt_count"] == 0
    assert recovered["payload"]["input"] == ["keep me"]
    assert store.requeue_dead_letter_async_job(job["job_id"], "ops-user") is False
    audit = store.audit_events(limit=10)
    assert any(event["actor"] == "ops-user"
        and event["action"] == f"async_job_requeued:{job['job_id']}" for event in audit)


@pytest.mark.asyncio
async def test_signed_job_webhook_delivery_records_attempt_history(tmp_path, monkeypatch):
    monkeypatch.setenv("JOB_EVENT_SECRET", "job-event-secret")
    store = create_usage_store(str(tmp_path / "job-webhook.sqlite3"))
    requests = []

    async def receive(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204)

    app = SimpleNamespace(state=SimpleNamespace(
        config={"server": {"webhooks_allow_private_targets": True}, "async_jobs": {"webhooks": {"search": {
            "url": "https://hooks.test/jobs", "signing_keys": {"2026-10": "JOB_EVENT_SECRET"},
            "active_key_id": "2026-10"}}}},
        webhook_circuits={}, store=store, http=httpx.AsyncClient(transport=httpx.MockTransport(receive))))
    job = store.create_async_job("embedding_batch", "tenant-a", "operator",
        {"application": "search", "input": ["private input must not be included"]}, progress_total=1)

    await _emit_async_job_event(app, {**job, "payload": {"application": "search"}}, "queued")
    await _deliver_webhook_batch(app)

    event = store.async_job_webhook_events(job["job_id"])[0]
    request = requests[0]
    fields = dict(part.split("=", 1) for part in request.headers["X-LLM-Gateway-Signature"].split(","))
    signed = fields["t"].encode() + b"." + request.content
    expected = hmac.new(b"job-event-secret", signed, hashlib.sha256).hexdigest()
    assert fields["kid"] == "2026-10"
    assert request.headers["X-LLM-Gateway-Key-Id"] == "2026-10"
    assert fields["v1"] == expected
    assert request.headers["X-LLM-Gateway-Event"] == "queued"
    assert request.headers["Idempotency-Key"] == event["event_id"]
    assert b"private input" not in request.content
    assert event["status"] == "delivered"
    assert len(event["delivery_history"]) == 1
    assert event["delivery_history"][0]["http_status"] == 204
    await app.state.http.aclose()


@pytest.mark.asyncio
async def test_webhook_retry_after_and_endpoint_circuit_breaker(tmp_path):
    store = create_usage_store(str(tmp_path / "webhook-circuit.sqlite3"))
    store.enqueue_webhook({"alert_id": "first", "application": "app"}, "https://hooks.test/events")
    store.enqueue_webhook({"alert_id": "second", "application": "app"}, "https://hooks.test/events")
    requests = []

    def unavailable(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(429, headers={"Retry-After": "120"})

    app = SimpleNamespace(state=SimpleNamespace(store=store,
        http=httpx.AsyncClient(transport=httpx.MockTransport(unavailable)),
        config={"server": {"webhooks_allow_private_targets": True,
            "webhook_circuit_failure_threshold": 1, "webhook_circuit_recovery_seconds": 30}},
        webhook_circuits={}))
    await _deliver_webhook_batch(app)
    assert len(requests) == 1
    with store._connect() as db:
        rows = {row["event_id"]: row for row in db.execute(
            "SELECT event_id, status, attempts, next_attempt_at FROM webhook_outbox").fetchall()}
    assert rows["first"]["status"] == "pending"
    assert rows["first"]["attempts"] == 1
    assert (datetime.fromisoformat(rows["first"]["next_attempt_at"]) - datetime.now(UTC)).total_seconds() >= 118
    assert rows["second"]["status"] == "pending"
    assert rows["second"]["attempts"] == 0
    exposition = metrics_response()[0].decode()
    assert "llm_gateway_webhook_deliveries_total" in exposition
    assert "llm_gateway_webhook_delivery_duration_seconds" in exposition
    assert "llm_gateway_webhook_circuit_state" in exposition
    await app.state.http.aclose()


@pytest.mark.asyncio
async def test_webhook_redirect_is_not_followed(tmp_path):
    store = create_usage_store(str(tmp_path / "webhook-redirect.sqlite3"))
    store.enqueue_webhook({"alert_id": "redirect", "application": "app"}, "https://hooks.test/events")
    requests = []

    def redirect(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(302, headers={"Location": "http://127.0.0.1/admin"})

    app = SimpleNamespace(state=SimpleNamespace(store=store,
        http=httpx.AsyncClient(transport=httpx.MockTransport(redirect)),
        config={"server": {"webhooks_allow_private_targets": True}}, webhook_circuits={}))
    await _deliver_webhook_batch(app)
    assert len(requests) == 1
    assert store.webhook_events()[0]["status"] == "dead_letter"
    await app.state.http.aclose()


@pytest.mark.parametrize("url", [
    "https://user:pass@example.com/hook", "https://localhost/hook",
    "https://127.0.0.1/hook", "https://[::1]/hook", "http://example.com:8080/hook",
])
def test_webhook_target_validation_rejects_unsafe_destinations(url):
    with pytest.raises(UnsafeWebhookTarget):
        validate_webhook_url(url, resolve_dns=False)


def test_webhook_target_validation_rejects_private_dns_results(monkeypatch):
    monkeypatch.setattr("llm_gateway.webhook_security.socket.getaddrinfo", lambda *args, **kwargs: [
        (2, 1, 6, "", ("8.8.8.8", 443)),
        (2, 1, 6, "", ("10.0.0.8", 443)),
    ])
    with pytest.raises(UnsafeWebhookTarget, match="private_webhook_target_not_allowed"):
        validate_webhook_url("https://hooks.example.com/events")


@pytest.mark.asyncio
async def test_admin_job_retry_requires_exact_confirmation_and_dead_letter(tmp_path):
    cfg = config(str(tmp_path / "job-retry-api.sqlite3"))
    async with gateway_client(cfg, httpx.MockTransport(lambda request: success_response())) as client:
        store = create_usage_store(cfg["server"]["database"])
        job = store.create_async_job("embedding_batch", None, "admin",
            {"application": "search", "input": ["one"]}, 1)
        mismatch = await client.post(f"/v1/admin/async-jobs/{job['job_id']}/retry?confirm_job_id=wrong")
        assert mismatch.status_code == 400
        not_dead = await client.post(
            f"/v1/admin/async-jobs/{job['job_id']}/retry?confirm_job_id={job['job_id']}")
        assert not_dead.status_code == 409
        assert (await client.get(f"/v1/admin/async-jobs/{job['job_id']}/webhooks")).json() == {
            "job_id": job["job_id"], "events": []}


def test_async_job_idempotency_and_tenant_queue_limit(tmp_path):
    store = create_usage_store(str(tmp_path / "idempotent-jobs.sqlite3"))
    first = store.create_async_job("embedding_batch", "tenant-a", "search", {"input": ["one"]}, 1,
        idempotency_key="request-1", idempotency_scope="tenant-a:search", max_queued_for_tenant=1)
    replay = store.create_async_job("embedding_batch", "tenant-a", "search", {"input": ["one"]}, 1,
        idempotency_key="request-1", idempotency_scope="tenant-a:search", max_queued_for_tenant=1)
    assert replay["job_id"] == first["job_id"]
    with pytest.raises(ValueError, match="Idempotency-Key"):
        store.create_async_job("embedding_batch", "tenant-a", "search", {"input": ["different"]}, 1,
            idempotency_key="request-1", idempotency_scope="tenant-a:search", max_queued_for_tenant=1)
    with pytest.raises(ValueError, match="queue limit"):
        store.create_async_job("embedding_batch", "tenant-a", "search", {"input": ["two"]}, 1,
            idempotency_key="request-2", idempotency_scope="tenant-a:search", max_queued_for_tenant=1)


def test_async_jobs_dispatch_fairly_across_tenants(tmp_path):
    store = create_usage_store(str(tmp_path / "fair-jobs.sqlite3"))
    first_a = store.create_async_job("embedding_batch", "tenant-a", "a", {}, 1)
    second_a = store.create_async_job("embedding_batch", "tenant-a", "a", {}, 1)
    first_b = store.create_async_job("embedding_batch", "tenant-b", "b", {}, 1)
    first_c = store.create_async_job("embedding_batch", "tenant-c", "c", {}, 1)
    assert store.pending_async_jobs(4) == [first_a["job_id"], first_b["job_id"],
        first_c["job_id"], second_a["job_id"]]
    assert store.claim_async_job(first_a["job_id"])
    assert store.pending_async_jobs(3, {"tenant-a": 1}) == [first_b["job_id"], first_c["job_id"]]


def test_fair_dispatch_remembers_tenants_that_recently_ran(tmp_path):
    store = create_usage_store(str(tmp_path / "fair-job-turns.sqlite3"))
    first_a = store.create_async_job("embedding_batch", "tenant-a", "a", {}, 1)
    second_a = store.create_async_job("embedding_batch", "tenant-a", "a", {}, 1)
    first_b = store.create_async_job("embedding_batch", "tenant-b", "b", {}, 1)
    assert store.claim_async_job(first_a["job_id"])
    store.update_async_job(first_a["job_id"], status="completed")
    assert store.pending_async_jobs(1) == [first_b["job_id"]]
    assert second_a["status"] == "queued"


@pytest.mark.asyncio
async def test_provider_health_and_circuits_are_operation_scoped():
    from llm_gateway.circuit import ProviderCircuitBreaker
    from llm_gateway.provider_health import ProviderHealth

    breaker = ProviderCircuitBreaker(["provider"], failure_threshold=1, recovery_seconds=30)
    assert await breaker.allow("provider", operation="embeddings")
    await breaker.failure("provider", operation="embeddings")
    assert not await breaker.allow("provider", operation="embeddings")
    assert await breaker.allow("provider", operation="chat")
    health = ProviderHealth()
    await health.record("provider", 100, False, operation="embeddings")
    await health.record("provider", 20, True, operation="chat")
    snapshot = await health.snapshot_scoped(["provider"], ["chat", "embeddings", "rerank"])
    assert snapshot["provider"]["embeddings"]["failures"] == 1
    assert snapshot["provider"]["chat"]["successes"] == 1
    assert "rerank" not in snapshot["provider"]


def test_tenant_job_concurrency_is_enforced_and_retention_prunes_artifacts(tmp_path):
    database = str(tmp_path / "tenant-job-controls.sqlite3")
    store = create_usage_store(database)
    first = store.create_async_job("embedding_batch", "tenant-a", "search", {"input": ["a"]}, 1)
    second = store.create_async_job("embedding_batch", "tenant-a", "search", {"input": ["b"]}, 1)
    assert store.claim_async_job(first["job_id"], max_concurrent_for_tenant=1)
    assert not store.claim_async_job(second["job_id"], max_concurrent_for_tenant=1)
    store.update_async_job(first["job_id"], status="completed", manifest={"artifact_key": "tenant-a/old.json"})
    with sqlite3.connect(database) as db:
        db.execute("UPDATE async_jobs SET completed_at='2000-01-01T00:00:00+00:00' WHERE job_id=?",
                   (first["job_id"],))
    assert store.expired_tenant_async_job_artifacts("tenant-a", 1) == ["tenant-a/old.json"]
    assert store.prune_tenant_async_jobs("tenant-a", 1) == 1
    assert store.async_job(first["job_id"]) is None
    assert store.claim_async_job(second["job_id"], max_concurrent_for_tenant=1)


@pytest.mark.asyncio
async def test_tenant_embedding_job_size_limits_are_enforced(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPPORT_TENANT_KEY", "tenant-secret")
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    cfg = config(str(tmp_path / "tenant-job-limits.sqlite3"))
    cfg["server"]["async_embedding_threshold"] = 1
    cfg["providers"]["primary"]["models"]["embed-model"] = {"input_cost_per_1m": 1}
    cfg["embedding_routes"] = {"retrieval": {"provider": "primary", "model": "embed-model"}}
    cfg["tenants"] = {"support": {"api_key_env": "SUPPORT_TENANT_KEY", "job_policy": {
        "max_concurrent_jobs": 1, "max_job_items": 2, "max_job_input_bytes": 5}}}
    async with gateway_client(cfg, httpx.MockTransport(lambda _: httpx.Response(200))) as client:
        headers = {"Authorization": "Bearer tenant-secret"}
        too_many = await client.post("/v1/embeddings/jobs", headers=headers, json={
            "model": "retrieval", "input": ["a", "b", "c"], "application": "search"})
        too_large = await client.post("/v1/embeddings/jobs", headers=headers, json={
            "model": "retrieval", "input": ["123", "456"], "application": "search"})
        auto_job_too_many = await client.post("/v1/embeddings", headers=headers, json={
            "model": "retrieval", "input": ["a", "b", "c"], "application": "search"})
        assert too_many.status_code == 413
        assert too_large.status_code == 413
        assert auto_job_too_many.status_code == 413


@pytest.mark.asyncio
async def test_embedding_provider_circuit_state_is_reported_and_skips_open_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    cfg = config(str(tmp_path / "embedding-breaker.sqlite3"))
    cfg["circuit_breaker"] = {"failure_threshold": 1, "recovery_seconds": 30}
    cfg["providers"]["primary"]["models"]["embed-model"] = {"input_cost_per_1m": 1}
    cfg["providers"]["backup"]["models"]["embed-model"] = {"input_cost_per_1m": 1}
    cfg["embedding_routes"] = {"retrieval": {"provider": "primary", "model": "embed-model",
        "fallbacks": [{"provider": "backup", "model": "embed-model"}]}}
    calls = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if request.url.host == "primary.test" and request.url.path.endswith("/embeddings"):
            return httpx.Response(503)
        if request.url.path.endswith("/chat/completions"):
            return success_response()
        inputs = json.loads(request.content)["input"]
        return httpx.Response(200, json={"model": "embed-model", "data": [
            {"index": i, "embedding": [1.0]} for i, _ in enumerate(inputs)],
            "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)}})

    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        for text in ("one", "two"):
            result = await client.post("/v1/embeddings", json={"model": "retrieval", "input": text})
            assert result.status_code == 200
            assert result.json()["gateway"]["provider"] == "backup"
        health = (await client.get("/v1/admin/providers")).json()["providers"]
        assert health["primary"]["operations"]["embeddings"]["state"] == "open"
        assert health["primary"]["operations"]["embeddings"]["failures"] == 1
        assert health["primary"]["operations"]["chat"]["state"] == "closed"
        assert calls.count("primary.test") == 1
        chat = await client.post("/v1/chat/completions", json={"model": "simple",
            "messages": [{"role": "user", "content": "chat still works"}]})
        assert chat.status_code == 200


@pytest.mark.asyncio
async def test_native_reranker_circuit_state_is_reported_and_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("COHERE_KEY", "cohere-secret")
    cfg = config(str(tmp_path / "rerank-breaker.sqlite3"))
    cfg["circuit_breaker"] = {"failure_threshold": 1, "recovery_seconds": 30}
    cfg["providers"]["primary"]["models"]["embed-model"] = {"input_cost_per_1m": 1}
    cfg["providers"]["cohere"] = {"kind": "cohere", "base_url": "https://cohere.test/v2",
        "api_key_env": "COHERE_KEY", "models": {"rank-v1": {"rerank_cost_per_1k_documents": 1}}}
    cfg["embedding_routes"] = {"retrieval": {"provider": "primary", "model": "embed-model",
        "reranker": {"provider": "cohere", "model": "rank-v1", "fallback": "cosine"}}}
    calls = 0

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.host == "cohere.test":
            calls += 1
            return httpx.Response(503)
        inputs = json.loads(request.content)["input"]
        return httpx.Response(200, json={"model": "embed-model", "data": [
            {"index": i, "embedding": [1.0, float(i)]} for i, _ in enumerate(inputs)],
            "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)}})

    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        for _ in range(2):
            result = await client.post("/v1/rerank", json={"model": "retrieval", "query": "q",
                "documents": ["a", "b"]})
            assert result.status_code == 200
            assert result.json()["gateway"]["reranker"] == "cosine_similarity"
        health = (await client.get("/v1/admin/providers")).json()["providers"]
        assert health["cohere"]["operations"]["rerank"]["state"] == "open"
        assert health["cohere"]["operations"]["rerank"]["failures"] == 1
        assert health["cohere"]["operations"]["chat"]["state"] == "closed"
        assert calls == 1


@pytest.mark.asyncio
async def test_retrieval_evaluation_job_runs_baseline_and_candidate_rerankers(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEY", "admin-secret")
    monkeypatch.setenv("COHERE_KEY", "cohere-secret")
    cfg = config(str(tmp_path / "retrieval-eval-jobs.sqlite3"))
    cfg["providers"]["cohere"] = {"kind": "cohere", "base_url": "https://cohere.test/v2",
        "api_key_env": "COHERE_KEY", "models": {"rank-v1": {"rerank_cost_per_1k_documents": 1}}}
    cfg["embedding_routes"] = {
        "baseline": {"provider": "primary", "model": "embed-model",
            "reranker": {"provider": "cohere", "model": "rank-v1"}},
        "candidate": {"provider": "primary", "model": "embed-model",
            "reranker": {"provider": "cohere", "model": "rank-v1"}},
    }

    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/rerank")
        return httpx.Response(200, json={"results": [{"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.1}], "meta": {"billed_units": {"search_units": 1}}})

    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        headers = {"Authorization": "Bearer admin-secret"}
        queued = await client.post("/v1/admin/retrieval/evaluation-jobs", headers=headers, json={
            "dataset_name": "help-center", "dataset_version": "v1", "dataset_sha256": "c" * 64,
            "baseline_route": "baseline", "candidate_route": "candidate", "application": "search",
            "cases": [{"query": "refund", "documents": ["shipping", "refund policy"],
                "document_ids": ["d1", "d2"], "relevant_document_ids": ["d2"]}]})
        assert queued.status_code == 202
        job_id = queued.json()["job_id"]
        for _ in range(100):
            status = await client.get(f"/v1/admin/retrieval/evaluation-jobs/{job_id}", headers=headers)
            if status.json()["status"] in {"completed", "failed", "dead_letter"}:
                break
            await asyncio.sleep(0.01)
        assert status.json()["status"] == "completed"
        result = (await client.get(f"/v1/admin/retrieval/evaluation-jobs/{job_id}/results",
            headers=headers)).json()
        assert result["comparison"]["routes"]["candidate"]["candidate"]["retrieval_metrics"]["mrr"] == 1
        candidate_summary = result["candidate"]["summary"]
        assert candidate_summary["estimated_cost_usd"] > 0
        assert candidate_summary["average_latency_ms"] >= 0
        assert candidate_summary["retrieval_metrics"]["recall@5"] == 1
        assert result["comparison_id"]


@pytest.mark.asyncio
async def test_tenant_export_and_deletion_can_run_as_audited_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEY", "admin-secret")
    cfg = config(str(tmp_path / "tenant-data-jobs.sqlite3"))
    cfg["tenants"] = {"tenant-a": {"api_key_env": "TENANT_A_KEY",
        "data_policy": {"usage_retention_days": 90}}}
    store = create_usage_store(cfg["server"]["database"])
    store.record({"request_id": "job-request", "occurred_at": datetime.now(UTC).isoformat(),
        "application": "tenant-a:search", "route": "retrieval", "provider": "primary",
        "model": "embed-model", "prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2,
        "cost_usd": 0.01, "status": "success", "latency_ms": 10, "fallback_used": 0,
        "cached": 0, "operation": "embeddings"})
    async with gateway_client(cfg, httpx.MockTransport(lambda _: success_response())) as client:
        headers = {"Authorization": "Bearer admin-secret"}
        queued = await client.get("/v1/admin/tenants/tenant-a/data-export?async=true", headers=headers)
        assert queued.status_code == 202
        job_id = queued.json()["job_id"]
        for _ in range(100):
            status = await client.get(f"/v1/admin/tenant-data-jobs/{job_id}", headers=headers)
            if status.json()["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.01)
        assert status.json()["status"] == "completed"
        assert status.json()["manifest"]["record_counts"]["requests"] == 1
        exported = await client.get(f"/v1/admin/tenant-data-jobs/{job_id}/results", headers=headers)
        assert exported.json()["download_url"].endswith("/download")
        export_key = exported.json()["manifest"]["artifact_key"]
        export_path = tmp_path / "tenant-data-jobs.sqlite3.artifacts" / export_key
        assert export_path.exists()
        downloaded = await client.get(exported.json()["download_url"], headers=headers)
        assert downloaded.json()["requests"][0]["request_id"] == "job-request"

        deletion = await client.delete("/v1/admin/tenants/tenant-a/data?confirm_tenant_id=tenant-a&async=true",
                                       headers=headers)
        assert deletion.status_code == 202
        delete_id = deletion.json()["job_id"]
        for _ in range(100):
            status = await client.get(f"/v1/admin/tenant-data-jobs/{delete_id}", headers=headers)
            if status.json()["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.01)
        assert status.json()["status"] == "completed"
        assert status.json()["manifest"]["deleted_record_counts"]["requests"] == 1
        assert store.async_job(job_id) is None
        assert not export_path.exists()
        assert store.tenant_data_audits("tenant-a")[0]["action"] == "delete_job_completed"


@pytest.mark.asyncio
async def test_rerank_uses_native_provider_and_tracks_cost(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("COHERE_KEY", "cohere-secret")
    cfg = config(str(tmp_path / "native-rerank.sqlite3"))
    cfg["providers"]["cohere"] = {"kind": "cohere", "base_url": "https://cohere.test/v2",
        "api_key_env": "COHERE_KEY", "models": {"rank-v1": {
            "rerank_cost_per_1k_documents": 2, "pricing_version": "rank-price-v1"}}}
    cfg["embedding_routes"] = {"retrieval": {"provider": "primary", "model": "embed-model",
        "reranker": {"provider": "cohere", "model": "rank-v1", "fallback": "cosine"}}}

    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/rerank"
        return httpx.Response(200, json={"results": [{"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.1}], "meta": {"billed_units": {"search_units": 1}}})

    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        result = await client.post("/v1/rerank", json={"model": "retrieval", "query": "q",
            "documents": ["first", "second"], "application": "search"})
        assert result.status_code == 200
        body = result.json()
        assert body["gateway"]["reranker"] == "cohere"
        assert body["gateway"]["cost_usd"] == pytest.approx(0.004)
        assert body["results"][0]["document"] == "second"
        report = (await client.get("/v1/admin/report", params={"application": "search"})).json()
        assert report["cost_usd"] == pytest.approx(0.004)


@pytest.mark.asyncio
async def test_native_rerank_falls_back_to_cosine_before_returning_output(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("COHERE_KEY", "cohere-secret")
    cfg = config(str(tmp_path / "native-rerank-fallback.sqlite3"))
    cfg["providers"]["cohere"] = {"kind": "cohere", "base_url": "https://cohere.test/v2",
        "api_key_env": "COHERE_KEY", "models": {"rank-v1": {"rerank_cost_per_1k_documents": 2}}}
    cfg["providers"]["primary"]["models"]["embed-model"] = {"input_cost_per_1m": 1}
    cfg["embedding_routes"] = {"retrieval": {"provider": "primary", "model": "embed-model",
        "reranker": {"provider": "cohere", "model": "rank-v1", "fallback": "cosine"}}}

    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/rerank":
            return httpx.Response(503)
        values = json.loads(request.content)["input"]
        vectors = [[1.0, 0.0] if value in {"query", "best"} else [0.0, 1.0] for value in values]
        return httpx.Response(200, json={"model": "embed-model", "data": [
            {"index": index, "embedding": vector} for index, vector in enumerate(vectors)],
            "usage": {"prompt_tokens": len(values) * 2, "total_tokens": len(values) * 2}})

    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        result = await client.post("/v1/rerank", json={"model": "retrieval", "query": "query",
            "documents": ["other", "best"]})
        assert result.status_code == 200
        assert result.json()["gateway"]["reranker"] == "cosine_similarity"
        assert result.json()["gateway"]["native_fallback"] is True
        assert result.json()["results"][0]["index"] == 1


@pytest.mark.asyncio
async def test_embedding_cache_reuses_vectors_only_with_same_tenant_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("EMBED_CACHE_KEY", "stable-private-cache-key")
    cfg = config(str(tmp_path / "embedding-cache.sqlite3"))
    cfg["embedding_routes"] = {"retrieval": {"provider": "primary", "model": "embed-model"}}
    cfg["retrieval"] = {"embedding_cache": {"enabled": True, "ttl_seconds": 60,
        "hmac_key_env": "EMBED_CACHE_KEY"}}
    calls = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["input"])
        values = calls[-1]
        return httpx.Response(200, json={"model": "embed-model", "data": [
            {"index": index, "embedding": [1.0, float(index)]} for index, _ in enumerate(values)],
            "usage": {"prompt_tokens": len(values) * 2, "total_tokens": len(values) * 2}})

    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        first = await client.post("/v1/embeddings", json={"model": "retrieval", "input": "same text"})
        second = await client.post("/v1/embeddings", json={"model": "retrieval", "input": "same text"})
        assert first.status_code == second.status_code == 200
        assert second.json()["usage"]["total_tokens"] == 0
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_retrieval_evaluations_feed_quality_promotion_gates(tmp_path):
    cfg = config(str(tmp_path / "retrieval-evaluation.sqlite3"))
    cfg["embedding_routes"] = {"retrieval": {"provider": "primary", "model": "embed-model"}}
    async with gateway_client(cfg,
                              httpx.MockTransport(lambda _: success_response())) as client:
        # Evaluation reports are persisted in the same immutable store as chat evaluations.
        saved = await client.post("/v1/admin/retrieval/evaluations", json={
            "dataset_name": "retrieval-golden", "dataset_version": "candidate",
            "dataset_sha256": "a" * 64, "route": "retrieval", "cases": [
                {"ranked_document_ids": ["irrelevant", "relevant"], "relevant_document_ids": ["relevant"],
                 "cost_usd": 0.01, "latency_ms": 20},
                {"ranked_document_ids": ["relevant", "other"], "relevant_document_ids": ["relevant"],
                 "cost_usd": 0.02, "latency_ms": 30}]})
        assert saved.status_code == 200
        assert saved.json()["summary"]["retrieval_metrics"]["mrr"] == pytest.approx(0.75)
        baseline = await client.post("/v1/admin/retrieval/evaluations", json={
            "dataset_name": "retrieval-golden", "dataset_version": "baseline",
            "dataset_sha256": "b" * 64, "route": "retrieval", "cases": [
                {"ranked_document_ids": ["relevant", "irrelevant"], "relevant_document_ids": ["relevant"]}]})
        assert baseline.status_code == 200
        comparison = await client.post("/v1/admin/evaluations/compare", json={
            "baseline_run_id": baseline.json()["run_id"], "candidate_run_id": saved.json()["run_id"],
            "min_mrr": 0.8})
        assert comparison.status_code == 200
        assert comparison.json()["report"]["promotion_passed"] is False
        assert comparison.json()["report"]["routes"]["retrieval"]["retrieval_metrics"]["candidate"]["mrr"] == 0.75


@pytest.mark.asyncio
async def test_tenant_export_and_delete_preserve_audit_record(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEY", "admin-secret")
    cfg = config(str(tmp_path / "tenant-export.sqlite3"))
    cfg["tenants"] = {"tenant-a": {"api_key_env": "TENANT_A_KEY",
        "data_policy": {"usage_retention_days": 90}}}
    store = create_usage_store(cfg["server"]["database"])
    store.record({"request_id": "tenant-request", "occurred_at": datetime.now(UTC).isoformat(),
        "application": "tenant-a:search", "route": "retrieval", "provider": "primary",
        "model": "embed-model", "prompt_tokens": 4, "completion_tokens": 0, "total_tokens": 4,
        "cost_usd": 0.00001, "status": "success", "latency_ms": 10, "fallback_used": 0,
        "cached": 0, "operation": "embeddings"})
    store.record_tool_call_audits("tenant-request", "tenant-a", "tenant-a:search", [
        {"tool_name": "lookup", "outcome": "allowed"}])
    async with gateway_client(cfg, httpx.MockTransport(lambda request: success_response())) as client:
        headers = {"Authorization": "Bearer admin-secret"}
        exported = await client.get("/v1/admin/tenants/tenant-a/data-export", headers=headers)
        assert exported.status_code == 200
        assert len(exported.json()["data"]["requests"]) == 1
        assert exported.json()["data"]["audit_history"][-1]["action"] == "export"
        denied = await client.delete("/v1/admin/tenants/tenant-a/data", headers=headers,
            params={"confirm_tenant_id": "tenant-b"})
        assert denied.status_code == 400
        deleted = await client.delete("/v1/admin/tenants/tenant-a/data", headers=headers,
            params={"confirm_tenant_id": "tenant-a"})
        assert deleted.status_code == 200
        assert deleted.json()["audit_record_retained"] is True
        assert deleted.json()["deleted_record_counts"]["requests"] == 1
        assert store.report(None, None, "tenant-a:search")["requests"] == 0
        audits = store.tenant_data_audits("tenant-a")
        assert [item["action"] for item in audits] == ["delete", "export"]


@pytest.mark.asyncio
async def test_retrieval_application_request_quota_is_enforced(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    cfg = config(str(tmp_path / "retrieval-quota.sqlite3"))
    cfg["providers"]["primary"]["models"]["embed-model"] = {"input_cost_per_1m": 1.0}
    cfg["embedding_routes"] = {"retrieval": {"provider": "primary", "model": "embed-model"}}
    cfg["budgets"]["applications"]["search"] = {"max_requests_per_minute": 1}
    upstream_calls = []

    def upstream(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(request)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 2, "total_tokens": 2}})

    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        payload = {"model": "retrieval", "input": "query", "application": "search"}
        assert (await client.post("/v1/embeddings", json=payload)).status_code == 200
        limited = await client.post("/v1/embeddings", json=payload)
        assert limited.status_code == 429
        assert len(upstream_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_after_output", [False, True])
async def test_bedrock_stream_falls_back_only_before_output_is_emitted(tmp_path, monkeypatch, fail_after_output):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    cfg = config(str(tmp_path / f"bedrock-stream-{fail_after_output}.sqlite3"))
    cfg["server"]["max_retries"] = 0
    cfg["providers"]["primary"] = {"kind": "bedrock", "base_url": "https://bedrock.test",
        "region": "us-east-1", "models": {"primary-model": {}}}
    hosts = []

    def upstream(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        if request.url.host == "backup.test":
            events = [
                {"choices": [{"delta": {"content": "backup-answer"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}},
            ]
            return httpx.Response(200,
                content=("".join(f"data: {json.dumps(item)}\n\n" for item in events) + "data: [DONE]\n\n").encode(),
                headers={"content-type": "text/event-stream"})
        if not fail_after_output:
            return httpx.Response(503)
        body = bedrock_stream_frame("contentBlockDelta", {
            "contentBlockIndex": 0, "delta": {"text": "bedrock-answer"}})
        body += bedrock_stream_frame("internalServerException", {"message": "upstream failed"})
        return httpx.Response(200, content=body,
            headers={"content-type": "application/vnd.amazon.eventstream"})

    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        response = await client.post("/v1/chat/completions", json={"model": "simple", "stream": True,
            "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 200
        if fail_after_output:
            assert "bedrock-answer" in response.text
            assert "backup-answer" not in response.text
            assert hosts == ["bedrock.test"]
        else:
            assert "backup-answer" in response.text
            assert "bedrock-answer" not in response.text
            assert hosts == ["bedrock.test", "backup.test"]


@pytest.mark.asyncio
async def test_shadow_sample_records_comparison_and_combines_promotion_gates(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    cfg = config(str(tmp_path / "shadow.sqlite3"))
    cfg["routes"]["simple"]["shadow"] = {"provider": "backup", "model": "backup-model",
        "traffic_percent": 100, "promotion": {"min_samples": 1, "max_quality_loss": 0,
            "max_cost_regression_percent": 0, "max_latency_regression_percent": 100,
            "max_failure_rate": 0}}

    def upstream(request: httpx.Request) -> httpx.Response:
        body = success_response().json()
        if request.url.host == "backup.test":
            body["model"] = "backup-model"
            body["choices"][0]["message"]["content"] = "candidate answer"
        return httpx.Response(200, json=body)

    app = create_app(cfg, http_transport=httpx.MockTransport(upstream))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
        response = await client.post("/v1/chat/completions", json={"model": "simple",
            "application": "support-router", "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 200
        tasks = tuple(app.state.shadow_tasks)
        if tasks:
            await asyncio.gather(*tasks)
        report = await client.get("/v1/admin/shadow-reports", params={
            "application": "support-router", "route": "simple"})
        assert report.status_code == 200
        assert report.json()["samples"] == 1
        assert report.json()["candidate"]["quality_pass_rate"] == 1
        assert report.json()["promotion"]["passed"] is True

        baseline = await client.post("/v1/admin/evaluations", json={"dataset_name": "smoke",
            "dataset_version": "baseline", "dataset_sha256": "a" * 64, "case_count": 1,
            "report": {"routes": {"simple": {"requests": 1, "graded_cases": 1,
                "pass_rate": 1.0, "average_latency_ms": 10, "estimated_cost_usd": 0.01, "failures": 0}},
                "results": [{"route": "simple", "passed": True}]}})
        candidate = await client.post("/v1/admin/evaluations", json={"dataset_name": "smoke",
            "dataset_version": "candidate", "dataset_sha256": "b" * 64, "case_count": 1,
            "report": {"routes": {"simple": {"requests": 1, "graded_cases": 1,
                "pass_rate": 1.0, "average_latency_ms": 10, "estimated_cost_usd": 0.01, "failures": 0}},
                "results": [{"route": "simple", "passed": True}]}})
        comparison = await client.post("/v1/admin/evaluations/compare", json={
            "baseline_run_id": baseline.json()["run_id"],
            "candidate_run_id": candidate.json()["run_id"], "min_pass_rate": 0.9,
            "max_cost_regression_percent": 0, "max_latency_regression_percent": 0, "max_failure_rate": 0})
        assert comparison.json()["report"]["promotion_passed"] is True
        promotion = await client.post("/v1/admin/promotions/evaluate", json={
            "comparison_id": comparison.json()["comparison_id"], "application": "support-router",
            "route": "simple", "window_hours": 24})
        assert promotion.status_code == 200
        assert promotion.json()["promotion_passed"] is True


@pytest.mark.asyncio
async def test_shadow_candidate_obeys_tenant_provider_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-secret")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-secret")
    monkeypatch.setenv("RESTRICTED_TENANT_KEY", "restricted-secret")
    cfg = config(str(tmp_path / "tenant-shadow.sqlite3"))
    cfg["routes"]["simple"]["shadow"] = {"provider": "backup", "model": "backup-model",
        "traffic_percent": 100}
    cfg["tenants"] = {"restricted": {"api_key_env": "RESTRICTED_TENANT_KEY",
        "allowed_routes": ["simple"], "allowed_providers": ["primary"],
        "applications": {"support-router": {}}}}
    hosts = []

    def upstream(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return success_response()

    async with gateway_client(cfg, httpx.MockTransport(upstream)) as client:
        response = await client.post("/v1/chat/completions", headers={
            "Authorization": "Bearer restricted-secret"}, json={"model": "simple",
                "application": "support-router", "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 200
        assert hosts == ["primary.test"]
