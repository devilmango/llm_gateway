"""Provider adapter contract suite. Every adapter receives the same logical request."""

from __future__ import annotations

import json
import struct
import zlib
from typing import Any

import httpx
import pytest

from llm_gateway.capabilities import provider_capabilities
from llm_gateway.providers import ProviderCapabilityError, ProviderClient

ADAPTERS = ("openai", "vllm", "azure_openai", "anthropic", "gemini", "vertex", "bedrock")
STREAMING_ADAPTERS = ADAPTERS
TOOL = {"type": "function", "function": {"name": "lookup_order", "description": "Look up an order",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                   "required": ["order_id"]}}}
SCHEMA = {"type": "json_schema", "json_schema": {"name": "status",
    "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}}}}


def adapter_config(kind: str) -> dict[str, Any]:
    base_urls = {"openai": "https://openai.test/v1", "vllm": "https://vllm.test/v1",
        "azure_openai": "https://azure.test", "anthropic": "https://anthropic.test",
        "gemini": "https://gemini.test/v1beta", "vertex": "https://vertex.test/v1",
        "bedrock": "https://bedrock.test"}
    config: dict[str, Any] = {"kind": kind, "base_url": base_urls[kind],
        "request_id_header": "X-Contract-Request", "request_id_body_field": "metadata.gateway_request_id",
        "models": {"test-model": {}}}
    if kind in {"openai", "vllm", "azure_openai", "anthropic", "gemini"}:
        config["api_key_env"] = "CONFORMANCE_API_KEY"
    if kind == "azure_openai":
        config["api_version"] = "2025-01-01"
    if kind == "vertex":
        config.update({"project_id": "conformance-project", "region": "us-central1",
                       "access_token_env": "CONFORMANCE_VERTEX_TOKEN"})
    if kind == "bedrock":
        config["region"] = "us-east-1"
        config["access_key_env"] = "CONFORMANCE_AWS_ACCESS_KEY"
        config["secret_key_env"] = "CONFORMANCE_AWS_SECRET_KEY"
    return config


def completion_response(kind: str) -> dict[str, Any]:
    tool = {"id": "provider-call-1", "type": "function", "function": {
        "name": "lookup_order", "arguments": '{"order_id":"A-17"}'}}
    if kind in {"openai", "vllm", "azure_openai"}:
        return {"id": "response-1", "model": "test-model-served", "choices": [{"message": {
            "role": "assistant", "content": None, "tool_calls": [tool]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}
    if kind == "anthropic":
        return {"id": "response-1", "model": "test-model-served", "content": [{"type": "tool_use",
            "id": "provider-call-1", "name": "lookup_order", "input": {"order_id": "A-17"}}],
            "stop_reason": "tool_use", "usage": {"input_tokens": 3, "output_tokens": 2}}
    if kind in {"gemini", "vertex"}:
        return {"model": "test-model-served", "candidates": [{"content": {"parts": [{"functionCall": {
            "id": "provider-call-1", "name": "lookup_order", "args": {"order_id": "A-17"}}}]},
            "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 3,
                "candidatesTokenCount": 2, "totalTokenCount": 5}}
    return {"output": {"message": {"content": [{"toolUse": {
        "toolUseId": "provider-call-1", "name": "lookup_order", "input": {"order_id": "A-17"}}}]}},
        "stopReason": "tool_use", "usage": {"inputTokens": 3, "outputTokens": 2}}


def stream_response(kind: str) -> bytes:
    if kind == "bedrock":
        return b"".join(eventstream_frame(event_type, payload) for event_type, payload in [
            ("contentBlockStart", {"contentBlockIndex": 1, "start": {"toolUse": {
                "toolUseId": "tool-1", "name": "lookup_order"}}}),
            ("contentBlockDelta", {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"order_id":"A-17"}'}}}),
            ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "hello"}}),
            ("metadata", {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5}}}),
            ("messageStop", {"stopReason": "end_turn"}),
        ])
    if kind in {"openai", "vllm", "azure_openai"}:
        events = [
            {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
        ]
    elif kind == "anthropic":
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 3, "output_tokens": 0}}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": "hello"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
             "usage": {"output_tokens": 2}},
        ]
    else:
        events = [{"candidates": [{"content": {"parts": [{"text": "hello"}]},
            "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 3,
                "candidatesTokenCount": 2, "totalTokenCount": 5}}]
    return ("".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n").encode()


def eventstream_frame(event_type: str, payload: dict[str, Any]) -> bytes:
    name = b":event-type"
    value = event_type.encode()
    headers = bytes([len(name)]) + name + b"\x07" + struct.pack(">H", len(value)) + value
    headers += b"\x0d:message-type\x07" + struct.pack(">H", 5) + b"event"
    body = json.dumps(payload, separators=(",", ":")).encode()
    total = 12 + len(headers) + len(body) + 4
    prelude = struct.pack(">II", total, len(headers))
    return prelude + struct.pack(">I", zlib.crc32(prelude)) + headers + body + struct.pack(">I", zlib.crc32(prelude + struct.pack(">I", zlib.crc32(prelude)) + headers + body))


@pytest.fixture(autouse=True)
def adapter_credentials(monkeypatch):
    monkeypatch.setenv("CONFORMANCE_API_KEY", "test-api-key")
    monkeypatch.setenv("CONFORMANCE_VERTEX_TOKEN", "test-vertex-token")
    monkeypatch.setenv("CONFORMANCE_AWS_ACCESS_KEY", "test-access-key")
    monkeypatch.setenv("CONFORMANCE_AWS_SECRET_KEY", "test-secret-key")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ADAPTERS)
async def test_completion_tools_usage_structured_output_and_request_id_contract(kind: str):
    sent: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=completion_response(kind))

    providers = {"adapter": adapter_config(kind)}
    payload = {"messages": [{"role": "user", "content": "find order"}], "tools": [TOOL],
        "tool_choice": "auto"}
    capabilities = provider_capabilities(providers["adapter"])
    if capabilities["structured_output"]:
        payload["response_format"] = SCHEMA
    http = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    client = ProviderClient(http, providers, 2)
    try:
        result = await client.complete("adapter", "test-model", payload, request_id="gw_contract_1")
    finally:
        await http.aclose()

    assert result["tool_calls"][0]["function"]["name"] == "lookup_order"
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"order_id": "A-17"}
    assert result["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    request = sent[0]
    assert request.headers["X-Contract-Request"] == "gw_contract_1"
    provider_payload = json.loads(request.content)
    assert provider_payload["metadata"]["gateway_request_id"] == "gw_contract_1"
    if capabilities["tools"]:
        assert any(key in provider_payload for key in ("tools", "toolConfig"))
    if capabilities["structured_output"]:
        assert any(key in provider_payload for key in ("response_format", "output_config", "generationConfig"))
    else:
        with pytest.raises(ProviderCapabilityError, match="structured outputs"):
            client.validate_capabilities("adapter", {**payload, "response_format": SCHEMA})


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", STREAMING_ADAPTERS)
async def test_streaming_and_streaming_usage_contract(kind: str):
    sent: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, content=stream_response(kind),
            headers={"content-type": "text/event-stream"})

    providers = {"adapter": adapter_config(kind)}
    http = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    client = ProviderClient(http, providers, 2)
    items = []
    try:
        async for item in client.stream("adapter", "test-model",
                {"messages": [{"role": "user", "content": "hello"}]}, request_id="gw_stream_1"):
            items.append(item)
    finally:
        await http.aclose()
    assert {item.get("delta") for item in items if "delta" in item} == {"hello"}
    assert any(item.get("usage_progress", {}).get("total_tokens") == 5 for item in items)
    assert items[-1]["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    if kind == "bedrock":
        tool_deltas = [item["tool_call_delta"] for item in items if "tool_call_delta" in item]
        assert tool_deltas[0]["id"] == "tool-1"
        assert tool_deltas[0]["function"]["name"] == "lookup_order"
        arguments = "".join(delta["function"].get("arguments", "") for delta in tool_deltas)
        assert json.loads(arguments) == {"order_id": "A-17"}
    assert sent[0].headers["X-Contract-Request"] == "gw_stream_1"
    body = json.loads(sent[0].content)
    assert body["metadata"]["gateway_request_id"] == "gw_stream_1"
    assert body.get("stream") is True or kind in {"gemini", "vertex", "bedrock"}


@pytest.mark.parametrize("kind", ADAPTERS)
def test_capability_contract_matches_declared_adapter_matrix(kind: str):
    capabilities = provider_capabilities(adapter_config(kind))
    assert capabilities["tools"] is True
    assert capabilities["streaming"] is True
    assert capabilities["streaming_tools"] is True
    assert capabilities["structured_output"] is (kind != "bedrock")
    assert capabilities.get("embeddings", False) is (kind in {"openai", "vllm", "azure_openai", "gemini"})


@pytest.mark.asyncio
async def test_gemini_embeddings_adapter_normalizes_vectors_usage_and_request_id(monkeypatch):
    monkeypatch.setenv("CONFORMANCE_API_KEY", "test-key")
    sent: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"embeddings": [{"values": [0.25, 0.75]},
            {"values": [0.5, 0.5]}], "usageMetadata": {"promptTokenCount": 7}})

    providers = {"adapter": adapter_config("gemini")}
    http = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    client = ProviderClient(http, providers, 2)
    try:
        result = await client.embed("adapter", "test-model", ["first", "second"], "gw_embed_1")
    finally:
        await http.aclose()

    assert result == {"data": [[0.25, 0.75], [0.5, 0.5]], "model": "test-model",
        "usage": {"prompt_tokens": 7, "completion_tokens": 0, "total_tokens": 7}}
    assert "batchEmbedContents" in str(sent[0].url)
    assert sent[0].headers["X-Contract-Request"] == "gw_embed_1"
    body = json.loads(sent[0].content)
    assert [item["content"]["parts"][0]["text"] for item in body["requests"]] == ["first", "second"]
    assert provider_capabilities(providers["adapter"])["embeddings"] is True
