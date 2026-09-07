from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import struct
import time
import zlib
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit

import httpx

from .capabilities import provider_capabilities
from .observability import PROVIDER_LATENCY, PROVIDER_REQUESTS
from .secrets import SecretUnavailable, resolve_secret


class ProviderError(Exception):
    def __init__(self, message: str, *, retryable: bool = False, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class ProviderCapabilityError(ProviderError):
    """The adapter cannot serve this request shape; it is not an upstream failure."""


def _token_usage(body: dict[str, Any], kind: str) -> dict[str, int]:
    usage = body.get("usageMetadata", {}) if kind in {"gemini", "vertex"} else body.get("usage", {})
    if kind == "anthropic":
        prompt = int(usage.get("input_tokens", 0) or 0)
        completion = int(usage.get("output_tokens", 0) or 0)
    elif kind in {"gemini", "vertex"}:
        prompt = int(usage.get("promptTokenCount", 0) or 0)
        completion = int(usage.get("candidatesTokenCount", 0) or 0)
    elif kind == "bedrock":
        prompt = int(usage.get("inputTokens", 0) or 0)
        completion = int(usage.get("outputTokens", 0) or 0)
    else:
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": int(usage.get("total_tokens", 0) or usage.get("totalTokenCount", 0) or prompt + completion)}


class ProviderClient:
    def __init__(self, http: httpx.AsyncClient, providers: dict[str, Any], timeout: float):
        self.http = http
        self.providers = providers
        self.timeout = timeout

    @staticmethod
    def _apply_request_correlation(cfg: dict[str, Any], headers: dict[str, str],
                                   provider_payload: dict[str, Any], request_id: str | None) -> None:
        if not request_id:
            return
        header_name = cfg.get("request_id_header", "X-Gateway-Request-ID")
        if header_name:
            headers[str(header_name)] = request_id
        body_field = cfg.get("request_id_body_field")
        if body_field:
            if not isinstance(body_field, str) or any(not part for part in body_field.split(".")):
                raise ProviderError("Provider request_id_body_field must be a dot-separated path", retryable=False)
            parts = body_field.split(".")
            current = provider_payload
            for part in parts[:-1]:
                nested = current.get(part)
                if nested is None:
                    nested = {}
                    current[part] = nested
                if not isinstance(nested, dict):
                    raise ProviderError("Provider request ID metadata path conflicts with an existing value",
                                        retryable=False)
                current = nested
            current[parts[-1]] = request_id

    def validate_capabilities(self, provider_name: str, payload: dict[str, Any], streaming: bool = False) -> None:
        cfg = self.providers[provider_name]
        capabilities = provider_capabilities(cfg)
        if streaming and not capabilities.get("streaming", False):
            raise ProviderCapabilityError(f"Provider {provider_name!r} does not support streaming")
        if streaming and payload.get("tools") and not capabilities.get("streaming_tools", False):
            raise ProviderCapabilityError(f"Provider {provider_name!r} does not support streaming tool calls")
        if payload.get("tools") and not capabilities.get("tools", False):
            raise ProviderCapabilityError(f"Provider {provider_name!r} does not support tool calling")
        if payload.get("response_format") and not capabilities.get("structured_output", False):
            raise ProviderCapabilityError(f"Provider {provider_name!r} does not support structured outputs")
        parts = [part for message in payload.get("messages", [])
                 for part in (message.get("content") if isinstance(message.get("content"), list) else [])]
        if any(part.get("type") == "image_url" for part in parts) and not capabilities.get("images", False):
            raise ProviderCapabilityError(f"Provider {provider_name!r} does not support image inputs")
        audio_input = any(part.get("type") == "input_audio" for part in parts)
        if audio_input and not capabilities.get("audio_input", False):
            raise ProviderCapabilityError(f"Provider {provider_name!r} does not support audio input")
        if payload.get("modalities") and "audio" in payload["modalities"] and not capabilities.get("audio_output", False):
            raise ProviderCapabilityError(f"Provider {provider_name!r} does not support audio output")

    async def complete(self, provider_name: str, model: str, payload: dict[str, Any],
                       request_id: str | None = None) -> dict[str, Any]:
        cfg = self.providers[provider_name]
        kind = cfg["kind"]
        try:
            key = await resolve_secret(cfg.get("api_key_env"), cfg.get("api_key_file")) or None
        except SecretUnavailable as exc:
            raise ProviderError(f"Provider {provider_name!r} credential is unavailable") from exc
        if kind not in {"bedrock"} and not key and kind != "vertex" and not cfg.get("api_key_optional", False):
            raise ProviderError(f"Provider {provider_name!r} has no API key configured")
        base = (cfg.get("base_url") or "").rstrip("/")
        headers = {"Content-Type": "application/json"}
        params = None
        bedrock_signing: tuple[str, str, str, str, str] | None = None
        self.validate_capabilities(provider_name, payload)
        if kind in {"openai", "vllm"}:
            url = f"{base}/chat/completions"
            if key:
                headers["Authorization"] = f"Bearer {key}"
            provider_payload = {**payload, "model": model}
            normalize_kind = "openai"
        elif kind == "azure_openai":
            url = f"{base}/openai/deployments/{model}/chat/completions"
            params = {"api-version": cfg.get("api_version", "2024-10-21")}
            if key:
                headers["api-key"] = key
            provider_payload = {k: v for k, v in payload.items() if k != "model"}
            normalize_kind = "openai"
        elif kind == "anthropic":
            url = f"{base}/v1/messages"
            headers.update({"anthropic-version": "2023-06-01"})
            if key:
                headers["x-api-key"] = key
            provider_payload = self._anthropic_payload(model, payload)
            normalize_kind = kind
        elif kind == "gemini":
            url = f"{base}/models/{model}:generateContent"
            params = {"key": key} if key else None
            provider_payload = self._gemini_payload(payload)
            normalize_kind = kind
        elif kind == "vertex":
            region = cfg.get("region", "us-central1")
            project = cfg.get("project_id", "")
            if not project:
                raise ProviderError("Vertex provider requires project_id")
            root = base or f"https://{region}-aiplatform.googleapis.com/v1"
            url = f"{root}/projects/{project}/locations/{region}/publishers/google/models/{model}:generateContent"
            token_env = cfg.get("access_token_env", "GOOGLE_CLOUD_ACCESS_TOKEN")
            try:
                token = await resolve_secret(token_env, cfg.get("access_token_file"))
            except SecretUnavailable as exc:
                raise ProviderError("Vertex access token is unavailable") from exc
            if not token:
                raise ProviderError(f"Vertex provider requires an access token in {token_env}")
            headers["Authorization"] = f"Bearer {token}"
            provider_payload = self._gemini_payload(payload)
            normalize_kind = "gemini"
        elif kind == "bedrock":
            region = cfg.get("region") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
            if not region:
                raise ProviderError("Bedrock provider requires region or AWS_REGION")
            url = f"{base or f'https://bedrock-runtime.{region}.amazonaws.com'}/model/{quote(model, safe='-_.~')}/converse"
            provider_payload = self._bedrock_payload(payload)
            try:
                access_key = await resolve_secret(cfg.get("access_key_env", "AWS_ACCESS_KEY_ID"),
                                                  cfg.get("access_key_file"))
                secret_key = await resolve_secret(cfg.get("secret_key_env", "AWS_SECRET_ACCESS_KEY"),
                                                  cfg.get("secret_key_file"))
                session_token = await resolve_secret(cfg.get("session_token_env", "AWS_SESSION_TOKEN"),
                                                     cfg.get("session_token_file"))
            except SecretUnavailable as exc:
                raise ProviderError("Bedrock credential is unavailable") from exc
            bedrock_signing = (url, region, access_key, secret_key, session_token)
            normalize_kind = "bedrock"
        else:
            raise ProviderError(f"Unsupported provider kind {kind!r}")
        self._apply_request_correlation(cfg, headers, provider_payload, request_id)
        if bedrock_signing:
            signed_url, region, access_key, secret_key, session_token = bedrock_signing
            self._sign_aws_request(signed_url, provider_payload, headers, region, access_key, secret_key,
                                   session_token)
        started = time.perf_counter()
        outcome = "network_error"
        try:
            body = (json.dumps(provider_payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                    if kind == "bedrock" else None)
            response = await self.http.post(url, content=body, json=None if body is not None else provider_payload,
                                            headers=headers, params=params, timeout=self.timeout)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            PROVIDER_REQUESTS.labels(provider=provider_name, outcome=outcome).inc()
            PROVIDER_LATENCY.labels(provider=provider_name, outcome=outcome).observe(time.perf_counter() - started)
            raise ProviderError(f"{type(exc).__name__}: upstream request failed", retryable=True) from exc
        if response.status_code >= 400:
            outcome = "rate_limited" if response.status_code == 429 else "upstream_error"
            PROVIDER_REQUESTS.labels(provider=provider_name, outcome=outcome).inc()
            PROVIDER_LATENCY.labels(provider=provider_name, outcome=outcome).observe(time.perf_counter() - started)
            retryable = response.status_code == 429 or response.status_code >= 500
            # Keep upstream error bodies out of logs and client responses; providers can echo prompts/secrets.
            raise ProviderError(f"Upstream returned HTTP {response.status_code}", retryable=retryable, status_code=response.status_code)
        try:
            body = response.json()
        except ValueError as exc:
            outcome = "invalid_response"
            PROVIDER_REQUESTS.labels(provider=provider_name, outcome=outcome).inc()
            PROVIDER_LATENCY.labels(provider=provider_name, outcome=outcome).observe(time.perf_counter() - started)
            raise ProviderError("Provider returned invalid JSON", retryable=True) from exc
        try:
            result = self._normalize(body, normalize_kind, model)
        except ProviderError:
            outcome = "invalid_response"
            PROVIDER_REQUESTS.labels(provider=provider_name, outcome=outcome).inc()
            PROVIDER_LATENCY.labels(provider=provider_name, outcome=outcome).observe(time.perf_counter() - started)
            raise
        PROVIDER_REQUESTS.labels(provider=provider_name, outcome="success").inc()
        PROVIDER_LATENCY.labels(provider=provider_name, outcome="success").observe(time.perf_counter() - started)
        return result

    async def stream(self, provider_name: str, model: str, payload: dict[str, Any],
                     request_id: str | None = None):
        """Yield text deltas normalized from each supported provider's native SSE API."""
        cfg = self.providers[provider_name]
        kind = cfg["kind"]
        self.validate_capabilities(provider_name, payload, streaming=True)
        if kind == "bedrock":
            async for item in self._stream_bedrock(provider_name, model, payload, request_id):
                yield item
            return
        try:
            key = await resolve_secret(cfg.get("api_key_env"), cfg.get("api_key_file")) or None
        except SecretUnavailable as exc:
            raise ProviderError(f"Provider {provider_name!r} credential is unavailable") from exc
        if kind not in {"bedrock", "vertex"} and not key and not cfg.get("api_key_optional", False):
            raise ProviderError(f"Provider {provider_name!r} has no API key configured")
        headers = {"Content-Type": "application/json"}
        params = None
        base = (cfg.get("base_url") or "").rstrip("/")
        if kind in {"openai", "vllm"}:
            url = f"{base}/chat/completions"
            if key:
                headers["Authorization"] = f"Bearer {key}"
            request_payload = {**payload, "model": model, "stream": True, "stream_options": {"include_usage": True}}
        elif kind == "azure_openai":
            url = f"{base}/openai/deployments/{model}/chat/completions"
            params = {"api-version": cfg.get("api_version", "2024-10-21")}
            if key:
                headers["api-key"] = key
            request_payload = {k: v for k, v in payload.items() if k != "model"}
            request_payload.update({"stream": True, "stream_options": {"include_usage": True}})
        elif kind == "anthropic":
            url = f"{base}/v1/messages"
            headers["anthropic-version"] = "2023-06-01"
            if key:
                headers["x-api-key"] = key
            request_payload = self._anthropic_payload(model, payload)
            request_payload["stream"] = True
        elif kind == "gemini":
            url = f"{base}/models/{model}:streamGenerateContent"
            params = {"alt": "sse"}
            if key:
                params["key"] = key
            request_payload = self._gemini_payload(payload)
        elif kind == "vertex":
            region = cfg.get("region", "us-central1")
            project = cfg.get("project_id", "")
            token_env = cfg.get("access_token_env", "GOOGLE_CLOUD_ACCESS_TOKEN")
            try:
                token = await resolve_secret(token_env, cfg.get("access_token_file"))
            except SecretUnavailable as exc:
                raise ProviderError("Vertex access token is unavailable") from exc
            if not project or not token:
                raise ProviderError("Vertex streaming requires project_id and an access token")
            root = base or f"https://{region}-aiplatform.googleapis.com/v1"
            url = f"{root}/projects/{project}/locations/{region}/publishers/google/models/{model}:streamGenerateContent"
            params = {"alt": "sse"}
            headers["Authorization"] = f"Bearer {token}"
            request_payload = self._gemini_payload(payload)
        else:
            raise ProviderError(f"Unsupported provider kind {kind!r}")
        self._apply_request_correlation(cfg, headers, request_payload, request_id)
        started = time.perf_counter()
        outcome = "network_error"
        try:
            async with self.http.stream("POST", url, json=request_payload, headers=headers,
                                        params=params, timeout=self.timeout) as response:
                if response.status_code >= 400:
                    outcome = "rate_limited" if response.status_code == 429 else "upstream_error"
                    retryable = response.status_code == 429 or response.status_code >= 500
                    raise ProviderError(f"Upstream returned HTTP {response.status_code}",
                                        retryable=retryable, status_code=response.status_code)
                saw_delta = False
                usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                finish_reason = None
                anthropic_tool_indices: dict[int, int] = {}
                next_tool_index = 0
                next_gemini_tool_index = 0
                async for line in response.aiter_lines():
                    event_usage = None
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except ValueError as exc:
                        raise ProviderError("Provider returned invalid stream data", retryable=True) from exc
                    delta = None
                    if kind in {"openai", "vllm", "azure_openai"}:
                        event_usage = event.get("usage")
                        choices = event.get("choices", [])
                        choice = choices[0] if choices else {}
                        delta = choice.get("delta", {}).get("content")
                        finish_reason = choice.get("finish_reason") or finish_reason
                        if event_usage:
                            usage = {"prompt_tokens": int(event_usage.get("prompt_tokens", 0) or 0),
                                     "completion_tokens": int(event_usage.get("completion_tokens", 0) or 0),
                                     "total_tokens": int(event_usage.get("total_tokens", 0) or 0)}
                        for call in choice.get("delta", {}).get("tool_calls", []) or []:
                            saw_delta = True
                            yield {"tool_call_delta": {"index": call.get("index", 0),
                                **({"id": call["id"]} if call.get("id") else {}),
                                "type": call.get("type", "function"), "function": call.get("function", {})}}
                        audio_delta = choice.get("delta", {}).get("audio")
                        if audio_delta:
                            saw_delta = True
                            yield {"audio_delta": audio_delta}
                    elif kind == "anthropic":
                        event_type = event.get("type")
                        event_usage = event.get("message", {}).get("usage") if event_type == "message_start" else event.get("usage")
                        if event_type == "content_block_delta":
                            stream_delta = event.get("delta", {})
                            if stream_delta.get("type") == "text_delta":
                                delta = stream_delta.get("text")
                            elif stream_delta.get("type") == "input_json_delta":
                                saw_delta = True
                                block_index = int(event.get("index", 0))
                                yield {"tool_call_delta": {"index": anthropic_tool_indices.get(block_index, block_index),
                                    "type": "function", "function": {"arguments": stream_delta.get("partial_json", "")}}}
                        elif event_type == "content_block_start":
                            block = event.get("content_block", {})
                            if block.get("type") == "tool_use":
                                saw_delta = True
                                block_index = int(event.get("index", 0))
                                anthropic_tool_indices[block_index] = next_tool_index
                                next_tool_index += 1
                                yield {"tool_call_delta": {"index": anthropic_tool_indices[block_index],
                                    "id": block.get("id"), "type": "function",
                                    "function": {"name": block.get("name", ""),
                                                 "arguments": json.dumps(block.get("input", {}), separators=(",", ":")) if block.get("input") else ""}}}
                        elif event_type == "message_delta":
                            finish_reason = event.get("delta", {}).get("stop_reason") or finish_reason
                        if event_usage:
                            usage["prompt_tokens"] = int(event_usage.get("input_tokens", usage["prompt_tokens"]) or 0)
                            usage["completion_tokens"] = int(event_usage.get("output_tokens", usage["completion_tokens"]) or 0)
                            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                    elif kind in {"gemini", "vertex"}:
                        candidates = event.get("candidates", [])
                        candidate = candidates[0] if candidates else {}
                        parts = candidate.get("content", {}).get("parts", [])
                        delta = "".join(part.get("text", "") for part in parts if isinstance(part.get("text"), str))
                        for part in parts:
                            function_call = part.get("functionCall")
                            if function_call:
                                saw_delta = True
                                yield {"tool_call_delta": {"index": next_gemini_tool_index,
                                    **({"id": function_call["id"]} if function_call.get("id") else {}),
                                    "type": "function", "function": {"name": function_call.get("name", ""),
                                    "arguments": json.dumps(function_call.get("args", {}), separators=(",", ":"))}}}
                                next_gemini_tool_index += 1
                            audio_part = part.get("inlineData")
                            if audio_part and audio_part.get("mimeType", "").startswith("audio/"):
                                saw_delta = True
                                yield {"audio_delta": {"data": audio_part.get("data", ""),
                                                       "mime_type": audio_part.get("mimeType")}}
                        finish_reason = candidate.get("finishReason") or finish_reason
                        event_usage = event.get("usageMetadata")
                        if event_usage:
                            prompt = int(event_usage.get("promptTokenCount", 0) or 0)
                            completion = int(event_usage.get("candidatesTokenCount", 0) or 0)
                            usage = {"prompt_tokens": prompt, "completion_tokens": completion,
                                     "total_tokens": int(event_usage.get("totalTokenCount", 0) or prompt + completion)}
                    if kind == "openai" and event_usage:
                        usage = {"prompt_tokens": int(event_usage.get("prompt_tokens", 0) or 0),
                                 "completion_tokens": int(event_usage.get("completion_tokens", 0) or 0),
                                 "total_tokens": int(event_usage.get("total_tokens", 0) or 0)}
                    if event_usage:
                        yield {"usage_progress": dict(usage)}
                    if isinstance(delta, str) and delta:
                        saw_delta = True
                        yield {"delta": delta}
                if not saw_delta:
                    raise ProviderError("Provider returned no text completion", retryable=False)
                PROVIDER_REQUESTS.labels(provider=provider_name, outcome="success").inc()
                PROVIDER_LATENCY.labels(provider=provider_name, outcome="success").observe(time.perf_counter() - started)
                yield {"usage": usage, "finish_reason": finish_reason, "model": model}
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            PROVIDER_REQUESTS.labels(provider=provider_name, outcome=outcome).inc()
            PROVIDER_LATENCY.labels(provider=provider_name, outcome=outcome).observe(time.perf_counter() - started)
            raise ProviderError(f"{type(exc).__name__}: upstream request failed", retryable=True) from exc
        except ProviderError as exc:
            outcome = "rate_limited" if exc.status_code == 429 else "upstream_error"
            PROVIDER_REQUESTS.labels(provider=provider_name, outcome=outcome).inc()
            PROVIDER_LATENCY.labels(provider=provider_name, outcome=outcome).observe(time.perf_counter() - started)
            raise

    async def _stream_bedrock(self, provider_name: str, model: str, payload: dict[str, Any],
                              request_id: str | None = None):
        cfg = self.providers[provider_name]
        region = cfg.get("region") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        if not region:
            raise ProviderError("Bedrock provider requires region or AWS_REGION")
        base = (cfg.get("base_url") or f"https://bedrock-runtime.{region}.amazonaws.com").rstrip("/")
        url = f"{base}/model/{quote(model, safe='-_.~')}/converse-stream"
        request_payload = self._bedrock_payload(payload)
        try:
            access_key = await resolve_secret(cfg.get("access_key_env", "AWS_ACCESS_KEY_ID"),
                                              cfg.get("access_key_file"))
            secret_key = await resolve_secret(cfg.get("secret_key_env", "AWS_SECRET_ACCESS_KEY"),
                                              cfg.get("secret_key_file"))
            session_token = await resolve_secret(cfg.get("session_token_env", "AWS_SESSION_TOKEN"),
                                                 cfg.get("session_token_file"))
        except SecretUnavailable as exc:
            raise ProviderError("Bedrock credential is unavailable") from exc
        headers = {"Content-Type": "application/json", "Accept": "application/vnd.amazon.eventstream"}
        self._apply_request_correlation(cfg, headers, request_payload, request_id)
        self._sign_aws_request(url, request_payload, headers, region, access_key, secret_key, session_token)
        body = json.dumps(request_payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        started = time.perf_counter()
        outcome = "network_error"
        try:
            async with self.http.stream("POST", url, content=body, headers=headers, timeout=self.timeout) as response:
                if response.status_code >= 400:
                    outcome = "rate_limited" if response.status_code == 429 else "upstream_error"
                    raise ProviderError(f"Upstream returned HTTP {response.status_code}",
                                        retryable=response.status_code == 429 or response.status_code >= 500,
                                        status_code=response.status_code)
                saw_delta = False
                usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                finish_reason = None
                tool_indices: dict[int, int] = {}
                next_tool_index = 0
                async for event in self._bedrock_event_stream(response.aiter_bytes()):
                    event_type = event.get(":event-type", "")
                    if event.get(":message-type") == "exception" or event_type.endswith("Exception"):
                        raise ProviderError(f"Bedrock stream returned {event_type or 'an exception'}",
                                            retryable=event_type in {"throttlingException", "serviceUnavailableException"})
                    if event_type == "contentBlockStart":
                        block = event.get("start", {})
                        tool = block.get("toolUse")
                        if tool:
                            block_index = int(event.get("contentBlockIndex", 0))
                            tool_indices[block_index] = next_tool_index
                            yield {"tool_call_delta": {"index": next_tool_index,
                                "id": tool.get("toolUseId"), "type": "function",
                                "function": {"name": tool.get("name", ""), "arguments": ""}}}
                            next_tool_index += 1
                            saw_delta = True
                    elif event_type == "contentBlockDelta":
                        block_index = int(event.get("contentBlockIndex", 0))
                        delta = event.get("delta", {})
                        if isinstance(delta.get("text"), str) and delta["text"]:
                            yield {"delta": delta["text"]}
                            saw_delta = True
                        tool_delta = delta.get("toolUse")
                        if tool_delta and block_index in tool_indices:
                            yield {"tool_call_delta": {"index": tool_indices[block_index], "type": "function",
                                "function": {"arguments": tool_delta.get("input", "")}}}
                            saw_delta = True
                    elif event_type == "messageStop":
                        finish_reason = event.get("stopReason") or finish_reason
                    elif event_type == "metadata":
                        metadata = event.get("metadata", {})
                        token_usage = metadata.get("usage", {})
                        if token_usage:
                            prompt = int(token_usage.get("inputTokens", 0) or 0)
                            completion = int(token_usage.get("outputTokens", 0) or 0)
                            usage = {"prompt_tokens": prompt, "completion_tokens": completion,
                                "total_tokens": int(token_usage.get("totalTokens", prompt + completion) or prompt + completion)}
                            yield {"usage_progress": dict(usage)}
                if not saw_delta:
                    raise ProviderError("Provider returned no text completion", retryable=False)
                PROVIDER_REQUESTS.labels(provider=provider_name, outcome="success").inc()
                PROVIDER_LATENCY.labels(provider=provider_name, outcome="success").observe(time.perf_counter() - started)
                yield {"usage": usage, "finish_reason": finish_reason, "model": model}
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            PROVIDER_REQUESTS.labels(provider=provider_name, outcome=outcome).inc()
            PROVIDER_LATENCY.labels(provider=provider_name, outcome=outcome).observe(time.perf_counter() - started)
            raise ProviderError(f"{type(exc).__name__}: upstream request failed", retryable=True) from exc
        except ProviderError as exc:
            outcome = "rate_limited" if exc.status_code == 429 else "upstream_error"
            PROVIDER_REQUESTS.labels(provider=provider_name, outcome=outcome).inc()
            PROVIDER_LATENCY.labels(provider=provider_name, outcome=outcome).observe(time.perf_counter() - started)
            raise

    @staticmethod
    async def _bedrock_event_stream(chunks):
        """Decode AWS EventStream frames and validate their CRCs before parsing JSON events."""
        buffer = bytearray()
        async for chunk in chunks:
            buffer.extend(chunk)
            while len(buffer) >= 12:
                total_length, headers_length = struct.unpack_from(">II", buffer)
                if total_length < 16 or headers_length > total_length - 16:
                    raise ProviderError("Bedrock returned an invalid EventStream frame", retryable=False)
                if len(buffer) < total_length:
                    break
                frame = bytes(buffer[:total_length])
                del buffer[:total_length]
                if zlib.crc32(frame[:8]) != struct.unpack_from(">I", frame, 8)[0]:
                    raise ProviderError("Bedrock EventStream prelude checksum failed", retryable=False)
                if zlib.crc32(frame[:-4]) != struct.unpack_from(">I", frame, total_length - 4)[0]:
                    raise ProviderError("Bedrock EventStream message checksum failed", retryable=False)
                headers: dict[str, Any] = {}
                cursor = 12
                headers_end = 12 + headers_length
                while cursor < headers_end:
                    name_length = frame[cursor]
                    cursor += 1
                    name = frame[cursor:cursor + name_length].decode("utf-8")
                    cursor += name_length
                    value_type = frame[cursor]
                    cursor += 1
                    if value_type == 0:
                        value = True
                    elif value_type == 1:
                        value = False
                    elif value_type == 2:
                        value = struct.unpack_from(">b", frame, cursor)[0]
                        cursor += 1
                    elif value_type == 3:
                        value = struct.unpack_from(">h", frame, cursor)[0]
                        cursor += 2
                    elif value_type == 4:
                        value = struct.unpack_from(">i", frame, cursor)[0]
                        cursor += 4
                    elif value_type in {5, 8}:
                        value = struct.unpack_from(">q", frame, cursor)[0]
                        cursor += 8
                    elif value_type in {6, 7}:
                        value_length = struct.unpack_from(">H", frame, cursor)[0]
                        cursor += 2
                        raw = frame[cursor:cursor + value_length]
                        cursor += value_length
                        value = raw if value_type == 6 else raw.decode("utf-8")
                    elif value_type == 9:
                        value = frame[cursor:cursor + 16]
                        cursor += 16
                    else:
                        raise ProviderError("Bedrock EventStream used an unknown header type", retryable=False)
                    headers[name] = value
                try:
                    payload = json.loads(frame[headers_end:-4] or b"{}")
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ProviderError("Bedrock returned invalid EventStream JSON", retryable=False) from exc
                if not isinstance(payload, dict):
                    raise ProviderError("Bedrock EventStream event must be a JSON object", retryable=False)
                yield {**headers, **payload}
        if buffer:
            raise ProviderError("Bedrock returned a truncated EventStream frame", retryable=True)

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
        return ""

    @staticmethod
    def _bedrock_payload(payload: dict[str, Any]) -> dict[str, Any]:
        messages = []
        system = []
        tool_ids: dict[str, str] = {}
        for message in payload["messages"]:
            if message["role"] == "assistant":
                for call in message.get("tool_calls", []):
                    tool_ids[call.get("id", "")] = call.get("function", {}).get("name", "")
        for message in payload["messages"]:
            role = message["role"]
            content = message.get("content")
            if role in {"system", "developer"}:
                text = ProviderClient._content_text(content)
                if text:
                    system.append({"text": text})
                continue
            blocks: list[dict[str, Any]] = []
            if role == "tool":
                blocks.append({"toolResult": {"toolUseId": message["tool_call_id"],
                    "content": [{"text": content if isinstance(content, str) else json.dumps(content)}]}})
            else:
                if isinstance(content, str) and content:
                    blocks.append({"text": content})
                elif isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text":
                            blocks.append({"text": part["text"]})
                        elif part.get("type") == "image_url":
                            url = ProviderClient._image_url(part)
                            match = re.fullmatch(r"data:image/(png|jpeg|gif|webp);base64,(.+)", url, re.DOTALL)
                            if not match:
                                raise ProviderError("Bedrock image input requires a PNG/JPEG/GIF/WebP data URI")
                            blocks.append({"image": {"format": match.group(1),
                                "source": {"bytes": match.group(2)}}})
                if role == "assistant":
                    for call in message.get("tool_calls", []):
                        function = call.get("function", {})
                        try:
                            arguments = json.loads(function.get("arguments", "{}"))
                        except (TypeError, ValueError):
                            arguments = {}
                        blocks.append({"toolUse": {"toolUseId": call.get("id", ""),
                            "name": function.get("name", ""), "input": arguments}})
            if blocks:
                messages.append({"role": "assistant" if role == "assistant" else "user", "content": blocks})
        result: dict[str, Any] = {"messages": messages}
        if system:
            result["system"] = system
        inference = {}
        for source, destination in (("temperature", "temperature"), ("top_p", "topP"), ("max_tokens", "maxTokens")):
            if payload.get(source) is not None:
                inference[destination] = payload[source]
        if inference:
            result["inferenceConfig"] = inference
        if payload.get("tools") and payload.get("tool_choice") != "none":
            tools = []
            for tool in payload["tools"]:
                function = tool["function"]
                tools.append({"toolSpec": {"name": function["name"],
                    "description": function.get("description", ""),
                    "inputSchema": {"json": function.get("parameters", {"type": "object"})}}})
            tool_config: dict[str, Any] = {"tools": tools}
            choice = payload.get("tool_choice")
            if choice == "required":
                tool_config["toolChoice"] = {"any": {}}
            elif isinstance(choice, dict):
                tool_config["toolChoice"] = {"tool": {"name": choice["function"]["name"]}}
            result["toolConfig"] = tool_config
        return result

    @staticmethod
    def _sign_aws_request(url: str, payload: dict[str, Any], headers: dict[str, str], region: str,
                          access_key: str, secret_key: str, token: str = "") -> None:
        if not access_key or not secret_key:
            raise ProviderError("Bedrock requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        parsed = urlsplit(url)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        date = timestamp[:8]
        headers["Host"] = parsed.netloc
        headers["X-Amz-Date"] = timestamp
        if token:
            headers["X-Amz-Security-Token"] = token
        canonical_headers = "content-type:application/json\n" + f"host:{parsed.netloc}\n" + f"x-amz-date:{timestamp}\n"
        signed_headers = "content-type;host;x-amz-date"
        if token:
            canonical_headers += f"x-amz-security-token:{token}\n"
            signed_headers += ";x-amz-security-token"
        canonical_query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
        canonical_request = "\n".join(("POST", parsed.path or "/", canonical_query,
            canonical_headers, signed_headers, hashlib.sha256(body).hexdigest()))
        scope = f"{date}/{region}/bedrock/aws4_request"
        string_to_sign = "\n".join(("AWS4-HMAC-SHA256", timestamp, scope,
                                     hashlib.sha256(canonical_request.encode()).hexdigest()))
        def sign(key: bytes, value: str) -> bytes:
            return hmac.new(key, value.encode(), hashlib.sha256).digest()
        signing_key = sign(sign(sign(sign(("AWS4" + secret_key).encode(), date), region), "bedrock"), "aws4_request")
        signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
        headers["Authorization"] = (f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
                                     f"SignedHeaders={signed_headers}, Signature={signature}")

    @staticmethod
    def _image_url(part: dict[str, Any]) -> str:
        value = part.get("image_url", "")
        return value.get("url", "") if isinstance(value, dict) else value

    @staticmethod
    def _anthropic_payload(model: str, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload["messages"]
        system = "\n".join(ProviderClient._content_text(m.get("content")) for m in messages
                            if m["role"] in ("system", "developer") and m.get("content"))
        api_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            if role in ("system", "developer"):
                continue
            if role == "tool":
                content = [{"type": "tool_result", "tool_use_id": message["tool_call_id"],
                            "content": message.get("content") or ""}]
                api_messages.append({"role": "user", "content": content})
                continue
            content: list[dict[str, Any]] = []
            message_content = message.get("content")
            if isinstance(message_content, str) and message_content:
                content.append({"type": "text", "text": message_content})
            elif isinstance(message_content, list):
                for part in message_content:
                    if part.get("type") == "text":
                        content.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image_url":
                        url = ProviderClient._image_url(part)
                        match = re.fullmatch(r"data:(image/[\w.+-]+);base64,(.+)", url, re.DOTALL)
                        if match:
                            content.append({"type": "image", "source": {"type": "base64",
                                "media_type": match.group(1), "data": match.group(2)}})
                        else:
                            content.append({"type": "image", "source": {"type": "url", "url": url}})
            for call in message.get("tool_calls", []):
                function = call.get("function", {})
                try:
                    arguments = json.loads(function.get("arguments", "{}"))
                except (TypeError, ValueError):
                    arguments = {}
                content.append({"type": "tool_use", "id": call.get("id", ""),
                                "name": function.get("name", ""), "input": arguments})
            api_messages.append({"role": role, "content": content or (message.get("content") or "")})
        result: dict[str, Any] = {"model": model, "messages": api_messages,
                                  "max_tokens": payload.get("max_tokens", 1024)}
        if system:
            result["system"] = system
        for key in ("temperature", "top_p"):
            if payload.get(key) is not None:
                result[key] = payload[key]
        if payload.get("tools"):
            result["tools"] = []
            for tool in payload["tools"]:
                function = tool["function"]
                converted = {"name": function["name"],
                             "input_schema": function.get("parameters", {"type": "object"})}
                if function.get("description"):
                    converted["description"] = function["description"]
                if function.get("strict"):
                    converted["strict"] = True
                result["tools"].append(converted)
            choice = payload.get("tool_choice")
            if choice == "none":
                result.pop("tools")
            elif isinstance(choice, dict):
                result["tool_choice"] = {"type": "tool", "name": choice.get("function", {}).get("name")}
            elif choice == "required":
                result["tool_choice"] = {"type": "any"}
            elif choice == "auto":
                result["tool_choice"] = {"type": "auto"}
        response_format = payload.get("response_format")
        if response_format:
            if response_format["type"] == "json_object":
                instruction = "Respond with one valid JSON object only. Do not include markdown fences or surrounding prose."
                result["system"] = f"{result['system']}\n{instruction}" if result.get("system") else instruction
            else:
                result["output_config"] = {"format": {"type": "json_schema",
                    "schema": response_format["json_schema"]["schema"]}}
        return result

    @staticmethod
    def _gemini_payload(payload: dict[str, Any]) -> dict[str, Any]:
        system = [ProviderClient._content_text(m.get("content")) for m in payload["messages"]
                  if m["role"] in ("system", "developer")]
        contents = []
        function_names: dict[str, str] = {}
        for message in payload["messages"]:
            if message["role"] == "assistant":
                for call in message.get("tool_calls", []):
                    function_names[call.get("id", "")] = call.get("function", {}).get("name", "")
        for message in payload["messages"]:
            if message["role"] in ("system", "developer"):
                continue
            if message["role"] == "tool":
                name = message.get("name") or function_names.get(message.get("tool_call_id", ""), "")
                try:
                    response = json.loads(message.get("content") or "{}")
                except (TypeError, ValueError):
                    response = {"result": message.get("content") or ""}
                parts = [{"functionResponse": {"name": name, "response": response}}]
            else:
                parts = []
                message_content = message.get("content")
                if isinstance(message_content, str) and message_content:
                    parts.append({"text": message_content})
                elif isinstance(message_content, list):
                    for part in message_content:
                        if part.get("type") == "text":
                            parts.append({"text": part["text"]})
                        elif part.get("type") == "image_url":
                            url = ProviderClient._image_url(part)
                            match = re.fullmatch(r"data:(image/[\w.+-]+);base64,(.+)", url, re.DOTALL)
                            if match:
                                parts.append({"inlineData": {"mimeType": match.group(1), "data": match.group(2)}})
                            else:
                                parts.append({"fileData": {"mimeType": "image/jpeg", "fileUri": url}})
                        elif part.get("type") == "input_audio":
                            audio = part["input_audio"]
                            mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac",
                                    "ogg": "audio/ogg", "m4a": "audio/mp4"}[audio["format"]]
                            parts.append({"inlineData": {"mimeType": mime, "data": audio["data"]}})
                for call in message.get("tool_calls", []):
                    function = call.get("function", {})
                    try:
                        args = json.loads(function.get("arguments", "{}"))
                    except (TypeError, ValueError):
                        args = {}
                    function_call = {"name": function.get("name", ""), "args": args}
                    if call.get("id"):
                        function_call["id"] = call["id"]
                    parts.append({"functionCall": function_call})
            contents.append({"role": "model" if message["role"] == "assistant" else "user", "parts": parts})
        result: dict[str, Any] = {"contents": contents, "generationConfig": {}}
        if system:
            result["systemInstruction"] = {"parts": [{"text": "\n".join(system)}]}
        for src, dst in (("temperature", "temperature"), ("top_p", "topP"), ("max_tokens", "maxOutputTokens")):
            if payload.get(src) is not None:
                result["generationConfig"][dst] = payload[src]
        if not result["generationConfig"]:
            result.pop("generationConfig")
        tools = payload.get("tools")
        if tools:
            declarations = []
            for tool in tools:
                function = tool["function"]
                declaration = {"name": function["name"],
                               "parameters": function.get("parameters", {"type": "object"})}
                if function.get("description"):
                    declaration["description"] = function["description"]
                declarations.append(declaration)
            result["tools"] = [{"functionDeclarations": declarations}]
            choice = payload.get("tool_choice")
            mode = "NONE" if choice == "none" else "ANY" if choice == "required" or isinstance(choice, dict) else "AUTO"
            function_calling = {"mode": mode}
            if isinstance(choice, dict):
                function_calling["allowedFunctionNames"] = [choice.get("function", {}).get("name")]
            result["toolConfig"] = {"functionCallingConfig": function_calling}
        response_format = payload.get("response_format")
        if response_format:
            generation = result.setdefault("generationConfig", {})
            generation["responseMimeType"] = "application/json"
            if response_format["type"] == "json_schema":
                generation["responseSchema"] = response_format["json_schema"]["schema"]
        if payload.get("modalities") and "audio" in payload["modalities"]:
            generation = result.setdefault("generationConfig", {})
            generation["responseModalities"] = ["TEXT", "AUDIO"]
            if payload.get("audio", {}).get("voice"):
                generation["speechConfig"] = {"voiceConfig": {"prebuiltVoiceConfig": {
                    "voiceName": payload["audio"]["voice"]}}}
        return result

    @staticmethod
    def _normalize(body: dict[str, Any], kind: str, model: str) -> dict[str, Any]:
        tool_calls: list[dict[str, Any]] = []
        audio_parts: list[dict[str, Any]] = []
        if kind == "anthropic":
            text_parts = []
            for block in body.get("content", []):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append({"id": block.get("id"), "type": "function",
                        "function": {"name": block.get("name"),
                                     "arguments": json.dumps(block.get("input", {}), separators=(",", ":"))}})
            text = "".join(text_parts)
            finish = body.get("stop_reason")
            response_id = body.get("id")
        elif kind in {"gemini", "vertex"}:
            candidates = body.get("candidates", [])
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            text_parts = []
            for part in parts:
                if part.get("text"):
                    text_parts.append(part["text"])
                if part.get("functionCall"):
                    call = part["functionCall"]
                    tool_calls.append({"id": call.get("id") or f"call_{len(tool_calls)}", "type": "function",
                        "function": {"name": call.get("name", ""),
                                     "arguments": json.dumps(call.get("args", {}), separators=(",", ":"))}})
                if part.get("inlineData", {}).get("mimeType", "").startswith("audio/"):
                    audio_parts.append({"data": part["inlineData"].get("data", ""),
                                        "mime_type": part["inlineData"].get("mimeType")})
            text = "".join(text_parts)
            finish = candidates[0].get("finishReason") if candidates else None
            response_id = None
        elif kind == "bedrock":
            output = body.get("output", {}).get("message", {})
            text_parts = []
            for block in output.get("content", []):
                if block.get("text"):
                    text_parts.append(block["text"])
                if block.get("toolUse"):
                    tool = block["toolUse"]
                    tool_calls.append({"id": tool.get("toolUseId"), "type": "function",
                        "function": {"name": tool.get("name", ""),
                                     "arguments": json.dumps(tool.get("input", {}), separators=(",", ":"))}})
            text = "".join(text_parts)
            finish = body.get("stopReason")
            response_id = None
        else:
            choices = body.get("choices", [])
            message = choices[0].get("message", {}) if choices else {}
            text = message.get("content") or ""
            tool_calls = message.get("tool_calls", []) or []
            finish = choices[0].get("finish_reason") if choices else None
            response_id = body.get("id")
            audio = message.get("audio")
            audio_parts = [audio] if audio else []
        if not text and not tool_calls and not audio_parts:
            raise ProviderError("Provider returned no text completion", retryable=False)
        usage_kind = "gemini" if kind == "vertex" else kind
        result = {"text": text, "finish_reason": finish, "usage": _token_usage(body, usage_kind),
                  "provider_response_id": response_id, "model": body.get("model", model)}
        if tool_calls:
            result["tool_calls"] = tool_calls
        if audio_parts:
            result["audio"] = audio_parts[0] if len(audio_parts) == 1 else audio_parts
        return result
