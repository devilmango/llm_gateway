from __future__ import annotations

import os
import time
from typing import Any

import httpx

from .observability import PROVIDER_LATENCY, PROVIDER_REQUESTS


class ProviderError(Exception):
    def __init__(self, message: str, *, retryable: bool = False, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


def _token_usage(body: dict[str, Any], kind: str) -> dict[str, int]:
    usage = body.get("usageMetadata", {}) if kind == "gemini" else body.get("usage", {})
    if kind == "anthropic":
        prompt = int(usage.get("input_tokens", 0) or 0)
        completion = int(usage.get("output_tokens", 0) or 0)
    elif kind == "gemini":
        prompt = int(usage.get("promptTokenCount", 0) or 0)
        completion = int(usage.get("candidatesTokenCount", 0) or 0)
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

    async def complete(self, provider_name: str, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        cfg = self.providers[provider_name]
        kind = cfg["kind"]
        key = os.getenv(cfg.get("api_key_env", "")) if cfg.get("api_key_env") else None
        if not key and not cfg.get("api_key_optional", False):
            raise ProviderError(f"Provider {provider_name!r} has no API key configured")
        base = cfg["base_url"].rstrip("/")
        headers = {"Content-Type": "application/json"}
        params = None
        if kind == "anthropic":
            url = f"{base}/v1/messages"
            headers.update({"anthropic-version": "2023-06-01"})
            if key:
                headers["x-api-key"] = key
            provider_payload = self._anthropic_payload(model, payload)
        elif kind == "gemini":
            url = f"{base}/models/{model}:generateContent"
            params = {"key": key} if key else None
            provider_payload = self._gemini_payload(payload)
        elif kind == "openai":
            url = f"{base}/chat/completions"
            if key:
                headers["Authorization"] = f"Bearer {key}"
            provider_payload = {**payload, "model": model}
        else:
            raise ProviderError(f"Unsupported provider kind {kind!r}")
        started = time.perf_counter()
        outcome = "network_error"
        try:
            response = await self.http.post(url, json=provider_payload, headers=headers, params=params, timeout=self.timeout)
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
            result = self._normalize(body, kind, model)
        except ProviderError:
            outcome = "invalid_response"
            PROVIDER_REQUESTS.labels(provider=provider_name, outcome=outcome).inc()
            PROVIDER_LATENCY.labels(provider=provider_name, outcome=outcome).observe(time.perf_counter() - started)
            raise
        PROVIDER_REQUESTS.labels(provider=provider_name, outcome="success").inc()
        PROVIDER_LATENCY.labels(provider=provider_name, outcome="success").observe(time.perf_counter() - started)
        return result

    @staticmethod
    def _anthropic_payload(model: str, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload["messages"]
        system = "\n".join(m["content"] for m in messages if m["role"] in ("system", "developer"))
        result: dict[str, Any] = {"model": model, "messages": [m for m in messages if m["role"] in ("user", "assistant")], "max_tokens": payload.get("max_tokens", 1024)}
        if system:
            result["system"] = system
        for key in ("temperature", "top_p"):
            if payload.get(key) is not None:
                result[key] = payload[key]
        return result

    @staticmethod
    def _gemini_payload(payload: dict[str, Any]) -> dict[str, Any]:
        system = [m["content"] for m in payload["messages"] if m["role"] in ("system", "developer")]
        contents = []
        for message in payload["messages"]:
            if message["role"] in ("system", "developer"):
                continue
            contents.append({"role": "model" if message["role"] == "assistant" else "user",
                             "parts": [{"text": message["content"]}]})
        result: dict[str, Any] = {"contents": contents, "generationConfig": {}}
        if system:
            result["systemInstruction"] = {"parts": [{"text": "\n".join(system)}]}
        for src, dst in (("temperature", "temperature"), ("top_p", "topP"), ("max_tokens", "maxOutputTokens")):
            if payload.get(src) is not None:
                result["generationConfig"][dst] = payload[src]
        if not result["generationConfig"]:
            result.pop("generationConfig")
        return result

    @staticmethod
    def _normalize(body: dict[str, Any], kind: str, model: str) -> dict[str, Any]:
        if kind == "anthropic":
            text = "".join(block.get("text", "") for block in body.get("content", []) if block.get("type") == "text")
            finish = body.get("stop_reason")
            response_id = body.get("id")
        elif kind == "gemini":
            candidates = body.get("candidates", [])
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            text = "".join(part.get("text", "") for part in parts)
            finish = candidates[0].get("finishReason") if candidates else None
            response_id = None
        else:
            choices = body.get("choices", [])
            text = choices[0].get("message", {}).get("content", "") if choices else ""
            finish = choices[0].get("finish_reason") if choices else None
            response_id = body.get("id")
        if not text:
            raise ProviderError("Provider returned no text completion", retryable=False)
        return {"text": text, "finish_reason": finish, "usage": _token_usage(body, kind),
                "provider_response_id": response_id, "model": body.get("model", model)}
