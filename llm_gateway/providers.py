from __future__ import annotations

import json
import os
import re
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

    async def stream(self, provider_name: str, model: str, payload: dict[str, Any]):
        """Yield text deltas normalized from each supported provider's native SSE API."""
        cfg = self.providers[provider_name]
        kind = cfg["kind"]
        key = os.getenv(cfg.get("api_key_env", "")) if cfg.get("api_key_env") else None
        if not key and not cfg.get("api_key_optional", False):
            raise ProviderError(f"Provider {provider_name!r} has no API key configured")
        headers = {"Content-Type": "application/json"}
        params = None
        base = cfg["base_url"].rstrip("/")
        if kind == "openai":
            url = f"{base}/chat/completions"
            if key:
                headers["Authorization"] = f"Bearer {key}"
            request_payload = {**payload, "model": model, "stream": True, "stream_options": {"include_usage": True}}
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
        else:
            raise ProviderError(f"Unsupported provider kind {kind!r}")
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
                async for line in response.aiter_lines():
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
                    if kind == "openai":
                        event_usage = event.get("usage")
                        choices = event.get("choices", [])
                        choice = choices[0] if choices else {}
                        delta = choice.get("delta", {}).get("content")
                        finish_reason = choice.get("finish_reason") or finish_reason
                        if event_usage:
                            usage = {"prompt_tokens": int(event_usage.get("prompt_tokens", 0) or 0),
                                     "completion_tokens": int(event_usage.get("completion_tokens", 0) or 0),
                                     "total_tokens": int(event_usage.get("total_tokens", 0) or 0)}
                    elif kind == "anthropic":
                        event_type = event.get("type")
                        event_usage = event.get("message", {}).get("usage") if event_type == "message_start" else event.get("usage")
                        if event_type == "content_block_delta":
                            delta = event.get("delta", {}).get("text")
                        elif event_type == "message_delta":
                            finish_reason = event.get("delta", {}).get("stop_reason") or finish_reason
                        if event_usage:
                            usage["prompt_tokens"] = int(event_usage.get("input_tokens", usage["prompt_tokens"]) or 0)
                            usage["completion_tokens"] = int(event_usage.get("output_tokens", usage["completion_tokens"]) or 0)
                            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                    else:
                        candidates = event.get("candidates", [])
                        candidate = candidates[0] if candidates else {}
                        parts = candidate.get("content", {}).get("parts", [])
                        delta = "".join(part.get("text", "") for part in parts if isinstance(part.get("text"), str))
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

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
        return ""

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
        return result

    @staticmethod
    def _normalize(body: dict[str, Any], kind: str, model: str) -> dict[str, Any]:
        tool_calls: list[dict[str, Any]] = []
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
        elif kind == "gemini":
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
            text = "".join(text_parts)
            finish = candidates[0].get("finishReason") if candidates else None
            response_id = None
        else:
            choices = body.get("choices", [])
            message = choices[0].get("message", {}) if choices else {}
            text = message.get("content") or ""
            tool_calls = message.get("tool_calls", []) or []
            finish = choices[0].get("finish_reason") if choices else None
            response_id = body.get("id")
        if not text and not tool_calls:
            raise ProviderError("Provider returned no text completion", retryable=False)
        result = {"text": text, "finish_reason": finish, "usage": _token_usage(body, kind),
                  "provider_response_id": response_id, "model": body.get("model", model)}
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result
