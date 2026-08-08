from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class CostReconciliationItem(BaseModel):
    request_id: str = Field(min_length=1, max_length=120)
    actual_cost_usd: float = Field(ge=0)


class CostReconciliationRequest(BaseModel):
    source: str = Field(min_length=1, max_length=120)
    records: list[CostReconciliationItem] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def unique_request_ids(self) -> CostReconciliationRequest:
        ids = [item.request_id for item in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("Each request_id may appear only once per reconciliation batch")
        return self


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def validate_message_content(self) -> ChatMessage:
        if self.role == "assistant" and (self.content is not None or self.tool_calls):
            return self
        if self.content is None:
            raise ValueError("Message content is required unless an assistant message contains tool_calls")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool_call_id is required for tool messages")
        if isinstance(self.content, list):
            for part in self.content:
                if not isinstance(part, dict) or part.get("type") not in {"text", "image_url", "input_audio"}:
                    raise ValueError("Multimodal content supports text, image_url, and input_audio parts")
                if part["type"] == "text" and not isinstance(part.get("text"), str):
                    raise ValueError("Text content parts require a text value")
                if part["type"] == "image_url":
                    image_url = part.get("image_url")
                    if isinstance(image_url, dict):
                        image_url = image_url.get("url")
                    if not isinstance(image_url, str) or not image_url.startswith(("https://", "http://", "data:image/")):
                        raise ValueError("image_url must be an HTTP(S) URL or image data URI")
                if part["type"] == "input_audio":
                    audio = part.get("input_audio")
                    if not isinstance(audio, dict) or not isinstance(audio.get("data"), str) or audio.get("format") not in {"wav", "mp3", "flac", "ogg", "m4a"}:
                        raise ValueError("input_audio requires base64 data and a supported format")
        return self


class ChatRequest(BaseModel):
    # `model` is a logical route such as simple or reasoning.
    model: str = "simple"
    messages: list[ChatMessage] = Field(min_length=1)
    application: str = Field(default="default", max_length=120)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=100_000)
    top_p: float | None = Field(default=None, gt=0, le=1)
    user: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    stream: bool = False
    tools: list[dict[str, Any]] | None = Field(default=None, max_length=128)
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    cost_center: str | None = Field(default=None, max_length=120)
    modalities: list[Literal["text", "audio"]] | None = None
    audio: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_generation_options(self) -> ChatRequest:
        if self.tools is not None:
            for tool in self.tools:
                function = tool.get("function")
                if (tool.get("type") != "function" or not isinstance(function, dict)
                        or not isinstance(function.get("name"), str) or not function.get("name")):
                    raise ValueError("Each tool must use type=function and include function.name")
                parameters = function.get("parameters", {"type": "object"})
                if not isinstance(parameters, dict) or parameters.get("type") != "object":
                    raise ValueError("Function parameters must be a JSON Schema object")
        if self.tool_choice is not None and self.tools is None:
            raise ValueError("tool_choice requires tools")
        if isinstance(self.tool_choice, str) and self.tool_choice not in {"auto", "none", "required"}:
            raise ValueError("tool_choice must be auto, none, required, or a named function")
        if isinstance(self.tool_choice, dict):
            function = self.tool_choice.get("function")
            function_name = function.get("name") if isinstance(function, dict) else None
            names = {tool.get("function", {}).get("name") for tool in self.tools or []}
            if self.tool_choice.get("type") != "function" or function_name not in names:
                raise ValueError("Named tool_choice must reference a function listed in tools")
        if self.modalities and "audio" in self.modalities and not self.audio:
            raise ValueError("Audio output requires audio voice/format settings")
        if self.audio is not None and (not self.audio.get("voice") or self.audio.get("format") not in {"wav", "mp3", "opus", "flac"}):
            raise ValueError("Audio output requires a voice and supported format")
        if self.response_format is not None:
            kind = self.response_format.get("type")
            if kind not in {"json_object", "json_schema"}:
                raise ValueError("response_format.type must be json_object or json_schema")
            if kind == "json_schema":
                schema_spec = self.response_format.get("json_schema", {})
                if (not isinstance(schema_spec, dict) or not schema_spec.get("name")
                        or not isinstance(schema_spec.get("schema"), dict)):
                    raise ValueError("json_schema response_format requires a name and schema object")
        return self


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatChoice(BaseModel):
    index: int = 0
    message: dict[str, Any]
    finish_reason: str | None = None


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage
    gateway: dict[str, Any]
