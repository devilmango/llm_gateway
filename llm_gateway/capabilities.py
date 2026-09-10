from __future__ import annotations

from typing import Any

CAPABILITIES: dict[str, dict[str, bool]] = {
    "openai": {"streaming": True, "tools": True, "structured_output": True, "images": True,
               "audio_input": True, "audio_output": True, "streaming_tools": True,
               "embeddings": True, "reranking": True},
    "azure_openai": {"streaming": True, "tools": True, "structured_output": True, "images": True,
                     "audio_input": True, "audio_output": True, "streaming_tools": True,
                     "embeddings": True, "reranking": True},
    "vllm": {"streaming": True, "tools": True, "structured_output": True, "images": True,
             "audio_input": False, "audio_output": False, "streaming_tools": True,
             "embeddings": True, "reranking": True},
    "anthropic": {"streaming": True, "tools": True, "structured_output": True, "images": True,
                  "audio_input": False, "audio_output": False, "streaming_tools": True},
    "gemini": {"streaming": True, "tools": True, "structured_output": True, "images": True,
               "audio_input": True, "audio_output": True, "streaming_tools": True,
               "embeddings": True, "reranking": True},
    "vertex": {"streaming": True, "tools": True, "structured_output": True, "images": True,
               "audio_input": True, "audio_output": True, "streaming_tools": True},
    "bedrock": {"streaming": True, "tools": True, "structured_output": False, "images": True,
                "audio_input": False, "audio_output": False, "streaming_tools": True},
}


def provider_capabilities(provider: dict[str, Any]) -> dict[str, bool]:
    kind = provider.get("kind", "")
    if kind not in CAPABILITIES:
        raise ValueError(f"Unsupported provider kind {kind!r}")
    overrides = provider.get("capabilities", {})
    unknown = set(overrides) - set(CAPABILITIES[kind])
    if unknown or any(not isinstance(value, bool) for value in overrides.values()):
        raise ValueError(f"Invalid capability override for provider kind {kind!r}")
    return {**CAPABILITIES[kind], **overrides}
