"""Factory for building LLM clients based on model specifications."""

from __future__ import annotations

from typing import Dict, Type

from askpanda_mcp.llm.base import LLMClient
from askpanda_mcp.llm.exceptions import LLMConfigError
from askpanda_mcp.llm.types import ModelSpec

from askpanda_mcp.llm.providers.anthropic_client import AnthropicLLMClient
from askpanda_mcp.llm.providers.gemini_client import GeminiLLMClient
from askpanda_mcp.llm.providers.openai_client import OpenAILLMClient
from askpanda_mcp.llm.providers.openai_compat_client import OpenAICompatLLMClient


_PROVIDER_MAP: Dict[str, Type[LLMClient]] = {
    "openai": OpenAILLMClient,
    "anthropic": AnthropicLLMClient,
    "gemini": GeminiLLMClient,
    "openai_compat": OpenAICompatLLMClient,
}


def build_client(model_spec: ModelSpec) -> LLMClient:
    """
    Build an LLM client for a given ModelSpec.

    Args:
        model_spec: Model configuration.

    Returns:
        LLMClient instance.

    Raises:
        LLMConfigError: If provider is unknown.
    """
    cls = _PROVIDER_MAP.get(model_spec.provider)
    if not cls:
        raise LLMConfigError(f"Unknown LLM provider: {model_spec.provider}")
    return cls(model_spec)
