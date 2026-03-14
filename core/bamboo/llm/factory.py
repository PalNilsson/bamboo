"""Factory for building LLM clients based on model specifications."""

from __future__ import annotations

from bamboo.llm.base import LLMClient
from bamboo.llm.exceptions import LLMConfigError
from bamboo.llm.types import ModelSpec

from bamboo.llm.providers.anthropic_client import AnthropicLLMClient
from bamboo.llm.providers.gemini_client import GeminiLLMClient
from bamboo.llm.providers.openai_client import OpenAILLMClient
from bamboo.llm.providers.mistral_client import MistralLLMClient
from bamboo.llm.providers.openai_compat_client import OpenAICompatLLMClient

_PROVIDER_MAP: dict[str, type[LLMClient]] = {
    "openai": OpenAILLMClient,
    "anthropic": AnthropicLLMClient,
    "gemini": GeminiLLMClient,
    "mistral": MistralLLMClient,
    "openai_compat": OpenAICompatLLMClient,
}


def build_client(model_spec: ModelSpec) -> LLMClient:
    """Build an LLM client for a given ModelSpec.

    Args:
        model_spec: Model configuration including provider and model identifiers.

    Returns:
        LLMClient instance for the specified provider.

    Raises:
        LLMConfigError: If the provider name in model_spec is not registered.
    """
    cls = _PROVIDER_MAP.get(model_spec.provider)
    if not cls:
        raise LLMConfigError(f"Unknown LLM provider: {model_spec.provider}")
    return cls(model_spec)
