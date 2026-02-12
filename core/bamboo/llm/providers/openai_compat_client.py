"""OpenAI-compatible LLM client implementation."""

from __future__ import annotations

from collections.abc import Sequence

from bamboo.llm.base import LLMClient
from bamboo.llm.exceptions import LLMProviderError
from bamboo.llm.types import GenerateParams, LLMResponse, Message


class OpenAICompatLLMClient(LLMClient):
    """OpenAI-compatible provider client (skeleton).

    Use this for vLLM/Ollama/LM Studio/Together/Fireworks or any OpenAI-compatible endpoint.
    """

    async def generate(self, messages: Sequence[Message], params: GenerateParams) -> LLMResponse:
        """Generate a response from an OpenAI-compatible provider.

        Args:
            messages: Ordered chat messages to send to the provider.
            params: Generation parameters (temperature, max tokens, etc.).

        Returns:
            LLMResponse containing the generated text.

        Raises:
            LLMProviderError: If the provider call fails.
        """
        try:
            return LLMResponse(text="[OpenAI-compatible not wired yet]")
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(str(exc)) from exc
