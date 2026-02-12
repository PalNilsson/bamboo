"""Gemini LLM client implementation."""

from __future__ import annotations

from collections.abc import Sequence

from bamboo.llm.base import LLMClient
from bamboo.llm.exceptions import LLMProviderError
from bamboo.llm.types import GenerateParams, LLMResponse, Message


class GeminiLLMClient(LLMClient):
    """Google Gemini provider client (skeleton)."""

    async def generate(self, messages: Sequence[Message], params: GenerateParams) -> LLMResponse:
        """Generate a response from the Gemini provider.

        Args:
            messages: Ordered chat messages to send to the provider.
            params: Generation parameters (temperature, max tokens, etc.).

        Returns:
            LLMResponse containing the generated text.

        Raises:
            LLMProviderError: If the provider call fails.
        """
        try:
            return LLMResponse(text="[Gemini not wired yet]")
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(str(exc)) from exc
