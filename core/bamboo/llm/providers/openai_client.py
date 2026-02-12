"""OpenAI LLM client implementation."""

from __future__ import annotations

from collections.abc import Sequence

from bamboo.llm.base import LLMClient
from bamboo.llm.exceptions import LLMProviderError
from bamboo.llm.types import GenerateParams, LLMResponse, Message


class OpenAILLMClient(LLMClient):
    """OpenAI provider client (skeleton).

    Note: Keep vendor imports inside methods to avoid hard dependency if not used.
    """

    async def generate(self, messages: Sequence[Message], params: GenerateParams) -> LLMResponse:
        """Generate a response from the OpenAI provider.

        Args:
            messages: Ordered chat messages to send to the provider.
            params: Generation parameters (temperature, max tokens, etc.).

        Returns:
            LLMResponse containing the generated text.

        Raises:
            LLMProviderError: If the provider call fails.
        """
        try:
            # TODO: import openai SDK lazily and call API here
            # Normalize response into LLMResponse.
            return LLMResponse(text="[OpenAI not wired yet]")
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(str(exc)) from exc
