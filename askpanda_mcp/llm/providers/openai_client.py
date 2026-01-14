"""OpenAI LLM client implementation."""

from __future__ import annotations

from typing import Sequence

from askpanda_mcp.llm.base import LLMClient
from askpanda_mcp.llm.exceptions import LLMProviderError
from askpanda_mcp.llm.types import GenerateParams, LLMResponse, Message


class OpenAILLMClient(LLMClient):
    """OpenAI provider client (skeleton).

    Note: Keep vendor imports inside methods to avoid hard dependency if not used.
    """

    async def generate(self, messages: Sequence[Message], params: GenerateParams) -> LLMResponse:
        try:
            # TODO: import openai SDK lazily and call API here
            # Normalize response into LLMResponse.
            return LLMResponse(text="[OpenAI not wired yet]")
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(str(exc)) from exc
