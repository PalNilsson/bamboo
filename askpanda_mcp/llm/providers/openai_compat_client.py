"""OpenAI-compatible LLM client implementation."""

from __future__ import annotations

from collections.abc import Sequence

from askpanda_mcp.llm.base import LLMClient
from askpanda_mcp.llm.exceptions import LLMProviderError
from askpanda_mcp.llm.types import GenerateParams, LLMResponse, Message


class OpenAICompatLLMClient(LLMClient):
    """OpenAI-compatible provider client (skeleton).

    Use this for vLLM/Ollama/LM Studio/Together/Fireworks or any OpenAI-compatible endpoint.
    """

    async def generate(self, _messages: Sequence[Message], _params: GenerateParams) -> LLMResponse:
        try:
            return LLMResponse(text="[OpenAI-compatible not wired yet]")
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(str(exc)) from exc
