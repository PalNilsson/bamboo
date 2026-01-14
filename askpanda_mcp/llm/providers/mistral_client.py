from __future__ import annotations

from typing import Sequence

from askpanda_mcp.llm.base import LLMClient
from askpanda_mcp.llm.exceptions import LLMProviderError
from askpanda_mcp.llm.types import GenerateParams, LLMResponse, Message


class MistralLLMClient(LLMClient):
    """Mistral provider client (skeleton)."""

    async def generate(self, messages: Sequence[Message], params: GenerateParams) -> LLMResponse:
        try:
            return LLMResponse(text="[Mistral not wired yet]")
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(str(exc)) from exc
