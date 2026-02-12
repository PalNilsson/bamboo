"""Base classes for LLM provider clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from bamboo.llm.types import GenerateParams, LLMResponse, Message, ModelSpec


class LLMClient(ABC):
    """
    Abstract base for provider clients.

    Implementations should:
    - Accept normalized messages and return a normalized response.
    - Hide vendor SDK types completely.
    """

    def __init__(self, model_spec: ModelSpec) -> None:
        """Initialize the client with a model spec.

        Args:
            model_spec: Model specification for this client.
        """
        self._model_spec = model_spec

    async def close(self) -> None:
        """Close any underlying network resources."""
        return

    @property
    def model_spec(self) -> ModelSpec:
        """Return the model spec used by this client."""
        return self._model_spec

    @abstractmethod
    async def generate(
        self,
        messages: Sequence[Message],
        params: GenerateParams,
    ) -> LLMResponse:
        """Generate a completion.

        Args:
            messages: Conversation messages in normalized format.
            params: Generation parameters.

        Returns:
            A normalized LLMResponse.
        """
        raise NotImplementedError
