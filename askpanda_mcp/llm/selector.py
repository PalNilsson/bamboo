"""Model selection logic based on task type and tenant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from askpanda_mcp.llm.registry import ModelRegistry
from askpanda_mcp.llm.types import ModelSpec


TaskType = Literal["route", "synthesize", "rag_answer", "log_analysis"]


@dataclass(frozen=True)
class LLMSelector:
    """Selects a model profile for a given task and tenant."""

    registry: ModelRegistry
    default_profile: str = "default"
    fast_profile: str = "fast"
    reasoning_profile: str = "reasoning"

    def select(self, task: TaskType, _tenant: str | None = None) -> ModelSpec:
        """Selects a ModelSpec for a task.

        Args:
            task: High-level task category.
            Optional tenant identifier (future: per-tenant overrides).

        Returns:
            ModelSpec for the chosen profile.
        """
        if task == "route":
            return self.registry.get(self.fast_profile)
        if task in {"log_analysis", "rag_answer"}:
            return self.registry.get(self.reasoning_profile)
        return self.registry.get(self.default_profile)
