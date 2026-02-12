"""Normalized types for LLM interactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict


Role = Literal["system", "user", "assistant", "tool"]


class Message(TypedDict, total=False):
    """Normalized chat message.

    Keys are intentionally minimal to keep cross-provider compatibility.
    """

    role: Role
    content: str
    name: str  # Optional display name for tool messages, etc.


@dataclass(frozen=True)
class ToolCall:
    """Represents a tool invocation requested by a model."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class TokenUsage:
    """Normalized token usage (may be partially filled depending on provider)."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    """Normalized model response."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: TokenUsage | None = None
    raw: Any = None


@dataclass(frozen=True)
class ModelSpec:
    """Concrete model configuration."""

    provider: str                 # "openai" | "anthropic" | "gemini" | "openai_compat"
    model: str                    # e.g. "gpt-4.1-mini"
    base_url: str | None = None
    api_key_env: str | None = None
    extra: dict[str, Any] | None = None  # provider-specific knobs (optional)


@dataclass(frozen=True)
class GenerateParams:
    """Generation parameters."""

    temperature: float = 0.2
    max_tokens: int | None = None
