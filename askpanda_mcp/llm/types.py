"""Normalized types for LLM interactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, TypedDict


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
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass(frozen=True)
class LLMResponse:
    """Normalized model response."""
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Optional[TokenUsage] = None
    raw: Any = None


@dataclass(frozen=True)
class ModelSpec:
    """Concrete model configuration."""
    provider: str                 # "openai" | "anthropic" | "gemini" | "openai_compat"
    model: str                    # e.g. "gpt-4.1-mini"
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    extra: dict[str, Any] = None  # provider-specific knobs (optional)


@dataclass(frozen=True)
class GenerateParams:
    """Generation parameters."""
    temperature: float = 0.2
    max_tokens: Optional[int] = None
