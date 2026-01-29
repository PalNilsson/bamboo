"""Common tool patterns and small helpers for AskPanDA tools.

This module defines the small runtime protocol used by MCP tools (`MCPTool`),
plus lightweight helpers for building MCP-compatible content payloads.

The `MCPTool` protocol describes the minimal shape a tool implementation must
expose so it can be registered with the MCP server in this project.
"""
from __future__ import annotations
from typing import Any, Protocol, cast
from collections.abc import Sequence

from askpanda_mcp.llm.types import Message

MCPContent = dict[str, Any]


class MCPTool(Protocol):
    """Protocol describing a minimal MCP tool implementation.

    Implementations must provide `get_definition` and an async `call` method.
    """

    @staticmethod
    def get_definition() -> dict[str, Any]:
        """Return a serializable definition describing the tool."""
        raise NotImplementedError

    async def call(self, arguments: dict[str, Any]) -> list[MCPContent]:
        """Execute the tool with the provided arguments."""
        raise NotImplementedError


def coerce_messages(raw: Sequence[Any]) -> list[Message]:
    """Coerce raw message objects into a list of {role, content} dicts.

    This helper is shared by multiple tools to normalize incoming message
    representations into the simple ``{{'role': str, 'content': str}}`` shape.
    """
    out: list[Message] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user"))
        content = str(item.get("content", ""))
        if not content:
            continue
        out.append(cast(Message, {"role": role, "content": content}))
    return out


def text_content(text: str) -> list[MCPContent]:
    """Create a simple text content payload for MCP responses.

    Args:
        text: Human-readable text to include in the response.
    """
    return [{"type": "text", "text": text}]


__all__ = ["MCPContent", "MCPTool", "text_content", "coerce_messages"]
