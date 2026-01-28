"""Server-side orchestration tool.

This module defines `askpanda_answer`, the intended **single entrypoint** tool
that all clients (Streamlit, Open WebUI, CLI, etc.) should call.

Initial routing rules:
  1) If `bypass_routing=True`, always call `askpanda_llm_answer`.
  2) Else, if the question contains the substring `task <integer>` (case-insensitive),
     route to `panda_task_status`.
  3) Otherwise, call `askpanda_llm_answer`.

The selection rule "task <integer> must always be present" is enforced here so it
is consistent across all clients.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from askpanda_mcp.tools.base import text_content
from askpanda_mcp.tools.llm_passthrough import askpanda_llm_answer_tool
from askpanda_mcp.tools.task_status import panda_task_status_tool


_TASK_PATTERN = re.compile(r"\btask\s+(\d+)\b", re.IGNORECASE)


def _extract_task_id(text: str) -> Optional[int]:
    """Extracts a task ID from free text.

    Args:
        text: Free-form user question.

    Returns:
        Task ID if found, otherwise None.
    """
    m = _TASK_PATTERN.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _coerce_messages(raw: Sequence[Any]) -> List[Dict[str, str]]:
    """Coerces raw message objects into a list of {role, content} dicts.

    Args:
        raw: Sequence of dict-like objects.

    Returns:
        Normalized list of messages.
    """
    out: List[Dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user"))
        content = str(item.get("content", ""))
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


class AskPandaAnswerTool:
    """Top-level orchestration tool."""

    @staticmethod
    def get_definition() -> Dict[str, Any]:
        """Returns the MCP tool definition."""
        return {
            "name": "askpanda_answer",
            "description": (
                "Server-side orchestration entrypoint. Routes to task tools when the "
                "question contains 'task <integer>', otherwise falls back to the default "
                "LLM passthrough."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "User question. Required if messages is empty.",
                    },
                    "messages": {
                        "type": "array",
                        "description": (
                            "Optional full chat history as a list of {role, content}."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                    },
                    "bypass_routing": {
                        "type": "boolean",
                        "description": "If true, bypass routing and call LLM passthrough.",
                        "default": False,
                    },
                    "include_raw": {
                        "type": "boolean",
                        "description": "If true, include raw payloads where supported.",
                        "default": False,
                    },
                },
            },
        }

    async def call(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Executes routing + delegation.

        Args:
            arguments: Tool arguments.

        Returns:
            MCP text content containing the final answer (prefixed with tool used).
        """
        question = str(arguments.get("question", "") or "").strip()
        messages_raw = arguments.get("messages") or []
        messages = _coerce_messages(messages_raw) if messages_raw else []
        bypass = bool(arguments.get("bypass_routing", False))
        include_raw = bool(arguments.get("include_raw", False))

        if not question and not messages:
            raise ValueError("Either 'question' or non-empty 'messages' must be provided.")

        if not question and messages:
            # Best-effort: use last user message as the routing hint.
            for msg in reversed(messages):
                if msg.get("role") == "user" and msg.get("content"):
                    question = msg["content"].strip()
                    break

        tool_used: str
        delegated: List[Dict[str, Any]]

        if bypass:
            tool_used = "askpanda_llm_answer"
            delegated = await askpanda_llm_answer_tool.call(
                {"messages": messages} if messages else {"question": question}
            )
        else:
            task_id = _extract_task_id(question)
            if task_id is not None:
                tool_used = "panda_task_status"
                delegated = await panda_task_status_tool.call(
                    {"task_id": task_id, "include_raw": include_raw}
                )
            else:
                tool_used = "askpanda_llm_answer"
                delegated = await askpanda_llm_answer_tool.call(
                    {"messages": messages} if messages else {"question": question}
                )

        # Flatten delegated response into a single text content item.
        body = ""
        try:
            if delegated and isinstance(delegated[0], dict):
                body = str(delegated[0].get("text", ""))
        except Exception:
            body = str(delegated)

        return text_content(f"[tool={tool_used}]\n\n{body}".strip())


askpanda_answer_tool = AskPandaAnswerTool()
