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
# pylint: disable=consider-using-alias
"""

from __future__ import annotations

import re
from typing import Any
from collections.abc import Sequence
from askpanda_mcp.llm.types import Message

from askpanda_mcp.tools.base import text_content, coerce_messages
from askpanda_mcp.tools.llm_passthrough import askpanda_llm_answer_tool
from askpanda_mcp.tools.task_status import panda_task_status_tool


_TASK_PATTERN = re.compile(r"\btask\s+(\d+)\b", re.IGNORECASE)


def _extract_task_id(text: str) -> int | None:
    """Extract a task ID from free text.

    Args:
        text: Free-form user question.

    Returns:
        Optional[int]: Task ID if found, otherwise None.
    """
    m = _TASK_PATTERN.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _coerce_messages(raw: Sequence[Any]) -> list[Message]:
    """Wrapper that delegates to shared coerce_messages helper."""
    return coerce_messages(raw)


class AskPandaAnswerTool:
    """Top-level orchestration tool.

    The tool routes incoming user prompts to more specific tools:
    - If ``bypass_routing`` is True, forwards to the LLM passthrough tool.
    - If the question contains a task id (e.g. "task 1234"), it delegates to
      the task-status tool.
    - Otherwise, it falls back to the LLM passthrough.
    """

    @staticmethod
    def get_definition() -> dict[str, Any]:
        """Return the MCP tool discovery definition.

        Returns:
            Dict[str, Any]: Tool definition including input schema used by
            MCP clients to validate and call this tool.
        """
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

    # The orchestration logic is intentionally compact; complexity is accepted here.
    # pylint: disable=too-complex
    async def call(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute routing and delegation to the appropriate tool.

        This method implements the orchestration policies described in the
        module docstring.

        Args:
            arguments: Mapping that should include either a ``question`` string
                or a non-empty ``messages`` list. Optional flags:
                ``bypass_routing`` (bool) and ``include_raw`` (bool).

        Returns:
            List[Dict[str, Any]]: MCP content list (text payload) produced by
            the delegated tool, wrapped with a short prefix indicating which
            tool was used.

        Raises:
            ValueError: If neither a question nor non-empty messages are provided.
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
                if msg.get("role") == "user":
                    content_val = msg.get("content")
                    if content_val:
                        question = str(content_val).strip()
                        break

        tool_used: str
        delegated: list[dict[str, Any]]

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
            elif delegated and isinstance(delegated[0], dict):
                body = str(delegated[0].get("content", ""))
        except (IndexError, TypeError, KeyError):
            body = str(delegated)

        return text_content(f"[tool={tool_used}]\n\n{body}".strip())


askpanda_answer_tool = AskPandaAnswerTool()
