"""Bamboo answer tool for LLM summarizing task metadata.

ATLAS-focused orchestration:
- Extract task id from question/messages
- Call task_status tool to get structured evidence
- Call bamboo_llm_answer (passthrough) with a prompt that includes:
    - original user question
    - compact JSON evidence
- Show a raw response preview ONLY when BigPanDA/tool errors occur
- Return a single text content entry
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from bamboo.llm.types import Message
from bamboo.tools.base import coerce_messages, text_content
from bamboo.tools.llm_passthrough import bamboo_llm_answer_tool
from bamboo.tools.task_status import panda_task_status_tool

_TASK_PATTERN = re.compile(r"(?i)\btask[:#/\-\s]*([0-9]{1,12})\b")


def _extract_task_id(text: str) -> int | None:
    """Extract a task ID from text.

    Args:
        text: Input text.

    Returns:
        The extracted task ID, or None if no task ID is found.
    """
    m = _TASK_PATTERN.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _coerce_messages(raw: Sequence[Any]) -> list[Message]:
    """Coerce raw message data into Message objects."""
    return coerce_messages(raw)


def _compact(obj: Any, limit: int = 6000) -> str:
    """Compact JSON for prompts and keep it bounded."""
    try:
        s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:  # pylint: disable=broad-exception-caught
        s = str(obj)
    if len(s) > limit:
        return s[:limit] + "…(truncated)"
    return s


def _is_bipanda_error(evidence: Any) -> bool:
    """Return True if evidence indicates an upstream BigPanDA/tool error.

    This is intentionally strict so we don't show raw previews for successful
    queries.

    Args:
        evidence: Evidence dict (typically from panda_task_status_tool).

    Returns:
        True if the evidence indicates an error or not-found condition.
    """
    if not isinstance(evidence, dict):
        return False

    if evidence.get("not_found") is True:
        return True

    if evidence.get("non_json") is True:
        return True

    if evidence.get("json_error"):
        return True

    http_status = evidence.get("http_status")
    if isinstance(http_status, int):
        return http_status >= 400
    if isinstance(http_status, str) and http_status.isdigit():
        return int(http_status) >= 400

    # Fall back to explicit exception markers if present.
    if evidence.get("exception"):
        return True

    err = evidence.get("error")
    if isinstance(err, str) and err.strip():
        return True

    return False


def _extract_raw_preview(tool_result: Any, evidence: Any, limit: int = 2000) -> str | None:
    """Best-effort extraction of raw upstream response text (for error display).

    We do not assume an exact schema; we look for common keys that might contain
    raw HTML/text/body returned by BigPanDA or an intermediate service.

    Args:
        tool_result: Full tool result returned by panda_task_status_tool.
        evidence: Evidence dict (tool_result.get("evidence", tool_result)).
        limit: Max characters to show.

    Returns:
        Preview string or None if nothing useful was found.
    """
    candidates: list[str] = []

    def add_from(d: Any) -> None:
        if not isinstance(d, dict):
            return
        for key in (
            "raw",
            "raw_text",
            "raw_body",
            "response_text",
            "body",
            "text",
            "html",
            "content",
            "error_body",
            "detail",
            "message",
        ):
            val = d.get(key)
            if isinstance(val, bytes):
                try:
                    val = val.decode("utf-8", errors="replace")
                except Exception:  # pylint: disable=broad-exception-caught
                    val = str(val)
            if isinstance(val, str) and val.strip():
                candidates.append(val.strip())

    add_from(tool_result)
    add_from(evidence)

    if isinstance(tool_result, dict):
        add_from(tool_result.get("upstream"))
        add_from(tool_result.get("response"))
        add_from(tool_result.get("error"))

    if not candidates:
        return None

    s = candidates[0]
    if len(s) > limit:
        s = s[:limit] + "…(truncated)"
    return s


class BambooAnswerTool:
    """MCP tool for answering questions about ATLAS tasks using LLM and task metadata."""

    @staticmethod
    def get_definition() -> dict[str, Any]:
        """Get the MCP tool definition for bamboo_answer."""
        return {
            "name": "bamboo_answer",
            "description": (
                "ATLAS Bamboo entrypoint. Uses tools + LLM to answer, summarizing task metadata when applicable."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "User question. Required if messages is empty."},
                    "messages": {
                        "type": "array",
                        "description": "Optional full chat history as a list of {role, content}.",
                        "items": {
                            "type": "object",
                            "properties": {"role": {"type": "string"}, "content": {"type": "string"}},
                            "required": ["role", "content"],
                        },
                    },
                    "bypass_routing": {"type": "boolean", "default": False},
                    "include_raw": {
                        "type": "boolean",
                        "default": False,
                        "description": "If True, forward include_jobs/include_raw to task tool when used.",
                    },
                },
            },
        }

    async def call(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Handle bamboo_answer tool invocation with optional task context."""
        question: str = str(arguments.get("question", "") or "").strip()
        messages_raw: list[Any] = arguments.get("messages") or []
        messages: list[Message] = _coerce_messages(messages_raw) if messages_raw else []
        bypass_routing: bool = bool(arguments.get("bypass_routing", False))
        include_raw: bool = bool(arguments.get("include_raw", False))

        if not question and messages:
            for msg in reversed(messages):
                if msg.get("role") == "user" and msg.get("content"):
                    question = str(msg["content"]).strip()
                    break

        if not question and not messages:
            raise ValueError("Either 'question' or non-empty 'messages' must be provided.")

        if bypass_routing:
            delegated = await bamboo_llm_answer_tool.call({"messages": messages} if messages else {"question": question})
            body = str(delegated[0].get("text", "")) if delegated and isinstance(delegated[0], dict) else str(delegated)
            return text_content(body)

        task_id = _extract_task_id(question)
        if task_id is None:
            delegated = await bamboo_llm_answer_tool.call({"messages": messages} if messages else {"question": question})
            body = str(delegated[0].get("text", "")) if delegated and isinstance(delegated[0], dict) else str(delegated)
            return text_content(body)

        # Call task status tool.
        tool_args = {
            "task_id": task_id,
            "query": question,
            "include_jobs": include_raw,  # FIX: honor include_raw (was always True)
            "include_raw": include_raw,
        }
        tool_result: dict[str, Any] = await panda_task_status_tool.call(tool_args)
        evidence: dict[str, Any] = tool_result.get("evidence", tool_result)

        is_error = _is_bipanda_error(evidence)
        raw_preview = _extract_raw_preview(tool_result, evidence) if is_error else None

        system: str = (
            "You are AskPanDA for the ATLAS experiment.\n"
            "Given a user's question and a JSON evidence object from BigPanDA, write a concise, helpful answer.\n"
            "Rules:\n"
            "- If evidence.not_found is true or evidence.http_status==404: say the task was not found and suggest checking the ID.\n"
            "- If evidence indicates non-JSON/HTTP error: explain BigPanDA returned an error and include monitor_url.\n"
            "- Otherwise: summarize status, task name, owner, start/end times, dsinfo and dataset failures if present.\n"
            "- If job_counts is empty but datasets_summary exists, still describe datasets_summary.\n"
            "- Always include the BigPanDA monitor link if present.\n"
            "- Keep it under ~8 bullet points.\n"
        )

        prompt_user: str = f"User question:\n{question}\n\nEvidence JSON:\n{_compact(evidence)}\n"

        # Show raw preview ONLY on errors.
        if raw_preview:
            system += "\nOn error, include the Raw response preview verbatim at the end inside a fenced code block.\n"
            prompt_user += f"\nRaw response preview:\n{raw_preview}\n"

        delegated = await bamboo_llm_answer_tool.call(
            {"messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt_user}]}
        )
        body = str(delegated[0].get("text", "")) if delegated and isinstance(delegated[0], dict) else str(delegated)

        # Guarantee raw preview appears (errors only) even if LLM ignores instructions.
        if raw_preview and "Raw response preview" not in body:
            body = (
                f"{body.rstrip()}\n\n"
                "Raw response preview:\n"
                "```text\n"
                f"{raw_preview}\n"
                "```\n"
            )

        return text_content(body)


bamboo_answer_tool = BambooAnswerTool()
