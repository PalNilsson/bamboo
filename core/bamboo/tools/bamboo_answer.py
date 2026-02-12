"""Bamboo answer tool for LLM summarizing task metadata.

ATLAS-focused orchestration:
- Extract task id from question/messages
- Call task_status tool to get structured evidence
- Call bamboo_llm_answer (passthrough) with a prompt that includes:
    - original user question
    - compact JSON evidence
- Return a single text content entry
"""
from __future__ import annotations

import json
import re
from typing import Any
from collections.abc import Sequence

from bamboo.llm.types import Message
from bamboo.tools.base import text_content, coerce_messages
from bamboo.tools.llm_passthrough import bamboo_llm_answer_tool
from bamboo.tools.task_status import panda_task_status_tool


# More permissive extraction
_TASK_PATTERN = re.compile(r"(?i)\btask[:#/\-\s]*([0-9]{1,12})\b")


def _extract_task_id(text: str) -> int | None:
    """Extract task ID from text using regex pattern matching.

    Args:
        text: Input text to search for task ID pattern.

    Returns:
        Extracted task ID as integer, or None if not found or invalid.
    """
    m = _TASK_PATTERN.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _coerce_messages(raw: Sequence[Any]) -> list[Message]:
    """Coerce raw message data into Message objects.

    Args:
        raw: Sequence of raw message data.

    Returns:
        List of Message objects.
    """
    return coerce_messages(raw)


def _compact(obj: Any, limit: int = 6000) -> str:
    """Compact JSON for prompt; keep it bounded.

    Args:
        obj: Object to compact as JSON.
        limit: Maximum string length before truncation. Defaults to 6000.

    Returns:
        Compacted JSON string, truncated if necessary with '…(truncated)'.
    """
    try:
        s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:  # pylint: disable=broad-exception-caught
        s = str(obj)
    if len(s) > limit:
        return s[:limit] + "…(truncated)"
    return s


class BambooAnswerTool:
    """MCP tool for answering questions about ATLAS tasks using LLM and task metadata.

    This tool extracts task IDs from user questions and integrates with the
    task_status tool to provide evidence for LLM-generated answers.
    """

    @staticmethod
    def get_definition() -> dict[str, Any]:
        """Get the MCP tool definition for bamboo_answer.

        Returns:
            Dictionary containing tool metadata including name, description,
            and JSON schema for input validation.
        """
        return {
            "name": "bamboo_answer",
            "description": "ATLAS Bamboo entrypoint. Uses tools + LLM to answer, summarizing task metadata when applicable.",
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
                    "include_raw": {"type": "boolean", "default": False, "description": "Forward include_jobs/include_raw to task tool when used."},
                },
            },
        }

    async def call(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Handle bamboo_answer tool invocation with optional task context.

        Extracts task ID from question if present and calls task_status tool
        to get structured evidence. Uses LLM to summarize task metadata and
        answer user question.

        Args:
            arguments: Tool input dict with keys:
                - question: User question (optional if messages provided).
                - messages: Optional list of {role, content} chat history.
                - bypass_routing: If True, skip task detection and use LLM directly.
                - include_raw: If True, include raw job details from task tool.

        Returns:
            List containing a single dict with "type": "text" and "text" key.

        Raises:
            ValueError: If neither question nor messages are provided.
        """
        question: str = str(arguments.get("question", "") or "").strip()
        messages_raw: list[Any] = arguments.get("messages") or []
        messages: list[Message] = _coerce_messages(messages_raw) if messages_raw else []
        bypass: bool = bool(arguments.get("bypass_routing", False))
        include_raw: bool = bool(arguments.get("include_raw", False))

        if not question and not messages:
            raise ValueError("Either 'question' or non-empty 'messages' must be provided.")

        if not question and messages:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content_val = msg.get("content")
                    if content_val:
                        question = str(content_val).strip()
                        break

        if bypass:
            delegated: list[dict[str, Any]] = await bamboo_llm_answer_tool.call({"messages": messages} if messages else {"question": question})
            body: str = str(delegated[0].get("text", "")) if delegated and isinstance(delegated[0], dict) else str(delegated)
            return text_content(body)

        task_id: int | None = _extract_task_id(question)
        if task_id is None:
            delegated = await bamboo_llm_answer_tool.call({"messages": messages} if messages else {"question": question})
            body = str(delegated[0].get("text", "")) if delegated and isinstance(delegated[0], dict) else str(delegated)
            return text_content(body)

        # Call task status tool (our impl returns dict with evidence/text)
        tool_result: dict[str, Any] = await panda_task_status_tool.call({"task_id": task_id, "query": question, "include_jobs": True if include_raw else True})
        evidence: dict[str, Any] = tool_result.get("evidence", tool_result)

        # If task not found / non-json, ask LLM to explain clearly with next steps.
        system: str = (
            "You are AskPanDA for the ATLAS experiment. "
            "Given a user's question and a JSON evidence object from BigPanDA, "
            "write a concise, helpful answer.\n"
            "Rules:\n"
            "- If evidence.not_found is true or evidence.http_status==404: say the task was not found and suggest checking the ID.\n"
            "- If evidence indicates non-JSON/HTTP error: explain BigPanDA returned an error and include monitor_url.\n"
            "- Otherwise: summarize status, task name, owner, start/end times, dsinfo and dataset failures if present.\n"
            "- If job_counts is empty but datasets_summary exists, still describe datasets_summary.\n"
            "- Always include the BigPanDA monitor link.\n"
            "- Keep it under ~8 bullet points.\n"
        )

        prompt_user: str = (
            f"User question:\n{question}\n\n"
            f"Evidence JSON:\n{_compact(evidence)}\n"
        )

        delegated = await bamboo_llm_answer_tool.call(
            {"messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt_user}]}
        )
        body = str(delegated[0].get("text", "")) if delegated and isinstance(delegated[0], dict) else str(delegated)
        return text_content(body)


bamboo_answer_tool = BambooAnswerTool()
