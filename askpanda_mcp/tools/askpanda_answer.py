from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# NOTE:
# Adjust these imports to match your actual module paths / function names.
# The intent is:
#   - task tool: a callable that can accept (question=...) or (task_id=...)
#   - llm passthrough tool: callable that can accept (question=...) or (messages=...)

from askpanda_mcp.tools.task_status import panda_task_status  # <-- update if your symbol differs
from askpanda_mcp.tools.llm_passthrough import askpanda_llm_answer  # <-- update if your symbol differs


_TASK_PATTERN = re.compile(r"\btask\s+(\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class AnswerResult:
    """Result returned by the orchestration tool."""

    answer: str
    tool_used: str
    task_id: Optional[int] = None


def _extract_task_id(question: str) -> Optional[int]:
    """Extract a task id from a question.

    The substring 'task <integer>' must be present for task routing.

    Args:
        question: User question.

    Returns:
        Task ID if present, otherwise None.
    """
    m = _TASK_PATTERN.search(question or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


async def askpanda_answer(
    question: str,
    messages: Optional[List[Dict[str, str]]] = None,
    bypass_routing: bool = False,
    include_raw: bool = False,
) -> Dict[str, Any]:
    """Server-side orchestration entrypoint.

    This should become the single tool that *all* clients call. It performs
    deterministic routing and then delegates to specific tools.

    Routing logic (initial version):
      - If bypass_routing=True -> always call askpanda_llm_answer
      - Else if 'task <integer>' appears in question -> call panda_task_status
      - Else -> call askpanda_llm_answer

    Args:
        question: User's question text.
        messages: Optional full chat history. When provided, it can be used by the
            LLM fallback. (Task tool routing is based only on question.)
        bypass_routing: If True, bypass tool routing and call LLM directly.
        include_raw: If True, pass through to tools that support raw payload output.

    Returns:
        Dict with keys:
          - answer: str
          - tool_used: str
          - task_id: Optional[int]
    """
    q = (question or "").strip()
    if not q and not (messages or []):
        raise ValueError("Either 'question' or non-empty 'messages' must be provided.")

    if bypass_routing:
        # Prefer messages if provided; otherwise send the question.
        if messages:
            llm_resp = await askpanda_llm_answer(messages=messages)
        else:
            llm_resp = await askpanda_llm_answer(question=q)

        return AnswerResult(answer=_coerce_answer(llm_resp), tool_used="askpanda_llm_answer").__dict__

    task_id = _extract_task_id(q)
    if task_id is not None:
        tool_resp = await panda_task_status(task_id=task_id, question=q, include_raw=include_raw)
        return AnswerResult(
            answer=_coerce_answer(tool_resp),
            tool_used="panda_task_status",
            task_id=task_id,
        ).__dict__

    # Fallback: LLM
    if messages:
        llm_resp = await askpanda_llm_answer(messages=messages)
    else:
        llm_resp = await askpanda_llm_answer(question=q)

    return AnswerResult(answer=_coerce_answer(llm_resp), tool_used="askpanda_llm_answer").__dict__


def _coerce_answer(tool_return: Any) -> str:
    """Normalize tool returns to a user-facing string.

    Tools in your codebase may return:
      - str
      - dict with 'answer' or 'text'
      - MCP content objects

    This helper keeps the orchestration tool stable as you iterate.

    Args:
        tool_return: Raw return value from a tool.

    Returns:
        A string answer.
    """
    if tool_return is None:
        return ""

    if isinstance(tool_return, str):
        return tool_return

    if isinstance(tool_return, dict):
        for k in ("answer", "text", "content"):
            v = tool_return.get(k)
            if isinstance(v, str):
                return v
        return str(tool_return)

    # Fallback for any MCP content wrappers / objects
    return str(tool_return)
