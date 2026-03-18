"""Bamboo answer tool — ATLAS-focused orchestration.

Routing (in priority order)
----------------------------
1. **Job log analysis** — job ID + analysis keywords → ``panda_log_analysis``
   → LLM synthesis.
2. **Job status** — bare job ID → ``panda_job_status`` → LLM synthesis.
3. **Task status** — task ID → ``panda_task_status`` → LLM synthesis.
4. **General / fallback** — no recognised ID → ``panda_doc_search`` (RAG)
   → LLM synthesis.  The retrieved context is injected into the prompt so
   the LLM answers from documentation rather than parametric memory.  If
   the RAG tool is unavailable or returns no hits the LLM is still called
   with a clear note that no documentation context was found.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from bamboo.llm.exceptions import LLMConfigError, LLMError, LLMRateLimitError, LLMTimeoutError
from bamboo.llm.types import Message
from bamboo.tools.base import MCPContent, coerce_messages, text_content
from bamboo.tools.doc_rag import panda_doc_search_tool
from bamboo.tools.doc_bm25 import panda_doc_bm25_tool
from bamboo.tools.topic_guard import check_topic
from bamboo.tools.llm_passthrough import bamboo_llm_answer_tool
from bamboo.tools.task_status import panda_task_status_tool
from bamboo.tools.job_status import panda_job_status_tool  # type: ignore[import-untyped]
from bamboo.tools.log_analysis import panda_log_analysis_tool  # type: ignore[import-untyped]
from bamboo.tracing import (
    EVENT_GUARD,
    EVENT_RETRIEVAL,
    EVENT_SYNTHESIS,
    span,
)

# Matches "task 123", "task:123", "task-123" etc. (4-12 digits)
_TASK_PATTERN = re.compile(r"(?i)\btask[:#/\-\s]+([0-9]{4,12})\b")
# Matches "job 123", "job:123", "pandaid 123", "panda id 123" etc.
_JOB_PATTERN = re.compile(r"(?i)\b(?:job|pandaid|panda[\s_-]?id)[:#/\-\s]+([0-9]{4,12})\b")
# Matches "analyse/analyze/why did ... job 123 fail"
_LOG_PATTERN = re.compile(
    r"(?i)(?:analyz?e|analys[ei]|why|fail|log|diagnos)[^.]{0,60}"
    r"\bjob[:#/\-\s]+([0-9]{4,12})\b"
)


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


def _extract_job_id(text: str) -> int | None:
    """Extract a job (PanDA) ID from text.

    Args:
        text: Input text.

    Returns:
        The extracted job ID, or None if not found.
    """
    m = _JOB_PATTERN.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _is_log_analysis_request(text: str) -> bool:
    """Return True if the question is asking for log/failure analysis.

    Args:
        text: User question text.

    Returns:
        True if analysis keywords are present alongside a job reference.
    """
    return bool(_LOG_PATTERN.search(text or ""))


def _compact(obj: Any, limit: int = 6000) -> str:
    """Compact JSON for prompts, bounded to ``limit`` characters."""
    try:
        s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:  # pylint: disable=broad-exception-caught
        s = str(obj)
    if len(s) > limit:
        return s[:limit] + "…(truncated)"
    return s


def _is_bigpanda_error(evidence: Any) -> bool:
    """Return True if evidence indicates an upstream BigPanDA / tool error.

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
    if isinstance(http_status, int) and http_status >= 400:
        return True
    if isinstance(http_status, str) and http_status.isdigit() and int(http_status) >= 400:
        return True
    if evidence.get("exception"):
        return True
    err = evidence.get("error")
    if isinstance(err, str) and err.strip():
        return True
    return False


def _extract_raw_preview(tool_result: Any, evidence: Any, limit: int = 2000) -> str | None:
    """Best-effort extraction of raw upstream response text for error display.

    Args:
        tool_result: Full tool result returned by panda_task_status_tool.
        evidence: Evidence dict.
        limit: Max characters to include.

    Returns:
        Preview string or None if nothing useful was found.
    """
    candidates: list[str] = []

    def _add_from(d: Any) -> None:
        if not isinstance(d, dict):
            return
        for key in ("raw", "raw_text", "raw_body", "response_text", "body",
                    "text", "html", "content", "error_body", "detail", "message"):
            val = d.get(key)
            if isinstance(val, bytes):
                try:
                    val = val.decode("utf-8", errors="replace")
                except Exception:  # pylint: disable=broad-exception-caught
                    val = str(val)
            if isinstance(val, str) and val.strip():
                candidates.append(val.strip())

    _add_from(tool_result)
    _add_from(evidence)
    if isinstance(tool_result, dict):
        _add_from(tool_result.get("upstream"))
        _add_from(tool_result.get("response"))
        _add_from(tool_result.get("error"))

    if not candidates:
        return None
    s = candidates[0]
    if len(s) > limit:
        s = s[:limit] + "…(truncated)"
    return s


def _extract_delegated_text(delegated: Any) -> str:
    """Extract the text body from a delegated bamboo_llm_answer_tool result.

    Args:
        delegated: Raw return value from ``bamboo_llm_answer_tool.call()``.

    Returns:
        Plain text string from the first content block.
    """
    if delegated and isinstance(delegated[0], dict):
        return str(delegated[0].get("text", ""))
    return str(delegated)


async def _call_llm(
    system: str,
    user: str,
    history: list[Message] | None = None,
) -> str:
    """Call the default LLM with a system + user prompt and return the text.

    Prior conversation turns (``history``) are inserted between the system
    prompt and the synthesised user message so the model can resolve follow-up
    questions.  History should contain **raw** question/answer pairs, not the
    synthesised prompts that embed retrieved context or task JSON.

    Args:
        system: System prompt string.
        user: Synthesised user prompt for the current turn.
        history: Optional list of prior ``{role, content}`` turns to inject
            between the system prompt and the current user message.  Must
            contain only ``"user"`` and ``"assistant"`` roles.

    Returns:
        LLM response text.
    """
    messages: list[Message] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})

    delegated = await bamboo_llm_answer_tool.call({
        "messages": messages,
        "max_tokens": 2048,
    })
    return _extract_delegated_text(delegated)


# ---------------------------------------------------------------------------
# RAG retrieval helpers (module-level so they are not re-defined per-call)
# ---------------------------------------------------------------------------

_NO_CONTEXT_SIGNALS: tuple[str, ...] = (
    "not installed",
    "chromadb path not found",
    "failed to connect",
    "no results found",
    "no keyword matches",
    "required and must not be empty",
)


def _extract_rag_context(result: object) -> str:
    """Return text from a retrieval tool result if it contains useful context.

    Args:
        result: Raw return value from a retrieval tool call.

    Returns:
        Extracted text string, or empty string if the result is an error or
        contains a no-context signal on its first line.
    """
    if isinstance(result, Exception) or not result:
        return ""
    if not isinstance(result, list) or not isinstance(result[0], dict):
        return ""
    text = str(result[0].get("text", ""))
    first_line = text.split("\n")[0].lower()
    if any(s in first_line for s in _NO_CONTEXT_SIGNALS):
        return ""
    return text


def _rag_hit_count(result: object, context: str) -> int:
    """Return the number of non-empty result lines, or -1 on retrieval error.

    Args:
        result: Raw return value from a retrieval tool call.
        context: Extracted context string for this result.

    Returns:
        Number of non-empty lines in the context, or -1 if the result was an
        exception.
    """
    if isinstance(result, Exception):
        return -1
    return len([ln for ln in context.splitlines() if ln.strip()])


async def _run_vector_search(question: str) -> str:
    """Run vector search inside its own tracing span.

    Args:
        question: User question to search for.

    Returns:
        Extracted context string, or empty string on failure.
    """
    async with span(EVENT_RETRIEVAL, tool="panda_doc_search", backend="vector") as _s:
        try:
            result = await panda_doc_search_tool.call({"query": question, "top_k": 20})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            result = exc  # type: ignore[assignment]
        ctx = _extract_rag_context(result)
        _s.set(hits=_rag_hit_count(result, ctx))
    return ctx


async def _run_bm25_search(question: str) -> str:
    """Run BM25 keyword search inside its own tracing span.

    Args:
        question: User question to search for.

    Returns:
        Extracted context string, or empty string on failure.
    """
    async with span(EVENT_RETRIEVAL, tool="panda_doc_bm25", backend="bm25") as _s:
        try:
            result = await panda_doc_bm25_tool.call({"query": question, "top_k": 10})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            result = exc  # type: ignore[assignment]
        ctx = _extract_rag_context(result)
        _s.set(hits=_rag_hit_count(result, ctx))
    return ctx


async def _retrieve_rag_context(question: str) -> str:
    """Run vector and BM25 searches concurrently and merge results.

    Args:
        question: User question to retrieve context for.

    Returns:
        Merged context string, or empty string if both searches fail or return
        no useful content.
    """
    try:
        vec_ctx, bm25_ctx = await asyncio.gather(
            _run_vector_search(question),
            _run_bm25_search(question),
        )
        if vec_ctx and bm25_ctx:
            return f"{vec_ctx}\n\n--- Keyword search results ---\n{bm25_ctx}"
        return vec_ctx or bm25_ctx
    except Exception:  # pylint: disable=broad-exception-caught
        return ""  # retrieval failure is non-fatal


def _extract_history(messages: list[Message], current_question: str) -> list[Message]:
    """Extract prior conversation turns from a full messages list.

    The current question (last user message) is excluded so it is not
    duplicated when the synthesised user prompt is appended by ``_call_llm``.
    Only ``"user"`` and ``"assistant"`` role messages are kept; ``"system"``
    messages from the client are dropped because ``_call_llm`` builds its own
    system prompt.

    Only the **last** user turn whose content matches ``current_question`` is
    stripped — earlier turns with the same text (repeated questions) are
    preserved.

    Args:
        messages: Full coerced chat history including the current turn.
        current_question: The question that was derived from the last user
            message, used to identify and strip the final user turn.

    Returns:
        List of prior ``{role, content}`` Message dicts in chronological
        order, suitable for passing as the ``history`` argument to
        ``_call_llm``.
    """
    allowed_roles = {"user", "assistant"}
    # Find the index of the *last* user turn that matches current_question.
    tail_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if (
            msg.get("role") == "user" and
            str(msg.get("content", "")).strip() == current_question.strip()
        ):
            tail_idx = i
            break

    prior: list[Message] = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = str(msg.get("content", "")).strip()
        if role not in allowed_roles or not content:
            continue
        if i == tail_idx:
            continue  # skip only the one identified current-question turn
        prior.append({"role": role, "content": content})  # type: ignore[typeddict-item]
    return prior


def _friendly_llm_error(exc: LLMError) -> str:
    """Return a concise, user-readable explanation of an LLM provider error.

    Inspects the exception type and message to produce actionable advice
    rather than exposing raw SDK internals to the user.

    Args:
        exc: An :class:`~bamboo.llm.exceptions.LLMError` subclass instance.

    Returns:
        A plain-text string suitable for display in the TUI or returned as
        tool output.
    """
    raw = str(exc)

    if isinstance(exc, LLMConfigError):
        return (
            "⚙️  LLM not configured — check your API key environment variables "
            "(e.g. MISTRAL_API_KEY, OPENAI_API_KEY) and restart the server."
        )

    if isinstance(exc, LLMRateLimitError):
        return (
            "⏳  Rate limit reached on the LLM provider. "
            "Please wait a moment and try again."
        )

    if isinstance(exc, LLMTimeoutError):
        return (
            "⏱️  The LLM provider did not respond in time. "
            "This is usually transient — please try again in a moment."
        )

    # Classify common transient HTTP conditions by message content.
    raw_lower = raw.lower()
    _overload_signals = ("503", "502", "overflow", "overloaded",
                         "upstream connect error", "reset before headers",
                         "reset reason")
    if any(s in raw_lower for s in _overload_signals):
        return (
            "🔄  The LLM provider is temporarily overloaded (service unavailable). "
            "This is not a problem with your question — please try again in a few seconds."
        )
    if any(s in raw_lower for s in ("429", "rate limit", "rate_limit", "too many requests")):
        return (
            "⏳  Rate limit reached on the LLM provider. "
            "Please wait a moment and try again."
        )
    _auth_signals = ("401", "403", "unauthorized", "forbidden", "invalid api key",
                     "authentication")
    if any(s in raw_lower for s in _auth_signals):
        return (
            "🔑  Authentication failed with the LLM provider. "
            "Check that your API key is correct and has not expired."
        )
    if any(s in raw_lower for s in ("timeout", "timed out", "deadline")):
        return (
            "⏱️  The request to the LLM provider timed out. "
            "This is usually transient — please try again."
        )

    # Generic provider error — include a sanitised excerpt of the raw message
    # (strip the redundant "Mistral error after retries: " prefix if present).
    _known_prefixes = (
        "mistral error after retries: ",
        "openai error after retries: ",
        "openai-compatible error after retries: ",
        "anthropic error after retries: ",
        "gemini error after retries: ",
        "llm provider error: ",
        "provider error: ",
    )
    for prefix in _known_prefixes:
        if raw_lower.startswith(prefix):
            raw = raw[len(prefix):]
            break
    # Keep it short — the full SDK traceback is not useful to a user.
    excerpt = raw[:200] + ("…" if len(raw) > 200 else "")
    return (
        f"⚠️  The LLM provider returned an error: {excerpt}\n"
        "This may be transient — please try again. "
        "If the problem persists, check the server logs."
    )


class BambooAnswerTool:
    """MCP tool for answering questions about ATLAS tasks using LLM and task metadata."""

    @staticmethod
    def get_definition() -> dict[str, Any]:
        """Return the MCP tool definition for bamboo_answer.

        Returns:
            Tool definition dict.
        """
        return {
            "name": "bamboo_answer",
            "description": (
                "ATLAS Bamboo entrypoint. Uses tools + LLM to answer, "
                "summarising task metadata when applicable."
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
                        "description": "Optional full chat history as a list of {role, content}.",
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
                        "default": False,
                        "description": "If true, skip task-ID extraction and send directly to LLM.",
                    },
                    "include_jobs": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include job records when fetching task status (adds ?jobs=1).",
                    },
                    "include_raw": {
                        "type": "boolean",
                        "default": False,
                        "description": "If true, include a raw response preview in error output.",
                    },
                },
            },
        }

    async def call(self, arguments: dict[str, Any]) -> list[MCPContent]:
        """Handle bamboo_answer tool invocation with optional task context.

        LLM provider errors (overloaded service, rate limits, auth failures,
        timeouts) are caught and returned as a friendly user-readable message
        rather than propagated as exceptions — tools must always return a
        result.

        Args:
            arguments: Tool arguments.

        Returns:
            List[MCPContent]: One-element MCP text content list.

        Raises:
            ValueError: If neither question nor messages is provided.
        """
        question: str = str(arguments.get("question", "") or "").strip()
        messages_raw: list[Any] = arguments.get("messages") or []
        messages: list[Message] = coerce_messages(messages_raw) if messages_raw else []
        bypass_routing: bool = bool(arguments.get("bypass_routing", False))
        include_jobs: bool = bool(arguments.get("include_jobs", True))
        include_raw: bool = bool(arguments.get("include_raw", False))

        # Derive question from the last user message if not supplied directly.
        if not question and messages:
            for msg in reversed(messages):
                if msg.get("role") == "user" and msg.get("content"):
                    question = str(msg.get("content", "")).strip()
                    break

        if not question and not messages:
            raise ValueError("Either 'question' or non-empty 'messages' must be provided.")

        try:
            history = _extract_history(messages, question) if messages else []
            return await self._route(
                question=question,
                history=history,
                bypass_routing=bypass_routing,
                include_jobs=include_jobs,
                include_raw=include_raw,
            )
        except LLMError as exc:
            return text_content(_friendly_llm_error(exc))

    async def _route(
        self,
        question: str,
        history: list[Message],
        bypass_routing: bool,
        include_jobs: bool,
        include_raw: bool,
    ) -> list[MCPContent]:
        """Dispatch to the appropriate synthesis method based on question content.

        Args:
            question: Extracted or derived user question string.
            history: Prior conversation turns (user/assistant pairs) excluding
                the current question, ready to be threaded into ``_call_llm``.
            bypass_routing: Skip ID extraction and go directly to the LLM.
            include_jobs: Pass ``?jobs=1`` when fetching task metadata.
            include_raw: Append raw BigPanDA response snippet on errors.

        Returns:
            List[MCPContent]: One-element MCP text content list.
        """
        if bypass_routing:
            # Bypass path already uses the full messages list directly.
            msgs: list[Message] = []
            if history:
                msgs.extend(history)
            msgs.append({"role": "user", "content": question})
            delegated = await bamboo_llm_answer_tool.call(
                {"messages": msgs} if msgs else {"question": question}
            )
            return text_content(_extract_delegated_text(delegated))

        # Topic guard must run before any tool or LLM call.
        async with span(EVENT_GUARD, tool="topic_guard") as _guard_span:
            guard = await check_topic(question)
            _guard_span.set(
                allowed=guard.allowed,
                reason=guard.reason,
                llm_used=guard.llm_used,
            )
        if not guard.allowed:
            return text_content(guard.rejection_message)

        task_id = _extract_task_id(question)
        job_id = _extract_job_id(question)

        if job_id and _is_log_analysis_request(question):
            return await self._synthesise_log_analysis(question, job_id, history)
        if job_id and not task_id:
            return await self._synthesise_job(question, job_id, include_raw, history)
        if not task_id:
            return await self._synthesise_rag(question, history)
        return await self._synthesise_task(question, task_id, include_jobs, include_raw, history)

    async def _synthesise_log_analysis(
        self, question: str, job_id: int, history: list[Message]
    ) -> list[MCPContent]:
        """Fetch job log analysis and synthesise a diagnostic answer.

        Args:
            question: Original user question.
            job_id: Extracted PanDA job ID.
            history: Prior conversation turns to inject into the LLM prompt.

        Returns:
            List[MCPContent]: One-element MCP text content list.
        """
        tool_result = await panda_log_analysis_tool.call({
            "job_id": job_id,
            "query": question,
            "context": "",
        })
        evidence = tool_result.get("evidence", tool_result)
        system = (
            "You are AskPanDA for the ATLAS experiment.\n"
            "Given a user's question and a JSON evidence object containing PanDA job "
            "log analysis, write a concise diagnostic answer.\n"
            "Rules:\n"
            "- State the failure classification clearly.\n"
            "- Quote relevant log excerpts if present.\n"
            "- Suggest concrete next steps based on the failure type.\n"
            "- Include the BigPanDA monitor link.\n"
            "- Keep it under ~10 bullet points.\n"
        )
        user = f"User question:\n{question}\n\nEvidence JSON:\n{_compact(evidence)}\n"
        async with span(EVENT_SYNTHESIS, tool="bamboo_answer", route="log_analysis"):
            body = await _call_llm(system, user, history)
        return text_content(body)

    async def _synthesise_job(
        self, question: str, job_id: int, include_raw: bool, history: list[Message]
    ) -> list[MCPContent]:
        """Fetch job status and synthesise a concise status answer.

        Args:
            question: Original user question.
            job_id: Extracted PanDA job ID.
            include_raw: Append raw BigPanDA response preview on errors.
            history: Prior conversation turns to inject into the LLM prompt.

        Returns:
            List[MCPContent]: One-element MCP text content list.
        """
        tool_result = await panda_job_status_tool.call({
            "job_id": job_id,
            "query": question,
        })
        evidence = tool_result.get("evidence", tool_result)
        is_error = _is_bigpanda_error(evidence)
        raw_preview = _extract_raw_preview(tool_result, evidence) if is_error and include_raw else None
        system = (
            "You are AskPanDA for the ATLAS experiment.\n"
            "Given a user's question and a JSON evidence object from BigPanDA, "
            "write a concise, helpful answer about the job status.\n"
            "Rules:\n"
            "- If evidence.not_found is true: say the job was not found and suggest "
            "checking the ID.\n"
            "- Otherwise: summarise status, site, queue, pilot error, and timing.\n"
            "- Always include the BigPanDA monitor URL as plain text (not a Markdown "
            "hyperlink), e.g.: Monitor: https://bigpanda.cern.ch/job/12345/\n"
            "- Keep it under ~8 bullet points.\n"
        )
        user = f"User question:\n{question}\n\nEvidence JSON:\n{_compact(evidence)}\n"
        if raw_preview:
            system += "\nOn error, include the Raw response preview verbatim at the end inside a fenced code block.\n"
            user += f"\nRaw response preview:\n{raw_preview}\n"
        async with span(EVENT_SYNTHESIS, tool="bamboo_answer", route="job"):
            body = await _call_llm(system, user, history)
        if raw_preview and "Raw response preview" not in body:
            body = f"{body.rstrip()}\n\nRaw response preview:\n```text\n{raw_preview}\n```\n"
        return text_content(body)

    async def _synthesise_rag(self, question: str, history: list[Message]) -> list[MCPContent]:
        """Retrieve documentation context and synthesise a grounded answer.

        Runs vector and BM25 retrieval concurrently, then calls the LLM with
        the merged context.  If retrieval fails or returns nothing, the LLM is
        still called with an explicit note that no documentation was found.

        Args:
            question: Original user question.
            history: Prior conversation turns to inject into the LLM prompt.

        Returns:
            List[MCPContent]: One-element MCP text content list.
        """
        rag_context = await _retrieve_rag_context(question)

        if rag_context:
            system = (
                "You are AskPanDA, an expert assistant for the PanDA workload management "
                "system and ATLAS experiment workflows at CERN.\n"
                "You are given a user question and relevant excerpts retrieved from the "
                "PanDA/Bamboo documentation knowledge base.\n"
                "Rules:\n"
                "- Base your answer primarily on the retrieved documentation excerpts.\n"
                "- If the excerpts fully answer the question, do not add unreferenced claims.\n"
                "- If the excerpts are only partially relevant, supplement with your general "
                "knowledge but clearly distinguish what comes from documentation vs. general "
                "knowledge.\n"
                "- Be concise and precise. Prefer bullet points for multi-part answers.\n"
                "- Do not fabricate PanDA-specific details (task IDs, queue names, error "
                "codes) that are not in the excerpts.\n"
            )
            user = (
                f"User question:\n{question}\n\n"
                f"Retrieved documentation excerpts:\n{rag_context}\n"
            )
        else:
            system = (
                "You are AskPanDA, an expert assistant for the PanDA workload management "
                "system and ATLAS experiment workflows at CERN.\n"
                "No relevant documentation excerpts were found for this question.\n"
                "Rules:\n"
                "- Do NOT answer from general knowledge or make up any PanDA-specific "
                "details such as error codes, queue names, or configuration values.\n"
                "- Tell the user that the documentation knowledge base did not contain "
                "enough information to answer this question reliably.\n"
                "- Suggest they consult the official PanDA documentation or BigPanDA "
                "monitor directly.\n"
                "- If you can point to a plausible documentation URL or resource name, "
                "do so — but do not invent specific technical values.\n"
            )
            user = f"User question:\n{question}\n"

        async with span(EVENT_SYNTHESIS, tool="bamboo_answer", route="rag"):
            body = await _call_llm(system, user, history)
        return text_content(body)

    async def _synthesise_task(
        self, question: str, task_id: int, include_jobs: bool, include_raw: bool,
        history: list[Message]
    ) -> list[MCPContent]:
        """Fetch task metadata and synthesise an answer grounded in it.

        Args:
            question: Original user question.
            task_id: Extracted PanDA task ID.
            include_jobs: Pass ``?jobs=1`` to the task status tool.
            include_raw: Append raw BigPanDA response preview on errors.
            history: Prior conversation turns to inject into the LLM prompt.

        Returns:
            List[MCPContent]: One-element MCP text content list.
        """
        tool_result = await panda_task_status_tool.call({
            "task_id": task_id,
            "query": question,
            "include_jobs": include_jobs,
        })
        evidence = tool_result.get("evidence", tool_result)
        is_error = _is_bigpanda_error(evidence)
        raw_preview = _extract_raw_preview(tool_result, evidence) if is_error else None

        system = (
            "You are AskPanDA, an assistant for the ATLAS experiment at CERN.\n"
            "You are given a user question and JSON metadata for a PanDA task fetched from BigPanDA.\n"
            "Answer the user's specific question using only data explicitly present in the metadata.\n"
            "Rules:\n"
            "- If evidence.not_found is true or evidence.http_status==404: clearly state that the task ID\n"
            "  was not found in BigPanDA. Say the task does not exist or the ID is incorrect. Do not\n"
            "  include a monitor link.\n"
            "- If evidence indicates a non-JSON or HTTP error (but not 404): explain that BigPanDA returned\n"
            "  an unexpected response and include the monitor_url so the user can check manually.\n"
            "- Otherwise: answer the question directly using only fields present in the metadata.\n"
            "- NEVER infer, guess, or derive values not explicitly in the data. If a requested value is\n"
            "  absent, say it is not available in the metadata rather than inventing it.\n"
            "- The Job list section below the metadata lists actual PanDA job IDs. Use ONLY those IDs\n"
            "  when answering questions about pandaids/job IDs. If the section says no jobs were\n"
            "  returned, say so — never derive job IDs from dataset IDs or any other field.\n"
            "- Be concise. Include the BigPanDA monitor URL as plain text at the end in non-error cases,\n"
            "  e.g.: Monitor: https://bigpanda.cern.ch/task/12345/\n"
        )
        if raw_preview and include_raw:
            system += "\nInclude the raw BigPanDA response snippet verbatim at the end inside a fenced code block.\n"

        user = self._build_task_prompt(question, evidence, raw_preview)

        async with span(EVENT_SYNTHESIS, tool="bamboo_answer", route="task"):
            body = await _call_llm(system, user, history)

        if raw_preview and include_raw:
            body = (
                f"{body.rstrip()}\n\n"
                "**BigPanDA raw response snippet:**\n"
                "```text\n"
                f"{raw_preview}\n"
                "```\n"
            )
        return text_content(body)

    @staticmethod
    def _build_task_prompt(
        question: str, evidence: Any, raw_preview: str | None
    ) -> str:
        """Build the user prompt for task status synthesis.

        Strips the raw jobs array from the payload and replaces it with a
        compact pandaid/status list to avoid prompt truncation and LLM
        hallucination of job IDs.

        Args:
            question: Original user question.
            evidence: Evidence dict from panda_task_status_tool.
            raw_preview: Optional raw response snippet to append.

        Returns:
            Formatted user prompt string.
        """
        payload = evidence.get("payload") if isinstance(evidence, dict) else None
        if isinstance(payload, dict):
            _job_keys = ("jobs", "jobList", "joblist", "job_list", "jobSummary")
            jobs_list = next(
                (payload.get(k) for k in _job_keys if isinstance(payload.get(k), list)),
                None,
            )
            payload_slim = {k: v for k, v in payload.items() if k not in _job_keys}
            evidence_for_prompt: Any = payload_slim

            if not jobs_list:
                job_context = (
                    "\n\nJob list: No job records were returned for this task."
                    " Do not infer PanDA job IDs from any other field."
                )
            else:
                pandaids = [
                    j.get("pandaid") or j.get("PandaID") or j.get("panda_id")
                    for j in jobs_list if isinstance(j, dict)
                ]
                pandaids = [str(p) for p in pandaids if p is not None]
                statuses = [
                    j.get("jobStatus") or j.get("status") or ""
                    for j in jobs_list if isinstance(j, dict)
                ]
                job_rows = [
                    f"  pandaid={pid} status={st}"
                    for pid, st in zip(pandaids, statuses)
                ]
                tail = "\n  …(truncated)" if len(jobs_list) > 200 else ""
                job_context = (
                    f"\n\nJob list ({len(jobs_list)} jobs):\n"
                    f"{chr(10).join(job_rows[:200])}{tail}"
                )
        else:
            evidence_for_prompt = evidence
            job_context = ""

        user = (
            f"User question:\n{question}\n"
            f"\nTask metadata (JSON):\n{_compact(evidence_for_prompt)}"
            f"{job_context}\n"
        )
        if raw_preview:
            user += f"\nBigPanDA raw response snippet:\n{raw_preview}\n"
        return user


bamboo_answer_tool = BambooAnswerTool()

__all__ = ["BambooAnswerTool", "bamboo_answer_tool", "_extract_history"]
