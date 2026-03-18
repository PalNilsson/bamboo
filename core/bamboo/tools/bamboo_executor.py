"""Plan executor for Bamboo — runs a validated :class:`~bamboo.tools.planner.Plan`.

This module provides the execution layer that sits between the planner and the
LLM synthesiser.  Given a ``Plan`` produced by :mod:`bamboo.tools.planner`, it:

1. Iterates ``plan.tool_calls`` in order.
2. Resolves each tool via the core ``TOOLS`` registry or the plugin entry-point
   loader.
3. Validates arguments with :func:`~bamboo.core._validate_arguments`.
4. Calls ``await tool.call(args)`` and collects ``list[MCPContent]``.
5. Unpacks JSON evidence from evidence tools.
6. Selects a synthesis system prompt based on which tools were called.
7. Synthesises a final natural-language answer via the LLM.

All functions are intentionally **pure orchestration** — no experiment-specific
logic lives here.  Synthesis prompts are kept as module-level constants so they
can be updated independently of routing logic.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from bamboo.llm.types import Message
from bamboo.tools.base import MCPContent, text_content
from bamboo.tools.llm_passthrough import bamboo_llm_answer_tool
from bamboo.tools.loader import find_tool_by_name
from bamboo.tools.planner import Plan
from bamboo.tracing import EVENT_PLAN, EVENT_RETRIEVAL, EVENT_SYNTHESIS, span

# ---------------------------------------------------------------------------
# Synthesis system prompt constants (moved from bamboo_answer.py)
# ---------------------------------------------------------------------------

_SYSTEM_LOG_ANALYSIS: str = (
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

_SYSTEM_JOB: str = (
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

_SYSTEM_TASK: str = (
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

_SYSTEM_RAG: str = (
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

_SYSTEM_RAG_NO_CONTEXT: str = (
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

_SYSTEM_GENERIC: str = (
    "You are AskPanDA, an expert assistant for the PanDA workload management "
    "system and ATLAS experiment workflows at CERN.\n"
    "You have been given the results of one or more tool calls. Synthesise "
    "a clear, concise answer to the user's question based solely on the "
    "evidence provided.\n"
    "Rules:\n"
    "- Do not infer or fabricate values not present in the evidence.\n"
    "- If the evidence shows errors or empty results, explain that clearly.\n"
    "- Include relevant monitor URLs when available.\n"
    "- Be concise and prefer bullet points for multi-part answers.\n"
)

# ---------------------------------------------------------------------------
# RAG helpers (moved from bamboo_answer.py)
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
    from bamboo.tools.doc_rag import panda_doc_search_tool  # avoid circular at module level

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
    from bamboo.tools.doc_bm25 import panda_doc_bm25_tool  # avoid circular at module level

    async with span(EVENT_RETRIEVAL, tool="panda_doc_bm25", backend="bm25") as _s:
        try:
            result = await panda_doc_bm25_tool.call({"query": question, "top_k": 10})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            result = exc  # type: ignore[assignment]
        ctx = _extract_rag_context(result)
        _s.set(hits=_rag_hit_count(result, ctx))
    return ctx


async def retrieve_rag_context(question: str) -> str:
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
        return ""


# ---------------------------------------------------------------------------
# Shared LLM call helpers (moved from bamboo_answer.py)
# ---------------------------------------------------------------------------


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


# Maximum characters kept per assistant history message before truncation.
# A 400-char excerpt preserves enough for the model to resolve follow-up
# references without bloating multi-turn synthesis prompts.
_HISTORY_ASSISTANT_MAX_CHARS: int = 400


def _truncate_history(history: list[Message]) -> list[Message]:
    """Return a copy of history with long assistant messages truncated.

    User messages are kept verbatim (they're short questions).  Assistant
    messages are capped at ``_HISTORY_ASSISTANT_MAX_CHARS`` characters so
    that a long prior answer does not dominate the synthesis prompt on the
    next turn.

    Args:
        history: Prior conversation turns with ``role`` and ``content`` keys.

    Returns:
        New list with assistant content truncated where necessary.
    """
    out: list[Message] = []
    for msg in history:
        if msg.get("role") == "assistant":
            content = str(msg.get("content", ""))
            if len(content) > _HISTORY_ASSISTANT_MAX_CHARS:
                content = content[:_HISTORY_ASSISTANT_MAX_CHARS] + "…(truncated)"
            out.append({"role": "assistant", "content": content})
        else:
            out.append(msg)
    return out


async def call_llm(
    system: str,
    user: str,
    history: list[Message] | None = None,
    max_tokens: int = 2048,
) -> str:
    """Call the default LLM with a system + user prompt and return the text.

    Prior conversation turns (``history``) are inserted between the system
    prompt and the synthesised user message so the model can resolve follow-up
    questions.  Long assistant messages in history are truncated via
    :func:`_truncate_history` to prevent synthesis prompts from growing
    unbounded across multi-turn conversations.

    Args:
        system: System prompt string.
        user: Synthesised user prompt for the current turn.
        history: Optional list of prior ``{role, content}`` turns to inject
            between the system prompt and the current user message.  Must
            contain only ``"user"`` and ``"assistant"`` roles.
        max_tokens: Maximum tokens for the LLM response (default 2048).

    Returns:
        LLM response text.
    """
    messages: list[Message] = [{"role": "system", "content": system}]
    if history:
        messages.extend(_truncate_history(history))
    messages.append({"role": "user", "content": user})

    delegated = await bamboo_llm_answer_tool.call({
        "messages": messages,
        "max_tokens": max_tokens,
    })
    return _extract_delegated_text(delegated)


def unpack_tool_result(result: list[MCPContent]) -> dict[str, Any]:
    """Deserialise a JSON-wrapped MCPContent result from an internal tool.

    Internal tools (job_status, log_analysis, task_status) return a
    one-element ``list[MCPContent]`` whose ``text`` field contains the
    JSON-serialised ``{evidence, text}`` dict.  This helper unpacks that
    layer so callers can access ``result.get("evidence", ...)`` as before.

    Falls back to an empty dict if the result cannot be parsed, so callers
    always receive a dict regardless of upstream errors.

    Args:
        result: Raw return value from an internal tool ``call()`` method.

    Returns:
        Deserialised dict, or ``{}`` on parse failure.
    """
    try:
        if result and isinstance(result[0], dict):
            text = result[0].get("text", "")
            if isinstance(text, str) and text.strip().startswith("{"):
                return json.loads(text)  # type: ignore[no-any-return]
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    return {}


# ---------------------------------------------------------------------------
# Synthesis prompt selection
# ---------------------------------------------------------------------------


def _pick_synthesis_prompt(tool_names: list[str]) -> str:
    """Select the most appropriate synthesis system prompt for a set of tools.

    The priority order mirrors the original hard-wired routing logic in
    ``bamboo_answer._route()``, ensuring that specialist prompts are
    preferred over generic ones when a dedicated prompt exists.

    Args:
        tool_names: Names of the tools that were actually called during
            plan execution (in call order).

    Returns:
        System prompt string for the LLM synthesis step.
    """
    if "panda_log_analysis" in tool_names:
        return _SYSTEM_LOG_ANALYSIS
    if "panda_job_status" in tool_names:
        return _SYSTEM_JOB
    if "panda_task_status" in tool_names:
        return _SYSTEM_TASK
    if any(t in tool_names for t in ("panda_doc_search", "panda_doc_bm25")):
        return _SYSTEM_RAG
    return _SYSTEM_GENERIC


# ---------------------------------------------------------------------------
# Plan execution
# ---------------------------------------------------------------------------


def _resolve_tool(tool_name: str, namespace: str | None, tools: dict[str, Any]) -> Any:
    """Resolve a tool object by name from the registry or entry points.

    Tries the static TOOLS registry first, then namespace-qualified entry-point
    lookup, then unqualified suffix-based lookup.

    Args:
        tool_name: Unqualified or qualified tool name to resolve.
        namespace: Optional namespace hint (e.g. ``"atlas"``).
        tools: The core TOOLS registry dict.

    Returns:
        Resolved tool object, or ``None`` if not found.
    """
    tool_obj: Any = tools.get(tool_name)
    if tool_obj is None and namespace:
        resolved = find_tool_by_name(tool_name, namespace=namespace)
        if resolved is not None:
            tool_obj = resolved.obj
    if tool_obj is None:
        resolved = find_tool_by_name(tool_name)
        if resolved is not None:
            tool_obj = resolved.obj
    return tool_obj


def _build_synthesis_prompt(
    called_tool_names: list[str],
    evidence_parts: list[str],
    question: str,
    errors: list[str],
    original_question: str | None = None,
) -> tuple[str, str]:
    """Build the system and user prompts for the synthesis LLM call.

    Selects a specialist prompt for known tool sets and falls back to a
    generic multi-tool prompt for unknown combinations.  RAG evidence is
    presented as documentation excerpts; other evidence as a merged block.

    When ``original_question`` differs from ``question`` (i.e. the user sent a
    content-free follow-up and ``question`` is the reformulated RAG query), the
    user prompt instructs the LLM to **expand** the prior answer rather than
    re-answer the original question from scratch.

    Args:
        called_tool_names: Names of tools that completed successfully.
        evidence_parts: One evidence string per successful tool call.
        question: Question used for retrieval (may be reformulated from history).
        errors: Error messages from any failed tool calls.
        original_question: The user's actual phrasing if it differs from
            ``question`` (e.g. "Tell me more please").  When provided, the
            synthesis prompt uses expansion framing instead of answer framing.

    Returns:
        Tuple of ``(system_prompt, user_prompt)`` strings.
    """
    rag_tools = {"panda_doc_search", "panda_doc_bm25"}
    plan_is_rag = any(t in rag_tools for t in called_tool_names)
    is_followup = (
        original_question is not None and original_question != question
    )

    if plan_is_rag:
        rag_context = "\n\n".join(evidence_parts)
        if rag_context:
            system = _SYSTEM_RAG
            if is_followup:
                user = (
                    f"The user asked a follow-up: {repr(original_question)}\n"
                    f"They want you to expand on the topic: {repr(question)}\n"
                    f"Using the retrieved documentation excerpts below, provide "
                    f"a more detailed explanation than before. "
                    f"Do not simply repeat what was said — go deeper, but be "
                    f"concise: aim for 200-300 words maximum.\n\n"
                    f"Retrieved documentation excerpts:\n{rag_context}\n"
                )
            else:
                user = (
                    f"User question:\n{question}\n\n"
                    f"Retrieved documentation excerpts:\n{rag_context}\n"
                )
        else:
            system = _SYSTEM_RAG_NO_CONTEXT
            user = f"User question:\n{question}\n"
    else:
        system = _pick_synthesis_prompt(called_tool_names)
        evidence_block = "\n\n".join(evidence_parts)
        user = (
            f"User question:\n{question}\n\n"
            f"Evidence from tool calls:\n{evidence_block}\n"
        )
        if errors:
            user += f"\nNote: the following tool calls failed: {'; '.join(errors)}\n"

    return system, user


async def execute_plan(
    plan: Plan,
    question: str,
    history: list[Message],
    include_raw: bool = False,
    original_question: str | None = None,
) -> list[MCPContent]:
    """Execute a validated Plan and return a synthesised answer.

    Iterates ``plan.tool_calls`` in order, calls each tool, unpacks evidence,
    merges all evidence into a single synthesised LLM call.

    Unknown tools, validation failures, and individual tool call exceptions are
    handled gracefully — partial evidence from successful calls is still used
    for synthesis.  Only when *all* calls fail is a top-level error returned.

    Args:
        plan: Validated :class:`~bamboo.tools.planner.Plan` from the planner.
        question: Question string used for retrieval (may be reformulated).
        history: Prior conversation turns to inject into the LLM prompt.
        include_raw: If ``True``, include raw tool-result previews in the
            synthesised answer when errors are detected.
        original_question: The user's actual phrasing when ``question`` has
            been reformulated (e.g. for content-free follow-ups).  Passed to
            :func:`_build_synthesis_prompt` to enable expansion framing.

    Returns:
        One-element ``list[MCPContent]`` with the synthesised text answer.
    """
    # Lazy import to avoid circular dependency at module load time.
    from bamboo.core import TOOLS  # pylint: disable=import-outside-toplevel
    from bamboo.core import _validate_arguments  # pylint: disable=import-outside-toplevel

    evidence_parts: list[str] = []
    called_tool_names: list[str] = []
    errors: list[str] = []

    async with span(EVENT_PLAN, tool="bamboo_executor", plan=plan.model_dump()):
        pass  # Emit the plan as a trace event so the TUI /plan command can find it.

    for tc in plan.tool_calls:
        tool_name: str = tc.tool
        args: dict[str, Any] = dict(tc.arguments)

        tool_obj = _resolve_tool(tool_name, tc.namespace, TOOLS)
        if tool_obj is None:
            errors.append(f"Unknown tool: {tool_name}")
            continue

        # Validate arguments against the tool schema.
        get_def_fn = getattr(tool_obj, "get_definition", None)
        if callable(get_def_fn):
            try:
                tool_def: dict[str, Any] = get_def_fn()  # type: ignore[assignment]
            except Exception:  # pylint: disable=broad-exception-caught
                tool_def = {}
            err = _validate_arguments(tool_def, args)
            if err:
                errors.append(f"Invalid args for {tool_name}: {err}")
                continue

        # Call the tool, catching any exceptions so one failure is non-fatal.
        try:
            raw_result: list[MCPContent] = await tool_obj.call(args)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            errors.append(f"Tool {tool_name} raised: {exc!s}")
            continue

        called_tool_names.append(tool_name)

        # Unpack JSON evidence; fall back to raw text if unpacking yields nothing.
        unpacked = unpack_tool_result(raw_result)
        if unpacked:
            evidence_parts.append(f"[{tool_name}]\n{_compact_json(unpacked)}")
        else:
            raw_text = raw_result[0].get("text", "") if raw_result else ""
            if raw_text:
                evidence_parts.append(f"[{tool_name}]\n{raw_text}")

    if not called_tool_names:
        error_summary = "; ".join(errors) if errors else "No tool calls in plan."
        return text_content(f"All tool calls failed: {error_summary}")

    system, user = _build_synthesis_prompt(
        called_tool_names, evidence_parts, question, errors,
        original_question=original_question,
    )

    async with span(EVENT_SYNTHESIS, tool="bamboo_executor",
                    tools=called_tool_names, route=plan.route.value):
        # Cap tokens at 600 for follow-up expansions to keep latency under
        # 10s at typical provider throughput; use full 2048 otherwise.
        synthesis_max_tokens = 600 if original_question is not None else 2048
        body = await call_llm(system, user, history, max_tokens=synthesis_max_tokens)

    return text_content(body)


def _compact_json(obj: Any, limit: int = 6000) -> str:
    """Compact JSON for prompts, bounded to ``limit`` characters.

    Args:
        obj: Any JSON-serialisable object.
        limit: Maximum character count before truncation.

    Returns:
        Compact JSON string, truncated with an ellipsis if over ``limit``.
    """
    try:
        s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:  # pylint: disable=broad-exception-caught
        s = str(obj)
    if len(s) > limit:
        return s[:limit] + "…(truncated)"
    return s


__all__ = [
    "execute_plan",
    "call_llm",
    "unpack_tool_result",
    "retrieve_rag_context",
    "_pick_synthesis_prompt",
    "_SYSTEM_LOG_ANALYSIS",
    "_SYSTEM_JOB",
    "_SYSTEM_TASK",
    "_SYSTEM_RAG",
    "_SYSTEM_RAG_NO_CONTEXT",
    "_SYSTEM_GENERIC",
]
