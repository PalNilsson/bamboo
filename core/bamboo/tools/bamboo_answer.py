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

import json
import re
from typing import Any

from bamboo.llm.types import Message
from bamboo.tools.base import MCPContent, coerce_messages, text_content
from bamboo.tools.doc_rag import panda_doc_search_tool
from bamboo.tools.doc_bm25 import panda_doc_bm25_tool
from bamboo.tools.topic_guard import check_topic
from bamboo.tools.llm_passthrough import bamboo_llm_answer_tool
from bamboo.tools.task_status import panda_task_status_tool
from bamboo.tools.job_status import panda_job_status_tool  # type: ignore[import-untyped]
from bamboo.tools.log_analysis import panda_log_analysis_tool  # type: ignore[import-untyped]

# Matches "task 123", "task:123", "task-123" etc. (4-12 digits)
_TASK_PATTERN = re.compile(r"(?i)\btask[:#/\-\s]+([0-9]{4,12})\b")
# Matches "job 123", "job:123", "pandaid 123", "panda id 123" etc.
_JOB_PATTERN = re.compile(r"(?i)\b(?:job|pandaid|panda[\s_-]?id)[:#/\-\s]+([0-9]{4,12})\b")
# Matches "analyse/analyze/why did ... job 123 fail"
_LOG_PATTERN = re.compile(
    r"(?i)(?:analys[ei]|why|fail|log|diagnos)[^.]{0,60}"
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

    async def call(self, arguments: dict[str, Any]) -> list[MCPContent]:  # noqa: C901
        """Handle bamboo_answer tool invocation with optional task context.

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

        # include_jobs and include_raw are independent flags.
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

        if bypass_routing:
            delegated = await bamboo_llm_answer_tool.call(
                {"messages": messages} if messages else {"question": question}
            )
            body = str(delegated[0].get("text", "")) if delegated and isinstance(delegated[0], dict) else str(delegated)
            return text_content(body)

        # --- Topic guard — must run before any tool or LLM call ---
        guard = await check_topic(question)
        if not guard.allowed:
            return text_content(guard.rejection_message)

        task_id = _extract_task_id(question)
        job_id = _extract_job_id(question)
        want_log_analysis = _is_log_analysis_request(question)

        # --- Job log analysis ---
        if job_id and want_log_analysis:
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
            prompt_user = f"User question:\n{question}\n\nEvidence JSON:\n{_compact(evidence)}\n"
            delegated = await bamboo_llm_answer_tool.call(
                {"messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt_user},
                ], "max_tokens": 2048}
            )
            body = str(delegated[0].get("text", "")) if delegated and isinstance(delegated[0], dict) else str(delegated)
            return text_content(body)

        # --- Job status ---
        if job_id and not task_id:
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
                "- Always include the BigPanDA monitor URL as plain text (not a Markdown hyperlink), e.g.: Monitor: https://bigpanda.cern.ch/job/12345/\n"
                "- Keep it under ~8 bullet points.\n"
            )
            prompt_user = f"User question:\n{question}\n\nEvidence JSON:\n{_compact(evidence)}\n"
            if raw_preview:
                system += "\nOn error, include the Raw response preview verbatim at the end inside a fenced code block.\n"
                prompt_user += f"\nRaw response preview:\n{raw_preview}\n"
            delegated = await bamboo_llm_answer_tool.call(
                {"messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt_user},
                ], "max_tokens": 2048}
            )
            body = str(delegated[0].get("text", "")) if delegated and isinstance(delegated[0], dict) else str(delegated)
            if raw_preview and "Raw response preview" not in body:
                body = f"{body.rstrip()}\n\nRaw response preview:\n```text\n{raw_preview}\n```\n"
            return text_content(body)

        # --- No recognised ID — RAG first, then LLM synthesis ---
        # This is the default path for general knowledge questions such as
        # "What is PanDA?" or "How does brokerage work?".  We always attempt
        # doc retrieval first; the LLM then synthesises an answer grounded in
        # the retrieved context.  If RAG is unavailable or returns nothing the
        # LLM is still called but is explicitly told no context was found, so
        # it can answer from general knowledge while being transparent about it.
        if not task_id:
            rag_context: str = ""
            try:
                import asyncio as _asyncio
                # Run vector search and BM25 search concurrently.
                _vec_result, _bm25_result = await _asyncio.gather(
                    panda_doc_search_tool.call({"query": question, "top_k": 20}),
                    panda_doc_bm25_tool.call({"query": question, "top_k": 10}),
                    return_exceptions=True,
                )

                _no_context_signals = (
                    "not installed",
                    "chromadb path not found",
                    "failed to connect",
                    "no results found",
                    "no keyword matches",
                    "required and must not be empty",
                )

                def _extract_context(result: object) -> str:
                    """Return text from a tool result if it contains useful context."""
                    if isinstance(result, Exception) or not result:
                        return ""
                    if not isinstance(result, list) or not isinstance(result[0], dict):
                        return ""
                    text = str(result[0].get("text", ""))
                    first_line = text.split("\n")[0].lower()
                    if any(s in first_line for s in _no_context_signals):
                        return ""
                    return text

                _vec_context = _extract_context(_vec_result)
                _bm25_context = _extract_context(_bm25_result)

                # Merge: vector results first (conceptual), BM25 second
                # (keyword/exact-match) so the LLM draws from both.
                if _vec_context and _bm25_context:
                    rag_context = (
                        f"{_vec_context}\n\n"
                        f"--- Keyword search results ---\n{_bm25_context}"
                    )
                elif _vec_context:
                    rag_context = _vec_context
                elif _bm25_context:
                    rag_context = _bm25_context
            except Exception:  # pylint: disable=broad-exception-caught
                pass  # Search failure is non-fatal; we continue to LLM

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
                prompt_user = (
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
                prompt_user = f"User question:\n{question}\n"

            delegated = await bamboo_llm_answer_tool.call(
                {"messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt_user},
                ], "max_tokens": 2048}
            )
            body = (
                str(delegated[0].get("text", ""))
                if delegated and isinstance(delegated[0], dict)
                else str(delegated)
            )
            return text_content(body)

        # --- Task status ---
        tool_args: dict[str, Any] = {
            "task_id": task_id,
            "query": question,
            "include_jobs": include_jobs,
        }
        tool_result = await panda_task_status_tool.call(tool_args)
        evidence = tool_result.get("evidence", tool_result)

        is_error = _is_bigpanda_error(evidence)
        # Always extract the raw preview on errors so the LLM sees what BigPanDA
        # actually returned — useful both for not-found detection and debugging.
        # The include_raw flag additionally appends it verbatim to the user-facing
        # response as a fenced code block.
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
        # Build the prompt context from the full payload, but handle the jobs list
        # carefully: it can be large and gets truncated, causing hallucinations.
        # Instead, extract job pandaids explicitly and present them as a clean list,
        # then include the rest of the payload without the raw jobs array.
        payload = evidence.get("payload") if isinstance(evidence, dict) else None
        if isinstance(payload, dict):
            _job_keys = ("jobs", "jobList", "joblist", "job_list", "jobSummary")
            jobs_list = next(
                (payload.get(k) for k in _job_keys
                 if isinstance(payload.get(k), list)),
                None,
            )
            # Build a compact payload without the raw jobs array to avoid truncation.
            payload_slim = {k: v for k, v in payload.items() if k not in _job_keys}
            evidence_for_prompt: Any = payload_slim
            # Add a clean job summary (or explicit absence note) so the LLM knows
            # whether job IDs are available — preventing hallucination.
            if not jobs_list:
                job_context = (
                    "\n\nJob list: No job records were returned for this task."
                    " Do not infer PanDA job IDs from any other field."
                )
            elif jobs_list:
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
                    f"{"\n".join(job_rows[:200])}{tail}"
                )
        else:
            evidence_for_prompt = evidence
            job_context = ""
        prompt_user = (
            f"User question:\n{question}\n"
            f"\nTask metadata (JSON):\n{_compact(evidence_for_prompt)}"
            f"{job_context}\n"
        )
        if raw_preview:
            prompt_user += f"\nBigPanDA raw response snippet:\n{raw_preview}\n"
        if raw_preview and include_raw:
            system += "\nInclude the raw BigPanDA response snippet verbatim at the end inside a fenced code block.\n"

        delegated = await bamboo_llm_answer_tool.call(
            {"messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt_user}],
             "max_tokens": 2048}
        )
        body = str(delegated[0].get("text", "")) if delegated and isinstance(delegated[0], dict) else str(delegated)

        # Always append the raw snippet when include_raw is set — do not rely on
        # the LLM to include it, as it often ignores that instruction.
        if raw_preview and include_raw:
            body = (
                f"{body.rstrip()}\n\n"
                "**BigPanDA raw response snippet:**\n"
                "```text\n"
                f"{raw_preview}\n"
                "```\n"
            )

        return text_content(body)


bamboo_answer_tool = BambooAnswerTool()

__all__ = ["BambooAnswerTool", "bamboo_answer_tool"]
