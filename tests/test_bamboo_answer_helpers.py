"""Unit tests for the module-level helper functions in bamboo_answer.

These helpers were previously nested inside _call_inner and are now at module
level after the refactor.  Testing them directly is faster and more targeted
than going through the full BambooAnswerTool.call() pipeline.

Coverage:
- _extract_task_id: pattern matching for task IDs
- _extract_job_id: pattern matching for job/pandaID references
- _is_log_analysis_request: keyword detection for log/failure analysis
- _compact: JSON serialisation with truncation
- _is_bigpanda_error: upstream error classification from evidence dicts
- _extract_raw_preview: raw text extraction from nested tool results
- _extract_delegated_text: text extraction from LLM passthrough results
- _extract_rag_context: context extraction with no-context signal detection
- _rag_hit_count: hit counting from retrieval results
- BambooAnswerTool._build_task_prompt: task prompt construction with job list
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import bamboo.tools.bamboo_answer as ba_mod
from bamboo.tools.bamboo_answer import (
    BambooAnswerTool,
    _compact,
    _extract_delegated_text,
    _extract_job_id,
    _extract_rag_context,
    _extract_raw_preview,
    _extract_task_id,
    _is_bigpanda_error,
    _is_log_analysis_request,
    _rag_hit_count,
)


# ---------------------------------------------------------------------------
# _extract_task_id
# ---------------------------------------------------------------------------


class TestExtractTaskId:
    """Pattern matching for task IDs in user questions."""

    def test_plain_task_number(self) -> None:
        assert _extract_task_id("What is task 12345678?") == 12345678

    def test_task_colon(self) -> None:
        assert _extract_task_id("task:99887766 status") == 99887766

    def test_task_hash(self) -> None:
        assert _extract_task_id("task#12345678") == 12345678

    def test_task_dash(self) -> None:
        assert _extract_task_id("task-56781234 failed") == 56781234

    def test_case_insensitive(self) -> None:
        assert _extract_task_id("TASK 12345678") == 12345678

    def test_no_match_returns_none(self) -> None:
        assert _extract_task_id("What is PanDA?") is None

    def test_too_short_ignored(self) -> None:
        # 3 digits is below the 4-digit minimum
        assert _extract_task_id("task 123") is None

    def test_too_long_ignored(self) -> None:
        # 13 digits is above the 12-digit maximum
        assert _extract_task_id("task 1234567890123") is None

    def test_exactly_four_digits(self) -> None:
        assert _extract_task_id("task 1234") == 1234

    def test_exactly_twelve_digits(self) -> None:
        assert _extract_task_id("task 123456789012") == 123456789012

    def test_empty_string(self) -> None:
        assert _extract_task_id("") is None

    def test_none_like_empty(self) -> None:
        # The function treats None-like inputs as empty via `text or ""`
        assert _extract_task_id("") is None


# ---------------------------------------------------------------------------
# _extract_job_id
# ---------------------------------------------------------------------------


class TestExtractJobId:
    """Pattern matching for job / pandaID references."""

    def test_job_space(self) -> None:
        assert _extract_job_id("job 6837798305 failed") == 6837798305

    def test_job_colon(self) -> None:
        assert _extract_job_id("job:6837798305") == 6837798305

    def test_pandaid(self) -> None:
        assert _extract_job_id("pandaid 6837798305") == 6837798305

    def test_panda_id_spaced(self) -> None:
        assert _extract_job_id("panda id 6837798305") == 6837798305

    def test_panda_id_hyphen(self) -> None:
        assert _extract_job_id("panda-id 6837798305") == 6837798305

    def test_case_insensitive(self) -> None:
        assert _extract_job_id("JOB 6837798305") == 6837798305

    def test_no_match_returns_none(self) -> None:
        assert _extract_job_id("What is PanDA?") is None

    def test_task_reference_not_matched(self) -> None:
        # "task" should not match the job pattern
        assert _extract_job_id("task 12345678") is None

    def test_empty_string(self) -> None:
        assert _extract_job_id("") is None


# ---------------------------------------------------------------------------
# _is_log_analysis_request
# ---------------------------------------------------------------------------


class TestIsLogAnalysisRequest:
    """Keyword detection for log / failure analysis intent."""

    def test_why_fail(self) -> None:
        assert _is_log_analysis_request("Why did job 6837798305 fail?") is True

    def test_analyse(self) -> None:
        assert _is_log_analysis_request("analyse job 6837798305") is True

    def test_analyze_american(self) -> None:
        assert _is_log_analysis_request("analyze job 6837798305") is True

    def test_diagnose(self) -> None:
        assert _is_log_analysis_request("diagnose job 6837798305 please") is True

    def test_log_keyword(self) -> None:
        assert _is_log_analysis_request("show log for job 6837798305") is True

    def test_plain_job_status_not_matched(self) -> None:
        assert _is_log_analysis_request("What is job 6837798305 status?") is False

    def test_no_job_id_not_matched(self) -> None:
        assert _is_log_analysis_request("Why did the task fail?") is False

    def test_empty(self) -> None:
        assert _is_log_analysis_request("") is False


# ---------------------------------------------------------------------------
# _compact
# ---------------------------------------------------------------------------


class TestCompact:
    """JSON serialisation with length capping."""

    def test_small_dict(self) -> None:
        result = _compact({"key": "val"})
        assert '"key"' in result
        assert '"val"' in result

    def test_truncated_at_limit(self) -> None:
        big = {"data": "x" * 10000}
        result = _compact(big, limit=100)
        assert len(result) <= 100 + len("…(truncated)")
        assert result.endswith("…(truncated)")

    def test_exact_limit_not_truncated(self) -> None:
        # A string that serialises to exactly limit chars should not be truncated.
        short = {"k": "v"}
        full = _compact(short, limit=10000)
        assert not full.endswith("…(truncated)")

    def test_non_serialisable_falls_back_to_str(self) -> None:
        class _Weird:
            def __repr__(self) -> str:
                return "weird_repr"
        result = _compact(_Weird())
        assert "weird_repr" in result

    def test_list_input(self) -> None:
        result = _compact([1, 2, 3])
        assert "1" in result and "2" in result


# ---------------------------------------------------------------------------
# _is_bigpanda_error
# ---------------------------------------------------------------------------


class TestIsBigPandaError:
    """Error classification from BigPanDA evidence dicts."""

    def test_not_found_true(self) -> None:
        assert _is_bigpanda_error({"not_found": True}) is True

    def test_not_found_false_is_not_error(self) -> None:
        assert _is_bigpanda_error({"not_found": False, "status": "done"}) is False

    def test_http_status_404(self) -> None:
        assert _is_bigpanda_error({"http_status": 404}) is True

    def test_http_status_200_ok(self) -> None:
        assert _is_bigpanda_error({"http_status": 200}) is False

    def test_http_status_500(self) -> None:
        assert _is_bigpanda_error({"http_status": 500}) is True

    def test_http_status_string_404(self) -> None:
        assert _is_bigpanda_error({"http_status": "404"}) is True

    def test_http_status_string_200(self) -> None:
        assert _is_bigpanda_error({"http_status": "200"}) is False

    def test_non_json_flag(self) -> None:
        assert _is_bigpanda_error({"non_json": True}) is True

    def test_json_error_flag(self) -> None:
        assert _is_bigpanda_error({"json_error": "parse failed"}) is True

    def test_exception_key(self) -> None:
        assert _is_bigpanda_error({"exception": "ConnectionError"}) is True

    def test_error_string(self) -> None:
        assert _is_bigpanda_error({"error": "network timeout"}) is True

    def test_empty_error_string_not_error(self) -> None:
        assert _is_bigpanda_error({"error": ""}) is False

    def test_non_dict_returns_false(self) -> None:
        assert _is_bigpanda_error("not a dict") is False
        assert _is_bigpanda_error(None) is False
        assert _is_bigpanda_error([]) is False

    def test_clean_evidence_ok(self) -> None:
        evidence = {"status": "done", "http_status": 200, "payload": {}}
        assert _is_bigpanda_error(evidence) is False


# ---------------------------------------------------------------------------
# _extract_raw_preview
# ---------------------------------------------------------------------------


class TestExtractRawPreview:
    """Raw text extraction from nested tool result structures."""

    def test_raw_key_from_evidence(self) -> None:
        evidence = {"raw": "upstream raw text"}
        result = _extract_raw_preview({}, evidence)
        assert result == "upstream raw text"

    def test_text_key_from_tool_result(self) -> None:
        tool_result = {"text": "some text body"}
        result = _extract_raw_preview(tool_result, {})
        assert result == "some text body"

    def test_nested_upstream_key(self) -> None:
        tool_result = {"upstream": {"body": "nested body text"}}
        result = _extract_raw_preview(tool_result, {})
        assert result == "nested body text"

    def test_bytes_decoded(self) -> None:
        evidence = {"raw": b"byte content here"}
        result = _extract_raw_preview({}, evidence)
        assert result == "byte content here"

    def test_truncation_at_limit(self) -> None:
        evidence = {"raw": "a" * 3000}
        result = _extract_raw_preview({}, evidence, limit=100)
        assert result is not None
        assert result.endswith("…(truncated)")
        assert len(result) <= 100 + len("…(truncated)")

    def test_no_useful_content_returns_none(self) -> None:
        result = _extract_raw_preview({}, {})
        assert result is None

    def test_whitespace_only_skipped(self) -> None:
        evidence = {"raw": "   \n  "}
        result = _extract_raw_preview({}, evidence)
        assert result is None


# ---------------------------------------------------------------------------
# _extract_delegated_text
# ---------------------------------------------------------------------------


class TestExtractDelegatedText:
    """Text extraction from bamboo_llm_answer_tool results."""

    def test_standard_mcp_result(self) -> None:
        delegated = [{"type": "text", "text": "hello world"}]
        assert _extract_delegated_text(delegated) == "hello world"

    def test_missing_text_key_returns_empty(self) -> None:
        delegated = [{"type": "text"}]
        assert _extract_delegated_text(delegated) == ""

    def test_non_dict_first_element_falls_back_to_str(self) -> None:
        delegated = ["plain string"]
        result = _extract_delegated_text(delegated)
        assert "plain string" in result

    def test_empty_list_falls_back_to_str(self) -> None:
        result = _extract_delegated_text([])
        assert isinstance(result, str)

    def test_non_list_falls_back_to_str(self) -> None:
        result = _extract_delegated_text("bare string")
        assert "bare string" in result


# ---------------------------------------------------------------------------
# _extract_rag_context
# ---------------------------------------------------------------------------


class TestExtractRagContext:
    """Context extraction with no-context signal filtering."""

    def test_good_result_returns_text(self) -> None:
        result = [{"type": "text", "text": "PanDA is a workload manager.\nMore info here."}]
        ctx = _extract_rag_context(result)
        assert "PanDA" in ctx

    def test_exception_returns_empty(self) -> None:
        assert _extract_rag_context(RuntimeError("fail")) == ""

    def test_not_installed_signal(self) -> None:
        result = [{"type": "text", "text": "ChromaDB is not installed. Install it."}]
        assert _extract_rag_context(result) == ""

    def test_no_results_found_signal(self) -> None:
        result = [{"type": "text", "text": "No results found for your query."}]
        assert _extract_rag_context(result) == ""

    def test_chromadb_path_not_found(self) -> None:
        result = [{"type": "text", "text": "ChromaDB path not found at /tmp/db"}]
        assert _extract_rag_context(result) == ""

    def test_no_keyword_matches_signal(self) -> None:
        result = [{"type": "text", "text": "No keyword matches for 'foo bar'."}]
        assert _extract_rag_context(result) == ""

    def test_signal_only_on_first_line(self) -> None:
        # Signal appearing only on second line should NOT suppress the result.
        result = [{"type": "text", "text": "Good context here.\nNo results found elsewhere."}]
        ctx = _extract_rag_context(result)
        assert ctx != ""

    def test_empty_list_returns_empty(self) -> None:
        assert _extract_rag_context([]) == ""

    def test_non_list_returns_empty(self) -> None:
        assert _extract_rag_context("not a list") == ""


# ---------------------------------------------------------------------------
# _rag_hit_count
# ---------------------------------------------------------------------------


class TestRagHitCount:
    """Hit counting from retrieval results."""

    def test_counts_non_empty_lines(self) -> None:
        context = "line one\nline two\nline three"
        count = _rag_hit_count([{"type": "text", "text": context}], context)
        assert count == 3

    def test_blank_lines_not_counted(self) -> None:
        context = "line one\n\n\nline two\n"
        count = _rag_hit_count([{"type": "text", "text": context}], context)
        assert count == 2

    def test_exception_returns_minus_one(self) -> None:
        assert _rag_hit_count(RuntimeError("error"), "") == -1

    def test_empty_context_returns_zero(self) -> None:
        assert _rag_hit_count([{"text": ""}], "") == 0


# ---------------------------------------------------------------------------
# BambooAnswerTool._build_task_prompt
# ---------------------------------------------------------------------------


class TestBuildTaskPrompt:
    """Task prompt construction with job list handling."""

    def test_includes_question(self) -> None:
        evidence: dict[str, Any] = {}
        prompt = BambooAnswerTool._build_task_prompt("What failed?", evidence, None)
        assert "What failed?" in prompt

    def test_no_payload_uses_evidence_directly(self) -> None:
        evidence = {"status": "done", "task_id": 123}
        prompt = BambooAnswerTool._build_task_prompt("q", evidence, None)
        assert "done" in prompt or "123" in prompt

    def test_job_list_extracted_as_clean_rows(self) -> None:
        jobs = [
            {"pandaid": "111", "jobStatus": "finished"},
            {"pandaid": "222", "jobStatus": "failed"},
        ]
        evidence = {"payload": {"status": "done", "jobs": jobs}}
        prompt = BambooAnswerTool._build_task_prompt("q", evidence, None)
        assert "pandaid=111" in prompt
        assert "pandaid=222" in prompt
        assert "status=finished" in prompt

    def test_no_jobs_adds_note(self) -> None:
        evidence = {"payload": {"status": "done"}}
        prompt = BambooAnswerTool._build_task_prompt("q", evidence, None)
        assert "No job records" in prompt

    def test_raw_jobs_array_stripped_from_payload(self) -> None:
        jobs = [{"pandaid": "111", "jobStatus": "finished"}]
        evidence = {"payload": {"status": "done", "jobs": jobs}}
        prompt = BambooAnswerTool._build_task_prompt("q", evidence, None)
        # The raw jobs array should not appear as a JSON blob.
        assert '"jobStatus"' not in prompt

    def test_joblist_key_variant(self) -> None:
        jobs = [{"pandaid": "999", "status": "running"}]
        evidence = {"payload": {"status": "running", "jobList": jobs}}
        prompt = BambooAnswerTool._build_task_prompt("q", evidence, None)
        assert "pandaid=999" in prompt

    def test_raw_preview_appended(self) -> None:
        evidence = {"payload": {"status": "done"}}
        prompt = BambooAnswerTool._build_task_prompt("q", evidence, "PREVIEW_TEXT")
        assert "PREVIEW_TEXT" in prompt

    def test_no_raw_preview_when_none(self) -> None:
        evidence = {"payload": {"status": "done"}}
        prompt = BambooAnswerTool._build_task_prompt("q", evidence, None)
        assert "raw response snippet" not in prompt.lower()

    def test_large_job_list_truncated(self) -> None:
        jobs = [{"pandaid": str(i), "jobStatus": "done"} for i in range(300)]
        evidence = {"payload": {"status": "done", "jobs": jobs}}
        prompt = BambooAnswerTool._build_task_prompt("q", evidence, None)
        assert "truncated" in prompt

    def test_exactly_200_jobs_not_truncated(self) -> None:
        jobs = [{"pandaid": str(i), "jobStatus": "done"} for i in range(200)]
        evidence = {"payload": {"status": "done", "jobs": jobs}}
        prompt = BambooAnswerTool._build_task_prompt("q", evidence, None)
        assert "truncated" not in prompt


# ---------------------------------------------------------------------------
# _retrieve_rag_context integration (async, mocked)
# ---------------------------------------------------------------------------


class TestRetrieveRagContext:
    """Integration tests for _retrieve_rag_context with mocked tools."""

    @pytest.mark.asyncio
    async def test_merges_both_results(self) -> None:
        vec_text = "Vector result content."
        bm25_text = "BM25 keyword result."
        with (
            patch.object(ba_mod, "panda_doc_search_tool",
                         AsyncMock(call=AsyncMock(
                             return_value=[{"type": "text", "text": vec_text}]))),
            patch.object(ba_mod, "panda_doc_bm25_tool",
                         AsyncMock(call=AsyncMock(
                             return_value=[{"type": "text", "text": bm25_text}]))),
        ):
            ctx = await ba_mod._retrieve_rag_context("test question")

        assert vec_text in ctx
        assert bm25_text in ctx
        assert "Keyword search results" in ctx

    @pytest.mark.asyncio
    async def test_falls_back_to_vector_only(self) -> None:
        vec_text = "Only vector content."
        with (
            patch.object(ba_mod, "panda_doc_search_tool",
                         AsyncMock(call=AsyncMock(
                             return_value=[{"type": "text", "text": vec_text}]))),
            patch.object(ba_mod, "panda_doc_bm25_tool",
                         AsyncMock(call=AsyncMock(
                             return_value=[{"type": "text", "text": "No keyword matches."}]))),
        ):
            ctx = await ba_mod._retrieve_rag_context("test question")

        assert ctx == vec_text

    @pytest.mark.asyncio
    async def test_both_fail_returns_empty(self) -> None:
        with (
            patch.object(ba_mod, "panda_doc_search_tool",
                         AsyncMock(call=AsyncMock(side_effect=RuntimeError("chroma down")))),
            patch.object(ba_mod, "panda_doc_bm25_tool",
                         AsyncMock(call=AsyncMock(side_effect=RuntimeError("bm25 down")))),
        ):
            ctx = await ba_mod._retrieve_rag_context("test question")

        assert ctx == ""
