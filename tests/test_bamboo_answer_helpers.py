"""Unit tests for module-level helper functions after the executor refactor.

After the refactor the helper functions were split across two modules:

- ``bamboo_answer`` retains: ``_extract_task_id``, ``_extract_job_id``,
  ``_is_log_analysis_request``, ``_extract_history``.
- ``bamboo_executor`` owns: ``_compact_json`` (was ``_compact``),
  ``unpack_tool_result`` (was ``_unpack_tool_result``),
  ``_extract_delegated_text``, ``_extract_rag_context``, ``_rag_hit_count``,
  ``retrieve_rag_context`` (was ``_retrieve_rag_context``).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import bamboo.tools.bamboo_executor as ex_mod
from bamboo.tools.bamboo_answer import (
    _extract_job_id,
    _extract_task_id,
    _is_log_analysis_request,
)
from bamboo.tools.bamboo_executor import (
    _compact_json,
    _extract_delegated_text,
    _extract_rag_context,
    _rag_hit_count,
    retrieve_rag_context,
    unpack_tool_result,
)


class TestExtractTaskId:
    """Pattern matching for task IDs in user questions."""

    def test_plain_task_number(self) -> None:
        """Standard 'task <id>' form is matched."""
        assert _extract_task_id("What is task 12345678?") == 12345678

    def test_task_colon(self) -> None:
        """'task:<id>' form is matched."""
        assert _extract_task_id("task:99887766 status") == 99887766

    def test_task_hash(self) -> None:
        """'task#<id>' form is matched."""
        assert _extract_task_id("task#12345678") == 12345678

    def test_task_dash(self) -> None:
        """'task-<id>' form is matched."""
        assert _extract_task_id("task-56781234 failed") == 56781234

    def test_case_insensitive(self) -> None:
        """Pattern matching is case-insensitive."""
        assert _extract_task_id("TASK 12345678") == 12345678

    def test_no_match_returns_none(self) -> None:
        """Non-task questions return None."""
        assert _extract_task_id("What is PanDA?") is None

    def test_too_short_ignored(self) -> None:
        """IDs below the 4-digit minimum are not matched."""
        assert _extract_task_id("task 123") is None

    def test_too_long_ignored(self) -> None:
        """IDs above the 12-digit maximum are not matched."""
        assert _extract_task_id("task 1234567890123") is None

    def test_exactly_four_digits(self) -> None:
        """Exactly 4-digit IDs are at the lower boundary."""
        assert _extract_task_id("task 1234") == 1234

    def test_exactly_twelve_digits(self) -> None:
        """Exactly 12-digit IDs are at the upper boundary."""
        assert _extract_task_id("task 123456789012") == 123456789012

    def test_empty_string(self) -> None:
        """Empty input returns None."""
        assert _extract_task_id("") is None


class TestExtractJobId:
    """Pattern matching for job / pandaID references."""

    def test_job_space(self) -> None:
        """'job <id>' form is matched."""
        assert _extract_job_id("job 6837798305 failed") == 6837798305

    def test_job_colon(self) -> None:
        """'job:<id>' form is matched."""
        assert _extract_job_id("job:6837798305") == 6837798305

    def test_pandaid(self) -> None:
        """'pandaid <id>' form is matched."""
        assert _extract_job_id("pandaid 6837798305") == 6837798305

    def test_panda_id_spaced(self) -> None:
        """'panda id <id>' form is matched."""
        assert _extract_job_id("panda id 6837798305") == 6837798305

    def test_panda_id_hyphen(self) -> None:
        """'panda-id <id>' form is matched."""
        assert _extract_job_id("panda-id 6837798305") == 6837798305

    def test_case_insensitive(self) -> None:
        """Pattern matching is case-insensitive."""
        assert _extract_job_id("JOB 6837798305") == 6837798305

    def test_no_match_returns_none(self) -> None:
        """Non-job questions return None."""
        assert _extract_job_id("What is PanDA?") is None

    def test_task_reference_not_matched(self) -> None:
        """'task' keyword does not match the job pattern."""
        assert _extract_job_id("task 12345678") is None

    def test_empty_string(self) -> None:
        """Empty input returns None."""
        assert _extract_job_id("") is None


class TestIsLogAnalysisRequest:
    """Keyword detection for log / failure analysis intent."""

    def test_why_fail(self) -> None:
        """'why ... fail' phrasing is detected."""
        assert _is_log_analysis_request("Why did job 6837798305 fail?") is True

    def test_analyse(self) -> None:
        """'analyse' triggers log analysis detection."""
        assert _is_log_analysis_request("analyse job 6837798305") is True

    def test_analyze_american(self) -> None:
        """'analyze' (American spelling) also triggers detection."""
        assert _is_log_analysis_request("analyze job 6837798305") is True

    def test_diagnose(self) -> None:
        """'diagnose' triggers log analysis detection."""
        assert _is_log_analysis_request("diagnose job 6837798305 please") is True

    def test_log_keyword(self) -> None:
        """'log' keyword triggers detection."""
        assert _is_log_analysis_request("show log for job 6837798305") is True

    def test_plain_job_status_not_matched(self) -> None:
        """A plain status question without analysis keywords is not matched."""
        assert _is_log_analysis_request("What is job 6837798305 status?") is False

    def test_no_job_id_not_matched(self) -> None:
        """Failure keywords without a job ID are not matched."""
        assert _is_log_analysis_request("Why did the task fail?") is False

    def test_empty(self) -> None:
        """Empty input returns False."""
        assert _is_log_analysis_request("") is False


class TestCompactJson:
    """JSON serialisation with length capping (executor helper)."""

    def test_small_dict(self) -> None:
        """Small dicts are serialised normally."""
        result = _compact_json({"key": "val"})
        assert '"key"' in result and '"val"' in result

    def test_truncated_at_limit(self) -> None:
        """Output is capped at limit + truncation suffix."""
        big = {"data": "x" * 10000}
        result = _compact_json(big, limit=100)
        assert len(result) <= 100 + len("…(truncated)")
        assert result.endswith("…(truncated)")

    def test_exact_limit_not_truncated(self) -> None:
        """Output at or below limit is never truncated."""
        assert not _compact_json({"k": "v"}, limit=10000).endswith("…(truncated)")

    def test_non_serialisable_falls_back_to_str(self) -> None:
        """Non-JSON-serialisable objects fall back to repr/str."""
        class _Weird:
            def __repr__(self) -> str:
                return "weird_repr"
        assert "weird_repr" in _compact_json(_Weird())

    def test_list_input(self) -> None:
        """Lists are serialised correctly."""
        result = _compact_json([1, 2, 3])
        assert "1" in result and "2" in result


class TestUnpackToolResult:
    """JSON unwrapping from MCPContent list results."""

    def test_valid_json_text_is_parsed(self) -> None:
        """A valid JSON text block is deserialised."""
        result = unpack_tool_result([{"type": "text", "text": '{"evidence": {"status": "done"}}'}])
        assert result == {"evidence": {"status": "done"}}

    def test_non_json_text_returns_empty(self) -> None:
        """Non-JSON text returns an empty dict."""
        assert unpack_tool_result([{"type": "text", "text": "plain text"}]) == {}

    def test_empty_list_returns_empty(self) -> None:
        """Empty result list returns an empty dict."""
        assert unpack_tool_result([]) == {}

    def test_missing_text_key_returns_empty(self) -> None:
        """Missing text key in content block returns an empty dict."""
        assert unpack_tool_result([{"type": "text"}]) == {}


class TestExtractDelegatedText:
    """Text extraction from bamboo_llm_answer_tool results."""

    def test_standard_mcp_result(self) -> None:
        """Standard MCPContent dict is unwrapped."""
        assert _extract_delegated_text([{"type": "text", "text": "hello world"}]) == "hello world"

    def test_missing_text_key_returns_empty(self) -> None:
        """Missing text key returns empty string."""
        assert _extract_delegated_text([{"type": "text"}]) == ""

    def test_non_dict_first_element_falls_back_to_str(self) -> None:
        """Non-dict first element is stringified."""
        assert "plain string" in _extract_delegated_text(["plain string"])

    def test_empty_list_falls_back_to_str(self) -> None:
        """Empty list falls back to str representation."""
        assert isinstance(_extract_delegated_text([]), str)

    def test_non_list_falls_back_to_str(self) -> None:
        """Non-list input is stringified."""
        assert "bare string" in _extract_delegated_text("bare string")  # type: ignore[arg-type]


class TestExtractRagContext:
    """Context extraction with no-context signal filtering."""

    def test_good_result_returns_text(self) -> None:
        """Useful retrieval text is returned unchanged."""
        result = [{"type": "text", "text": "PanDA is a workload manager.\nMore info here."}]
        assert "PanDA" in _extract_rag_context(result)

    def test_exception_returns_empty(self) -> None:
        """An exception object returns empty string."""
        assert _extract_rag_context(RuntimeError("fail")) == ""

    def test_not_installed_signal(self) -> None:
        """'not installed' on the first line suppresses the result."""
        assert _extract_rag_context([{"type": "text", "text": "ChromaDB is not installed."}]) == ""

    def test_no_results_found_signal(self) -> None:
        """'no results found' on the first line suppresses the result."""
        assert _extract_rag_context([{"type": "text", "text": "No results found for your query."}]) == ""

    def test_chromadb_path_not_found(self) -> None:
        """'chromadb path not found' on the first line suppresses the result."""
        assert _extract_rag_context([{"type": "text", "text": "ChromaDB path not found at /tmp/db"}]) == ""

    def test_no_keyword_matches_signal(self) -> None:
        """'no keyword matches' on the first line suppresses the result."""
        assert _extract_rag_context([{"type": "text", "text": "No keyword matches for 'foo bar'."}]) == ""

    def test_signal_only_on_second_line_not_suppressed(self) -> None:
        """Suppression signals on lines after the first are ignored."""
        result = [{"type": "text", "text": "Good context here.\nNo results found elsewhere."}]
        assert _extract_rag_context(result) != ""

    def test_empty_list_returns_empty(self) -> None:
        """Empty list returns empty string."""
        assert _extract_rag_context([]) == ""

    def test_non_list_returns_empty(self) -> None:
        """Non-list input returns empty string."""
        assert _extract_rag_context("not a list") == ""


class TestRagHitCount:
    """Hit counting from retrieval results."""

    def test_counts_non_empty_lines(self) -> None:
        """Non-empty lines are counted."""
        context = "line one\nline two\nline three"
        assert _rag_hit_count([{"type": "text", "text": context}], context) == 3

    def test_blank_lines_not_counted(self) -> None:
        """Blank lines are excluded from the count."""
        context = "line one\n\n\nline two\n"
        assert _rag_hit_count([{"type": "text", "text": context}], context) == 2

    def test_exception_returns_minus_one(self) -> None:
        """Exception result returns -1."""
        assert _rag_hit_count(RuntimeError("error"), "") == -1

    def test_empty_context_returns_zero(self) -> None:
        """Empty context returns 0."""
        assert _rag_hit_count([{"text": ""}], "") == 0


class TestRetrieveRagContext:
    """Integration tests for retrieve_rag_context with mocked search functions."""

    @pytest.mark.asyncio
    async def test_merges_both_results(self) -> None:
        """Both vector and BM25 results are merged with a separator."""
        vec_text = "Vector result content."
        bm25_text = "BM25 keyword result."
        with (
            patch.object(ex_mod, "_run_vector_search", new=AsyncMock(return_value=vec_text)),
            patch.object(ex_mod, "_run_bm25_search", new=AsyncMock(return_value=bm25_text)),
        ):
            ctx = await retrieve_rag_context("test question")
        assert vec_text in ctx
        assert bm25_text in ctx
        assert "Keyword search results" in ctx

    @pytest.mark.asyncio
    async def test_falls_back_to_vector_only(self) -> None:
        """When BM25 returns empty, only vector context is returned."""
        vec_text = "Only vector content."
        with (
            patch.object(ex_mod, "_run_vector_search", new=AsyncMock(return_value=vec_text)),
            patch.object(ex_mod, "_run_bm25_search", new=AsyncMock(return_value="")),
        ):
            ctx = await retrieve_rag_context("test question")
        assert ctx == vec_text

    @pytest.mark.asyncio
    async def test_both_fail_returns_empty(self) -> None:
        """When both searches raise, an empty string is returned gracefully."""
        with (
            patch.object(ex_mod, "_run_vector_search", new=AsyncMock(side_effect=RuntimeError("down"))),
            patch.object(ex_mod, "_run_bm25_search", new=AsyncMock(side_effect=RuntimeError("down"))),
        ):
            ctx = await retrieve_rag_context("test question")
        assert ctx == ""
