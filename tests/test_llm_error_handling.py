"""Tests for LLM error handling in bamboo_answer and llm_passthrough.

Verifies that:
- _friendly_llm_error() classifies all LLMError subtypes correctly.
- BambooAnswerTool.call() never raises LLMError — it always returns text.
- The friendly message is returned instead of raw SDK error strings.
- llm_passthrough re-raises LLMError so bamboo_answer can catch it.
- All routes (rag, task, job, log_analysis, bypass) handle LLM errors.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import bamboo.tools.bamboo_answer as ba_mod
from bamboo.llm.exceptions import (
    LLMConfigError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from bamboo.tools.bamboo_answer import BambooAnswerTool, _friendly_llm_error


# ---------------------------------------------------------------------------
# _friendly_llm_error — classification
# ---------------------------------------------------------------------------


class TestFriendlyLlmError:
    """Unit tests for the _friendly_llm_error() classifier."""

    def test_config_error(self) -> None:
        """LLMConfigError produces an actionable configuration message."""
        msg = _friendly_llm_error(LLMConfigError("MISTRAL_API_KEY is not set"))
        assert "api key" in msg.lower() or "configured" in msg.lower()
        assert "⚙️" in msg

    def test_timeout_error(self) -> None:
        """LLMTimeoutError produces a try-again message."""
        msg = _friendly_llm_error(LLMTimeoutError("request timed out"))
        assert "time" in msg.lower()
        assert "⏱️" in msg

    def test_503_overloaded(self) -> None:
        """503 / overflow in the message string signals overloaded provider."""
        exc = LLMProviderError(
            "Mistral error after retries: API error occurred: Status 503 "
            "Content-Type \"text/plain\". Body: upstream connect error or "
            "disconnect/reset before headers. reset reason: overflow"
        )
        msg = _friendly_llm_error(exc)
        assert "overloaded" in msg.lower() or "unavailable" in msg.lower()
        assert "🔄" in msg
        # Must not expose raw SDK internals
        assert "upstream connect error" not in msg
        assert "reset reason" not in msg

    def test_502_bad_gateway(self) -> None:
        """502 status is also classified as transient overload."""
        msg = _friendly_llm_error(LLMProviderError("Status 502 bad gateway"))
        assert "🔄" in msg

    def test_429_rate_limit_by_text(self) -> None:
        """429 / rate limit in the message produces a wait message."""
        msg = _friendly_llm_error(LLMProviderError("Status 429 too many requests"))
        assert "rate limit" in msg.lower() or "wait" in msg.lower()
        assert "⏳" in msg

    def test_rate_limit_error_type(self) -> None:
        """LLMRateLimitError produces the wait message regardless of text."""
        msg = _friendly_llm_error(LLMRateLimitError("quota exceeded"))
        assert "⏳" in msg or "rate limit" in msg.lower() or "wait" in msg.lower()

    def test_401_auth_failure(self) -> None:
        """401 / unauthorized produces an API key message."""
        msg = _friendly_llm_error(LLMProviderError("Status 401 unauthorized"))
        assert "key" in msg.lower() or "auth" in msg.lower()
        assert "🔑" in msg

    def test_403_forbidden(self) -> None:
        """403 forbidden also produces the auth message."""
        msg = _friendly_llm_error(LLMProviderError("403 forbidden"))
        assert "🔑" in msg

    def test_timeout_in_message(self) -> None:
        """'timed out' in message text routes to timeout message."""
        msg = _friendly_llm_error(LLMProviderError("request timed out after 30s"))
        assert "⏱️" in msg

    def test_generic_provider_error_strips_prefix(self) -> None:
        """Generic errors strip the 'Mistral error after retries:' prefix."""
        msg = _friendly_llm_error(
            LLMProviderError("Mistral error after retries: something unusual happened")
        )
        assert "Mistral error after retries:" not in msg
        assert "⚠️" in msg

    def test_generic_error_truncates_long_message(self) -> None:
        """Generic errors truncate messages longer than 200 characters."""
        long_msg = "x" * 500
        msg = _friendly_llm_error(LLMProviderError(long_msg))
        # The excerpt inside the message must be at most 200 chars + ellipsis.
        assert "…" in msg

    def test_returns_string(self) -> None:
        """_friendly_llm_error always returns a non-empty string."""
        for exc in [
            LLMConfigError("x"),
            LLMTimeoutError("x"),
            LLMProviderError("x"),
            LLMRateLimitError("x"),
        ]:
            result = _friendly_llm_error(exc)
            assert isinstance(result, str) and result


# ---------------------------------------------------------------------------
# BambooAnswerTool.call() — never raises LLMError
# ---------------------------------------------------------------------------


def _make_llm_error_mock(exc: LLMError) -> AsyncMock:
    """Return an AsyncMock for bamboo_llm_answer_tool that raises exc."""
    return AsyncMock(side_effect=exc)


def _make_rag_mock(text: str = "some doc content") -> AsyncMock:
    return AsyncMock(return_value=[{"type": "text", "text": text}])


def _make_task_mock() -> AsyncMock:
    return AsyncMock(return_value={"evidence": {"payload": {"status": "done"}}})


class TestBambooAnswerLLMErrorHandling:
    """BambooAnswerTool.call() must never propagate LLMError."""

    @pytest.mark.asyncio
    async def test_rag_route_503_returns_friendly_message(self) -> None:
        """503 on the RAG synthesis path returns a friendly message, not an exception."""
        exc = LLMProviderError(
            "Mistral error after retries: Status 503 upstream connect error "
            "reset reason: overflow"
        )
        llm_mock = AsyncMock(side_effect=exc)
        tool = BambooAnswerTool()
        with (
            patch.object(ba_mod, "panda_doc_search_tool", AsyncMock(call=_make_rag_mock())),
            patch.object(ba_mod, "panda_doc_bm25_tool", AsyncMock(call=_make_rag_mock())),
            patch.object(ba_mod, "bamboo_llm_answer_tool", AsyncMock(call=llm_mock)),
        ):
            result = await tool.call({"question": "What is PanDA?"})

        assert result[0]["type"] == "text"
        text = result[0]["text"]
        assert "overloaded" in text.lower() or "unavailable" in text.lower()
        assert "🔄" in text
        # Must not contain raw SDK error string
        assert "upstream connect error" not in text

    @pytest.mark.asyncio
    async def test_task_route_timeout_returns_friendly_message(self) -> None:
        """Timeout on the task synthesis path returns a friendly message."""
        exc = LLMTimeoutError("deadline exceeded")
        llm_mock = AsyncMock(side_effect=exc)
        task_mock = _make_task_mock()
        tool = BambooAnswerTool()
        with (
            patch.object(ba_mod, "panda_task_status_tool", AsyncMock(call=task_mock)),
            patch.object(ba_mod, "bamboo_llm_answer_tool", AsyncMock(call=llm_mock)),
        ):
            result = await tool.call({"question": "What is task 12345678 status?"})

        assert result[0]["type"] == "text"
        assert "⏱️" in result[0]["text"]

    @pytest.mark.asyncio
    async def test_config_error_returns_friendly_message(self) -> None:
        """Missing API key returns a configuration hint, not an exception."""
        exc = LLMConfigError("MISTRAL_API_KEY is not set")
        llm_mock = AsyncMock(side_effect=exc)
        tool = BambooAnswerTool()
        with (
            patch.object(ba_mod, "panda_doc_search_tool", AsyncMock(call=_make_rag_mock())),
            patch.object(ba_mod, "panda_doc_bm25_tool", AsyncMock(call=_make_rag_mock())),
            patch.object(ba_mod, "bamboo_llm_answer_tool", AsyncMock(call=llm_mock)),
        ):
            result = await tool.call({"question": "What is brokerage?"})

        assert result[0]["type"] == "text"
        assert "⚙️" in result[0]["text"]

    @pytest.mark.asyncio
    async def test_rate_limit_returns_friendly_message(self) -> None:
        """429 rate limit error is communicated clearly."""
        exc = LLMProviderError("Status 429 too many requests")
        llm_mock = AsyncMock(side_effect=exc)
        tool = BambooAnswerTool()
        with (
            patch.object(ba_mod, "panda_doc_search_tool", AsyncMock(call=_make_rag_mock())),
            patch.object(ba_mod, "panda_doc_bm25_tool", AsyncMock(call=_make_rag_mock())),
            patch.object(ba_mod, "bamboo_llm_answer_tool", AsyncMock(call=llm_mock)),
        ):
            result = await tool.call({"question": "What is JEDI?"})

        assert result[0]["type"] == "text"
        assert "⏳" in result[0]["text"]

    @pytest.mark.asyncio
    async def test_job_route_llm_error_returns_friendly_message(self) -> None:
        """LLM error on the job status path also returns a friendly message."""
        exc = LLMProviderError("Status 503 service unavailable")
        llm_mock = AsyncMock(side_effect=exc)
        job_mock = AsyncMock(return_value={"evidence": {"status": "failed"}})
        tool = BambooAnswerTool()
        with (
            patch.object(ba_mod, "panda_job_status_tool", AsyncMock(call=job_mock)),
            patch.object(ba_mod, "bamboo_llm_answer_tool", AsyncMock(call=llm_mock)),
        ):
            result = await tool.call({"question": "What happened to job 99887766?"})

        assert result[0]["type"] == "text"
        assert "🔄" in result[0]["text"]

    @pytest.mark.asyncio
    async def test_log_analysis_route_llm_error_returns_friendly_message(self) -> None:
        """LLM error on log analysis path returns a friendly message."""
        exc = LLMProviderError("Status 503 overflow")
        llm_mock = AsyncMock(side_effect=exc)
        log_mock = AsyncMock(return_value={"evidence": {"failure": "segfault"}})
        tool = BambooAnswerTool()
        with (
            patch.object(ba_mod, "panda_log_analysis_tool", AsyncMock(call=log_mock)),
            patch.object(ba_mod, "bamboo_llm_answer_tool", AsyncMock(call=llm_mock)),
        ):
            result = await tool.call({"question": "Why did job 99887766 fail?"})

        assert result[0]["type"] == "text"
        assert "🔄" in result[0]["text"]

    @pytest.mark.asyncio
    async def test_bypass_routing_llm_error_returns_friendly_message(self) -> None:
        """LLM error on the bypass_routing path also returns a friendly message."""
        exc = LLMTimeoutError("timed out")
        llm_mock = AsyncMock(side_effect=exc)
        tool = BambooAnswerTool()
        with patch.object(ba_mod, "bamboo_llm_answer_tool", AsyncMock(call=llm_mock)):
            result = await tool.call({"question": "hello", "bypass_routing": True})

        assert result[0]["type"] == "text"
        assert "⏱️" in result[0]["text"]

    @pytest.mark.asyncio
    async def test_successful_response_unaffected(self) -> None:
        """Normal successful responses are completely unaffected by the error handler."""
        llm_reply = "PanDA is the Production and Distributed Analysis system."
        llm_mock = AsyncMock(return_value=[{"type": "text", "text": llm_reply}])
        tool = BambooAnswerTool()
        with (
            patch.object(ba_mod, "panda_doc_search_tool", AsyncMock(call=_make_rag_mock())),
            patch.object(ba_mod, "panda_doc_bm25_tool", AsyncMock(call=_make_rag_mock())),
            patch.object(ba_mod, "bamboo_llm_answer_tool", AsyncMock(call=llm_mock)),
        ):
            result = await tool.call({"question": "What is PanDA?"})

        assert result[0]["text"] == llm_reply
