"""Tests for LLM error handling in bamboo_answer.

After the deterministic fast-path refactor, _route() calls execute_plan()
directly (no planner LLM call). LLMError can surface from inside execute_plan
(specifically from call_llm). The BambooAnswerTool.call() boundary catches all
LLMError and converts it to a friendly message.

We mock execute_plan at the bamboo.tools.bamboo_answer level to inject errors.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bamboo.llm.exceptions import (
    LLMConfigError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from bamboo.tools.bamboo_answer import BambooAnswerTool, _friendly_llm_error


def _mock_guard(allowed: bool = True) -> MagicMock:
    """Return a mock topic-guard result."""
    g = MagicMock()
    g.allowed = allowed
    g.reason = "ok"
    g.rejection_message = "Off-topic."
    g.llm_used = False
    return g


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
        """503 / overflow signals overloaded provider."""
        exc = LLMProviderError(
            "Mistral error after retries: Status 503 upstream connect error "
            "reset reason: overflow"
        )
        msg = _friendly_llm_error(exc)
        assert "overloaded" in msg.lower() or "unavailable" in msg.lower()
        assert "🔄" in msg
        assert "upstream connect error" not in msg
        assert "reset reason" not in msg

    def test_502_bad_gateway(self) -> None:
        """502 status is classified as transient overload."""
        assert "🔄" in _friendly_llm_error(LLMProviderError("Status 502 bad gateway"))

    def test_429_rate_limit_by_text(self) -> None:
        """429 / rate limit produces a wait message."""
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
        assert "🔑" in _friendly_llm_error(LLMProviderError("403 forbidden"))

    def test_timeout_in_message(self) -> None:
        """'timed out' in message routes to timeout message."""
        assert "⏱️" in _friendly_llm_error(LLMProviderError("request timed out after 30s"))

    def test_generic_provider_error_strips_prefix(self) -> None:
        """Generic errors strip the 'Mistral error after retries:' prefix."""
        msg = _friendly_llm_error(
            LLMProviderError("Mistral error after retries: something unusual happened")
        )
        assert "Mistral error after retries:" not in msg
        assert "⚠️" in msg

    def test_generic_error_truncates_long_message(self) -> None:
        """Generic errors truncate messages longer than 200 characters."""
        assert "…" in _friendly_llm_error(LLMProviderError("x" * 500))

    def test_returns_string(self) -> None:
        """_friendly_llm_error always returns a non-empty string."""
        for exc in [LLMConfigError("x"), LLMTimeoutError("x"),
                    LLMProviderError("x"), LLMRateLimitError("x")]:
            result = _friendly_llm_error(exc)
            assert isinstance(result, str) and result


class TestBambooAnswerLLMErrorHandling:
    """BambooAnswerTool.call() must never propagate LLMError."""

    @pytest.mark.asyncio
    async def test_503_from_execute_plan_returns_friendly_message(self) -> None:
        """503 raised from execute_plan returns a friendly overloaded message."""
        exc = LLMProviderError(
            "Mistral error after retries: Status 503 upstream connect error reset reason: overflow"
        )
        guard_mock = AsyncMock(return_value=_mock_guard())
        exec_mock = AsyncMock(side_effect=exc)
        tool = BambooAnswerTool()
        with (
            patch("bamboo.tools.bamboo_answer.check_topic", guard_mock),
            patch("bamboo.tools.bamboo_answer.execute_plan", exec_mock),
        ):
            result = await tool.call({"question": "What is PanDA?"})
        assert result[0]["type"] == "text"
        text = result[0]["text"]
        assert "overloaded" in text.lower() or "unavailable" in text.lower()
        assert "🔄" in text
        assert "upstream connect error" not in text

    @pytest.mark.asyncio
    async def test_timeout_from_execute_plan_returns_friendly_message(self) -> None:
        """Timeout raised from execute_plan returns a friendly timeout message."""
        guard_mock = AsyncMock(return_value=_mock_guard())
        exec_mock = AsyncMock(side_effect=LLMTimeoutError("deadline exceeded"))
        tool = BambooAnswerTool()
        with (
            patch("bamboo.tools.bamboo_answer.check_topic", guard_mock),
            patch("bamboo.tools.bamboo_answer.execute_plan", exec_mock),
        ):
            result = await tool.call({"question": "What is task 12345678 status?"})
        assert "⏱️" in result[0]["text"]

    @pytest.mark.asyncio
    async def test_config_error_from_execute_plan_returns_friendly_message(self) -> None:
        """Missing API key surfaces as a configuration hint, not an exception."""
        guard_mock = AsyncMock(return_value=_mock_guard())
        exec_mock = AsyncMock(side_effect=LLMConfigError("MISTRAL_API_KEY is not set"))
        tool = BambooAnswerTool()
        with (
            patch("bamboo.tools.bamboo_answer.check_topic", guard_mock),
            patch("bamboo.tools.bamboo_answer.execute_plan", exec_mock),
        ):
            result = await tool.call({"question": "What is brokerage?"})
        assert "⚙️" in result[0]["text"]

    @pytest.mark.asyncio
    async def test_rate_limit_from_execute_plan_returns_friendly_message(self) -> None:
        """429 rate limit error surfaces clearly."""
        guard_mock = AsyncMock(return_value=_mock_guard())
        exec_mock = AsyncMock(side_effect=LLMProviderError("Status 429 too many requests"))
        tool = BambooAnswerTool()
        with (
            patch("bamboo.tools.bamboo_answer.check_topic", guard_mock),
            patch("bamboo.tools.bamboo_answer.execute_plan", exec_mock),
        ):
            result = await tool.call({"question": "What is JEDI?"})
        assert "⏳" in result[0]["text"]

    @pytest.mark.asyncio
    async def test_bypass_routing_llm_error_returns_friendly_message(self) -> None:
        """LLM error on the bypass_routing path returns a friendly message."""
        llm_mock = AsyncMock(side_effect=LLMTimeoutError("timed out"))
        tool = BambooAnswerTool()
        with patch("bamboo.tools.bamboo_answer.bamboo_llm_answer_tool") as mock_llm:
            mock_llm.call = llm_mock
            result = await tool.call({"question": "hello", "bypass_routing": True})
        assert "⏱️" in result[0]["text"]

    @pytest.mark.asyncio
    async def test_successful_response_unaffected(self) -> None:
        """Normal successful responses are unaffected by the error handler."""
        llm_reply = "PanDA is the Production and Distributed Analysis system."
        guard_mock = AsyncMock(return_value=_mock_guard())
        exec_mock = AsyncMock(return_value=[{"type": "text", "text": llm_reply}])
        tool = BambooAnswerTool()
        with (
            patch("bamboo.tools.bamboo_answer.check_topic", guard_mock),
            patch("bamboo.tools.bamboo_answer.execute_plan", exec_mock),
        ):
            result = await tool.call({"question": "What is PanDA?"})
        assert result[0]["text"] == llm_reply
