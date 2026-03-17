"""Tests for the OpenAI, OpenAI-compatible, Anthropic, and Gemini LLM provider clients.

All tests mock the vendor SDKs so no real API credentials or network access
are required.  Each test exercises one behavioural contract:

- Happy path: correct text and token usage extracted from the mock response.
- Missing API key: raises LLMConfigError before any network call.
- Missing SDK: raises LLMConfigError with an install hint.
- Rate-limit (429): raises LLMRateLimitError immediately (no retry).
- Timeout: raises LLMTimeoutError.
- Retriable error: retries up to the configured limit, then raises LLMProviderError.
- Message normalisation: system prompt handled correctly per provider API shape.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bamboo.llm.exceptions import (
    LLMConfigError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from bamboo.llm.types import GenerateParams, Message, ModelSpec


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _spec(provider: str, model: str = "test-model") -> ModelSpec:
    """Build a minimal ModelSpec for testing."""
    return ModelSpec(provider=provider, model=model)


def _params(temperature: float = 0.2, max_tokens: int | None = None) -> GenerateParams:
    """Build GenerateParams for testing."""
    return GenerateParams(temperature=temperature, max_tokens=max_tokens)


_MESSAGES: list[Message] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is PanDA?"},
]


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

class TestOpenAIClient:
    """Tests for OpenAILLMClient."""

    def _make_mock_response(self, text: str = "test reply",
                            prompt_tokens: int = 10,
                            completion_tokens: int = 5) -> MagicMock:
        """Build a minimal mock that looks like an openai ChatCompletion."""
        usage = MagicMock(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        message = MagicMock(content=text)
        choice = MagicMock(message=message)
        return MagicMock(choices=[choice], usage=usage)

    @pytest.mark.asyncio
    async def test_happy_path_returns_text_and_tokens(self, monkeypatch: Any) -> None:
        """Successful call returns the response text and parsed token usage."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_response = self._make_mock_response("PanDA is a workload manager.", 20, 8)

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        from bamboo.llm.providers.openai_client import OpenAILLMClient
        client = OpenAILLMClient(_spec("openai"))

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client._client = None  # force re-init
            resp = await client.generate(_MESSAGES, _params(max_tokens=100))

        assert resp.text == "PanDA is a workload manager."
        assert resp.usage is not None
        assert resp.usage.input_tokens == 20
        assert resp.usage.output_tokens == 8
        assert resp.usage.total_tokens == 28

    @pytest.mark.asyncio
    async def test_missing_api_key_raises_config_error(self, monkeypatch: Any) -> None:
        """Missing OPENAI_API_KEY raises LLMConfigError before any network call."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from bamboo.llm.providers.openai_client import OpenAILLMClient
        client = OpenAILLMClient(_spec("openai"))
        client._client = None

        with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
            await client.generate(_MESSAGES, _params())

    @pytest.mark.asyncio
    async def test_missing_sdk_raises_config_error(self, monkeypatch: Any) -> None:
        """ImportError for openai SDK is surfaced as LLMConfigError with install hint."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from bamboo.llm.providers.openai_client import OpenAILLMClient
        client = OpenAILLMClient(_spec("openai"))
        client._client = None

        with patch.dict("sys.modules", {"openai": None}):  # type: ignore[dict-item]
            with pytest.raises(LLMConfigError, match="pip install"):
                await client.generate(_MESSAGES, _params())

    @pytest.mark.asyncio
    async def test_rate_limit_raises_immediately(self, monkeypatch: Any) -> None:
        """A 429 error raises LLMRateLimitError without retrying."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ASKPANDA_OPENAI_RETRIES", "3")

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Status 429 too many requests")
        )
        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        from bamboo.llm.providers.openai_client import OpenAILLMClient
        client = OpenAILLMClient(_spec("openai"))

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client._client = mock_client
            with pytest.raises(LLMRateLimitError):
                await client.generate(_MESSAGES, _params())

        # Should have only been called once (no retry on rate-limit).
        mock_client.chat.completions.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_timeout_raises_llm_timeout_error(self, monkeypatch: Any) -> None:
        """asyncio.TimeoutError is converted to LLMTimeoutError."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )

        from bamboo.llm.providers.openai_client import OpenAILLMClient
        client = OpenAILLMClient(_spec("openai"))
        client._client = mock_client

        with pytest.raises(LLMTimeoutError):
            await client.generate(_MESSAGES, _params())

    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self, monkeypatch: Any) -> None:
        """Transient errors trigger retries up to the configured limit."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ASKPANDA_OPENAI_RETRIES", "2")
        monkeypatch.setenv("ASKPANDA_OPENAI_BACKOFF_SECONDS", "0")

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("connection reset")
        )

        from bamboo.llm.providers.openai_client import OpenAILLMClient
        client = OpenAILLMClient(_spec("openai"))
        client._client = mock_client

        with pytest.raises(LLMProviderError, match="OpenAI error after retries"):
            await client.generate(_MESSAGES, _params())

        assert mock_client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_system_message_excluded_from_messages(self, monkeypatch: Any) -> None:
        """System messages are passed through to OpenAI chat messages (OpenAI supports them)."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_response = self._make_mock_response("answer")
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        from bamboo.llm.providers.openai_client import OpenAILLMClient
        client = OpenAILLMClient(_spec("openai"))

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client._client = None
            await client.generate(_MESSAGES, _params())

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        roles = [m["role"] for m in call_kwargs["messages"]]
        assert "system" in roles


# ---------------------------------------------------------------------------
# OpenAI-compat client
# ---------------------------------------------------------------------------

class TestOpenAICompatClient:
    """Tests for OpenAICompatLLMClient."""

    @pytest.mark.asyncio
    async def test_happy_path(self, monkeypatch: Any) -> None:
        """Successful call with custom base_url returns response text."""
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "token-xyz")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="vLLM response"))]
        mock_response.usage = MagicMock(
            prompt_tokens=5, completion_tokens=3, total_tokens=8
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        spec = ModelSpec(
            provider="openai_compat",
            model="llama-3-8b",
            base_url="http://localhost:8000/v1",
        )
        from bamboo.llm.providers.openai_compat_client import OpenAICompatLLMClient
        client = OpenAICompatLLMClient(spec)

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client._client = None
            resp = await client.generate(_MESSAGES, _params())

        assert resp.text == "vLLM response"
        assert resp.usage is not None
        assert resp.usage.input_tokens == 5

    @pytest.mark.asyncio
    async def test_missing_base_url_raises_config_error(self, monkeypatch: Any) -> None:
        """Missing base URL raises LLMConfigError."""
        monkeypatch.delenv("ASKPANDA_OPENAI_COMPAT_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)

        from bamboo.llm.providers.openai_compat_client import OpenAICompatLLMClient
        client = OpenAICompatLLMClient(_spec("openai_compat"))
        client._client = None

        with pytest.raises(LLMConfigError, match="base URL"):
            await client.generate(_MESSAGES, _params())

    @pytest.mark.asyncio
    async def test_rate_limit_raises_immediately(self, monkeypatch: Any) -> None:
        """429 raises LLMRateLimitError without retrying."""
        monkeypatch.setenv("ASKPANDA_OPENAI_COMPAT_RETRIES", "3")
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("429 rate limit exceeded")
        )
        spec = ModelSpec(
            provider="openai_compat",
            model="llama-3",
            base_url="http://localhost:8000/v1",
        )
        from bamboo.llm.providers.openai_compat_client import OpenAICompatLLMClient
        client = OpenAICompatLLMClient(spec)
        client._client = mock_client

        with pytest.raises(LLMRateLimitError):
            await client.generate(_MESSAGES, _params())

        mock_client.chat.completions.create.assert_awaited_once()


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------

class TestAnthropicClient:
    """Tests for AnthropicLLMClient."""

    def _make_mock_response(
            self, text: str = "claude reply",
            input_tokens: int = 15,
            output_tokens: int = 6) -> MagicMock:
        """Build a minimal mock that looks like an Anthropic Message response."""
        content_block = MagicMock(text=text)
        usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
        return MagicMock(content=[content_block], usage=usage)

    @pytest.mark.asyncio
    async def test_happy_path_extracts_text_and_tokens(self, monkeypatch: Any) -> None:
        """Successful call returns text and input/output token counts."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        mock_response = self._make_mock_response("ATLAS is the experiment.", 20, 10)

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        from bamboo.llm.providers.anthropic_client import AnthropicLLMClient
        client = AnthropicLLMClient(_spec("anthropic", "claude-3-5-haiku-latest"))

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client._client = None
            resp = await client.generate(_MESSAGES, _params(max_tokens=512))

        assert resp.text == "ATLAS is the experiment."
        assert resp.usage is not None
        assert resp.usage.input_tokens == 20
        assert resp.usage.output_tokens == 10
        assert resp.usage.total_tokens is None  # Anthropic doesn't return total

    @pytest.mark.asyncio
    async def test_system_message_passed_separately(self, monkeypatch: Any) -> None:
        """The system message is extracted and passed as the 'system' parameter."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        mock_response = self._make_mock_response("ok")
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        from bamboo.llm.providers.anthropic_client import AnthropicLLMClient
        client = AnthropicLLMClient(_spec("anthropic"))

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client._client = None
            await client.generate(_MESSAGES, _params(max_tokens=100))

        call_kwargs = mock_client.messages.create.call_args[1]
        # System prompt must not appear in the messages list.
        msg_roles = [m["role"] for m in call_kwargs["messages"]]
        assert "system" not in msg_roles
        # It must be in the 'system' keyword argument.
        assert "system" in call_kwargs
        assert "helpful assistant" in call_kwargs["system"]

    @pytest.mark.asyncio
    async def test_missing_api_key_raises_config_error(self, monkeypatch: Any) -> None:
        """Missing ANTHROPIC_API_KEY raises LLMConfigError."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from bamboo.llm.providers.anthropic_client import AnthropicLLMClient
        client = AnthropicLLMClient(_spec("anthropic"))
        client._client = None

        with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
            await client.generate(_MESSAGES, _params())

    @pytest.mark.asyncio
    async def test_missing_sdk_raises_config_error(self, monkeypatch: Any) -> None:
        """ImportError for anthropic SDK surfaces as LLMConfigError with install hint."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        from bamboo.llm.providers.anthropic_client import AnthropicLLMClient
        client = AnthropicLLMClient(_spec("anthropic"))
        client._client = None

        with patch.dict("sys.modules", {"anthropic": None}):  # type: ignore[dict-item]
            with pytest.raises(LLMConfigError, match="pip install"):
                await client.generate(_MESSAGES, _params())

    @pytest.mark.asyncio
    async def test_rate_limit_raises_immediately(self, monkeypatch: Any) -> None:
        """429 raises LLMRateLimitError immediately."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("ASKPANDA_ANTHROPIC_RETRIES", "3")

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=Exception("429 rate_limit_error")
        )
        from bamboo.llm.providers.anthropic_client import AnthropicLLMClient
        client = AnthropicLLMClient(_spec("anthropic"))
        client._client = mock_client

        with pytest.raises(LLMRateLimitError):
            await client.generate(_MESSAGES, _params())

        mock_client.messages.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self, monkeypatch: Any) -> None:
        """Transient 5xx errors are retried up to the configured limit."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("ASKPANDA_ANTHROPIC_RETRIES", "2")
        monkeypatch.setenv("ASKPANDA_ANTHROPIC_BACKOFF_SECONDS", "0")

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=Exception("503 service unavailable")
        )
        from bamboo.llm.providers.anthropic_client import AnthropicLLMClient
        client = AnthropicLLMClient(_spec("anthropic"))
        client._client = mock_client

        with pytest.raises(LLMProviderError, match="Anthropic error after retries"):
            await client.generate(_MESSAGES, _params())

        assert mock_client.messages.create.await_count == 2

    @pytest.mark.asyncio
    async def test_system_only_messages_adds_user_turn(self, monkeypatch: Any) -> None:
        """When only a system message is given, a minimal user turn is synthesised."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        mock_response = self._make_mock_response("ok")
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        system_only: list[Message] = [{"role": "system", "content": "You are helpful."}]

        from bamboo.llm.providers.anthropic_client import AnthropicLLMClient
        client = AnthropicLLMClient(_spec("anthropic"))

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client._client = None
            await client.generate(system_only, _params())

        call_kwargs = mock_client.messages.create.call_args[1]
        assert len(call_kwargs["messages"]) >= 1
        assert call_kwargs["messages"][0]["role"] == "user"


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

class TestGeminiClient:
    """Tests for GeminiLLMClient."""

    def _make_mock_genai(
            self, text: str = "gemini reply",
            input_tokens: int = 12,
            output_tokens: int = 5) -> MagicMock:
        """Build a mock google.generativeai module."""
        usage_meta = MagicMock(
            prompt_token_count=input_tokens,
            candidates_token_count=output_tokens,
            total_token_count=input_tokens + output_tokens,
        )
        response = MagicMock(text=text, usage_metadata=usage_meta)
        model_mock = MagicMock()
        model_mock.generate_content_async = AsyncMock(return_value=response)
        genai = MagicMock()
        genai.GenerativeModel.return_value = model_mock
        genai.configure = MagicMock()
        return genai

    @pytest.mark.asyncio
    async def test_happy_path_returns_text_and_tokens(self, monkeypatch: Any) -> None:
        """Successful call returns text and parsed usage metadata."""
        monkeypatch.setenv("GEMINI_API_KEY", "ai-test-key")
        mock_genai = self._make_mock_genai("Gemini says: PanDA is a system.", 15, 7)

        from bamboo.llm.providers.gemini_client import GeminiLLMClient
        client = GeminiLLMClient(_spec("gemini", "gemini-1.5-flash"))

        with patch.dict("sys.modules", {"google.generativeai": mock_genai, "google": MagicMock()}):
            client._genai = None
            client._genai = mock_genai  # inject directly to skip import
            resp = await client.generate(_MESSAGES, _params(max_tokens=200))

        assert resp.text == "Gemini says: PanDA is a system."
        assert resp.usage is not None
        assert resp.usage.input_tokens == 15
        assert resp.usage.output_tokens == 7

    @pytest.mark.asyncio
    async def test_system_message_passed_as_system_instruction(self, monkeypatch: Any) -> None:
        """System message is passed as system_instruction to GenerativeModel."""
        monkeypatch.setenv("GEMINI_API_KEY", "ai-test-key")
        mock_genai = self._make_mock_genai()

        from bamboo.llm.providers.gemini_client import GeminiLLMClient
        client = GeminiLLMClient(_spec("gemini"))
        client._genai = mock_genai

        await client.generate(_MESSAGES, _params())

        call_kwargs = mock_genai.GenerativeModel.call_args[1]
        assert "system_instruction" in call_kwargs
        assert "helpful assistant" in call_kwargs["system_instruction"]

    @pytest.mark.asyncio
    async def test_missing_api_key_raises_config_error(self, monkeypatch: Any) -> None:
        """Missing GEMINI_API_KEY raises LLMConfigError."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from bamboo.llm.providers.gemini_client import GeminiLLMClient
        client = GeminiLLMClient(_spec("gemini"))
        client._genai = None

        with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
            await client.generate(_MESSAGES, _params())

    @pytest.mark.asyncio
    async def test_rate_limit_raises_immediately(self, monkeypatch: Any) -> None:
        """429/quota error raises LLMRateLimitError immediately."""
        monkeypatch.setenv("GEMINI_API_KEY", "ai-test-key")
        monkeypatch.setenv("ASKPANDA_GEMINI_RETRIES", "3")

        mock_genai = MagicMock()
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(
            side_effect=Exception("429 resource_exhausted quota exceeded")
        )
        mock_genai.GenerativeModel.return_value = mock_model

        from bamboo.llm.providers.gemini_client import GeminiLLMClient
        client = GeminiLLMClient(_spec("gemini"))
        client._genai = mock_genai

        with pytest.raises(LLMRateLimitError):
            await client.generate(_MESSAGES, _params())

        mock_model.generate_content_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self, monkeypatch: Any) -> None:
        """Transient errors are retried up to the configured limit."""
        monkeypatch.setenv("GEMINI_API_KEY", "ai-test-key")
        monkeypatch.setenv("ASKPANDA_GEMINI_RETRIES", "2")
        monkeypatch.setenv("ASKPANDA_GEMINI_BACKOFF_SECONDS", "0")

        mock_genai = MagicMock()
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(
            side_effect=Exception("503 service unavailable")
        )
        mock_genai.GenerativeModel.return_value = mock_model

        from bamboo.llm.providers.gemini_client import GeminiLLMClient
        client = GeminiLLMClient(_spec("gemini"))
        client._genai = mock_genai

        with pytest.raises(LLMProviderError, match="Gemini error after retries"):
            await client.generate(_MESSAGES, _params())

        assert mock_model.generate_content_async.await_count == 2
