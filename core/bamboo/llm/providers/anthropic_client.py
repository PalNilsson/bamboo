"""Anthropic LLM client implementation.

Uses the official ``anthropic`` SDK with the async ``AsyncAnthropic`` client.
Install the dependency with::

    pip install -r requirements-anthropic.txt
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from typing import Any

from bamboo.llm.base import LLMClient
from bamboo.llm.exceptions import LLMConfigError, LLMProviderError, LLMRateLimitError, LLMTimeoutError
from bamboo.llm.types import GenerateParams, LLMResponse, Message, ModelSpec, TokenUsage


_ANTHROPIC_CONCURRENCY = asyncio.Semaphore(int(os.getenv("ASKPANDA_ANTHROPIC_CONCURRENCY", "4")))

# Anthropic requires a non-empty max_tokens; use a safe default when the
# caller does not specify one.
_DEFAULT_MAX_TOKENS = 4096


class AnthropicLLMClient(LLMClient):
    """Anthropic provider client.

    Uses the official ``anthropic`` Python SDK (``anthropic>=0.25``).  The
    SDK is imported lazily so installing it is only required when this
    provider is actually used.

    The Anthropic messages API separates the system prompt from the chat
    history.  This client extracts the first ``role="system"`` message and
    passes it as the ``system`` parameter; all remaining messages are sent
    as ``messages``.

    Retry behaviour mirrors the Mistral client: exponential backoff with
    ``ASKPANDA_ANTHROPIC_RETRIES`` attempts (default 3) and
    ``ASKPANDA_ANTHROPIC_BACKOFF_SECONDS`` initial wait (default 1.0 s).
    """

    def __init__(self, model_spec: ModelSpec) -> None:
        """Initialise the client with a model spec.

        Args:
            model_spec: Model specification for this client.
        """
        super().__init__(model_spec)
        self._client: Any | None = None
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        """Close any underlying client resources."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            self._client = None

    async def _get_client(self) -> Any:
        """Lazily initialise and return the AsyncAnthropic client.

        Returns:
            An ``anthropic.AsyncAnthropic`` instance.

        Raises:
            LLMConfigError: If the API key environment variable is not set or
                the SDK is not installed.
        """
        if self._client is not None:
            return self._client

        env_name = getattr(self._model_spec, "api_key_env", None) or "ANTHROPIC_API_KEY"
        api_key = os.getenv(env_name)
        if not api_key:
            raise LLMConfigError(
                f"Anthropic API key not found.  Set {env_name} in the environment."
            )

        async with self._lock:
            if self._client is None:
                try:
                    from anthropic import AsyncAnthropic  # type: ignore  # pylint: disable=import-outside-toplevel
                except ImportError as exc:
                    raise LLMConfigError(
                        "The 'anthropic' package is not installed.  "
                        "Run: pip install -r requirements-anthropic.txt"
                    ) from exc

                self._client = AsyncAnthropic(api_key=api_key)

        return self._client

    def _split_messages(
        self, messages: Sequence[Message]
    ) -> tuple[str, list[dict[str, str]]]:
        """Split a message list into a system prompt and chat turns.

        The Anthropic API accepts a separate ``system`` string and a
        ``messages`` list.  This method extracts the first system-role message
        as the system prompt and returns the rest as normalised turns.

        Args:
            messages: Sequence of normalised Message dicts.

        Returns:
            A ``(system_prompt, chat_messages)`` tuple.  ``system_prompt`` may
            be an empty string if no system message is present.
        """
        system_prompt = ""
        chat: list[dict[str, str]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system" and not system_prompt:
                system_prompt = content
                continue
            if role == "tool":
                role = "assistant"
                name_val = m.get("name")
                if name_val:
                    content = f"[tool:{name_val}]\n{content}"
            if role not in ("user", "assistant"):
                role = "user"
            chat.append({"role": role, "content": content})

        # Anthropic requires the first message to have role="user".
        # If only a system message was provided, add a minimal user turn.
        if not chat:
            chat = [{"role": "user", "content": "Please respond."}]

        return system_prompt, chat

    async def generate(self, messages: Sequence[Message], params: GenerateParams) -> LLMResponse:
        """Generate a completion using Anthropic.

        Args:
            messages: Normalised messages.
            params: Generation parameters.

        Returns:
            Normalised LLMResponse.

        Raises:
            LLMConfigError: If the model spec is invalid or the API key is missing.
            LLMRateLimitError: On HTTP 429 rate-limit responses.
            LLMTimeoutError: On request timeouts.
            LLMProviderError: For all other provider errors after retries are exhausted.
        """
        model = self._model_spec.model
        if not model:
            raise LLMConfigError("ModelSpec.model is empty for Anthropic provider.")

        system_prompt, chat_messages = self._split_messages(messages)
        max_tokens = params.max_tokens or _DEFAULT_MAX_TOKENS

        async with _ANTHROPIC_CONCURRENCY:
            tries = int(os.getenv("ASKPANDA_ANTHROPIC_RETRIES", "3"))
            backoff = float(os.getenv("ASKPANDA_ANTHROPIC_BACKOFF_SECONDS", "1.0"))
            last_err: Exception | None = None

            for _ in range(tries):
                try:
                    client = await self._get_client()

                    kwargs: dict[str, Any] = {
                        "model": model,
                        "messages": chat_messages,
                        "max_tokens": max_tokens,
                    }
                    if system_prompt:
                        kwargs["system"] = system_prompt
                    if params.temperature != 1.0:
                        # Anthropic default temperature is 1.0; only set if
                        # the caller explicitly lowered it.
                        kwargs["temperature"] = params.temperature

                    res = await client.messages.create(**kwargs)

                    # Extract text from the first content block.
                    text = ""
                    for block in getattr(res, "content", []):
                        block_text = getattr(block, "text", None)
                        if isinstance(block_text, str):
                            text = block_text.strip()
                            break

                    usage = None
                    try:
                        u = getattr(res, "usage", None)
                        if u is not None:
                            usage = TokenUsage(
                                input_tokens=getattr(u, "input_tokens", None),
                                output_tokens=getattr(u, "output_tokens", None),
                                total_tokens=None,  # Anthropic does not return total
                            )
                    except Exception:  # pylint: disable=broad-exception-caught
                        usage = None

                    return LLMResponse(text=text, usage=usage, raw=res)

                except asyncio.TimeoutError as exc:
                    raise LLMTimeoutError(str(exc)) from exc
                except (LLMConfigError, LLMRateLimitError):
                    raise  # permanent failures — do not retry
                except Exception as exc:  # noqa: BLE001  pylint: disable=broad-exception-caught
                    exc_str = str(exc).lower()
                    if "429" in exc_str or "rate limit" in exc_str or "rate_limit" in exc_str:
                        raise LLMRateLimitError(str(exc)) from exc
                    last_err = exc
                    await asyncio.sleep(backoff)
                    backoff *= 2

            raise LLMProviderError(
                f"Anthropic error after retries: {last_err!s}"
            ) from last_err
