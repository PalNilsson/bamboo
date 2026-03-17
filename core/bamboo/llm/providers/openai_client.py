"""OpenAI LLM client implementation.

Uses the official ``openai`` SDK with the async ``AsyncOpenAI`` client.
Install the dependency with::

    pip install -r requirements-openai.txt
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from typing import Any

from bamboo.llm.base import LLMClient
from bamboo.llm.exceptions import LLMConfigError, LLMProviderError, LLMRateLimitError, LLMTimeoutError
from bamboo.llm.types import GenerateParams, LLMResponse, Message, ModelSpec, TokenUsage


_OPENAI_CONCURRENCY = asyncio.Semaphore(int(os.getenv("ASKPANDA_OPENAI_CONCURRENCY", "8")))


class OpenAILLMClient(LLMClient):
    """OpenAI provider client.

    Uses the official ``openai`` Python SDK (``openai>=1.0``).  The SDK is
    imported lazily so installing it is only required when this provider is
    actually used.

    Retry behaviour mirrors the Mistral client: exponential backoff with
    ``ASKPANDA_OPENAI_RETRIES`` attempts (default 3) and
    ``ASKPANDA_OPENAI_BACKOFF_SECONDS`` initial wait (default 1.0 s).
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
        """Lazily initialise and return the AsyncOpenAI client.

        Returns:
            An ``openai.AsyncOpenAI`` instance.

        Raises:
            LLMConfigError: If the API key environment variable is not set.
        """
        if self._client is not None:
            return self._client

        env_name = getattr(self._model_spec, "api_key_env", None) or "OPENAI_API_KEY"
        api_key = os.getenv(env_name)
        if not api_key:
            raise LLMConfigError(
                f"OpenAI API key not found.  Set {env_name} in the environment."
            )

        async with self._lock:
            if self._client is None:
                try:
                    from openai import AsyncOpenAI  # type: ignore  # pylint: disable=import-outside-toplevel
                except ImportError as exc:
                    raise LLMConfigError(
                        "The 'openai' package is not installed.  "
                        "Run: pip install -r requirements-openai.txt"
                    ) from exc

                kwargs: dict[str, Any] = {"api_key": api_key}
                base_url = getattr(self._model_spec, "base_url", None)
                if base_url:
                    kwargs["base_url"] = base_url
                self._client = AsyncOpenAI(**kwargs)

        return self._client

    def _normalize_messages(self, messages: Sequence[Message]) -> list[dict[str, str]]:
        """Convert normalised messages into OpenAI chat message dicts.

        Args:
            messages: Sequence of normalised Message dicts.

        Returns:
            List of dicts with 'role' and 'content' keys accepted by the
            OpenAI chat completions API.
        """
        out: list[dict[str, str]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "tool":
                role = "assistant"
                name_val = m.get("name")
                if name_val:
                    content = f"[tool:{name_val}]\n{content}"
            if role not in ("system", "user", "assistant"):
                role = "user"
            out.append({"role": role, "content": content})
        return out

    async def generate(self, messages: Sequence[Message], params: GenerateParams) -> LLMResponse:
        """Generate a completion using OpenAI.

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
            raise LLMConfigError("ModelSpec.model is empty for OpenAI provider.")

        openai_messages = self._normalize_messages(messages)

        async with _OPENAI_CONCURRENCY:
            tries = int(os.getenv("ASKPANDA_OPENAI_RETRIES", "3"))
            backoff = float(os.getenv("ASKPANDA_OPENAI_BACKOFF_SECONDS", "1.0"))
            last_err: Exception | None = None

            for _ in range(tries):
                try:
                    client = await self._get_client()

                    kwargs: dict[str, Any] = {
                        "model": model,
                        "messages": openai_messages,
                        "temperature": params.temperature,
                    }
                    if params.max_tokens is not None:
                        kwargs["max_tokens"] = params.max_tokens

                    res = await client.chat.completions.create(**kwargs)

                    text = (res.choices[0].message.content or "").strip()

                    usage = None
                    try:
                        u = getattr(res, "usage", None)
                        if u is not None:
                            usage = TokenUsage(
                                input_tokens=getattr(u, "prompt_tokens", None),
                                output_tokens=getattr(u, "completion_tokens", None),
                                total_tokens=getattr(u, "total_tokens", None),
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
                    # Surface rate-limit errors immediately — no retry benefit.
                    if "429" in exc_str or "rate limit" in exc_str or "rate_limit" in exc_str:
                        raise LLMRateLimitError(str(exc)) from exc
                    last_err = exc
                    await asyncio.sleep(backoff)
                    backoff *= 2

            raise LLMProviderError(f"OpenAI error after retries: {last_err!s}") from last_err
