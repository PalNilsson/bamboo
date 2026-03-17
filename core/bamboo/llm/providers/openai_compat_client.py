"""OpenAI-compatible LLM client implementation.

Works with any server that exposes an OpenAI-compatible ``/v1/chat/completions``
endpoint — vLLM, Ollama, LM Studio, Together AI, Fireworks AI, Anyscale, etc.

Configure via environment variables::

    LLM_DEFAULT_PROVIDER=openai_compat
    LLM_DEFAULT_MODEL=meta-llama/Llama-3-8b-instruct
    ASKPANDA_OPENAI_COMPAT_BASE_URL=http://localhost:8000/v1
    OPENAI_COMPAT_API_KEY=token-abc123          # omit or set to "none" if not required

Install the dependency with::

    pip install -r requirements-openai.txt       # reuses the openai SDK
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from typing import Any

from bamboo.llm.base import LLMClient
from bamboo.llm.exceptions import LLMConfigError, LLMProviderError, LLMRateLimitError, LLMTimeoutError
from bamboo.llm.types import GenerateParams, LLMResponse, Message, TokenUsage


_OPENAI_COMPAT_CONCURRENCY = asyncio.Semaphore(
    int(os.getenv("ASKPANDA_OPENAI_COMPAT_CONCURRENCY", "8"))
)


class OpenAICompatLLMClient(LLMClient):
    """OpenAI-compatible provider client.

    Delegates to the ``openai`` SDK's ``AsyncOpenAI`` with a custom ``base_url``.
    This makes it compatible with any server that implements the OpenAI chat
    completions API, including local inference engines.
    """

    def __init__(self, model_spec: Any) -> None:
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
        """Lazily initialise and return an AsyncOpenAI client pointed at the compat endpoint.

        Returns:
            An ``openai.AsyncOpenAI`` instance configured with the custom base URL.

        Raises:
            LLMConfigError: If the base URL is not configured or the SDK is not installed.
        """
        if self._client is not None:
            return self._client

        base_url = (
            getattr(self._model_spec, "base_url", None) or
            os.getenv("ASKPANDA_OPENAI_COMPAT_BASE_URL") or
            os.getenv("OPENAI_COMPAT_BASE_URL") or
            ""
        ).strip()

        if not base_url:
            raise LLMConfigError(
                "OpenAI-compatible base URL not configured.  "
                "Set ASKPANDA_OPENAI_COMPAT_BASE_URL in the environment."
            )

        env_name = getattr(self._model_spec, "api_key_env", None) or "OPENAI_COMPAT_API_KEY"
        api_key = os.getenv(env_name) or "none"  # many local servers don't check the key

        async with self._lock:
            if self._client is None:
                try:
                    from openai import AsyncOpenAI  # type: ignore  # pylint: disable=import-outside-toplevel
                except ImportError as exc:
                    raise LLMConfigError(
                        "The 'openai' package is not installed.  "
                        "Run: pip install -r requirements-openai.txt"
                    ) from exc

                self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        return self._client

    def _normalize_messages(self, messages: Sequence[Message]) -> list[dict[str, str]]:
        """Convert normalised messages into OpenAI chat message dicts.

        Args:
            messages: Sequence of normalised Message dicts.

        Returns:
            List of dicts with 'role' and 'content' keys.
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
        """Generate a completion using an OpenAI-compatible endpoint.

        Args:
            messages: Normalised messages.
            params: Generation parameters.

        Returns:
            Normalised LLMResponse.

        Raises:
            LLMConfigError: If the base URL is not set or the SDK is missing.
            LLMRateLimitError: On HTTP 429 responses.
            LLMTimeoutError: On request timeouts.
            LLMProviderError: For all other errors after retries are exhausted.
        """
        model = self._model_spec.model
        if not model:
            raise LLMConfigError("ModelSpec.model is empty for openai_compat provider.")

        compat_messages = self._normalize_messages(messages)

        async with _OPENAI_COMPAT_CONCURRENCY:
            tries = int(os.getenv("ASKPANDA_OPENAI_COMPAT_RETRIES", "3"))
            backoff = float(os.getenv("ASKPANDA_OPENAI_COMPAT_BACKOFF_SECONDS", "1.0"))
            last_err: Exception | None = None

            for _ in range(tries):
                try:
                    client = await self._get_client()

                    kwargs: dict[str, Any] = {
                        "model": model,
                        "messages": compat_messages,
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
                    if "429" in exc_str or "rate limit" in exc_str or "rate_limit" in exc_str:
                        raise LLMRateLimitError(str(exc)) from exc
                    last_err = exc
                    await asyncio.sleep(backoff)
                    backoff *= 2

            raise LLMProviderError(
                f"OpenAI-compatible error after retries: {last_err!s}"
            ) from last_err
