"""Exceptions for LLM providers."""

class LLMError(RuntimeError):
    """Base exception for all LLM provider errors."""


class LLMConfigError(LLMError):
    """Raised when configuration is missing or invalid."""


class LLMRateLimitError(LLMError):
    """Raised when a provider rate-limits requests."""


class LLMTimeoutError(LLMError):
    """Raised on request timeouts."""


class LLMProviderError(LLMError):
    """Raised for provider-specific unexpected errors."""
