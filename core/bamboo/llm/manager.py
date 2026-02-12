"""LLM client manager.

This module provides a small cache for LLM provider clients, keyed by ModelSpec.
It allows the service to reuse underlying HTTP connection pools across requests
and to close all clients cleanly on shutdown (ASGI lifespan).
"""

from __future__ import annotations

import asyncio

from bamboo.llm.base import LLMClient
from bamboo.llm.factory import build_client
from bamboo.llm.types import ModelSpec


def _spec_key(spec: ModelSpec) -> tuple[str, str, str, str]:
    """Create a stable cache key for a ModelSpec.

    Args:
        spec: Model specification.

    Returns:
        A tuple used as dictionary key.
    """
    extra_str = ""
    if spec.extra:
        # Deterministic ordering for stable caching.
        extra_str = str(sorted(spec.extra.items()))
    return (spec.provider, spec.model, spec.base_url or "", extra_str)


class LLMClientManager:
    """Caches LLMClient instances and closes them on shutdown.

    This manager is intended to be shared per-process (per service instance).
    It is safe for concurrent use.
    """

    def __init__(self) -> None:
        """Initialize the manager with an empty client cache."""
        self._clients: dict[tuple[str, str, str, str], LLMClient] = {}
        self._lock = asyncio.Lock()

    async def get_client(self, spec: ModelSpec) -> LLMClient:
        """Return a cached LLMClient for the given ModelSpec.

        If a client for this spec does not exist yet, it is created via the
        provider factory.

        Args:
            spec: Model specification.

        Returns:
            Cached or newly created LLMClient instance.
        """
        key = _spec_key(spec)
        async with self._lock:
            client = self._clients.get(key)
            if client is None:
                client = build_client(spec)
                self._clients[key] = client
            return client

    async def close_all(self) -> None:
        """Close and clear all cached clients.

        This should be called during application shutdown (e.g. ASGI lifespan
        shutdown) to ensure underlying HTTP resources are released cleanly.
        """
        async with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()

        for client in clients:
            try:
                await client.close()
            except Exception:  # pylint: disable=broad-exception-caught - best-effort close
                # Best-effort close: don't fail shutdown due to provider cleanup.
                pass
