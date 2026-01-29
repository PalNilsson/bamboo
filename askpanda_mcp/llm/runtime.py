"""Runtime LLM context.

This module provides a tiny, explicit place to store process-wide LLM objects
constructed during server startup.

Why this exists:
  - MCP tool instances in this repo are created as simple singletons and do not
    receive the Server/app instance.
  - We still want tools to be able to access the configured model selector and
    the shared LLM client cache (connection pooling).

The server startup code should call the setters exactly once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from askpanda_mcp.llm.manager import LLMClientManager
from askpanda_mcp.llm.selector import LLMSelector

# Module-level globals are an explicit pattern here (process-wide singletons).
# pylint: disable=global-statement
_llm_manager: LLMClientManager | None = None
_llm_selector: LLMSelector | None = None


def set_llm_manager(manager: LLMClientManager) -> None:
    """Sets the process-wide LLM client manager.

    Args:
        manager: LLM client manager instance.
    """
    global _llm_manager
    _llm_manager = manager


def get_llm_manager() -> LLMClientManager:
    """Returns the process-wide LLM client manager.

    Returns:
        The LLMClientManager.

    Raises:
        RuntimeError: If the manager has not been initialized.
    """
    if _llm_manager is None:
        raise RuntimeError("LLM manager is not initialized. Did core.create_server() run?")
    return _llm_manager


def set_llm_selector(selector: LLMSelector) -> None:
    """Sets the process-wide LLM selector.

    Args:
        selector: LLMSelector instance.
    """
    global _llm_selector
    _llm_selector = selector


def get_llm_selector() -> LLMSelector:
    """Returns the process-wide LLM selector.

    Returns:
        The LLMSelector.

    Raises:
        RuntimeError: If the selector has not been initialized.
    """
    if _llm_selector is None:
        raise RuntimeError("LLM selector is not initialized. Did core.create_server() run?")
    return _llm_selector
