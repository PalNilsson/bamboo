"""Core MCP server wiring for AskPanDA.

This module defines:
- A tool registry (MCP tools exposed by the server).
- Server factory `create_server()` used by entrypoints.
- Prompt handlers used by MCP clients.

The server instance is also used as a convenient place to attach shared, long-lived
resources (e.g. LLM model registry/selector, LLM client manager).
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

from mcp.server import Server
from mcp.types import ListToolsResult, Tool

from askpanda_mcp.config import Config
from askpanda_mcp.prompts.templates import (
    get_askpanda_system_prompt,
    get_failure_triage_prompt,
)
from askpanda_mcp.tools.doc_rag import panda_doc_search_tool
from askpanda_mcp.tools.health import askpanda_health_tool
from askpanda_mcp.tools.log_analysis import panda_log_analysis_tool
from askpanda_mcp.tools.pilot_monitor import panda_pilot_status_tool
from askpanda_mcp.tools.queue_info import panda_queue_info_tool
from askpanda_mcp.tools.task_status import panda_task_status_tool

from askpanda_mcp.llm.config_loader import build_model_registry_from_config
from askpanda_mcp.llm.manager import LLMClientManager
from askpanda_mcp.llm.registry import ModelRegistry
from askpanda_mcp.llm.selector import LLMSelector


# NOTE: The tool objects are expected to implement:
# - get_definition() -> dict[str, Any]
# - call(arguments: dict[str, Any]) -> Any
TOOLS: Dict[str, Any] = {
    "askpanda_health": askpanda_health_tool,
    "panda_doc_search": panda_doc_search_tool,
    "panda_queue_info": panda_queue_info_tool,
    "panda_task_status": panda_task_status_tool,
    "panda_log_analysis": panda_log_analysis_tool,
    "panda_pilot_status": panda_pilot_status_tool,
}


def _get_config_object() -> Any:
    """Returns a Config instance if constructible; otherwise returns Config class.

    Some codebases model configuration as a dataclass/instance, while others
    keep Config as a static namespace with class attributes. This helper
    supports both patterns.

    Returns:
        A Config instance or the Config class.
    """
    try:
        return Config()  # type: ignore[call-arg]
    except TypeError:
        return Config


def create_server() -> Server:
    """Creates and configures the MCP Server.

    The returned server has the following additional attributes attached:
    - model_registry: ModelRegistry
    - llm_selector: LLMSelector
    - llm_manager: LLMClientManager

    Returns:
        Configured MCP server instance.
    """
    app = Server(Config.SERVER_NAME)

    cfg = _get_config_object()

    # Phase 0: Multi-LLM configuration (registry + selector) and a shared client manager.
    model_registry: ModelRegistry = build_model_registry_from_config(cfg)
    llm_selector = LLMSelector(
        registry=model_registry,
        default_profile=getattr(cfg, "LLM_DEFAULT_PROFILE", "default"),
        fast_profile=getattr(cfg, "LLM_FAST_PROFILE", "fast"),
        reasoning_profile=getattr(cfg, "LLM_REASONING_PROFILE", "reasoning"),
    )
    llm_manager = LLMClientManager()

    # Attach shared objects to the server for easy access by tools/orchestration.
    app.model_registry = model_registry  # type: ignore[attr-defined]
    app.llm_selector = llm_selector  # type: ignore[attr-defined]
    app.llm_manager = llm_manager  # type: ignore[attr-defined]

    @app.list_tools()
    async def list_tools() -> Any:
        """Lists tool definitions exposed by this MCP server."""
        defs = [tool.get_definition() for tool in TOOLS.values()]

        # If Tool is a real class/model, return Tool objects.
        if inspect.isclass(Tool):
            return [Tool(**d) for d in defs]

        # Otherwise, try wrapping in ListToolsResult (often a model even if Tool is TypedDict).
        if inspect.isclass(ListToolsResult):
            return ListToolsResult(tools=defs)

        # Last resort: plain dicts
        return defs

    @app.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
        """Calls a named tool.

        Args:
            name: Tool name.
            arguments: Tool arguments payload.

        Returns:
            Tool result payload.

        Raises:
            ValueError: If the tool name is unknown.
        """
        tool = TOOLS.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")
        return await tool.call(arguments or {})

    @app.list_prompts()
    async def list_prompts() -> list[dict[str, Any]]:
        """Lists prompts exposed by this MCP server."""
        return [
            {
                "name": "askpanda_system",
                "description": "System prompt for AskPanDA.",
                "arguments": [],
            },
            {
                "name": "failure_triage",
                "description": "Prompt template for failure triage.",
                "arguments": [
                    {"name": "error_snippet", "description": "Error text snippet."},
                ],
            },
        ]

    @app.get_prompt()
    async def get_prompt(name: str, arguments: dict[str, Any]) -> Any:
        """Resolves a named prompt.

        Args:
            name: Prompt name.
            arguments: Prompt arguments.

        Returns:
            Prompt result, as expected by MCP.

        Raises:
            ValueError: If the prompt name is unknown.
        """
        if name == "askpanda_system":
            return await get_askpanda_system_prompt()
        if name == "failure_triage":
            return await get_failure_triage_prompt(arguments or {})
        raise ValueError(f"Unknown prompt: {name}")

    return app
