"""Core MCP server wiring.

This module creates the MCP Server instance, registers tools/prompts, and
initializes process-wide resources (LLM selection + client caching).
"""

from __future__ import annotations

import inspect
from typing import Any, cast

from mcp.server import Server
from mcp.types import ListToolsResult, Tool

from askpanda_mcp.config import Config

# Phase 0: multi-LLM wiring
from askpanda_mcp.llm.config_loader import build_model_registry_from_config
from askpanda_mcp.llm.manager import LLMClientManager
from askpanda_mcp.llm.selector import LLMSelector
from askpanda_mcp.llm.runtime import set_llm_manager, set_llm_selector

from askpanda_mcp.tools.health import askpanda_health_tool
from askpanda_mcp.tools.doc_rag import panda_doc_search_tool
from askpanda_mcp.tools.queue_info import panda_queue_info_tool
from askpanda_mcp.tools.task_status import panda_task_status_tool
from askpanda_mcp.tools.log_analysis import panda_log_analysis_tool
from askpanda_mcp.tools.pilot_monitor import panda_pilot_status_tool
from askpanda_mcp.tools.llm_passthrough import askpanda_llm_answer_tool
from askpanda_mcp.tools.askpanda_answer import askpanda_answer_tool
from askpanda_mcp.prompts.templates import (
    get_askpanda_system_prompt,
    get_failure_triage_prompt,
)

TOOLS = {
    "askpanda_health": askpanda_health_tool,
    "askpanda_llm_answer": askpanda_llm_answer_tool,
    "askpanda_answer": askpanda_answer_tool,
    "panda_doc_search": panda_doc_search_tool,
    "panda_queue_info": panda_queue_info_tool,
    "panda_task_status": panda_task_status_tool,
    "panda_log_analysis": panda_log_analysis_tool,
    "panda_pilot_status": panda_pilot_status_tool,
}


def create_server() -> Server:  # pylint: disable=too-complex
    """Create and configure the MCP Server instance.

    This function wires up multi-LLM selection (model registry, selector, and
    per-process LLM client manager), registers available tools and prompts on
    the Server "app" instance, and publishes shared runtime state so tool
    singletons can access it.
    """
    app: Server = Server(Config.SERVER_NAME)

    # ---- Phase 0: initialize multi-LLM selection + per-process client cache ----
    # Support both Config being a class of constants or a dataclass-like type.
    try:
        config_obj: Config | type[Config] = Config()  # type: ignore[call-arg]
    except TypeError:
        config_obj = Config  # type: ignore[assignment]

    model_registry: Any = build_model_registry_from_config(config_obj)
    llm_selector: LLMSelector = LLMSelector(
        registry=model_registry,
        default_profile=getattr(config_obj, "LLM_DEFAULT_PROFILE", "default"),
        fast_profile=getattr(config_obj, "LLM_FAST_PROFILE", "fast"),
        reasoning_profile=getattr(config_obj, "LLM_REASONING_PROFILE", "reasoning"),
    )
    llm_manager: LLMClientManager = LLMClientManager()

    # Attach for visibility (HTTP shutdown handler can close these).
    setattr(app, "model_registry", model_registry)
    setattr(app, "llm_selector", llm_selector)
    setattr(app, "llm_manager", llm_manager)

    # Also publish into runtime context so simple tool singletons can access it.
    set_llm_selector(llm_selector)
    set_llm_manager(llm_manager)

    @app.list_tools()
    async def list_tools() -> Any:
        """Return the set of registered tools.

        The MCP `Tool` representation may be a runtime model/class or a
        TypedDict; handle both by inspecting the imported `Tool` symbol.

        Returns:
            Union[List[Tool], ListToolsResult, List[Dict[str, Any]]]: The tool
            list in the appropriate shape for the MCP server/client contract.
        """
        defs: list[dict[str, Any]] = [tool.get_definition() for tool in TOOLS.values()]

        # If Tool is a real class/model, return Tool objects.
        if inspect.isclass(Tool):
            return [Tool(**d) for d in defs]

        # Otherwise, try wrapping in ListToolsResult (often a model even if Tool is TypedDict).
        if inspect.isclass(ListToolsResult):
            return ListToolsResult(tools=cast(list[Tool], defs))

        # Last resort: plain dicts
        return defs

    @app.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a registered tool by name.

        Args:
            name: Name of the tool to call.
            arguments: JSON-like mapping of arguments for the tool.

        Returns:
            Any: Result produced by the called tool.

        Raises:
            ValueError: If the requested tool name is unknown.
        """
        tool: Any | None = TOOLS.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")
        return await tool.call(arguments or {})

    @app.list_prompts()
    async def list_prompts() -> Any:
        """List available prompts and their metadata.

        Returns:
            List[Dict[str, Any]]: Each entry contains prompt `name`,
            `description` and optional `arguments` specification.
        """
        return [
            {"name": "askpanda_system", "description": "Core system prompt"},
            {
                "name": "failure_triage",
                "description": "Failure triage template",
                "arguments": [
                    {
                        "name": "log_text",
                        "description": "Log snippet",
                        "required": True,
                    }
                ],
            },
        ]

    @app.get_prompt()
    async def get_prompt(name: str, arguments: dict[str, str] | None) -> Any:
        """Return the requested prompt payload.

        Args:
            name: Prompt name (e.g. 'askpanda_system' or 'failure_triage').
            arguments: Arguments mapping used to fill template values.

        Returns:
            Dict[str, Any]: Prompt payload (typically a `messages` list).

        Raises:
            ValueError: If the requested prompt name is unknown.
        """
        if name == "askpanda_system":
            return await get_askpanda_system_prompt()
        if name == "failure_triage":
            return await get_failure_triage_prompt((arguments or {}).get("log_text", ""))
        raise ValueError(f"Unknown prompt: {name}")

    return app
