"""Core MCP server wiring.

This module creates the MCP Server instance, registers tools/prompts, and
initializes process-wide resources (LLM selection + client caching).
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

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


def create_server() -> Server:
    """Creates and configures the MCP server.

    Returns:
        Configured MCP Server instance.
    """
    app = Server(Config.SERVER_NAME)

    # ---- Phase 0: initialize multi-LLM selection + per-process client cache ----
    # Support both Config being a class of constants or a dataclass-like type.
    try:
        config_obj = Config()  # type: ignore[call-arg]
    except TypeError:
        config_obj = Config

    model_registry = build_model_registry_from_config(config_obj)
    llm_selector = LLMSelector(
        registry=model_registry,
        default_profile=getattr(config_obj, "LLM_DEFAULT_PROFILE", "default"),
        fast_profile=getattr(config_obj, "LLM_FAST_PROFILE", "fast"),
        reasoning_profile=getattr(config_obj, "LLM_REASONING_PROFILE", "reasoning"),
    )
    llm_manager = LLMClientManager()

    # Attach for visibility (HTTP shutdown handler can close these).
    app.model_registry = model_registry  # type: ignore[attr-defined]
    app.llm_selector = llm_selector      # type: ignore[attr-defined]
    app.llm_manager = llm_manager        # type: ignore[attr-defined]

    # Also publish into runtime context so simple tool singletons can access it.
    set_llm_selector(llm_selector)
    set_llm_manager(llm_manager)

    @app.list_tools()
    async def list_tools() -> Any:
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
    async def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
        tool = TOOLS.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")
        return await tool.call(arguments or {})

    @app.list_prompts()
    async def list_prompts() -> Any:
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
    async def get_prompt(name: str, arguments: Dict[str, Any]) -> Any:
        if name == "askpanda_system":
            return await get_askpanda_system_prompt()
        if name == "failure_triage":
            return await get_failure_triage_prompt(arguments.get("log_text", ""))
        raise ValueError(f"Unknown prompt: {name}")

    return app
