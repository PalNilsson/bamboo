import inspect
from mcp.server import Server
from mcp.types import Tool, ListToolsResult

from askpanda_mcp.config import Config
from askpanda_mcp.tools.health import askpanda_health_tool
from askpanda_mcp.tools.doc_rag import panda_doc_search_tool
from askpanda_mcp.tools.queue_info import panda_queue_info_tool
from askpanda_mcp.tools.task_status import panda_task_status_tool
from askpanda_mcp.tools.log_analysis import panda_log_analysis_tool
from askpanda_mcp.tools.pilot_monitor import panda_pilot_status_tool
from askpanda_mcp.prompts.templates import (
    get_askpanda_system_prompt,
    get_failure_triage_prompt,
)

TOOLS = {
    "askpanda_health": askpanda_health_tool,
    "panda_doc_search": panda_doc_search_tool,
    "panda_queue_info": panda_queue_info_tool,
    "panda_task_status": panda_task_status_tool,
    "panda_log_analysis": panda_log_analysis_tool,
    "panda_pilot_status": panda_pilot_status_tool,
}


def create_server() -> Server:
    app = Server(Config.SERVER_NAME)

    @app.list_tools()
    async def list_tools():
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
    async def call_tool(name: str, arguments: dict):
        tool = TOOLS.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")
        return await tool.call(arguments or {})

    @app.list_prompts()
    async def list_prompts():
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
    async def get_prompt(name: str, arguments: dict):
        if name == "askpanda_system":
            return await get_askpanda_system_prompt()
        if name == "failure_triage":
            return await get_failure_triage_prompt(arguments.get("log_text", ""))
        raise ValueError(f"Unknown prompt: {name}")

    return app
