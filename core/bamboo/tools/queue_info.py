"""Dummy queue/site info tool (maps to your previous QueueQueryAgent)."""
from __future__ import annotations
from typing import Any
import json
import os

from bamboo.tools.base import text_content
from bamboo.config import Config


class PandaQueueInfoTool:
    """Tool that returns site/queue metadata from a local queuedata.json file.

    This is a simple, local/dummy implementation that reads `queuedata.json`
    (path configured via `Config.QUEUE_DATA_PATH`) and returns a pretty-printed
    JSON payload for the requested site. It is useful for local development and
    testing before hooking up a real PanDA/atlas service.
    """

    @staticmethod
    def get_definition() -> dict[str, Any]:
        """Return the tool discovery definition.

        The returned mapping contains the tool `name`, `description` and an
        `inputSchema` that clients can use to validate calls.

        Returns:
            Dict[str, Any]: MCP-compatible tool discovery definition.
        """
        return {
            "name": "panda_queue_info",
            "description": (
                "Look up queue and site configuration for a named ATLAS "
                "computing site (e.g. BNL-ATLAS, CERN-PROD). Returns resource "
                "limits, queue names, and site parameters. Use when a question "
                "asks about a specific site's capabilities or configuration."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "site": {"type": "string", "description": "Site name, e.g. BNL-ATLAS"},
                },
                "required": ["site"],
                "additionalProperties": False,
            },
        }

    async def call(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Return queue metadata for a site from the configured queuedata file.

        Args:
            arguments: Mapping with a required ``site`` key.

        Returns:
            List[Dict[str, Any]]: A one-element MCP text content payload with
            pretty-printed queue metadata, or an error message if the file
            cannot be read or the site is unknown.
        """
        site = arguments.get("site", "")
        path = Config.QUEUE_DATA_PATH
        # Resolve relative to package if user kept defaults
        if not os.path.isabs(path):
            here = os.path.dirname(__file__)
            path = os.path.abspath(os.path.join(here, "..", path))
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, PermissionError, json.JSONDecodeError) as exc:
            return text_content(f"Could not read queue data at {path}: {exc}")
        info = data.get(site)
        if not info:
            known = ", ".join(sorted(data.keys()))
            return text_content(f"Unknown site '{site}'. Known sites: {known}")
        pretty = json.dumps(info, indent=2)
        return text_content(f"Queue info for {site} (dummy)\n\n{pretty}")


panda_queue_info_tool = PandaQueueInfoTool()
