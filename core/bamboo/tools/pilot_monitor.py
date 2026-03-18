"""Dummy pilot monitor tool (maps to your previous PilotMonitorAgent)."""
from __future__ import annotations
from typing import Any
from .base import text_content


class PandaPilotStatusTool:
    """Dummy tool that returns pilot counts and basic status for a given site.

    This placeholder is intended for development and testing. It returns a
    small text payload with simulated pilot metrics for the supplied site and
    lookback window.
    """

    @staticmethod
    def get_definition() -> dict[str, Any]:
        """Return the tool discovery definition.

        The returned mapping includes the tool `name`, a brief `description` and
        an `inputSchema` describing required and optional parameters.

        Returns:
            Dict[str, Any]: MCP-compatible tool discovery definition.
        """
        return {
            "name": "panda_pilot_status",
            "description": "Return pilot counts/failures for a site (dummy implementation).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "site": {"type": "string", "description": "Site name, e.g. BNL-ATLAS"},
                    "window_minutes": {"type": "integer", "description": "Lookback window in minutes", "default": 60},
                },
                "required": ["site"],
                "additionalProperties": False,
            },
        }

    async def call(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Return simulated pilot status for the requested site.

        Args:
            arguments: Mapping containing the `site` string and optional
                `window_minutes` integer.

        Returns:
            List[Dict[str, Any]]: A one-element MCP content list (text) with
            simulated pilot metrics.
        """
        site = arguments.get("site", "")
        window = int(arguments.get("window_minutes", 60))
        # Dummy numbers
        return text_content(
            f"Pilot status for {site} (dummy)\n"
            f"- window_minutes: {window}\n"
            f"- pilots_running: 128\n"
            f"- pilots_idle: 12\n"
            f"- pilots_failed: 3\n"
            "Replace with real Grafana/Harvester/PanDA monitor queries."
        )


panda_pilot_status_tool = PandaPilotStatusTool()
