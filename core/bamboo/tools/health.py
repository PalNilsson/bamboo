"""Healthcheck tool."""
from __future__ import annotations
from typing import Any
from bamboo.tools.base import text_content
from bamboo.config import Config


class HealthTool:
    """Provide a simple health/status reporting tool for Bamboo.

    This tool is intended for discovery by clients and quick operational
    checks. It returns a small text payload containing the server name,
    version, and flags for enabled integrations.
    """

    @staticmethod
    def get_definition() -> dict[str, Any]:
        """Return the tool discovery definition.

        The returned mapping includes the tool `name`, a short `description`,
        and an `inputSchema` describing expected inputs (none for this tool).

        Returns:
            Dict[str, Any]: Tool definition compatible with MCP discovery.
        """
        return {
            "name": "bamboo_health",
            "description": "Return server name/version and enabled integrations.",
            "inputSchema": {"type": "object", "properties": {}},
        }

    async def call(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute the health check and return a text payload.

        Args:
            arguments: JSON-like mapping of arguments supplied by the caller.
                This tool ignores arguments and always returns the server
                status text.

        Returns:
            List[Dict[str, Any]]: A one-element list containing the health
            status text block produced by ``text_content``.
        """
        # arguments intentionally unused; keep parameter for MCP compatibility
        del arguments  # pragma: no cover - explicit ignore

        return text_content(
            f"Bamboo MCP Server OK\n"
            f"- name: {Config.SERVER_NAME}\n"
            f"- version: {Config.SERVER_VERSION}\n"
            f"- ENABLE_REAL_PANDA: {Config.ENABLE_REAL_PANDA}\n"
            f"- ENABLE_REAL_LLM: {Config.ENABLE_REAL_LLM}"
        )


bamboo_health_tool = HealthTool()
