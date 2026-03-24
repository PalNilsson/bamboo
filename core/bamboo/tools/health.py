"""Healthcheck tool."""
from __future__ import annotations
from typing import Any
from bamboo.tools.base import text_content
from bamboo.config import Config


def _llm_info() -> str:
    """Return a short LLM provider/model string for the health response.

    Returns:
        ``"provider=<p> model=<m>"`` or ``"not configured"`` on any error.
    """
    try:
        from bamboo.tools.llm_passthrough import get_llm_info  # pylint: disable=import-outside-toplevel
        info = get_llm_info()
        return info if info else "not configured"
    except Exception:  # pylint: disable=broad-exception-caught
        return "not configured"


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
            "description": (
                "Check that the Bamboo MCP server is running and return its "
                "version, configured LLM provider, and which optional "
                "integrations (RAG, tracing) are enabled. Use for diagnostics "
                "or to confirm the server is reachable."
            ),
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
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

        # Instantiate Config() so env vars are read at call time, not import time.
        try:
            cfg: Config | type[Config] = Config()  # type: ignore[call-arg]
        except TypeError:
            cfg = Config  # type: ignore[assignment]

        return text_content(
            f"Bamboo MCP Server OK\n"
            f"- name: {cfg.SERVER_NAME}\n"
            f"- version: {cfg.SERVER_VERSION}\n"
            f"- ENABLE_REAL_PANDA: {cfg.ENABLE_REAL_PANDA}\n"
            f"- ENABLE_REAL_LLM: {cfg.ENABLE_REAL_LLM}\n"
            f"- llm_info: {_llm_info()}"
        )


bamboo_health_tool = HealthTool()
