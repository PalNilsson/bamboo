"""ATLAS UI manifest tool.

Returns UI branding metadata used by Bamboo clients (Textual / Streamlit / etc).

This module deliberately avoids importing from ``bamboo.tools.base`` so it
can be loaded standalone (e.g. during testing of just the plugin package).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any, Sequence


def _text_content(text: str) -> list[dict[str, Any]]:
    """Create a minimal MCP text content payload.

    This is a local copy of the ``text_content`` helper from
    ``bamboo.tools.base`` so the plugin package does not have a hard
    import dependency on bamboo core.

    Args:
        text: Response text.

    Returns:
        One-element list compatible with the MCP content format.
    """
    return [{"type": "text", "text": text}]


def _load_banner_lines() -> Sequence[str]:
    """Load ASCII banner lines from banner.txt shipped with this plugin."""
    try:
        pkg = __package__ or __name__.rpartition(".")[0]
        banner_path = resources.files(pkg).joinpath("banner.txt")
        txt = banner_path.read_text(encoding="utf-8")
        lines = [ln.rstrip("\n") for ln in txt.splitlines()]
        return [ln for ln in lines if ln.strip() != ""]
    except Exception:  # pylint: disable=broad-exception-caught
        return (
            r"""
    _        _     ____            ____     _
   / \   ___| | __|  _ \ ___ _ __ |  _ \   / \
  / _ \ / __| |/ /  |_) /_` | '_ \| | | | / _ \
 / ___ \\__ \   <|  __/ (_| | | | | |_| |/ ___ \
/_/   \_\___/_|\_\_|   \__,_|_| |_|____//_/   \_\
""".strip("\n").splitlines()
        )


@dataclass(frozen=True)
class AtlasUiManifestTool:
    """Tool that returns UI metadata for ATLAS / AskPanDA."""

    @staticmethod
    def get_definition() -> dict[str, Any]:
        """Return MCP tool definition.

        Returns:
            Tool definition dict.
        """
        return {
            "name": "atlas.ui_manifest",
            "description": "Return UI branding metadata (banner, display name, help text, accent).",
            "inputSchema": {"type": "object", "properties": {}},
        }

    async def call(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the manifest payload as JSON text.

        Args:
            arguments: Unused; present for MCP interface compatibility.

        Returns:
            One-element MCP content list with JSON-encoded manifest.
        """
        _ = arguments  # unused

        payload = {
            "plugin_id": "atlas",
            "display_name": "AskPanDA \u2013 ATLAS",
            "help": "Enter to send \u2022 /help \u2022 /plugin <id> \u2022 /tools \u2022 /debug on|off",
            "banner": _load_banner_lines(),
            "accent": "cyan",
        }
        return _text_content(json.dumps(payload, ensure_ascii=False))


atlas_ui_manifest_tool = AtlasUiManifestTool()

__all__ = ["AtlasUiManifestTool", "atlas_ui_manifest_tool"]
