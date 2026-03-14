"""
ATLAS UI manifest tool.

Returns UI branding metadata used by Bamboo clients (Textual / Streamlit / etc).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

from bamboo.tools.base import text_content


def _load_banner_lines() -> list[str]:
    """Load ASCII banner lines from banner.txt shipped with this plugin."""
    # Prefer banner.txt in the same Python package as this module.
    try:
        pkg = __package__ or __name__.rpartition(".")[0]
        banner_path = resources.files(pkg).joinpath("banner.txt")
        txt = banner_path.read_text(encoding="utf-8")
        lines = [ln.rstrip("\n") for ln in txt.splitlines()]
        return [ln for ln in lines if ln.strip() != ""]
    except Exception:
        # Hard fallback (safe, no escape warnings)
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
        """Return MCP tool definition."""
        return {
            "name": "atlas.ui_manifest",
            "description": "Return UI branding metadata (banner, display name, help text, accent).",
            "inputSchema": {"type": "object", "properties": {}},
        }

    async def call(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the manifest payload as JSON text."""
        _ = arguments  # unused

        payload = {
            "plugin_id": "atlas",
            "display_name": "AskPanDA – ATLAS",
            "help": "Enter to send • /help • /plugin <id> • /tools • /debug on|off",
            "banner": _load_banner_lines(),
            "accent": "cyan",
        }
        return text_content(json.dumps(payload, ensure_ascii=False))


atlas_ui_manifest_tool = AtlasUiManifestTool()
