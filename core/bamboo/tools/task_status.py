"""PanDA task status tool — public re-export for core.py and tests.

The canonical implementation lives in ``bamboo.tools.task_status_atlas``.
This module simply re-exports it so the rest of the codebase can use the
stable name ``bamboo.tools.task_status`` without caring about which
ATLAS-specific file holds the logic.
"""
from __future__ import annotations

from .task_status_atlas import (  # noqa: F401  (re-export)
    PandaTaskStatusTool,
    get_definition,
    panda_task_status_tool,
)

__all__ = ["PandaTaskStatusTool", "panda_task_status_tool", "get_definition"]
