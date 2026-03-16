"""ATLAS PanDA task status tool — askpanda_atlas plugin package.

Delegates to the canonical implementation in
``askpanda_atlas.task_status_impl``. Falls back to
``FallbackTaskStatusTool`` (from ``_fallback_tool``) when the plugin
implementation is not available.
"""
from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

try:
    from askpanda_atlas.task_status_impl import (  # noqa: F401
        PandaTaskStatusTool,
        get_definition,
        panda_task_status_tool,
    )
except ImportError:
    from askpanda_atlas._fallback_tool import (  # type: ignore[no-redef]  # noqa: F401,F811
        FallbackTaskStatusTool as PandaTaskStatusTool,
        get_definition,
    )
    panda_task_status_tool = PandaTaskStatusTool()  # type: ignore[assignment]  # noqa: F811

__all__ = ["PandaTaskStatusTool", "panda_task_status_tool", "get_definition"]
