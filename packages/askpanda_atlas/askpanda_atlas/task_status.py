"""ATLAS PanDA task status tool — askpanda_atlas plugin package.

Delegates to the canonical implementation in ``bamboo.tools.task_status_atlas``
when bamboo core is installed.  Falls back to ``FallbackTaskStatusTool`` (from
``_fallback_tool``) when core is not available, e.g. during standalone testing.
"""
from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

_core_available = False
try:
    from bamboo.tools.task_status_atlas import (  # noqa: F401
        PandaTaskStatusTool,
        get_definition,
        panda_task_status_tool,
    )
    _core_available = True
except ImportError:
    pass

if not _core_available:
    from askpanda_atlas._fallback_tool import (  # type: ignore[no-redef]  # noqa: F401,F811
        FallbackTaskStatusTool as PandaTaskStatusTool,
        get_definition,
    )
    panda_task_status_tool = PandaTaskStatusTool()  # type: ignore[assignment]  # noqa: F811

__all__ = ["PandaTaskStatusTool", "panda_task_status_tool", "get_definition"]
