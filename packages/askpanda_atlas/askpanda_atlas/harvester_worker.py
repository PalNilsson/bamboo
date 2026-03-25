"""Harvester worker (pilot) stats tool — askpanda_atlas plugin package.

Delegates to the canonical implementation in
``askpanda_atlas.harvester_worker_impl``.  The fallback path (ImportError)
is reserved for environments where ``requests`` is not installed; in that
case the tool is not registered and Bamboo falls back gracefully to the LLM
planner which will route to documentation search.
"""
from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

try:
    from askpanda_atlas.harvester_worker_impl import (  # noqa: F401
        PandaHarvesterWorkersTool,
        get_definition,
        panda_harvester_workers_tool,
    )
except ImportError as _exc:
    _logger.warning(
        "panda_harvester_workers tool unavailable (missing dependency): %s", _exc
    )
    raise

__all__ = [
    "PandaHarvesterWorkersTool",
    "get_definition",
    "panda_harvester_workers_tool",
]
