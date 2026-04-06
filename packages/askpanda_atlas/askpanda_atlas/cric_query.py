"""CRIC NL-to-SQL query tool — askpanda_atlas plugin package.

Delegates to the canonical implementation in
``askpanda_atlas.cric_query_impl``.  The fallback path (ImportError) is
reserved for environments where ``duckdb`` or ``sqlglot`` are not installed;
in that case the tool is not registered and Bamboo falls back gracefully to
the LLM planner which will route to documentation search.
"""
from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

try:
    from askpanda_atlas.cric_query_impl import (  # noqa: F401
        CricQueryTool,
        get_definition,
        cric_query_tool,
    )
except ImportError as _exc:
    _logger.warning(
        "cric_query tool unavailable (missing dependency): %s", _exc
    )
    raise

__all__ = ["CricQueryTool", "cric_query_tool", "get_definition"]
