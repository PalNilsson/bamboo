"""Implementation of ``panda_task_status`` using the richer jobs endpoint.

Fetches ``GET /jobs/?jeditaskid={task_id}&json`` from BigPanDA, parses the
response with :mod:`askpanda_epic.panda_task_schema`, and returns a compact
evidence dict that is safe to embed in an LLM synthesis prompt.

Public surface (mirrors ``log_analysis_impl`` pattern exactly):

- ``get_definition()`` — MCP tool definition dict
- ``PandaTaskStatusTool`` — MCP tool class with ``get_definition()`` and
  async ``call()``
- ``panda_task_status_tool`` — singleton instance

Design rules (per Bamboo architecture):

* All ``bamboo.tools.base`` imports are **deferred inside ``call()``** —
  never at module level.  This keeps every pure helper importable without
  bamboo installed, which is required by the fallback shim.
* Blocking HTTP is wrapped in ``asyncio.to_thread()``.
* ``call()`` never raises — errors are returned as ``text_content`` payloads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

_JOBS_PATH: str = "/jobs/"


def _jobs_url(base_url: str, task_id: int) -> str:
    """Build the BigPanDA jobs endpoint URL for a given task ID.

    Args:
        base_url: BigPanDA base URL, e.g. ``"https://bigpanda.cern.ch"``.
        task_id: JEDI task identifier.

    Returns:
        Full URL string ready for an HTTP GET request.
    """
    return f"{base_url.rstrip('/')}{_JOBS_PATH}?jeditaskid={task_id}&json"


# ---------------------------------------------------------------------------
# Core synchronous fetch — no bamboo dependency
# ---------------------------------------------------------------------------


def fetch_and_analyse(base_url: str, task_id: int) -> dict[str, Any]:
    """Fetch task jobs from BigPanDA and return a compact evidence dict.

    Synchronous; call via ``asyncio.to_thread()`` from async contexts.
    Imports from :mod:`askpanda_epic` locally so the function remains
    usable from fallback stubs that lack bamboo core.

    ``fetch_jsonish`` returns a 4-tuple
    ``(status_code, content_type, body_text, parsed_json_or_none)``.
    Non-2xx responses and non-JSON bodies are treated as errors.

    Args:
        base_url: BigPanDA base URL (no trailing slash).
        task_id: JEDI task identifier to query.

    Returns:
        Evidence dictionary produced by
        :func:`~askpanda_epic.panda_task_schema.build_evidence`, augmented
        with ``"task_id"`` and ``"endpoint"`` keys.

    Raises:
        RuntimeError: If the HTTP response is non-2xx or not a JSON dict.
    """
    from askpanda_epic._cache import cached_fetch_jsonish  # type: ignore[import]
    from askpanda_epic.panda_task_schema import (  # type: ignore[import]
        PandaTaskData,
        build_evidence,
    )

    url = _jobs_url(base_url, task_id)
    logger.debug("panda_task_status: fetching %s", url)

    status, ctype, body, payload = cached_fetch_jsonish(url)

    if status < 200 or status >= 300:
        raise RuntimeError(
            f"HTTP {status} fetching task {task_id} from {url}"
        )
    if payload is None:
        snippet = body[:200] if body else ""
        raise RuntimeError(
            f"Non-JSON response (content-type={ctype!r}) for task {task_id}: {snippet!r}"
        )

    task = PandaTaskData(payload)
    evidence = build_evidence(task)
    evidence["task_id"] = task_id
    evidence["endpoint"] = url
    # Store the verbatim BigPanDA response alongside the evidence so the
    # bamboo_last_evidence tool can serve it via /json without re-fetching.
    # raw_payload is intentionally NOT sent to the LLM — it is stripped by
    # BambooLastEvidenceTool.call() when mode='evidence'.
    evidence["raw_payload"] = payload

    logger.debug(
        "panda_task_status: task_id=%d total_jobs=%d",
        task_id,
        evidence.get("total_jobs", 0),
    )
    return evidence


# ---------------------------------------------------------------------------
# MCP tool definition
# ---------------------------------------------------------------------------


def get_definition() -> dict[str, Any]:
    """Return the MCP tool definition for ``panda_task_status``.

    Returns:
        Tool definition dict compatible with MCP discovery.
    """
    return {
        "name": "panda_task_status",
        "description": (
            "Fetch full job-level detail for a PanDA task from BigPanDA. "
            "Returns a task summary, per-status / per-site / per-error-code "
            "job counts, a sample of failed and finished jobs with error "
            "diagnostics, and (for small tasks) the complete list of PanDA "
            "job IDs. Use this tool when the user asks about a task, its "
            "overall status, which jobs failed, or which sites are involved."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "The JEDI task ID (jeditaskid) to query.",
                },
                "query": {
                    "type": "string",
                    "description": "Original user question (used by synthesiser).",
                },
                "include_jobs": {
                    "type": "boolean",
                    "description": "Unused; accepted for planner compatibility.",
                },
            },
            "required": ["task_id"],
            "additionalProperties": False,
        },
    }


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------


class PandaTaskStatusTool:
    """MCP tool that fetches and summarises a PanDA task from BigPanDA.

    Uses the richer ``GET /jobs/?jeditaskid={id}&json`` endpoint and
    returns a compact evidence dict structured for LLM summarisation.
    """

    def __init__(self) -> None:
        """Initialise with the cached tool definition."""
        self._def: dict[str, Any] = get_definition()

    def get_definition(self) -> dict[str, Any]:
        """Return the MCP tool definition.

        Returns:
            Tool definition dictionary.
        """
        return self._def

    async def call(self, arguments: dict[str, Any]) -> list[Any]:
        """Fetch task data and return structured evidence as MCP content.

        ``bamboo.tools.base`` is imported here (deferred) so the rest of
        this module remains importable when bamboo core is not installed.
        All blocking HTTP is offloaded via ``asyncio.to_thread``.

        Args:
            arguments: Dict with required ``"task_id"`` (int) and optional
                ``"query"`` (str) and ``"include_jobs"`` (bool).

        Returns:
            One-element MCP content list containing the JSON-serialised
            evidence dict, or an error payload if anything goes wrong.
        """
        from bamboo.tools.base import text_content  # deferred — see module docstring

        base_url: str = os.environ.get("PANDA_BASE_URL", "https://bigpanda.cern.ch")

        task_id_raw = arguments.get("task_id")
        if task_id_raw is None:
            return text_content(json.dumps({
                "evidence": {"error": "task_id argument is required."},
            }))

        try:
            task_id = int(task_id_raw)
        except (ValueError, TypeError):
            return text_content(json.dumps({
                "evidence": {
                    "error": f"task_id must be an integer, got {task_id_raw!r}.",
                },
            }))

        try:
            evidence = await asyncio.to_thread(fetch_and_analyse, base_url, task_id)
            return text_content(json.dumps({"evidence": evidence}))
        except Exception as exc:  # noqa: BLE001
            logger.exception("panda_task_status failed for task_id=%d", task_id)
            return text_content(json.dumps({
                "evidence": {
                    "task_id": task_id,
                    "error": repr(exc),
                },
                "text": f"Error fetching task {task_id}: {exc}",
            }))


panda_task_status_tool = PandaTaskStatusTool()

__all__ = [
    "PandaTaskStatusTool",
    "fetch_and_analyse",
    "get_definition",
    "panda_task_status_tool",
]
