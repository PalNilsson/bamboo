"""Implementation of ``panda_task_status`` using the richer jobs endpoint.

Fetches ``GET /jobs/?jeditaskid={task_id}&json`` from BigPanDA, parses the
response with :mod:`askpanda_atlas.panda_task_schema`, and returns a compact
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


def _task_url(base_url: str, task_id: int) -> str:
    """Build the BigPanDA task endpoint URL for a given task ID.

    Args:
        base_url: BigPanDA base URL.
        task_id: JEDI task identifier.

    Returns:
        Full URL string ready for an HTTP GET request.
    """
    return f"{base_url.rstrip('/')}/task/{task_id}/?json"


def _trace(record: dict[str, Any]) -> None:
    """Write a debug record to the bamboo trace file if configured.

    Uses the same NDJSON format as bamboo.tracing so the record appears in
    the TUI's /tracing output.  This is the only way to surface debug info
    from plugin code when the server runs under BAMBOO_QUIET=1 (stdio TUI).

    Args:
        record: Dict of fields to include in the trace line.
    """
    trace_file = os.environ.get("BAMBOO_TRACE_FILE", "")
    if not trace_file or not os.environ.get("BAMBOO_TRACE"):
        return
    try:
        record["bamboo_trace"] = True
        import json as _json
        with open(trace_file, "a", encoding="utf-8") as fh:
            fh.write(_json.dumps(record, default=str) + "\n")
    except Exception:  # pylint: disable=broad-exception-caught
        pass


def _fetch_task_meta(base_url: str, task_id: int) -> dict[str, Any]:
    """Fetch task-level metadata from the BigPanDA task endpoint.

    Retrieves the ``/task/{id}/?json`` endpoint which returns task-level fields
    including ``status``, ``superstatus``, ``taskname``, ``username``,
    ``creationdate``, ``starttime``, and ``endtime``.  These are distinct from
    the job-level fields returned by the jobs endpoint.

    The task endpoint response can have status either nested under a ``"task"``
    key or at the top level — both shapes are handled.

    Failures are logged to the trace file and an empty dict is returned so the
    caller can proceed with job-level data alone.

    Args:
        base_url: BigPanDA base URL (no trailing slash).
        task_id: JEDI task identifier to query.

    Returns:
        Dict with task-level fields, or empty dict on failure.
    """
    url = _task_url(base_url, task_id)
    logger.debug("panda_task_status: fetching task meta %s", url)

    try:
        try:
            from askpanda_atlas._cache import cached_fetch_jsonish  # type: ignore[import]
            fetch_fn = cached_fetch_jsonish
        except ImportError:
            from askpanda_atlas._fallback_http import fetch_jsonish  # type: ignore[import]
            fetch_fn = fetch_jsonish  # type: ignore[assignment]

        status, _ctype, _body, payload = fetch_fn(url)
        if status < 200 or status >= 300 or payload is None:
            logger.warning(
                "panda_task_status: task meta HTTP %d for task %d", status, task_id
            )
            _trace({"event": "task_meta", "task_id": task_id,
                    "error": f"HTTP {status}", "url": url})
            return {}

        # Status is nested under "task" key or at the top level.
        task = payload.get("task", {}) if isinstance(payload.get("task"), dict) else {}
        task_status = (task.get("status") if task else None) or payload.get("status")
        task_superstatus = (
            (task.get("superstatus") if task else None) or payload.get("superstatus")
        )

        _trace({
            "event": "task_meta",
            "task_id": task_id,
            "task_status": task_status,
            "task_superstatus": task_superstatus,
            "payload_keys": list(payload.keys()),
            "task_keys": list(task.keys()) if task else [],
        })

        def _tf(key: str) -> Any:
            """Get field from task dict with fallback to top-level payload."""
            return (task.get(key) if task else None) or payload.get(key)

        return {
            "task_status": task_status,
            "task_superstatus": task_superstatus,
            "taskname": _tf("taskname"),
            "username": _tf("username"),
            "creationdate": _tf("creationdate"),
            "starttime": _tf("starttime"),
            "endtime": _tf("endtime"),
            "task_monitor_url": f"{base_url.rstrip('/')}/task/{task_id}/",
            # dsinfo contains authoritative file/event counts (nfilesfinished,
            # nfilesfailed, etc.) — more reliable than job-level aggregates.
            "dsinfo": _tf("dsinfo"),
            "pctfinished": _tf("pctfinished"),
            "totev": _tf("totev"),
            "totevproc": _tf("totevproc"),
        }
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(
            "panda_task_status: task meta failed for task %d: %s", task_id, exc
        )
        _trace({"event": "task_meta", "task_id": task_id, "error": repr(exc), "url": url})
        return {}


# ---------------------------------------------------------------------------
# Core synchronous fetch — no bamboo dependency
# ---------------------------------------------------------------------------


def fetch_and_analyse(base_url: str, task_id: int) -> dict[str, Any]:
    """Fetch task jobs and task-level metadata from BigPanDA.

    Makes two HTTP requests:

    1. ``GET /jobs/?jeditaskid={id}&json`` — full job list for per-status /
       per-site / per-error-code counts via
       :func:`~askpanda_atlas.panda_task_schema.build_evidence`.
    2. ``GET /task/{id}/?json`` — task-level metadata including the definitive
       ``status`` and ``superstatus`` fields (e.g. ``"finished"``) regardless
       of individual job failures.  Best-effort: failure still returns job data.

    Synchronous; call via ``asyncio.to_thread()`` from async contexts.

    Args:
        base_url: BigPanDA base URL (no trailing slash).
        task_id: JEDI task identifier to query.

    Returns:
        Evidence dictionary with both task-level and job-level fields.

    Raises:
        RuntimeError: If the jobs HTTP response is non-2xx or not a JSON dict.
    """
    from askpanda_atlas._cache import cached_fetch_jsonish  # type: ignore[import]
    from askpanda_atlas.panda_task_schema import (  # type: ignore[import]
        PandaTaskData,
        build_evidence,
    )

    url = _jobs_url(base_url, task_id)
    logger.debug("panda_task_status: fetching jobs %s", url)

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

    # Fetch task-level status (best-effort — failures return empty dict).
    task_meta = _fetch_task_meta(base_url, task_id)
    if task_meta:
        evidence.update(task_meta)

    # raw_payload is NOT sent to the LLM — stripped by BambooLastEvidenceTool.
    evidence["raw_payload"] = payload

    logger.debug(
        "panda_task_status: task_id=%d total_jobs=%d task_status=%s",
        task_id,
        evidence.get("total_jobs", 0),
        evidence.get("task_status", "unknown"),
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
