"""ATLAS PanDA task status tool — core implementation.

Fetches task metadata from BigPanDA and returns structured evidence
suitable for LLM summarisation.

Interface
---------
- ``panda_task_status_tool.get_definition()`` — MCP tool definition
- ``await panda_task_status_tool.call(arguments)`` — returns dict with
  ``evidence`` and ``text`` keys

Evidence keys
-------------
task_id, monitor_url, fetched_url, http_status, content_type,
status, superstatus, taskname, username, creationdate, starttime,
endtime, dsinfo, datasets_summary, job_counts, payload (full JSON).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from bamboo.tools._panda_http import (
    datasets_summary,
    fetch_jsonish,
    get_base_url,
    job_counts_from_payload,
)

logger = logging.getLogger(__name__)


def get_definition() -> dict[str, Any]:
    """Return the MCP tool definition for the task status tool.

    Returns:
        Dict with name, description, inputSchema, examples, and tags.
    """
    return {
        "name": "panda_task_status",
        "description": (
            "Fetch PanDA task metadata from BigPanDA and return structured "
            "evidence for LLM summarisation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer", "description": "PanDA task ID (jeditaskid)"},
                "query": {"type": "string", "description": "Original user query (optional)"},
                "include_jobs": {
                    "type": "boolean",
                    "description": "Include job records in the response (default true, adds ?jobs=1).",
                },
                "timeout": {"type": "integer", "description": "HTTP timeout in seconds (default 30)"},
            },
            "required": ["task_id"],
            "additionalProperties": True,
        },
        "examples": [{"task_id": 48432100, "query": "What happened to task 48432100?"}],
        "tags": ["atlas", "panda", "bigpanda", "monitoring"],
    }


class PandaTaskStatusTool:
    """MCP tool for fetching PanDA task status and metadata from BigPanDA."""

    def __init__(self) -> None:
        """Initialise with the tool definition."""
        self._def: dict[str, Any] = get_definition()

    def get_definition(self) -> dict[str, Any]:
        """Return the MCP tool definition.

        Returns:
            Tool definition dictionary.
        """
        return self._def

    async def call(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Fetch task status and return structured evidence.

        Args:
            arguments: Dict with required ``task_id`` and optional
                ``query``, ``include_jobs``, ``timeout``.

        Returns:
            Dict with ``evidence`` (structured metadata) and ``text``
            (short human-readable summary).
        """
        if not isinstance(arguments, dict):
            return {"evidence": {"error": "arguments must be a dict", "provided": repr(arguments)}}

        task_id = arguments.get("task_id")
        if task_id is None:
            return {"evidence": {"error": "missing task_id", "provided": arguments}}

        try:
            task_id_int = int(task_id)
        except Exception:  # pylint: disable=broad-exception-caught
            return {"evidence": {"error": "task_id must be an integer", "provided": arguments}}

        include_jobs: bool = bool(arguments.get("include_jobs", True))

        timeout: int = 30
        try:
            timeout = int(arguments.get("timeout") or 30)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        base_url = get_base_url()
        monitor_url = f"{base_url}/task/{task_id_int}/"
        json_url = f"{monitor_url}?json" + ("&jobs=1" if include_jobs else "")

        try:
            http_status, content_type, text, payload = await asyncio.to_thread(
                fetch_jsonish, json_url, timeout
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            return {
                "evidence": {
                    "task_id": task_id_int,
                    "monitor_url": monitor_url,
                    "fetched_url": json_url,
                    "error": repr(e),
                },
                "text": f"Failed to fetch task {task_id_int} metadata (network error).",
            }

        # Non-JSON or HTTP error
        if payload is None:
            snippet = (text or "").strip().replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:400] + "…"
            evidence: dict[str, Any] = {
                "task_id": task_id_int,
                "monitor_url": monitor_url,
                "fetched_url": json_url,
                "http_status": http_status,
                "content_type": content_type,
                "response_snippet": snippet,
            }
            if http_status == 404:
                evidence["not_found"] = True
                msg = f"Task {task_id_int} was not found in BigPanDA (HTTP 404)."
            elif http_status >= 400:
                msg = f"BigPanDA returned HTTP {http_status} when fetching task {task_id_int}."
            else:
                msg = f"BigPanDA returned a non-JSON response for task {task_id_int}."
            return {"evidence": evidence, "text": msg}

        # Extract common task fields
        task: dict[str, Any] = payload.get("task", {}) if isinstance(payload.get("task"), dict) else {}
        status = (task.get("status") if task else None) or payload.get("status")

        evidence = {
            "task_id": task_id_int,
            "monitor_url": monitor_url,
            "fetched_url": json_url,
            "http_status": http_status,
            "content_type": content_type,
            "status": status,
            "superstatus": task.get("superstatus") if task else None,
            "taskname": task.get("taskname") if task else None,
            "username": task.get("username") if task else None,
            "creationdate": task.get("creationdate") if task else None,
            "starttime": task.get("starttime") if task else None,
            "endtime": task.get("endtime") if task else None,
            "dsinfo": task.get("dsinfo") if task else None,
            "datasets_summary": datasets_summary(payload),
            "job_counts": job_counts_from_payload(payload),
            "payload": payload,
        }

        summary = (
            f"Task {task_id_int} status: {status}."
            if status
            else f"Task {task_id_int} metadata fetched."
        )
        return {"evidence": evidence, "text": summary}


panda_task_status_tool = PandaTaskStatusTool()

__all__ = ["PandaTaskStatusTool", "panda_task_status_tool", "get_definition"]
