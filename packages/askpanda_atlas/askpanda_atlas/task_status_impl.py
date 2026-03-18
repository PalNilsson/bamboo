"""ATLAS PanDA task status tool — canonical implementation.

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

import json

import asyncio
import logging
from typing import Any

from askpanda_atlas._fallback_http import (
    datasets_summary,
    fetch_jsonish,
    get_base_url,
    job_counts_from_payload,
)
from bamboo.tools.base import MCPContent, text_content

logger = logging.getLogger(__name__)


def get_definition() -> dict[str, Any]:
    """Return the MCP tool definition for the task status tool.

    Returns:
        A dict with ``name``, ``description``, ``inputSchema``,
        ``examples``, and ``tags`` keys.
    """
    return {
        "name": "panda_task_status",
        "description": (
            "Get the current status, progress, and metadata of a PanDA task "
            "by its task ID (jeditaskid). Use when the question is about a "
            "specific task: its status, completion rate, dataset info, error "
            "counts, or list of jobs. For individual job questions, use "
            "panda_job_status instead."
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
            "additionalProperties": False,
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

    async def call(self, arguments: dict[str, Any]) -> list[MCPContent]:
        """Fetch task status and return structured evidence.

        The result is a one-element ``list[MCPContent]`` whose ``text`` field
        contains the JSON-serialised evidence dict.  Callers that need the raw
        evidence should parse ``json.loads(result[0]["text"])``.  This keeps
        the tool compliant with the MCP narrow-waist contract.

        Args:
            arguments: Dict with required ``task_id`` (int) and optional
                ``query`` (str), ``include_jobs`` (bool), ``timeout`` (int).

        Returns:
            One-element MCP content list containing the JSON-serialised
            evidence and text summary.
        """
        if not isinstance(arguments, dict):
            return text_content(json.dumps({"evidence": {"error": "arguments must be a dict", "provided": repr(arguments)}}))

        task_id = arguments.get("task_id")
        if task_id is None:
            return text_content(json.dumps({"evidence": {"error": "missing task_id", "provided": str(arguments)}}))

        try:
            task_id_int = int(task_id)
        except Exception:  # pylint: disable=broad-exception-caught
            return text_content(json.dumps({"evidence": {"error": "task_id must be an integer", "provided": str(arguments)}}))

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
            return text_content(json.dumps({
                "evidence": {
                    "task_id": task_id_int,
                    "monitor_url": monitor_url,
                    "fetched_url": json_url,
                    "error": repr(e),
                },
                "text": f"Failed to fetch task {task_id_int} metadata (network error).",
            }))

        # Non-JSON or HTTP error
        if payload is None:
            snippet = (text or "").strip().replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:400] + "…"
            # BigPanDA returns HTTP 200 with an HTML page for unknown tasks
            # rather than a 404, so detect the not-found condition from the body.
            snippet_lower = snippet.lower()
            html_not_found = (
                "text/html" in content_type and
                any(marker in snippet_lower for marker in (
                    "not found", "does not exist", "no such task",
                    "task not found", "unknown task",
                ))
            )
            evidence: dict[str, Any] = {
                "task_id": task_id_int,
                "monitor_url": monitor_url,
                "fetched_url": json_url,
                "http_status": http_status,
                "content_type": content_type,
                "response_snippet": snippet,
            }
            if http_status == 404 or html_not_found:
                evidence["not_found"] = True
                msg = f"Task {task_id_int} was not found in BigPanDA."
            elif http_status >= 400:
                msg = f"BigPanDA returned HTTP {http_status} when fetching task {task_id_int}."
            else:
                msg = f"BigPanDA returned a non-JSON response for task {task_id_int}."
            return text_content(json.dumps({"evidence": evidence, "text": msg}))

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
        return text_content(json.dumps({"evidence": evidence, "text": summary}))


panda_task_status_tool = PandaTaskStatusTool()

__all__ = ["PandaTaskStatusTool", "panda_task_status_tool", "get_definition"]
