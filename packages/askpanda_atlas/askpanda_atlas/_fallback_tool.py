"""Fallback PandaTaskStatusTool for standalone (no bamboo core) use.

Used only when ``askpanda_atlas.task_status_impl`` is not importable.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from ._fallback_http import (
    datasets_summary,
    fetch_jsonish,
    get_base_url,
    job_counts_from_payload,
)


def _text_content(data: dict) -> list[dict]:
    """Wrap a dict as a JSON-serialised MCP text content item.

    Args:
        data: Dict to serialise.

    Returns:
        One-element MCP content list.
    """
    return [{"type": "text", "text": json.dumps(data)}]


def get_definition() -> dict[str, Any]:
    """Return the MCP tool definition."""
    return {
        "name": "panda_task_status",
        "description": (
            "Get the current status, progress, and metadata of a PanDA task by its "
            "task ID (jeditaskid). Use when the question is about a specific task: "
            "its status, completion rate, dataset info, error counts, or list of jobs. "
            "For questions about an individual job, use panda_job_status instead."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer", "description": "PanDA task ID (jeditaskid)"},
                "query": {"type": "string", "description": "Original user query (optional)"},
                "include_jobs": {
                    "type": "boolean",
                    "description": "Include job records (default true)",
                },
                "timeout": {
                    "type": "integer",
                    "description": "HTTP timeout in seconds (default 30)",
                },
            },
            "required": ["task_id"],
            "additionalProperties": False,
        },
        "examples": [{"task_id": 48432100, "query": "What happened to task 48432100?"}],
        "tags": ["atlas", "panda", "bigpanda", "monitoring"],
    }


class FallbackTaskStatusTool:
    """Self-contained task status tool used when bamboo core is not installed."""

    def __init__(self) -> None:
        """Initialise with the tool definition."""
        self._def = get_definition()

    def get_definition(self) -> dict[str, Any]:
        """Return the MCP tool definition."""
        return self._def

    async def call(self, arguments: dict[str, Any]) -> list[dict]:
        """Fetch task status and return structured evidence.

        The result is a one-element list whose ``text`` field contains the
        JSON-serialised evidence dict, keeping the tool compliant with the
        MCP narrow-waist contract.

        Args:
            arguments: Dict with required ``task_id`` and optional fields.

        Returns:
            One-element MCP content list containing JSON-serialised evidence.
        """
        if not isinstance(arguments, dict):
            return _text_content({"evidence": {"error": "arguments must be a dict"}})
        task_id = arguments.get("task_id")
        if task_id is None:
            return _text_content({"evidence": {"error": "missing task_id"}})
        try:
            task_id_int = int(task_id)
        except Exception:  # pylint: disable=broad-exception-caught
            return _text_content({"evidence": {"error": "task_id must be an integer"}})

        include_jobs = bool(arguments.get("include_jobs", True))
        timeout = 30
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
            return _text_content({
                "evidence": {
                    "task_id": task_id_int,
                    "monitor_url": monitor_url,
                    "fetched_url": json_url,
                    "error": repr(e),
                },
                "text": f"Failed to fetch task {task_id_int} metadata (network error).",
            })

        if payload is None:
            return _text_content(
                self._non_json_response_dict(
                    task_id_int, monitor_url,
                    json_url, http_status, content_type, text,
                )
            )

        task = payload.get("task", {}) if isinstance(payload.get("task"), dict) else {}
        status = (task.get("status") if task else None) or payload.get("status")
        evidence: dict[str, Any] = {
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
        return _text_content({"evidence": evidence, "text": summary})

    @staticmethod
    def _non_json_response_dict(
        task_id_int: int,
        monitor_url: str,
        json_url: str,
        http_status: int,
        content_type: str,
        text: str,
    ) -> dict[str, Any]:
        """Build a structured error response for non-JSON / HTTP-error replies.

        Args:
            task_id_int: Numeric task ID.
            monitor_url: Human-facing BigPanDA monitor URL.
            json_url: The JSON API URL that was fetched.
            http_status: HTTP status code received.
            content_type: Content-Type header value.
            text: Raw response body text.

        Returns:
            Dict with ``evidence`` and ``text`` keys.
        """
        snippet = (text or "").strip().replace("\n", " ")[:400]
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
            msg = f"BigPanDA returned HTTP {http_status} for task {task_id_int}."
        else:
            msg = f"BigPanDA returned a non-JSON response for task {task_id_int}."
        return {"evidence": evidence, "text": msg}
