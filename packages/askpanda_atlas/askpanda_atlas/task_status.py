"""ATLAS PanDA task status tool implementation.

ATLAS PanDA task status tool implementation (canonical, async) extracted from legacy code.

Key features:
- Robust handling of non-JSON responses (HTML, empty, redirects, 404).
- Returns structured `evidence` optimized for LLM summarization:
    - task_id, monitor_url, fetched_url, http_status, content_type
    - status, superstatus, taskname, username, creation/end/start
    - dsinfo (if present)
    - datasets summary (counts, failures)
    - job_counts if jobs list is present (optional)
    - payload (full JSON) for deep inspection (can be truncated upstream)

Interface:
- panda_task_status_tool.get_definition() includes `inputSchema` (required by Streamlit tool registry)
- await panda_task_status_tool.call(arguments) returns dict with `evidence` and `text`
"""
from __future__ import annotations

import asyncio
import logging
from collections import Counter

import requests

logger = logging.getLogger(__name__)


def _default_base_url() -> str:
    """Get a base URL with safe fallbacks.

    Prefer a core helper if present; otherwise return the default BigPanDA URL.

    Returns:
        Base URL string for BigPanDA.
    """
    # Prefer core helper if present (Bamboo core / AskPanDA core)
    for modpath in ("bamboo.tools.https", "bamboo_core.tools.https", "tools.https"):
        try:
            m = __import__(modpath, fromlist=["get_base_url"])
            get_base_url = getattr(m, "get_base_url", None)
            if callable(get_base_url):
                return str(get_base_url())
        except Exception:  # pylint: disable=broad-exception-caught
            continue
    return "https://bigpanda.cern.ch"


def _fetch_jsonish(url: str, timeout: int = 30) -> tuple[int, str, str, dict | None]:
    """Fetch a URL expected to return JSON, handling non-JSON responses.

    Args:
        url: URL to fetch.
        timeout: HTTP request timeout in seconds.

    Returns:
        Tuple of (status_code, content_type, response_text, json_dict_or_none).
    """
    resp = requests.get(
        url,
        timeout=timeout,
        headers={
            "Accept": "application/json, text/json;q=0.9, */*;q=0.1",
            "User-Agent": "AskPanDA/1.0",
        },
        allow_redirects=True,
    )
    status = resp.status_code
    ctype = (resp.headers.get("content-type") or "").lower()
    text = resp.text or ""

    # If HTTP error, don't raise; caller returns structured evidence.
    if status < 200 or status >= 300:
        return status, ctype, text, None

    try:
        data = resp.json()
        if isinstance(data, dict):
            return status, ctype, text, data
        return status, ctype, text, {"_data": data}
    except Exception:  # pylint: disable=broad-exception-caught
        return status, ctype, text, None


def _job_counts_from_payload(payload: dict) -> dict[str, int]:
    """Count job statuses in the payload.

    Args:
        payload: Payload dict potentially containing job lists.

    Returns:
        Mapping of job status to count.
    """
    jobs = None
    for key in ("jobs", "jobList", "joblist"):
        if isinstance(payload.get(key), list):
            jobs = payload.get(key)
            break
    if not jobs:
        return {}
    statuses = []
    for j in jobs:
        if not isinstance(j, dict):
            continue
        s = j.get("jobStatus") or j.get("status") or j.get("job_status")
        if isinstance(s, str) and s:
            statuses.append(s)
    return dict(Counter(statuses))


def _datasets_summary(payload: dict) -> dict:
    """Summarize dataset statuses and file counts in the payload.

    Args:
        payload: Payload dict potentially containing datasets.

    Returns:
        Summary dict with counts and a small list of problematic datasets.
    """
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        return {}

    status_counts = Counter()
    nfilesfailed_total = 0
    nfilesfinished_total = 0
    nfileswaiting_total = 0
    nfilesmissing_total = 0

    worst = []  # keep a few problematic datasets
    for ds in datasets:
        if not isinstance(ds, dict):
            continue
        st = ds.get("status") or ""
        if isinstance(st, str) and st:
            status_counts[st] += 1
        for k in ("nfilesfailed", "nfilesfinished", "nfileswaiting", "nfilesmissing"):
            v = ds.get(k)
            if isinstance(v, int):
                if k == "nfilesfailed":
                    nfilesfailed_total += v
                elif k == "nfilesfinished":
                    nfilesfinished_total += v
                elif k == "nfileswaiting":
                    nfileswaiting_total += v
                elif k == "nfilesmissing":
                    nfilesmissing_total += v

        # capture datasets with failures
        nff = ds.get("nfilesfailed")
        if isinstance(nff, int) and nff > 0:
            worst.append({
                "datasetname": ds.get("datasetname"),
                "containername": ds.get("containername"),
                "type": ds.get("type"),
                "streamname": ds.get("streamname"),
                "status": ds.get("status"),
                "nfilesfailed": nff,
                "nfiles": ds.get("nfiles"),
            })

    worst = sorted(worst, key=lambda x: (x.get("nfilesfailed") or 0), reverse=True)[:5]

    return {
        "dataset_count": len(datasets),
        "status_counts": dict(status_counts),
        "nfilesfailed_total": nfilesfailed_total,
        "nfilesfinished_total": nfilesfinished_total,
        "nfileswaiting_total": nfileswaiting_total,
        "nfilesmissing_total": nfilesmissing_total,
        "worst_datasets": worst,
    }


def get_definition() -> dict:
    """Get the MCP tool definition.

    Returns:
        Dictionary describing the tool name, schema, examples, and tags.
    """
    return {
        "name": "task_status",
        "description": "Fetch PanDA task metadata from BigPanDA and return structured evidence for summarization.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer", "description": "PanDA task ID (jeditaskid)"},
                "query": {"type": "string", "description": "Original user query (optional)"},
                "include_jobs": {"type": "boolean", "description": "Attempt to include job records (default true)"},
                "timeout": {"type": "integer", "description": "HTTP timeout seconds (default 30)"},
            },
            "required": ["task_id"],
            "additionalProperties": True,
        },
        "examples": [{"task_id": 48432100, "query": "What happened to task 48432100?"}],
        "tags": ["atlas", "panda", "bigpanda", "monitoring"],
    }


class _Tool:
    """Tool wrapper exposing MCP-compatible definition and call interface."""

    def __init__(self):
        """Initialize the tool definition cache."""
        self._def = get_definition()

    def get_definition(self) -> dict:
        """Get the MCP tool definition.

        Returns:
            Tool definition dictionary.
        """
        return self._def

    async def call(self, arguments: dict) -> dict:
        """Fetch task status and return structured evidence.

        Args:
            arguments: Tool input dict with required 'task_id' and optional fields.

        Returns:
            Dict with 'evidence' and optional 'text' summary.
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

        include_jobs = arguments.get("include_jobs")
        if include_jobs is None:
            include_jobs = True

        timeout = arguments.get("timeout") or 30
        try:
            timeout = int(timeout)
        except Exception:  # pylint: disable=broad-exception-caught
            timeout = 30

        base_url = _default_base_url().rstrip("/")
        monitor_url = f"{base_url}/task/{task_id_int}/"
        json_url = f"{monitor_url}?json"
        if include_jobs:
            json_url = f"{monitor_url}?json&jobs=1"

        try:
            http_status, content_type, text, payload = await asyncio.to_thread(_fetch_jsonish, json_url, timeout)
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
            evidence = {
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
        task = payload.get("task") if isinstance(payload.get("task"), dict) else {}
        status = (task.get("status") if isinstance(task, dict) else None) or payload.get("status")
        superstatus = task.get("superstatus") if isinstance(task, dict) else None
        taskname = task.get("taskname") if isinstance(task, dict) else None
        username = task.get("username") if isinstance(task, dict) else None
        creationdate = task.get("creationdate") if isinstance(task, dict) else None
        starttime = task.get("starttime") if isinstance(task, dict) else None
        endtime = task.get("endtime") if isinstance(task, dict) else None
        dsinfo = task.get("dsinfo") if isinstance(task, dict) else None

        datasets_summary = _datasets_summary(payload)
        job_counts = _job_counts_from_payload(payload)

        evidence = {
            "task_id": task_id_int,
            "monitor_url": monitor_url,
            "fetched_url": json_url,
            "http_status": http_status,
            "content_type": content_type,
            "status": status,
            "superstatus": superstatus,
            "taskname": taskname,
            "username": username,
            "creationdate": creationdate,
            "starttime": starttime,
            "endtime": endtime,
            "dsinfo": dsinfo,
            "datasets_summary": datasets_summary,
            "job_counts": job_counts,
            "payload": payload,
        }

        # concise text (optional)
        summary = f"Task {task_id_int} status: {status}." if status else f"Task {task_id_int} metadata fetched."
        return {"evidence": evidence, "text": summary}


panda_task_status_tool = _Tool()
