"""ATLAS PanDA task status tool implementation.

Canonical async implementation extracted from legacy code for fetching PanDA
task metadata from BigPanDA and returning structured evidence for LLM summarization.

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
from typing import Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


def _default_base_url() -> str:
    """Get the best-effort base URL with safe fallback.

    Attempts to import base URL from tools.https module, falling back to
    the default CERN BigPanDA URL if import fails.

    Returns:
        Base URL for BigPanDA API.
    """
    try:
        from tools.https import get_base_url  # type: ignore
        return get_base_url()
    except Exception:
        return "https://bigpanda.cern.ch"


def _fetch_jsonish(url: str, timeout: int = 30) -> Tuple[int, str, str, Optional[dict]]:
    """Fetch a URL expected to return JSON, handling non-JSON responses robustly.

    Handles HTML, empty bodies, redirects, and HTTP errors gracefully without
    raising exceptions, allowing callers to process structured error responses.

    Args:
        url: URL to fetch.
        timeout: HTTP request timeout in seconds. Defaults to 30.

    Returns:
        Tuple of (status_code, content_type, response_text, json_dict_or_none).
        json_dict_or_none is None if response is not JSON or on HTTP errors.
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
    except Exception:
        return status, ctype, text, None


def _job_counts_from_payload(payload: dict) -> Dict[str, int]:
    """Extract and count job statuses from payload.

    Searches for jobs list in payload and counts occurrences of each job status.
    Handles multiple common key names (jobs, jobList, joblist).

    Args:
        payload: Payload dict potentially containing jobs list.

    Returns:
        Dictionary mapping job status strings to counts. Empty dict if no jobs found.
    """
    jobs = None
    for key in ("jobs", "jobList", "joblist"):
        if isinstance(payload.get(key), list):
            jobs = payload.get(key)
            break
    if not jobs:
        return {}
    statuses: list[str] = []
    for j in jobs:
        if not isinstance(j, dict):
            continue
        s = j.get("jobStatus") or j.get("status") or j.get("job_status")
        if isinstance(s, str) and s:
            statuses.append(s)
    return dict(Counter(statuses))


def _datasets_summary(payload: dict) -> dict:
    """Summarize dataset status and file counts from payload.

    Processes datasets list to extract status counts, file statistics, and
    identifies problematic datasets with failures.

    Args:
        payload: Payload dict potentially containing datasets list.

    Returns:
        Dictionary with summary stats including:
        - dataset_count: Total number of datasets
        - status_counts: Counter of dataset statuses
        - nfiles*_total: Aggregated file counts by status
        - worst_datasets: List of up to 5 datasets with failures, sorted by severity
    """
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        return {}

    status_counts: Counter[str] = Counter()
    nfilesfailed_total: int = 0
    nfilesfinished_total: int = 0
    nfileswaiting_total: int = 0
    nfilesmissing_total: int = 0

    worst: list[dict] = []  # keep a few problematic datasets
    for ds in datasets:
        if not isinstance(ds, dict):
            continue
        st = ds.get("status") or ""
        if isinstance(st, str) and st:
            status_counts[st] += 1
        for k, acc in (
            ("nfilesfailed", "nfilesfailed_total"),
            ("nfilesfinished", "nfilesfinished_total"),
            ("nfileswaiting", "nfileswaiting_total"),
            ("nfilesmissing", "nfilesmissing_total"),
        ):
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
    """Get the MCP tool definition for task status.

    Returns:
        Dictionary containing tool name, description, inputSchema with task_id
        requirement, examples, and tags for tool registry.
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
    """MCP tool for fetching PanDA task status and metadata from BigPanDA.

    Provides robust async interface for retrieving task information with
    structured evidence output optimized for LLM summarization.
    """

    def __init__(self) -> None:
        """Initialize the tool with its definition."""
        self._def: dict = get_definition()

    def get_definition(self) -> dict:
        """Get the MCP tool definition.

        Returns:
            Tool definition dictionary.
        """
        return self._def

    async def call(self, arguments: dict) -> dict:
        """Fetch task status and return structured evidence.

        Args:
            arguments: Tool input dict with required 'task_id' key and optional
                'query', 'include_jobs', and 'timeout' keys.

        Returns:
            Dictionary with 'evidence' (structured metadata) and 'text' (summary) keys.
        """
        if not isinstance(arguments, dict):
            return {"evidence": {"error": "arguments must be a dict", "provided": repr(arguments)}}

        task_id = arguments.get("task_id")
        if task_id is None:
            return {"evidence": {"error": "missing task_id", "provided": arguments}}

        try:
            task_id_int: int = int(task_id)
        except Exception:
            return {"evidence": {"error": "task_id must be an integer", "provided": arguments}}

        include_jobs: bool = bool(arguments.get("include_jobs", True))

        timeout: int = arguments.get("timeout") or 30
        try:
            timeout = int(timeout)
        except Exception:
            timeout = 30

        base_url: str = _default_base_url().rstrip("/")
        monitor_url: str = f"{base_url}/task/{task_id_int}/"
        json_url: str = f"{monitor_url}?json"
        if include_jobs:
            json_url = f"{monitor_url}?json&jobs=1"

        try:
            http_status: int
            content_type: str
            text: str
            payload: Optional[dict]
            http_status, content_type, text, payload = await asyncio.to_thread(_fetch_jsonish, json_url, timeout)
        except Exception as e:
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
            snippet: str = (text or "").strip().replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:400] + "…"
            evidence: dict = {
                "task_id": task_id_int,
                "monitor_url": monitor_url,
                "fetched_url": json_url,
                "http_status": http_status,
                "content_type": content_type,
                "response_snippet": snippet,
            }
            if http_status == 404:
                evidence["not_found"] = True
                msg: str = f"Task {task_id_int} was not found in BigPanDA (HTTP 404)."
            elif http_status >= 400:
                msg = f"BigPanDA returned HTTP {http_status} when fetching task {task_id_int}."
            else:
                msg = f"BigPanDA returned a non-JSON response for task {task_id_int}."
            return {"evidence": evidence, "text": msg}

        # Extract common task fields
        task: dict = payload.get("task", {}) if isinstance(payload.get("task"), dict) else {}
        status: Optional[str] = (task.get("status") if isinstance(task, dict) else None) or payload.get("status")
        superstatus: Optional[str] = task.get("superstatus") if isinstance(task, dict) else None
        taskname: Optional[str] = task.get("taskname") if isinstance(task, dict) else None
        username: Optional[str] = task.get("username") if isinstance(task, dict) else None
        creationdate: Optional[str] = task.get("creationdate") if isinstance(task, dict) else None
        starttime: Optional[str] = task.get("starttime") if isinstance(task, dict) else None
        endtime: Optional[str] = task.get("endtime") if isinstance(task, dict) else None
        dsinfo: Optional[str] = task.get("dsinfo") if isinstance(task, dict) else None

        datasets_summary: dict = _datasets_summary(payload) or {}
        job_counts: Dict[str, int] = _job_counts_from_payload(payload) or {}

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
        summary: str = f"Task {task_id_int} status: {status}." if status else f"Task {task_id_int} metadata fetched."
        return {"evidence": evidence, "text": summary}


panda_task_status_tool = _Tool()
