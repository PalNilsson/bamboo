"""PanDA task status tool.

This MCP tool fetches task metadata from the PanDA monitor JSON endpoints and returns
a concise summary (status, job counts, recent errors).

The tool is intended to be selected when the user prompt contains the substring
"task <integer>" (e.g. "what is the status of task 123?").
"""

from __future__ import annotations

import os
import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import httpx

from .base import text_content


_TASK_ID_RE = re.compile(r"\btask\s+(\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class TaskSummary:
    """A normalized task summary extracted from PanDA metadata."""

    task_id: int
    status: str
    job_counts: Mapping[str, int]
    error_codes: Tuple[int, ...]
    error_diags: Tuple[str, ...]
    monitor_url: str


def _panda_base_url() -> str:
    """Returns the PanDA monitor base URL.

    You can override the default using the environment variable `PANDA_BASE_URL`.

    Returns:
        Base URL as a string.
    """
    return os.getenv("PANDA_BASE_URL", "https://bigpanda.cern.ch").rstrip("/")


def extract_task_id_from_text(text: str) -> Optional[int]:
    """Extracts a task ID from a user prompt.

    Args:
        text: Free-form user prompt.

    Returns:
        The extracted task ID, or None if not found.
    """
    m = _TASK_ID_RE.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


async def _fetch_task_metadata(task_id: int, *, timeout_s: float = 30.0) -> Dict[str, Any]:
    """Fetch task metadata JSON from PanDA monitor.

    Uses the endpoint:
        {PANDA_BASE_URL}/task/<task_id>/?json

    Args:
        task_id: PanDA task ID.
        timeout_s: HTTP timeout in seconds.

    Returns:
        Parsed JSON as a dictionary.

    Raises:
        httpx.HTTPError: For network/HTTP errors.
        ValueError: If the response is not JSON/dict-like.
    """
    base = _panda_base_url()
    url = f"{base}/task/{task_id}/?json"

    # Retry a couple of times for transient issues.
    retries = int(os.getenv("ASKPANDA_PANDA_RETRIES", "2"))
    backoff = float(os.getenv("ASKPANDA_PANDA_BACKOFF_SECONDS", "0.8"))

    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, dict):
                    raise ValueError(f"Unexpected JSON type: {type(data)}")
                return data
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= retries:
                raise
            # simple exponential backoff
            import asyncio
            await asyncio.sleep(backoff * (2 ** attempt))

    assert last_exc is not None
    raise last_exc  # pragma: no cover


def _summarize_task(task_id: int, payload: Mapping[str, Any]) -> TaskSummary:
    """Builds a compact TaskSummary from PanDA task metadata JSON.

    Args:
        task_id: PanDA task ID.
        payload: The JSON payload returned by the monitor.

    Returns:
        TaskSummary.
    """
    base = _panda_base_url()
    monitor_url = f"{base}/task/{task_id}/"

    # Best-effort: field names can vary a bit across monitor versions
    status = str(payload.get("taskstatus") or payload.get("status") or "unknown")

    job_counts: Counter[str] = Counter()
    error_codes: deque[int] = deque(maxlen=50)
    error_diags: deque[str] = deque(maxlen=20)

    jobs = payload.get("jobs")
    if isinstance(jobs, list):
        for job in jobs:
            if not isinstance(job, dict):
                continue
            js = str(job.get("jobstatus") or job.get("status") or "unknown")
            job_counts[js] += 1

            # Collect a small sample of errors for quick diagnosis.
            for k, v in job.items():
                lk = k.lower()
                if "errorcode" in lk:
                    try:
                        iv = int(v)
                        if iv > 0:
                            error_codes.append(iv)
                    except Exception:
                        pass
                if "errordiag" in lk and isinstance(v, str) and v.strip():
                    error_diags.append(v.strip())

    # De-duplicate diags while preserving order (first occurrences)
    seen = set()
    uniq_diags = []
    for d in error_diags:
        if d in seen:
            continue
        seen.add(d)
        uniq_diags.append(d)

    return TaskSummary(
        task_id=task_id,
        status=status,
        job_counts=dict(job_counts),
        error_codes=tuple(error_codes),
        error_diags=tuple(uniq_diags),
        monitor_url=monitor_url,
    )


class PandaTaskStatusTool:
    """MCP tool that returns a summary for a PanDA task.

    Input:
        - task_id (int) OR
        - question (str) containing "task <integer>"

    Output:
        A human-readable summary plus a monitor URL.
    """

    @staticmethod
    def get_definition() -> Dict[str, Any]:
        """Returns the MCP tool definition."""
        return {
            "name": "panda_task_status",
            "description": (
                "Fetch PanDA task metadata and summarize status, job counts, and errors. "
                "Selection hint: the user prompt should include 'task <integer>'."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer", "description": "PanDA task ID"},
                    "question": {
                        "type": "string",
                        "description": "User prompt containing 'task <integer>' (used to extract task_id)",
                    },
                    "include_raw": {
                        "type": "boolean",
                        "description": "If true, append the raw JSON payload.",
                        "default": False,
                    },
                },
                "anyOf": [{"required": ["task_id"]}, {"required": ["question"]}],
            },
        }

    async def call(self, arguments: Dict[str, Any]) -> Any:
        """Executes the tool.

        Args:
            arguments: Tool arguments.

        Returns:
            MCP text content.
        """
        task_id = arguments.get("task_id")
        question = arguments.get("question") or ""
        include_raw = bool(arguments.get("include_raw", False))

        if task_id is None:
            task_id = extract_task_id_from_text(str(question))
        if task_id is None:
            return text_content(
                "Error: could not determine task id. Please include 'task <integer>' in the prompt "
                "or pass task_id explicitly."
            )

        try:
            payload = await _fetch_task_metadata(int(task_id))
        except httpx.HTTPStatusError as exc:
            return text_content(
                f"Error: PanDA monitor returned HTTP {exc.response.status_code} for task {task_id}.\n"
                f"URL: {_panda_base_url()}/task/{task_id}/?json"
            )
        except Exception as exc:  # noqa: BLE001
            return text_content(f"Error: failed to fetch task metadata for task {task_id}: {exc}")

        summary = _summarize_task(int(task_id), payload)

        lines = []
        lines.append(f"Task {summary.task_id}")
        lines.append(f"Status: {summary.status}")
        lines.append(f"Monitor: {summary.monitor_url}")
        lines.append("")
        if summary.job_counts:
            lines.append("Job counts:")
            for k, v in sorted(summary.job_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"- {k}: {v}")
        else:
            lines.append("Job counts: (not available in metadata)")
        lines.append("")

        if summary.error_codes:
            lines.append("Recent error codes (sample):")
            counts = Counter(summary.error_codes)
            for code, cnt in counts.most_common(10):
                lines.append(f"- {code}: {cnt}")
            lines.append("")
        if summary.error_diags:
            lines.append("Recent error diagnostics (sample):")
            for d in summary.error_diags[:10]:
                # Keep output readable
                d1 = d.replace("\n", " ").strip()
                if len(d1) > 400:
                    d1 = d1[:400] + "..."
                lines.append(f"- {d1}")
            lines.append("")
        if include_raw:
            import json
            lines.append("Raw payload:\n")
            lines.append(json.dumps(payload, indent=2)[:200000])  # guard against extreme size

        return text_content("\n".join(lines))


panda_task_status_tool = PandaTaskStatusTool()
