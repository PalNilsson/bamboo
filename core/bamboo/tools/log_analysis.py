"""PanDA job log analysis tool.

Downloads job metadata and pilot logs from BigPanDA via the
``bigpanda-downloader`` MCP server and returns a structured failure
analysis suitable for LLM summarisation.

The upstream ``analyze_bigpanda_job_failure`` tool returns the same JSON
metadata as ``download_bigpanda_metadata`` plus a pilot log section
separated by a ``===...===`` divider.  This tool parses both parts and
extracts key failure signals from the ``job`` dict and log text.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from bamboo.tools._mcp_caller import get_mcp_caller  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_SERVER = "bigpanda-downloader"
_TOOL = "analyze_bigpanda_job_failure"

# Error categories matched against combined job fields + log text
_FAILURE_PATTERNS: list[tuple[str, list[str]]] = [
    ("reassigned_by_jedi", ["reassigned by jedi", "toreassign"]),
    ("timeout", ["timeout", "timed out", "walltime", "cpu time exceeded", "tobekilled"]),
    ("segfault", ["segmentation fault", "sigsegv", "signal 11"]),
    ("disk_full", ["no space left", "disk quota", "disk full"]),
    ("memory", ["out of memory", "oom killer", "memory limit"]),
    ("network", ["connection refused", "network unreachable", "dns failure", "socket"]),
    ("input_missing", ["no such file", "file not found", "input file missing"]),
    ("pilot_error", ["piloterrorcode"]),
    ("software_error", ["exception", "traceback", "abort", "core dump"]),
]

_DIVIDER_RE = re.compile(r"={10,}")


def get_definition() -> dict[str, Any]:
    """Return the MCP tool definition for panda_log_analysis.

    Returns:
        Dict with name, description, inputSchema, examples, and tags.
    """
    return {
        "name": "panda_log_analysis",
        "description": (
            "Download and analyse a PanDA job failure from BigPanDA. "
            "Returns structured evidence including job metadata, error codes, "
            "pilot log excerpt, and failure classification."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "integer",
                    "description": "PanDA job ID (pandaid) to analyse.",
                },
                "query": {
                    "type": "string",
                    "description": "Original user query (optional).",
                },
                "context": {
                    "type": "string",
                    "description": "Optional context (site, task ID, release, etc.).",
                },
            },
            "required": ["job_id"],
            "additionalProperties": False,
        },
        "examples": [
            {"job_id": 6837798305, "query": "Why did job 6837798305 fail?"},
        ],
        "tags": ["atlas", "panda", "bigpanda", "job", "log", "failure", "diagnosis"],
    }


def _classify_failure(job: dict[str, Any], log_text: str) -> str:
    """Classify a job failure from job fields and log text.

    Args:
        job: The ``job`` dict from the BigPanDA response.
        log_text: Pilot log text (may be empty).

    Returns:
        A short failure category string.
    """
    # Build a combined search string from key error fields + log text
    search = " ".join([
        str(job.get("taskbuffererrordiag") or ""),
        str(job.get("piloterrordiag") or ""),
        str(job.get("exeerrordiag") or ""),
        str(job.get("jobsubstatus") or ""),
        str(job.get("commandtopilot") or ""),
        log_text,
    ]).lower()

    for category, keywords in _FAILURE_PATTERNS:
        if any(kw in search for kw in keywords):
            return category
    return "unknown"


def _parse_response(raw_text: str) -> tuple[dict[str, Any] | None, str]:
    """Split the analyze response into a job dict and pilot log text.

    The response format is::

        === BIGPANDA JOB FAILURE ANALYSIS FOR JOB NNN ===
        Job NNN metadata:
        { ... json ... }
        ===...===
        <pilot log or error message>

    Args:
        raw_text: Raw text from the MCP server.

    Returns:
        Tuple of (job_dict_or_None, pilot_log_text).
    """
    json_start = raw_text.find("{")
    if json_start < 0:
        return None, raw_text

    # Find the divider after the JSON block
    parts = _DIVIDER_RE.split(raw_text[json_start:], maxsplit=1)
    json_str = parts[0].strip()
    pilot_log = parts[1].strip() if len(parts) > 1 else ""

    try:
        data: dict[str, Any] = json.loads(json_str)
        return data.get("job"), pilot_log
    except json.JSONDecodeError:
        return None, raw_text


class PandaLogAnalysisTool:
    """MCP tool for downloading and analysing PanDA job failure logs."""

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
        """Download logs and return structured failure analysis.

        Args:
            arguments: Dict with required ``job_id`` and optional
                ``query`` and ``context``.

        Returns:
            Dict with ``evidence`` (structured analysis) and ``text``
            (human-readable summary).
        """
        if not isinstance(arguments, dict):
            return {"evidence": {"error": "arguments must be a dict", "provided": repr(arguments)}}

        job_id = arguments.get("job_id")
        if job_id is None:
            return {"evidence": {"error": "missing job_id", "provided": arguments}}

        try:
            job_id_int = int(job_id)
        except Exception:  # pylint: disable=broad-exception-caught
            return {"evidence": {"error": "job_id must be an integer", "provided": arguments}}

        context: str = str(arguments.get("context", "") or "")
        monitor_url = f"https://bigpanda.cern.ch/job?pandaid={job_id_int}"
        base_evidence: dict[str, Any] = {
            "job_id": job_id_int,
            "monitor_url": monitor_url,
            "context": context or None,
        }

        caller = get_mcp_caller()
        result = await caller.call(
            server_name=_SERVER,
            tool_name=_TOOL,
            arguments={"panda_id": str(job_id_int)},
        )

        if result["error"]:
            base_evidence["error"] = result["error"]
            return {
                "evidence": base_evidence,
                "text": f"Failed to analyse job {job_id_int}: {result['error']}",
            }

        raw_text: str = result["text"] or ""
        job, pilot_log = _parse_response(raw_text)

        if job is None:
            base_evidence["not_found"] = True
            base_evidence["raw"] = raw_text[:500]
            return {
                "evidence": base_evidence,
                "text": (
                    f"Could not retrieve analysis for job {job_id_int}. "
                    "The job ID may not exist."
                ),
            }

        classification = _classify_failure(job, pilot_log)
        pilot_log_failed = "❌" in pilot_log or "failed" in pilot_log.lower()

        evidence: dict[str, Any] = {
            **base_evidence,
            "failure_classification": classification,
            "jobstatus": job.get("jobstatus"),
            "jobsubstatus": job.get("jobsubstatus"),
            "computingsite": job.get("computingsite"),
            "cloud": job.get("cloud"),
            "atlasrelease": job.get("atlasrelease"),
            "jeditaskid": job.get("jeditaskid"),
            "attemptnr": job.get("attemptnr"),
            "maxattempt": job.get("maxattempt"),
            "commandtopilot": job.get("commandtopilot"),
            "piloterrorcode": job.get("piloterrorcode"),
            "piloterrordiag": job.get("piloterrordiag"),
            "exeerrorcode": job.get("exeerrorcode"),
            "exeerrordiag": job.get("exeerrordiag"),
            "taskbuffererrorcode": job.get("taskbuffererrorcode"),
            "taskbuffererrordiag": job.get("taskbuffererrordiag"),
            "ddmerrorcode": job.get("ddmerrorcode"),
            "ddmerrordiag": job.get("ddmerrordiag"),
            "starttime": job.get("starttime"),
            "endtime": job.get("endtime"),
            "duration": job.get("duration"),
            "pilot_log_available": not pilot_log_failed,
            "pilot_log_excerpt": pilot_log[:3000] if not pilot_log_failed else None,
            "pilot_log_error": pilot_log if pilot_log_failed else None,
        }

        summary = (
            f"Job {job_id_int} failure analysis complete. "
            f"Classification: {classification}."
        )
        if job.get("taskbuffererrordiag"):
            summary += f" Task buffer: {job['taskbuffererrordiag']}."
        if job.get("piloterrordiag"):
            summary += f" Pilot: {job['piloterrordiag']}."
        return {"evidence": evidence, "text": summary}


panda_log_analysis_tool = PandaLogAnalysisTool()

__all__ = ["PandaLogAnalysisTool", "panda_log_analysis_tool", "get_definition"]
