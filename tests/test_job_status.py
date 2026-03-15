"""Tests for panda_job_status tool."""
import asyncio
import json
from unittest.mock import AsyncMock

from bamboo.tools.job_status import panda_job_status_tool


SAMPLE_JOB = {
    "pandaid": 6837798305,
    "jobstatus": "closed",
    "jobsubstatus": "toreassign",
    "jobname": "user.test/.6837798305",
    "produsername": "Test User",
    "computingsite": "ROMANIA07_HTCondor",
    "cloud": "FR",
    "atlasrelease": "Atlas-25.2.66",
    "transformation": "runGen-00-00-02",
    "jeditaskid": 46703290,
    "attemptnr": 1,
    "maxattempt": 3,
    "creationtime": "2025-10-09 16:49:45",
    "starttime": None,
    "endtime": "2025-10-10 14:10:23",
    "duration": "0:0:00:00",
    "commandtopilot": "tobekilled",
    "piloterrorcode": 0,
    "piloterrordiag": "",
    "taskbuffererrorcode": 100,
    "taskbuffererrordiag": "reassigned by JEDI",
    "file_summary_str": "input: 6, size: 9.48GB; output: 1; log: 1",
}

SAMPLE_FILES = [
    {"type": "input", "status": "ready", "lfn": "file1.root"},
    {"type": "output", "status": "failed", "lfn": "output.root"},
    {"type": "log", "status": "failed", "lfn": "log.tgz"},
]

SAMPLE_RESPONSE = f"Job 6837798305 metadata:\n\n{json.dumps({'job': SAMPLE_JOB, 'files': SAMPLE_FILES, 'dsfiles': []})}"


def _make_caller(text=None, error=None):
    mock = AsyncMock()
    mock.call = AsyncMock(return_value={"text": text, "error": error})
    return mock


def test_job_status_success(monkeypatch):
    """Test successful job metadata fetch and evidence extraction."""
    monkeypatch.setattr("bamboo.tools.job_status.get_mcp_caller", lambda: _make_caller(text=SAMPLE_RESPONSE))
    result = asyncio.run(panda_job_status_tool.call({"job_id": 6837798305}))
    ev = result["evidence"]
    assert ev["job_id"] == 6837798305
    assert ev["jobstatus"] == "closed"
    assert ev["computingsite"] == "ROMANIA07_HTCondor"
    assert ev["taskbuffererrordiag"] == "reassigned by JEDI"
    assert ev["jeditaskid"] == 46703290
    assert "reassigned by JEDI" in result["text"]


def test_job_status_files_summary(monkeypatch):
    """Test that files_summary correctly counts types and statuses."""
    monkeypatch.setattr("bamboo.tools.job_status.get_mcp_caller", lambda: _make_caller(text=SAMPLE_RESPONSE))
    result = asyncio.run(panda_job_status_tool.call({"job_id": 6837798305}))
    fs = result["evidence"]["files_summary"]
    assert fs["total"] == 3
    assert fs["by_type"]["input"] == 1
    assert fs["by_type"]["output"] == 1
    assert fs["by_status"]["failed"] == 2
    assert "output.root" in fs["failed_files"]


def test_job_status_mcp_error(monkeypatch):
    """Test graceful handling when MCP server returns an error."""
    monkeypatch.setattr("bamboo.tools.job_status.get_mcp_caller",
                        lambda: _make_caller(error="Server not connected"))
    result = asyncio.run(panda_job_status_tool.call({"job_id": 9999}))
    assert "error" in result["evidence"]
    assert "Server not connected" in result["text"]


def test_job_status_missing_job_id():
    """Test validation when job_id is missing."""
    result = asyncio.run(panda_job_status_tool.call({}))
    assert result["evidence"]["error"] == "missing job_id"


def test_job_status_invalid_job_id():
    """Test validation when job_id is not an integer."""
    result = asyncio.run(panda_job_status_tool.call({"job_id": "notanint"}))
    assert "integer" in result["evidence"]["error"]


def test_job_status_not_found(monkeypatch):
    """Test handling when job JSON has no 'job' key."""
    empty = "Job 9999 metadata:\n\n" + json.dumps({"files": [], "dsfiles": []})
    monkeypatch.setattr("bamboo.tools.job_status.get_mcp_caller", lambda: _make_caller(text=empty))
    result = asyncio.run(panda_job_status_tool.call({"job_id": 9999}))
    assert result["evidence"].get("not_found") is True


def test_get_definition():
    """Test that get_definition returns required MCP fields."""
    d = panda_job_status_tool.get_definition()
    assert d["name"] == "panda_job_status"
    assert "job_id" in d["inputSchema"]["properties"]
    assert d["inputSchema"]["required"] == ["job_id"]
