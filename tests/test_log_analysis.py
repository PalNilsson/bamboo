"""Tests for panda_log_analysis tool."""
import asyncio
import json
from unittest.mock import AsyncMock

from bamboo.tools.log_analysis import panda_log_analysis_tool, _classify_failure, _parse_response


SAMPLE_JOB = {
    "pandaid": 6837798305,
    "jobstatus": "closed",
    "jobsubstatus": "toreassign",
    "computingsite": "ROMANIA07_HTCondor",
    "cloud": "FR",
    "atlasrelease": "Atlas-25.2.66",
    "jeditaskid": 46703290,
    "attemptnr": 1,
    "maxattempt": 3,
    "commandtopilot": "tobekilled",
    "piloterrorcode": 0,
    "piloterrordiag": "",
    "taskbuffererrorcode": 100,
    "taskbuffererrordiag": "reassigned by JEDI",
    "exeerrorcode": 0,
    "exeerrordiag": "",
    "starttime": None,
    "endtime": "2025-10-10 14:10:23",
    "duration": "0:0:00:00",
}

_JSON_BLOCK = json.dumps({"job": SAMPLE_JOB, "files": [], "dsfiles": []})
SAMPLE_RESPONSE = (
    "=== BIGPANDA JOB FAILURE ANALYSIS FOR JOB 6837798305 ===\n\n"
    f"Job 6837798305 metadata:\n\n{_JSON_BLOCK}\n\n"
    "============================================================\n\n"
    "❌ Pilot log download failed: Error downloading pilot log for job 6837798305 (exit code: 2)"
)

SAMPLE_RESPONSE_WITH_LOG = (
    "=== BIGPANDA JOB FAILURE ANALYSIS FOR JOB 1234 ===\n\n"
    f"Job 1234 metadata:\n\n{json.dumps({'job': {**SAMPLE_JOB, 'pandaid': 1234, 'jobsubstatus': 'failed',
                                                 'taskbuffererrordiag': '', 'commandtopilot': '', 'piloterrordiag': 'Segmentation fault'},
                                         'files': [], 'dsfiles': []})}\n\n"
    "============================================================\n\n"
    "pilot log content here: Segmentation fault in AthenaMP"
)


def _make_caller(text=None, error=None):
    mock = AsyncMock()
    mock.call = AsyncMock(return_value={"text": text, "error": error})
    return mock


def test_parse_response_with_pilot_log_error():
    """Test _parse_response separates JSON from pilot log error message."""
    job, pilot_log = _parse_response(SAMPLE_RESPONSE)
    assert job is not None
    assert job["pandaid"] == 6837798305
    assert "❌" in pilot_log


def test_parse_response_with_pilot_log_content():
    """Test _parse_response extracts pilot log when present."""
    job, pilot_log = _parse_response(SAMPLE_RESPONSE_WITH_LOG)
    assert job is not None
    assert "Segmentation fault" in pilot_log


def test_classify_failure_reassigned():
    """Test classification of reassigned-by-JEDI jobs."""
    result = _classify_failure(SAMPLE_JOB, "")
    assert result == "reassigned_by_jedi"


def test_classify_failure_segfault():
    """Test classification of segfault from pilot log."""
    job = {**SAMPLE_JOB, "taskbuffererrordiag": "", "commandtopilot": "", "jobsubstatus": "failed"}
    result = _classify_failure(job, "Segmentation fault in payload")
    assert result == "segfault"


def test_classify_failure_unknown():
    """Test fallback classification for unrecognised errors."""
    job = {k: "" for k in SAMPLE_JOB}
    result = _classify_failure(job, "some unrecognised error message")
    assert result == "unknown"


def test_log_analysis_success(monkeypatch):
    """Test successful log analysis with pilot log error (realistic case)."""
    monkeypatch.setattr("bamboo.tools.log_analysis.get_mcp_caller", lambda: _make_caller(text=SAMPLE_RESPONSE))
    result = asyncio.run(panda_log_analysis_tool.call({"job_id": 6837798305}))
    ev = result["evidence"]
    assert ev["job_id"] == 6837798305
    assert ev["failure_classification"] == "reassigned_by_jedi"
    assert ev["taskbuffererrordiag"] == "reassigned by JEDI"
    assert ev["pilot_log_available"] is False
    assert "reassigned by JEDI" in result["text"]


def test_log_analysis_with_real_log(monkeypatch):
    """Test analysis when pilot log is successfully downloaded."""
    monkeypatch.setattr("bamboo.tools.log_analysis.get_mcp_caller",
                        lambda: _make_caller(text=SAMPLE_RESPONSE_WITH_LOG))
    result = asyncio.run(panda_log_analysis_tool.call({"job_id": 1234}))
    ev = result["evidence"]
    assert ev["pilot_log_available"] is True
    assert ev["pilot_log_excerpt"] is not None
    assert "Segmentation fault" in ev["pilot_log_excerpt"]
    assert ev["failure_classification"] == "segfault"


def test_log_analysis_mcp_error(monkeypatch):
    """Test graceful handling when MCP server is not connected."""
    monkeypatch.setattr("bamboo.tools.log_analysis.get_mcp_caller",
                        lambda: _make_caller(error="bigpanda-downloader not connected"))
    result = asyncio.run(panda_log_analysis_tool.call({"job_id": 9999}))
    assert "error" in result["evidence"]
    assert "not connected" in result["text"]


def test_log_analysis_missing_job_id():
    """Test validation when job_id is missing."""
    result = asyncio.run(panda_log_analysis_tool.call({}))
    assert result["evidence"]["error"] == "missing job_id"


def test_get_definition():
    """Test that get_definition returns required MCP fields."""
    d = panda_log_analysis_tool.get_definition()
    assert d["name"] == "panda_log_analysis"
    assert "job_id" in d["inputSchema"]["properties"]
    assert d["inputSchema"]["required"] == ["job_id"]
