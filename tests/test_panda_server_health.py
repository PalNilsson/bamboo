"""Tests for panda_server_health tool.

Covers:
- Successful alive response (plain string).
- Successful alive response (JSON).
- Server reports itself not alive.
- MCP caller returns an error (server not connected).
- Malformed / empty response.
- get_definition() schema compliance.
- _parse_alive() edge cases.
"""
import asyncio
import json
from unittest.mock import AsyncMock

from askpanda_atlas.panda_server_health import (
    PandaServerHealthTool,
    _parse_alive,
    get_definition,
    panda_server_health_tool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unpack(result: list) -> dict:
    """Deserialise the JSON-wrapped MCPContent returned by the tool.

    Args:
        result: Return value of tool.call().

    Returns:
        Deserialised dict with ``evidence`` and ``text`` keys.
    """
    return json.loads(result[0]["text"])


def _make_caller(text: str | None = None, error: str | None = None) -> AsyncMock:
    """Build a mock MCPCaller that returns a fixed response.

    Args:
        text: Text to return in the ``text`` field.
        error: Error string to return in the ``error`` field.

    Returns:
        AsyncMock with a ``call`` coroutine method.
    """
    mock = AsyncMock()
    mock.call = AsyncMock(return_value={"text": text, "error": error})
    return mock


# ---------------------------------------------------------------------------
# _parse_alive unit tests
# ---------------------------------------------------------------------------

def test_parse_alive_true_string() -> None:
    """Parse a plain 'True' string as alive."""
    assert _parse_alive("True") is True


def test_parse_alive_false_string() -> None:
    """Parse a plain 'false' string as not alive."""
    assert _parse_alive("false") is False


def test_parse_alive_json_true() -> None:
    """Parse a JSON boolean true as alive."""
    assert _parse_alive("true") is True


def test_parse_alive_json_object_alive_true() -> None:
    """Parse a JSON object with alive=true as alive."""
    assert _parse_alive('{"alive": true}') is True


def test_parse_alive_json_object_alive_false() -> None:
    """Parse a JSON object with alive=false as not alive."""
    assert _parse_alive('{"alive": false}') is False


def test_parse_alive_empty_string() -> None:
    """Return False for an empty response."""
    assert _parse_alive("") is False


def test_parse_alive_arbitrary_non_empty_string() -> None:
    """Any non-empty non-falsy plain string is treated as alive."""
    assert _parse_alive("OK") is True
    assert _parse_alive("Server is running fine") is True


def test_parse_alive_down_string() -> None:
    """Explicit 'down' string is treated as not alive."""
    assert _parse_alive("down") is False


# ---------------------------------------------------------------------------
# Tool integration tests (MCPCaller mocked)
# ---------------------------------------------------------------------------

def test_health_alive_plain_string(monkeypatch) -> None:
    """Successful plain-string alive response builds correct evidence."""
    monkeypatch.setattr(
        "bamboo.tools._mcp_caller._mcp_caller",
        _make_caller(text="True"),
    )
    result = asyncio.run(panda_server_health_tool.call({}))
    res = _unpack(result)
    assert res["evidence"]["is_alive"] is True
    assert res["evidence"]["error"] is None
    assert "alive" in res["text"].lower()


def test_health_alive_json_response(monkeypatch) -> None:
    """Successful JSON alive response is parsed correctly."""
    monkeypatch.setattr(
        "bamboo.tools._mcp_caller._mcp_caller",
        _make_caller(text='{"alive": true, "message": "PanDA is up"}'),
    )
    result = asyncio.run(panda_server_health_tool.call({"query": "Is PanDA OK?"}))
    res = _unpack(result)
    assert res["evidence"]["is_alive"] is True
    assert res["evidence"]["error"] is None


def test_health_not_alive_response(monkeypatch) -> None:
    """Server responding 'false' results in is_alive=False."""
    monkeypatch.setattr(
        "bamboo.tools._mcp_caller._mcp_caller",
        _make_caller(text="false"),
    )
    result = asyncio.run(panda_server_health_tool.call({}))
    res = _unpack(result)
    assert res["evidence"]["is_alive"] is False
    assert "not appear to be alive" in res["text"]


def test_health_mcp_error(monkeypatch) -> None:
    """Graceful handling when the MCP server is not connected."""
    monkeypatch.setattr(
        "bamboo.tools._mcp_caller._mcp_caller",
        _make_caller(error="Upstream MCP server 'panda' is not connected."),
    )
    result = asyncio.run(panda_server_health_tool.call({}))
    res = _unpack(result)
    assert res["evidence"]["is_alive"] is False
    assert "not connected" in res["evidence"]["error"]
    assert "Could not reach" in res["text"]


def test_health_empty_response(monkeypatch) -> None:
    """Empty text response is treated as not alive."""
    monkeypatch.setattr(
        "bamboo.tools._mcp_caller._mcp_caller",
        _make_caller(text=""),
    )
    result = asyncio.run(panda_server_health_tool.call({}))
    res = _unpack(result)
    assert res["evidence"]["is_alive"] is False


def test_health_none_text_no_error(monkeypatch) -> None:
    """None text with no error is treated as not alive (graceful)."""
    monkeypatch.setattr(
        "bamboo.tools._mcp_caller._mcp_caller",
        _make_caller(text=None, error=None),
    )
    result = asyncio.run(panda_server_health_tool.call({}))
    res = _unpack(result)
    assert res["evidence"]["is_alive"] is False


# ---------------------------------------------------------------------------
# Narrow-waist contract
# ---------------------------------------------------------------------------

def test_call_returns_list(monkeypatch) -> None:
    """call() must return a list (narrow-waist contract)."""
    monkeypatch.setattr(
        "bamboo.tools._mcp_caller._mcp_caller",
        _make_caller(text="True"),
    )
    result = asyncio.run(panda_server_health_tool.call({}))
    assert isinstance(result, list)
    assert len(result) == 1


def test_call_text_contains_evidence_key(monkeypatch) -> None:
    """The text field must be JSON with an 'evidence' key."""
    monkeypatch.setattr(
        "bamboo.tools._mcp_caller._mcp_caller",
        _make_caller(text="True"),
    )
    result = asyncio.run(panda_server_health_tool.call({}))
    parsed = json.loads(result[0]["text"])
    assert "evidence" in parsed


# ---------------------------------------------------------------------------
# get_definition() schema checks
# ---------------------------------------------------------------------------

def test_get_definition_name() -> None:
    """Definition must expose the correct tool name."""
    assert get_definition()["name"] == "panda_server_health"


def test_get_definition_has_description() -> None:
    """Definition must have a non-empty description."""
    assert get_definition()["description"]


def test_get_definition_schema_type() -> None:
    """inputSchema must be type=object."""
    assert get_definition()["inputSchema"]["type"] == "object"


def test_get_definition_additional_properties_false() -> None:
    """inputSchema must close additionalProperties."""
    assert get_definition()["inputSchema"]["additionalProperties"] is False


def test_get_definition_required_is_empty_list() -> None:
    """No required fields — tool takes no required arguments."""
    assert get_definition()["inputSchema"]["required"] == []


def test_get_definition_no_properties() -> None:
    """Tool has no properties — query is passed via arguments dict at call time."""
    assert get_definition()["inputSchema"]["properties"] == {}


def test_tool_singleton_is_instance() -> None:
    """The module-level singleton must be a PandaServerHealthTool."""
    assert isinstance(panda_server_health_tool, PandaServerHealthTool)
