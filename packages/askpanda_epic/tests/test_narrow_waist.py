"""Architectural tests: MCP narrow-waist and schema contract compliance (askpanda_epic plugin).

Mirrors ``bamboo/tests/test_narrow_waist.py`` for the ePIC plugin.  The same
two contracts are verified against the ePIC tool set:

1. **Narrow-waist return type** — ``call()`` must return ``list[MCPContent]``,
   i.e. a non-empty list of ``{\"type\": ..., ...}`` dicts.

2. **Schema contract** — every ``get_definition()`` must declare a valid
   ``inputSchema`` with ``\"type\": \"object\"``.  All properties must carry
   ``\"type\"`` and ``\"description\"``.  Required fields must be declared, and
   ``\"additionalProperties\": False`` must be present (or ``anyOf`` used for
   tools with alternative required-field sets).

Tool singletons are imported directly to avoid pulling in ``bamboo.core``
(which imports ``mcp.server`` and requires the full MCP SDK at import time).
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from bamboo.tools.base import MCPContent  # noqa: F401
from askpanda_epic.task_status import panda_task_status_tool  # type: ignore[import]
from askpanda_epic.log_analysis import panda_log_analysis_tool  # type: ignore[import]
from askpanda_epic.doc_rag import epic_doc_search_tool  # type: ignore[import]
from askpanda_epic.doc_bm25 import epic_doc_bm25_tool  # type: ignore[import]
from askpanda_epic.ui_manifest import epic_ui_manifest_tool  # type: ignore[import]

# ---------------------------------------------------------------------------
# Tool registry — ePIC plugin tools only
# ---------------------------------------------------------------------------

TOOLS: dict[str, Any] = {
    "panda_task_status": panda_task_status_tool,
    "panda_log_analysis": panda_log_analysis_tool,
    "panda_doc_search": epic_doc_search_tool,
    "panda_doc_bm25": epic_doc_bm25_tool,
    "epic.ui_manifest": epic_ui_manifest_tool,
}

_TOOL_NAMES: list[str] = sorted(TOOLS.keys())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_mcp_content_list(value: Any) -> bool:
    """Return True if ``value`` is a valid ``list[MCPContent]``.

    A valid MCP content list is a non-empty list of dicts each containing at
    least a ``\"type\"`` key.  When ``type == \"text\"`` the ``\"text\"`` key must
    also be present.

    Args:
        value: The value returned by a tool's ``call()`` method.

    Returns:
        True if the value conforms to the narrow-waist contract.
    """
    if not isinstance(value, list) or len(value) == 0:
        return False
    for item in value:
        if not isinstance(item, dict):
            return False
        if "type" not in item:
            return False
        if item["type"] == "text" and "text" not in item:
            return False
    return True


# ---------------------------------------------------------------------------
# Minimal stub arguments — one valid input per tool
# ---------------------------------------------------------------------------

_STUB_ARGS: dict[str, dict[str, Any]] = {
    "panda_task_status": {"task_id": 1},
    "panda_log_analysis": {"job_id": 1},
    "panda_doc_search": {"query": "stub"},
    "panda_doc_bm25": {"query": "stub"},
    "epic.ui_manifest": {},
}

# ---------------------------------------------------------------------------
# Schema compliance tests — synchronous, no mocking needed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool_name", _TOOL_NAMES)
def test_tool_has_valid_schema(tool_name: str) -> None:
    """Every tool's inputSchema must declare type=object and well-typed properties.

    Args:
        tool_name: Name of the tool under test (injected by parametrize).
    """
    tool = TOOLS[tool_name]
    defn = tool.get_definition()

    assert isinstance(defn, dict), \
        f"{tool_name}: get_definition() must return a dict"
    assert defn.get("name"), \
        f"{tool_name}: definition must have a non-empty 'name'"
    assert defn.get("description"), \
        f"{tool_name}: definition must have a non-empty 'description'"

    schema = defn.get("inputSchema", {})
    assert isinstance(schema, dict), \
        f"{tool_name}: inputSchema must be a dict"
    assert schema.get("type") == "object", \
        f"{tool_name}: inputSchema must have type=object"

    props = schema.get("properties", {})
    for prop_name, prop_def in props.items():
        assert "type" in prop_def, \
            f"{tool_name}: property '{prop_name}' is missing 'type'"
        assert "description" in prop_def, \
            f"{tool_name}: property '{prop_name}' is missing 'description'"


@pytest.mark.parametrize("tool_name", _TOOL_NAMES)
def test_tool_schema_closes_additionalproperties(tool_name: str) -> None:
    """inputSchema must declare additionalProperties=False (or use anyOf).

    Args:
        tool_name: Name of the tool under test (injected by parametrize).
    """
    tool = TOOLS[tool_name]
    schema = tool.get_definition().get("inputSchema", {})

    has_additional = "additionalProperties" in schema
    has_any_of = "anyOf" in schema
    assert has_additional or has_any_of, (
        f"{tool_name}: inputSchema must declare 'additionalProperties' "
        f"(or use 'anyOf' for tools with alternative required-field sets)"
    )
    if has_additional:
        assert schema["additionalProperties"] is False, (
            f"{tool_name}: 'additionalProperties' must be False, "
            f"got {schema['additionalProperties']!r}"
        )


@pytest.mark.parametrize("tool_name", _TOOL_NAMES)
def test_tool_schema_declares_required_fields(tool_name: str) -> None:
    """Tools with properties must declare required fields or use anyOf.

    Args:
        tool_name: Name of the tool under test (injected by parametrize).
    """
    tool = TOOLS[tool_name]
    schema = tool.get_definition().get("inputSchema", {})
    props = schema.get("properties", {})

    if not props:
        return  # Zero-argument tools (ui_manifest) are exempt.

    has_required = bool(schema.get("required"))
    has_any_of = bool(schema.get("anyOf"))
    assert has_required or has_any_of, (
        f"{tool_name}: inputSchema has properties but no 'required' array "
        f"or 'anyOf' constraint"
    )


# ---------------------------------------------------------------------------
# Narrow-waist return type — async, mocks suppress all external I/O
# ---------------------------------------------------------------------------


def _build_patches() -> list[tuple[str, Any]]:
    """Build patch list to suppress all external I/O during call() tests.

    Returns:
        List of (dotted-target-string, mock-object) pairs.
    """
    fetch_mock = MagicMock(
        return_value=(200, "application/json", "{}", {"status": "done"})
    )

    return [
        # task_status fetches via the cache which calls _fallback_http.fetch_jsonish
        ("askpanda_epic._fallback_http.fetch_jsonish", fetch_mock),
        # log_analysis fetches metadata and logs directly
        ("askpanda_epic.log_analysis_impl._fetch_metadata",
         MagicMock(return_value={
             "job": {"jobstatus": "failed", "piloterrorcode": 0, "piloterrordiag": ""},
             "files": [], "dsfiles": [],
         })),
        ("askpanda_epic.log_analysis_impl._fetch_log_text", MagicMock(return_value=None)),
        # doc tools connect to ChromaDB — intercept at the client level
        ("askpanda_epic.doc_rag.EpicDocSearchTool._ensure_collection",
         MagicMock(return_value=None)),
        ("askpanda_epic.doc_rag.EpicDocSearchTool._collection",
         MagicMock(query=MagicMock(return_value={
             "documents": [["stub doc"]], "metadatas": [[{}]], "distances": [[0.1]],
         }))),
        ("askpanda_epic.doc_bm25.EpicDocBM25Tool._ensure_index",
         MagicMock(return_value=None)),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", _TOOL_NAMES)
async def test_tool_call_returns_mcp_content_list(tool_name: str) -> None:
    """Every tool's call() must return a valid list[MCPContent].

    This is the core narrow-waist architectural assertion.  A tool returning
    a bare dict instead of list[MCPContent] produces a malformed MCP response
    for any client invoking it directly over the protocol.

    Args:
        tool_name: Name of the tool under test (injected by parametrize).
    """
    tool = TOOLS[tool_name]
    args = _STUB_ARGS.get(tool_name, {})

    active: list[Any] = []
    for target, mock_obj in _build_patches():
        try:
            p = patch(target, mock_obj)
            p.start()
            active.append(p)
        except AttributeError:
            pass

    try:
        result = await tool.call(args)
    finally:
        for p in active:
            try:
                p.stop()
            except RuntimeError:
                pass

    assert _is_mcp_content_list(result), (
        f"Tool '{tool_name}' returned {type(result).__name__} instead of "
        f"list[MCPContent].\n"
        f"  Value: {result!r:.300}\n\n"
        f"  Fix: wrap the return value with text_content(json.dumps(payload))."
    )


# ---------------------------------------------------------------------------
# Evidence tools: JSON round-trip verification
# ---------------------------------------------------------------------------

_EVIDENCE_TOOLS = ["panda_task_status", "panda_log_analysis"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", _EVIDENCE_TOOLS)
async def test_evidence_tool_text_is_json_with_evidence_key(tool_name: str) -> None:
    """Evidence tools must encode their payload as JSON in the text field.

    The ``text`` field must be valid JSON containing at least an ``\"evidence\"``
    key so ``bamboo_answer._unpack_tool_result()`` can deserialise it.

    Args:
        tool_name: Name of the evidence tool under test.
    """
    tool = TOOLS[tool_name]
    args = _STUB_ARGS[tool_name]

    fetch_mock = MagicMock(
        return_value=(200, "application/json", "{}", {"status": "done"})
    )

    with (
        patch("askpanda_epic._fallback_http.fetch_jsonish", fetch_mock),
        patch("askpanda_epic.log_analysis_impl._fetch_metadata",
              return_value={
                  "job": {"jobstatus": "failed", "piloterrorcode": 0, "piloterrordiag": ""},
                  "files": [], "dsfiles": [],
              }),
        patch("askpanda_epic.log_analysis_impl._fetch_log_text", return_value=None),
    ):
        result = await tool.call(args)

    assert _is_mcp_content_list(result), \
        f"{tool_name}: call() must return list[MCPContent]"

    text = result[0].get("text", "")
    assert isinstance(text, str) and text.strip(), \
        f"{tool_name}: text field must be a non-empty string"

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"{tool_name}: text field is not valid JSON.\n"
            f"  Error: {exc}\n"
            f"  Text: {text[:300]!r}"
        )

    assert "evidence" in parsed, (
        f"{tool_name}: JSON payload must contain an 'evidence' key. "
        f"Got keys: {list(parsed)}"
    )
