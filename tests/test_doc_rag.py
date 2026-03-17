"""Tests for the ChromaDB-backed PandaDocSearchTool.

All tests monkeypatch the ``chromadb`` import so the real package is not
required at test time.  Each test resets the tool singleton's cached state
before running so tests are fully independent of each other.
"""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from bamboo.tools.doc_rag import PandaDocSearchTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chroma_module(collection: Any) -> types.ModuleType:
    """Build a minimal fake ``chromadb`` module with a ``PersistentClient``.

    Args:
        collection: Object to return from ``client.get_collection()``.

    Returns:
        types.ModuleType: Fake module that satisfies the import in
        :meth:`PandaDocSearchTool._ensure_collection`.
    """
    mod = types.ModuleType("chromadb")
    client = MagicMock()
    client.get_collection.return_value = collection
    mod.PersistentClient = MagicMock(return_value=client)  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_returns_formatted_hits(tmp_path, monkeypatch):
    """A successful query returns formatted snippets with scores and metadata."""
    # Arrange — fake collection that returns two hits
    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        "documents": [["First document text.", "Second document text."]],
        "metadatas": [[{"source": "doc_a.txt"}, {"source": "doc_b.txt"}]],
        "distances": [[0.1, 0.3]],
    }

    chroma_mod = _make_chroma_module(fake_collection)
    monkeypatch.setenv("BAMBOO_CHROMA_PATH", str(tmp_path))
    monkeypatch.setenv("BAMBOO_CHROMA_COLLECTION", "test_col")

    tool = PandaDocSearchTool()
    with patch.dict(sys.modules, {"chromadb": chroma_mod}):
        result = await tool.call({"query": "PanDA workflow", "top_k": 2})

    assert len(result) == 1
    assert result[0]["type"] == "text"
    text: str = result[0]["text"]

    assert "PanDA workflow" in text
    assert "[1]" in text
    assert "[2]" in text
    assert "doc_a.txt" in text
    assert "doc_b.txt" in text
    assert "First document text." in text
    # Scores: distance 0.1 → 90.0 %, distance 0.3 → 70.0 %
    assert "90.0%" in text
    assert "70.0%" in text


@pytest.mark.asyncio
async def test_chromadb_not_installed_returns_error_message(monkeypatch):
    """Returns a clear error message when chromadb is not installed."""
    tool = PandaDocSearchTool()

    # Remove chromadb from sys.modules and make the import fail.
    monkeypatch.delitem(sys.modules, "chromadb", raising=False)
    with patch.dict(sys.modules, {"chromadb": None}):  # type: ignore[dict-item]
        result = await tool.call({"query": "anything"})

    assert len(result) == 1
    text: str = result[0]["text"]
    assert "not installed" in text.lower() or "chromadb" in text.lower()
    assert result[0]["type"] == "text"


@pytest.mark.asyncio
async def test_missing_chroma_path_returns_error_message(monkeypatch):
    """Returns a clear error message when BAMBOO_CHROMA_PATH does not exist."""
    monkeypatch.setenv("BAMBOO_CHROMA_PATH", "/nonexistent/path/chroma_db")

    tool = PandaDocSearchTool()

    # Provide a real-looking chromadb module so the ImportError path is skipped.
    fake_col = MagicMock()
    chroma_mod = _make_chroma_module(fake_col)
    with patch.dict(sys.modules, {"chromadb": chroma_mod}):
        result = await tool.call({"query": "anything"})

    assert len(result) == 1
    text: str = result[0]["text"]
    assert "not found" in text.lower() or "chroma" in text.lower()
    assert result[0]["type"] == "text"


@pytest.mark.asyncio
async def test_empty_query_returns_error_message():
    """Returns an error message when the query string is empty."""
    tool = PandaDocSearchTool()
    result = await tool.call({"query": ""})

    assert len(result) == 1
    assert "required" in result[0]["text"].lower()


@pytest.mark.asyncio
async def test_no_results_returns_friendly_message(tmp_path, monkeypatch):
    """Returns a friendly message when the collection returns no documents."""
    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    chroma_mod = _make_chroma_module(fake_collection)
    monkeypatch.setenv("BAMBOO_CHROMA_PATH", str(tmp_path))

    tool = PandaDocSearchTool()
    with patch.dict(sys.modules, {"chromadb": chroma_mod}):
        result = await tool.call({"query": "obscure query with no matches"})

    assert len(result) == 1
    assert "no results" in result[0]["text"].lower()


def test_get_definition_shape():
    """get_definition returns the expected MCP tool definition structure."""
    defn = PandaDocSearchTool.get_definition()
    assert defn["name"] == "panda_doc_search"
    assert "query" in defn["inputSchema"]["properties"]
    assert "top_k" in defn["inputSchema"]["properties"]
    assert defn["inputSchema"]["required"] == ["query"]


def test_reset_clears_cached_state(tmp_path, monkeypatch):
    """_reset() clears _client and _collection so the next call re-initialises."""
    fake_collection = MagicMock()
    chroma_mod = _make_chroma_module(fake_collection)
    monkeypatch.setenv("BAMBOO_CHROMA_PATH", str(tmp_path))

    tool = PandaDocSearchTool()
    with patch.dict(sys.modules, {"chromadb": chroma_mod}):
        tool._ensure_collection()

    assert tool._collection is not None
    tool._reset()
    assert tool._client is None
    assert tool._collection is None
