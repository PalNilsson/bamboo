"""Tests for the RAG-first routing branch in BambooAnswerTool.

These tests cover the "no recognised task/job ID" path introduced to route
general PanDA knowledge questions through ``panda_doc_search_tool`` before
handing off to the LLM for synthesis.

All external I/O (RAG tool, LLM passthrough) is monkeypatched so tests run
fully offline without real ChromaDB or LLM credentials.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import bamboo.tools.bamboo_answer as ba_mod
from bamboo.tools.bamboo_answer import BambooAnswerTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm_mock(reply: str) -> AsyncMock:
    """Return an AsyncMock that simulates bamboo_llm_answer_tool.call."""
    mock = AsyncMock(return_value=[{"type": "text", "text": reply}])
    return mock


def _rag_mock(text: str) -> AsyncMock:
    """Return an AsyncMock that simulates panda_doc_search_tool.call."""
    mock = AsyncMock(return_value=[{"type": "text", "text": text}])
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_general_question_calls_rag_then_llm():
    """A question with no ID triggers RAG first, then LLM synthesis."""
    rag_text = (
        "PanDA Doc Search — top 2 result(s) for: 'What is PanDA?'\n\n"
        "[1] score=91.0%  distance=0.0900\n  PanDA is the workload manager …\n"
        "[2] score=85.0%  distance=0.1500\n  PanDA stands for Production …\n"
    )
    llm_reply = "PanDA is the Production and Distributed Analysis system used by ATLAS."

    rag_mock = _rag_mock(rag_text)
    llm_mock = _llm_mock(llm_reply)

    tool = BambooAnswerTool()
    with (
        patch.object(ba_mod, "panda_doc_search_tool", AsyncMock(call=rag_mock)),
        patch.object(ba_mod, "bamboo_llm_answer_tool", AsyncMock(call=llm_mock)),
    ):
        result = await tool.call({"question": "What is PanDA?"})

    # RAG must have been called with the question.
    rag_mock.assert_awaited_once()
    call_args = rag_mock.call_args[0][0]
    assert call_args["query"] == "What is PanDA?"
    assert call_args["top_k"] == 5

    # LLM must have been called; its reply is returned verbatim.
    llm_mock.assert_awaited_once()
    llm_call_msgs = llm_mock.call_args[0][0]["messages"]
    # System prompt should reference retrieved excerpts.
    assert any("retrieved" in m["content"].lower() for m in llm_call_msgs)
    # User prompt should embed the RAG context.
    user_msg = next(m for m in llm_call_msgs if m["role"] == "user")
    assert "PanDA Doc Search" in user_msg["content"]

    assert result[0]["type"] == "text"
    assert llm_reply in result[0]["text"]


@pytest.mark.asyncio
async def test_general_question_rag_unavailable_falls_back_gracefully():
    """When RAG returns an error message, the LLM is still called with a no-context prompt."""
    rag_text = "ChromaDB is not installed.  Install it with: pip install -r requirements-rag.txt"
    llm_reply = "Based on general knowledge: PanDA manages ATLAS workloads."

    rag_mock = _rag_mock(rag_text)
    llm_mock = _llm_mock(llm_reply)

    tool = BambooAnswerTool()
    with (
        patch.object(ba_mod, "panda_doc_search_tool", AsyncMock(call=rag_mock)),
        patch.object(ba_mod, "bamboo_llm_answer_tool", AsyncMock(call=llm_mock)),
    ):
        result = await tool.call({"question": "What is PanDA?"})

    rag_mock.assert_awaited_once()
    llm_mock.assert_awaited_once()

    # System prompt should NOT say "retrieved excerpts" — it's the no-context variant.
    llm_call_msgs = llm_mock.call_args[0][0]["messages"]
    system_msg = next(m for m in llm_call_msgs if m["role"] == "system")
    assert "no documentation excerpts" in system_msg["content"].lower()

    assert result[0]["type"] == "text"
    assert llm_reply in result[0]["text"]


@pytest.mark.asyncio
async def test_general_question_rag_exception_falls_back_gracefully():
    """If the RAG tool raises, the LLM is still called — RAG failure is non-fatal."""
    llm_reply = "PanDA is the workload manager for ATLAS."

    rag_mock = AsyncMock(side_effect=RuntimeError("unexpected chroma error"))
    llm_mock = _llm_mock(llm_reply)

    tool = BambooAnswerTool()
    with (
        patch.object(ba_mod, "panda_doc_search_tool", AsyncMock(call=rag_mock)),
        patch.object(ba_mod, "bamboo_llm_answer_tool", AsyncMock(call=llm_mock)),
    ):
        result = await tool.call({"question": "What is PanDA?"})

    llm_mock.assert_awaited_once()
    assert result[0]["type"] == "text"
    assert llm_reply in result[0]["text"]


@pytest.mark.asyncio
async def test_task_id_question_does_not_call_rag():
    """Questions containing a task ID must NOT trigger the RAG tool."""
    task_tool_mock = AsyncMock(return_value={"evidence": {"payload": {"status": "done"}}})
    llm_mock = _llm_mock("Task 12345678 finished.")
    rag_mock = _rag_mock("should not be called")

    tool = BambooAnswerTool()
    with (
        patch.object(ba_mod, "panda_doc_search_tool", AsyncMock(call=rag_mock)),
        patch.object(ba_mod, "panda_task_status_tool", AsyncMock(call=task_tool_mock)),
        patch.object(ba_mod, "bamboo_llm_answer_tool", AsyncMock(call=llm_mock)),
    ):
        await tool.call({"question": "What is the status of task 12345678?"})

    rag_mock.assert_not_awaited()
    task_tool_mock.assert_awaited_once()
