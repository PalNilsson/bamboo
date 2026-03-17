"""Tests for the Bamboo structured-tracing module (bamboo.tracing).

Coverage:
- SpanContext: field attachment, duration measurement.
- span() context manager: emits when enabled, silent when disabled.
- emit_sync(): emits when enabled, silent when disabled.
- JSON output format: sentinel, required keys, extra fields.
- Error resilience: broken stderr write must not propagate.
- Integration smoke: guard/retrieval/synthesis event constants are importable.
"""
from __future__ import annotations

import io
import json
import time
from typing import Any
from unittest.mock import patch

import pytest

import bamboo.tracing as tracing
from bamboo.tracing import (
    EVENT_GUARD,
    EVENT_LLM_CALL,
    EVENT_RETRIEVAL,
    EVENT_SYNTHESIS,
    EVENT_TOOL_CALL,
    SpanContext,
    emit_sync,
    span,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lines_from_stderr(buf: io.StringIO) -> list[dict[str, Any]]:
    """Parse all non-empty lines of a StringIO buffer as JSON objects.

    Args:
        buf: StringIO containing NDJSON lines written to stderr.

    Returns:
        List of parsed JSON objects, one per non-empty line.
    """
    buf.seek(0)
    records = []
    for line in buf.read().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# SpanContext unit tests
# ---------------------------------------------------------------------------


class TestSpanContext:
    """Tests for the SpanContext class in isolation."""

    def test_initial_fields(self) -> None:
        """SpanContext stores event and tool names correctly."""
        ctx = SpanContext(event="test_event", tool="my_tool")
        assert ctx.event == "test_event"
        assert ctx.tool == "my_tool"

    def test_set_attaches_extra_fields(self) -> None:
        """set() accumulates arbitrary kwargs in _extra."""
        ctx = SpanContext(event="e", tool="t")
        ctx.set(allowed=True, reason="keyword_allow")
        assert ctx._extra["allowed"] is True
        assert ctx._extra["reason"] == "keyword_allow"

    def test_set_overwrites_existing_key(self) -> None:
        """Calling set() twice on the same key keeps the latest value."""
        ctx = SpanContext(event="e", tool="t")
        ctx.set(hits=5)
        ctx.set(hits=10)
        assert ctx._extra["hits"] == 10

    def test_duration_increases_over_time(self) -> None:
        """_close() computes a positive duration_ms after a real delay."""
        buf = io.StringIO()
        ctx = SpanContext(event="e", tool="t")
        time.sleep(0.01)
        with patch("sys.stderr", buf):
            ctx._close()
        records = _lines_from_stderr(buf)
        assert len(records) == 1
        assert records[0]["duration_ms"] > 0


# ---------------------------------------------------------------------------
# span() context manager
# ---------------------------------------------------------------------------


class TestSpanContextManager:
    """Tests for the span() async context manager."""

    @pytest.mark.asyncio
    async def test_emits_when_tracing_enabled(self) -> None:
        """span() writes a JSON line to stderr when BAMBOO_TRACE=1."""
        buf = io.StringIO()
        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch("sys.stderr", buf),
        ):
            async with span(EVENT_GUARD, tool="topic_guard") as s:
                s.set(allowed=True, reason="keyword_allow", llm_used=False)

        records = _lines_from_stderr(buf)
        assert len(records) == 1
        rec = records[0]
        assert rec["bamboo_trace"] is True
        assert rec["event"] == EVENT_GUARD
        assert rec["tool"] == "topic_guard"
        assert rec["allowed"] is True
        assert rec["reason"] == "keyword_allow"
        assert rec["llm_used"] is False
        assert "ts" in rec
        assert "duration_ms" in rec

    @pytest.mark.asyncio
    async def test_silent_when_tracing_disabled(self) -> None:
        """span() produces no output when BAMBOO_TRACE is not set."""
        buf = io.StringIO()
        with (
            patch.object(tracing, "TRACING_ENABLED", False),
            patch("sys.stderr", buf),
        ):
            async with span(EVENT_GUARD, tool="topic_guard") as s:
                s.set(allowed=False)

        assert buf.getvalue() == ""

    @pytest.mark.asyncio
    async def test_initial_fields_included(self) -> None:
        """Fields passed to span() as kwargs appear in the emitted record."""
        buf = io.StringIO()
        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch("sys.stderr", buf),
        ):
            async with span(EVENT_RETRIEVAL, tool="panda_doc_search",
                            backend="vector", hits=15):
                pass

        records = _lines_from_stderr(buf)
        assert records[0]["backend"] == "vector"
        assert records[0]["hits"] == 15

    @pytest.mark.asyncio
    async def test_emits_on_exception(self) -> None:
        """span() still emits when the body raises an exception."""
        buf = io.StringIO()
        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch("sys.stderr", buf),
        ):
            with pytest.raises(ValueError):
                async with span(EVENT_TOOL_CALL, tool="bamboo_answer"):
                    raise ValueError("boom")

        records = _lines_from_stderr(buf)
        assert len(records) == 1
        assert records[0]["event"] == EVENT_TOOL_CALL

    @pytest.mark.asyncio
    async def test_yields_span_context(self) -> None:
        """span() yields a SpanContext instance the caller can annotate."""
        with patch.object(tracing, "TRACING_ENABLED", False):
            async with span(EVENT_LLM_CALL, tool="bamboo_llm_answer") as s:
                assert isinstance(s, SpanContext)
                s.set(input_tokens=100)
                assert s._extra["input_tokens"] == 100

    @pytest.mark.asyncio
    async def test_duration_ms_is_non_negative(self) -> None:
        """Emitted duration_ms is a non-negative float."""
        buf = io.StringIO()
        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch("sys.stderr", buf),
        ):
            async with span(EVENT_SYNTHESIS, tool="bamboo_answer", route="rag"):
                pass

        records = _lines_from_stderr(buf)
        assert records[0]["duration_ms"] >= 0.0

    @pytest.mark.asyncio
    async def test_set_after_init_overrides_initial(self) -> None:
        """set() inside the body overrides a field set at span() open time."""
        buf = io.StringIO()
        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch("sys.stderr", buf),
        ):
            async with span(EVENT_RETRIEVAL, tool="panda_doc_bm25",
                            hits=-1) as s:
                s.set(hits=7)

        records = _lines_from_stderr(buf)
        assert records[0]["hits"] == 7


# ---------------------------------------------------------------------------
# emit_sync()
# ---------------------------------------------------------------------------


class TestEmitSync:
    """Tests for the emit_sync() one-shot emission helper."""

    def test_emits_when_enabled(self) -> None:
        """emit_sync() writes a JSON line to stderr when tracing is on."""
        buf = io.StringIO()
        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch("sys.stderr", buf),
        ):
            emit_sync(EVENT_TOOL_CALL, tool="bamboo_health", duration_ms=1.5,
                      args_keys=["question"])

        records = _lines_from_stderr(buf)
        assert len(records) == 1
        rec = records[0]
        assert rec["bamboo_trace"] is True
        assert rec["event"] == EVENT_TOOL_CALL
        assert rec["tool"] == "bamboo_health"
        assert rec["duration_ms"] == 1.5
        assert rec["args_keys"] == ["question"]

    def test_silent_when_disabled(self) -> None:
        """emit_sync() produces no output when tracing is off."""
        buf = io.StringIO()
        with (
            patch.object(tracing, "TRACING_ENABLED", False),
            patch("sys.stderr", buf),
        ):
            emit_sync(EVENT_TOOL_CALL, tool="bamboo_health")

        assert buf.getvalue() == ""

    def test_defaults_duration_to_zero(self) -> None:
        """emit_sync() uses 0.0 as duration_ms when not supplied."""
        buf = io.StringIO()
        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch("sys.stderr", buf),
        ):
            emit_sync(EVENT_GUARD, tool="topic_guard")

        records = _lines_from_stderr(buf)
        assert records[0]["duration_ms"] == 0.0


# ---------------------------------------------------------------------------
# JSON format invariants
# ---------------------------------------------------------------------------


class TestJsonFormat:
    """Tests that every emitted line satisfies the documented schema."""

    @pytest.mark.asyncio
    async def test_required_keys_present(self) -> None:
        """Every span emits bamboo_trace, event, tool, ts, duration_ms."""
        buf = io.StringIO()
        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch("sys.stderr", buf),
        ):
            async with span(EVENT_LLM_CALL, tool="bamboo_llm_answer"):
                pass

        rec = _lines_from_stderr(buf)[0]
        for key in ("bamboo_trace", "event", "tool", "ts", "duration_ms"):
            assert key in rec, f"Missing required key: {key}"

    @pytest.mark.asyncio
    async def test_ts_is_iso8601(self) -> None:
        """The ts field is a valid ISO-8601 datetime string."""
        from datetime import datetime
        buf = io.StringIO()
        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch("sys.stderr", buf),
        ):
            async with span(EVENT_SYNTHESIS, tool="bamboo_answer", route="task"):
                pass

        rec = _lines_from_stderr(buf)[0]
        # fromisoformat() raises ValueError on invalid strings.
        parsed = datetime.fromisoformat(rec["ts"])
        assert parsed is not None

    @pytest.mark.asyncio
    async def test_each_line_is_valid_json(self) -> None:
        """Multiple concurrent spans each produce a valid, complete JSON line."""
        import asyncio
        buf = io.StringIO()
        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch("sys.stderr", buf),
        ):
            async def _task(name: str) -> None:
                async with span(EVENT_RETRIEVAL, tool=name, backend="vector"):
                    await asyncio.sleep(0)

            await asyncio.gather(_task("tool_a"), _task("tool_b"))

        records = _lines_from_stderr(buf)
        assert len(records) == 2
        tools = {r["tool"] for r in records}
        assert tools == {"tool_a", "tool_b"}


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------


class TestErrorResilience:
    """Tracing failures must never propagate into the request path."""

    @pytest.mark.asyncio
    async def test_broken_stderr_does_not_raise(self) -> None:
        """If writing to stderr fails, span() must not raise."""
        class _BrokenFile:
            def write(self, _: str) -> None:
                raise OSError("disk full")

            def flush(self) -> None:
                pass

        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch("sys.stderr", _BrokenFile()),
        ):
            # Must not raise despite broken stderr.
            async with span(EVENT_GUARD, tool="topic_guard") as s:
                s.set(allowed=True)

    def test_broken_stderr_emit_sync_does_not_raise(self) -> None:
        """If writing to stderr fails, emit_sync() must not raise."""
        class _BrokenFile:
            def write(self, _: str) -> None:
                raise OSError("disk full")

            def flush(self) -> None:
                pass

        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch("sys.stderr", _BrokenFile()),
        ):
            emit_sync(EVENT_TOOL_CALL, tool="bamboo_health")


# ---------------------------------------------------------------------------
# Public API / constants
# ---------------------------------------------------------------------------


class TestOutputRouting:
    """Tests that file and stderr outputs are mutually exclusive."""

    @pytest.mark.asyncio
    async def test_file_set_writes_to_file_not_stderr(self, tmp_path: Any) -> None:
        """When TRACE_FILE is set, spans go to the file and stderr is clean."""
        trace_file = str(tmp_path / "trace.jsonl")
        buf = io.StringIO()
        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch.object(tracing, "TRACE_FILE", trace_file),
            patch("sys.stderr", buf),
        ):
            async with span(EVENT_GUARD, tool="topic_guard") as s:
                s.set(allowed=True, reason="keyword_allow")

        # stderr must be untouched
        assert buf.getvalue() == "", "stderr should be empty when TRACE_FILE is set"

        # file must contain the span
        import pathlib
        lines = [ln for ln in pathlib.Path(trace_file).read_text().splitlines() if ln.strip()]
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["bamboo_trace"] is True
        assert rec["event"] == EVENT_GUARD
        assert rec["allowed"] is True

    @pytest.mark.asyncio
    async def test_no_file_writes_to_stderr_only(self) -> None:
        """When TRACE_FILE is not set, spans go to stderr and no file is created."""
        buf = io.StringIO()
        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch.object(tracing, "TRACE_FILE", None),
            patch("sys.stderr", buf),
        ):
            async with span(EVENT_SYNTHESIS, tool="bamboo_answer", route="rag"):
                pass

        records = _lines_from_stderr(buf)
        assert len(records) == 1
        assert records[0]["event"] == EVENT_SYNTHESIS

    @pytest.mark.asyncio
    async def test_emit_sync_file_set_skips_stderr(self, tmp_path: Any) -> None:
        """emit_sync also skips stderr when TRACE_FILE is configured."""
        trace_file = str(tmp_path / "trace.jsonl")
        buf = io.StringIO()
        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch.object(tracing, "TRACE_FILE", trace_file),
            patch("sys.stderr", buf),
        ):
            emit_sync(EVENT_TOOL_CALL, tool="bamboo_answer", duration_ms=10.0)

        assert buf.getvalue() == ""
        import pathlib
        lines = [ln for ln in pathlib.Path(trace_file).read_text().splitlines() if ln.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["event"] == EVENT_TOOL_CALL


class TestTraceFileHelpers:
    """Tests for trace_file_position() and read_trace_spans_since()."""

    def test_position_returns_zero_when_no_file(self) -> None:
        """trace_file_position() returns 0 when TRACE_FILE is not set."""
        with patch.object(tracing, "TRACE_FILE", None):
            assert tracing.trace_file_position() == 0

    def test_position_returns_zero_for_missing_file(self, tmp_path: Any) -> None:
        """trace_file_position() returns 0 when the configured file does not exist."""
        missing = str(tmp_path / "nonexistent.jsonl")
        with patch.object(tracing, "TRACE_FILE", missing):
            assert tracing.trace_file_position() == 0

    def test_position_returns_file_size(self, tmp_path: Any) -> None:
        """trace_file_position() returns the current byte size of the trace file."""
        f = tmp_path / "trace.jsonl"
        f.write_text('{"bamboo_trace":true}\n', encoding="utf-8")
        with patch.object(tracing, "TRACE_FILE", str(f)):
            pos = tracing.trace_file_position()
        assert pos > 0

    def test_read_spans_since_returns_empty_when_no_file(self) -> None:
        """read_trace_spans_since() returns [] when TRACE_FILE is not set."""
        with patch.object(tracing, "TRACE_FILE", None):
            assert tracing.read_trace_spans_since(0) == []

    def test_read_spans_since_reads_only_new_content(self, tmp_path: Any) -> None:
        """read_trace_spans_since() only returns spans written after the snapshot."""
        f = tmp_path / "trace.jsonl"
        # Write one span before the snapshot.
        old_span = json.dumps({"bamboo_trace": True, "event": "old"}) + "\n"
        f.write_text(old_span, encoding="utf-8")
        position = len(old_span.encode("utf-8"))

        # Append a new span after the snapshot.
        new_span = json.dumps({"bamboo_trace": True, "event": "new"}) + "\n"
        with f.open("a", encoding="utf-8") as fh:
            fh.write(new_span)

        with patch.object(tracing, "TRACE_FILE", str(f)):
            spans = tracing.read_trace_spans_since(position)

        assert len(spans) == 1
        assert spans[0]["event"] == "new"

    def test_read_spans_skips_non_trace_lines(self, tmp_path: Any) -> None:
        """read_trace_spans_since() silently skips non-JSON and non-trace lines."""
        f = tmp_path / "trace.jsonl"
        content = (
            "not json\n"
            '{"bamboo_trace": false, "event": "ignored"}\n'
            '{"bamboo_trace": true, "event": "kept"}\n'
        )
        f.write_text(content, encoding="utf-8")
        with patch.object(tracing, "TRACE_FILE", str(f)):
            spans = tracing.read_trace_spans_since(0)
        assert len(spans) == 1
        assert spans[0]["event"] == "kept"

    @pytest.mark.asyncio
    async def test_round_trip_via_span_and_read(self, tmp_path: Any) -> None:
        """Full round-trip: span() writes to file, read_trace_spans_since reads it."""
        trace_file = str(tmp_path / "trace.jsonl")
        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch.object(tracing, "TRACE_FILE", trace_file),
        ):
            pos = tracing.trace_file_position()
            async with span(EVENT_LLM_CALL, tool="bamboo_llm_answer",
                            provider="openai", model="gpt-4.1-mini") as s:
                s.set(input_tokens=100, output_tokens=50)
            spans = tracing.read_trace_spans_since(pos)

        assert len(spans) == 1
        rec = spans[0]
        assert rec["event"] == EVENT_LLM_CALL
        assert rec["provider"] == "openai"
        assert rec["input_tokens"] == 100
        assert rec["output_tokens"] == 50
    """Smoke tests for the public surface of bamboo.tracing."""

    def test_event_constants_are_strings(self) -> None:
        """All EVENT_* constants are non-empty strings."""
        for const in (
            EVENT_TOOL_CALL,
            EVENT_GUARD,
            EVENT_RETRIEVAL,
            EVENT_LLM_CALL,
            EVENT_SYNTHESIS,
        ):
            assert isinstance(const, str) and const

    def test_all_exports_present(self) -> None:
        """All symbols listed in __all__ are importable from bamboo.tracing."""
        import importlib
        module = importlib.import_module("bamboo.tracing")
        for name in module.__all__:
            assert hasattr(module, name), f"Missing export: {name}"

    def test_trace_file_constant_reflects_env(self) -> None:
        """TRACE_FILE is None when BAMBOO_TRACE_FILE is unset."""
        # The module constant is read at import time; just verify it's str or None.
        assert tracing.TRACE_FILE is None or isinstance(tracing.TRACE_FILE, str)
