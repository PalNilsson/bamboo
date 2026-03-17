"""Tests for the OpenTelemetry integration in bamboo.tracing.

All OTel SDK calls are mocked at the function level (_get_otel_tracer,
_start_otel_span) so no real opentelemetry package is required.

Tests verify:
- OTel is disabled when BAMBOO_OTEL_ENDPOINT is not set.
- Tracer initialisation is attempted when endpoint is set.
- Spans use the correct naming convention: "event:tool".
- Initial fields and set() fields appear as OTel span attributes.
- span.end() is called in the finally block, even on exception.
- Parent/child: nested spans reference the outer span via ContextVar.
- ContextVar is restored after a span exits.
- Missing/broken SDK is caught; NDJSON output is unaffected.
- NDJSON and OTel are both emitted when both are configured.
- emit_sync() does not create OTel spans.
"""
from __future__ import annotations

import io
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import bamboo.tracing as tracing
from bamboo.tracing import (
    EVENT_GUARD,
    EVENT_LLM_CALL,
    EVENT_RETRIEVAL,
    EVENT_SYNTHESIS,
    EVENT_TOOL_CALL,
    emit_sync,
    span,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ndjson_lines(buf: io.StringIO) -> list[dict[str, Any]]:
    """Parse all trace JSON lines from a StringIO buffer."""
    buf.seek(0)
    out = []
    for line in buf.read().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _make_mock_otel_span() -> MagicMock:
    """Return a fresh mock that behaves like an OTel span."""
    return MagicMock()


# ---------------------------------------------------------------------------
# OTel disabled
# ---------------------------------------------------------------------------


class TestOtelDisabled:
    """OTel path is inactive when BAMBOO_OTEL_ENDPOINT is not set."""

    @pytest.mark.asyncio
    async def test_no_otel_span_started_without_endpoint(self) -> None:
        """_start_otel_span is never called when endpoint is absent."""
        mock_start = MagicMock(return_value=None)
        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch.object(tracing, "OTEL_ENDPOINT", None),
            patch.object(tracing, "_start_otel_span", mock_start),
            patch("sys.stderr", io.StringIO()),
        ):
            async with span(EVENT_GUARD, tool="topic_guard") as s:
                s.set(allowed=True)

        mock_start.assert_not_called()
        assert s._otel_span is None

    @pytest.mark.asyncio
    async def test_ndjson_still_emitted_without_otel(self) -> None:
        """NDJSON output works normally when OTel is not configured."""
        buf = io.StringIO()
        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch.object(tracing, "OTEL_ENDPOINT", None),
            patch.object(tracing, "TRACE_FILE", None),
            patch("sys.stderr", buf),
        ):
            async with span(EVENT_GUARD, tool="topic_guard") as s:
                s.set(allowed=True, reason="keyword_allow")

        records = _ndjson_lines(buf)
        assert len(records) == 1
        assert records[0]["event"] == EVENT_GUARD
        assert records[0]["allowed"] is True


# ---------------------------------------------------------------------------
# OTel enabled — initialisation
# ---------------------------------------------------------------------------


class TestOtelInitialisation:
    """Tracer is called on the first span when endpoint is set."""

    @pytest.mark.asyncio
    async def test_start_otel_span_called_when_endpoint_set(self) -> None:
        """_start_otel_span is called when BAMBOO_OTEL_ENDPOINT is set."""
        mock_otel_span = _make_mock_otel_span()
        mock_start = MagicMock(return_value=mock_otel_span)
        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch.object(tracing, "OTEL_ENDPOINT", "http://localhost:4317"),
            patch.object(tracing, "_start_otel_span", mock_start),
            patch("sys.stderr", io.StringIO()),
        ):
            async with span(EVENT_TOOL_CALL, tool="bamboo_answer"):
                pass

        mock_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_sdk_does_not_raise(self) -> None:
        """_start_otel_span returning None (SDK missing) does not raise."""
        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch.object(tracing, "OTEL_ENDPOINT", "http://localhost:4317"),
            patch.object(tracing, "_start_otel_span", MagicMock(return_value=None)),
            patch("sys.stderr", io.StringIO()),
        ):
            async with span(EVENT_GUARD, tool="topic_guard") as s:
                s.set(allowed=True)

        assert s._otel_span is None

    @pytest.mark.asyncio
    async def test_start_span_exception_does_not_propagate(self) -> None:
        """If _start_otel_span raises internally it returns None and never propagates."""
        # _start_otel_span already wraps itself in try/except; this verifies
        # that the span() context manager also handles a None otel_span gracefully.
        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch.object(tracing, "OTEL_ENDPOINT", "http://localhost:4317"),
            patch.object(tracing, "_start_otel_span", MagicMock(return_value=None)),
            patch.object(tracing, "TRACE_FILE", None),
            patch("sys.stderr", io.StringIO()),
        ):
            async with span(EVENT_GUARD, tool="topic_guard") as s:
                s.set(allowed=False)

        # Span closed cleanly; NDJSON was emitted via stderr mock (consumed above).

    @pytest.mark.asyncio
    async def test_get_otel_tracer_import_error_caught(self) -> None:
        """ImportError inside _get_otel_tracer is caught; tracer returns None."""
        # Reset module-level state so _get_otel_tracer actually runs.
        with (
            patch.object(tracing, "_otel_ready", False),
            patch.object(tracing, "_otel_tracer", None),
            patch.object(tracing, "OTEL_ENDPOINT", "http://localhost:4317"),
            patch.dict("sys.modules", {"opentelemetry": None}),  # type: ignore[dict-item]
            patch("sys.stderr", io.StringIO()),
        ):
            result = tracing._get_otel_tracer()

        assert result is None


# ---------------------------------------------------------------------------
# OTel span naming and attributes
# ---------------------------------------------------------------------------


class TestOtelSpanShape:
    """Span names and attributes match the documented contract."""

    @pytest.mark.asyncio
    async def test_span_name_is_event_colon_tool(self) -> None:
        """OTel span name is 'event:tool', e.g. 'guard:topic_guard'."""
        mock_otel_span = _make_mock_otel_span()
        mock_start = MagicMock(return_value=mock_otel_span)

        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch.object(tracing, "OTEL_ENDPOINT", "http://localhost:4317"),
            patch.object(tracing, "_start_otel_span", mock_start),
            patch("sys.stderr", io.StringIO()),
        ):
            async with span(EVENT_GUARD, tool="topic_guard"):
                pass

        name_used = mock_start.call_args.kwargs["name"]
        assert name_used == "guard:topic_guard"

    @pytest.mark.asyncio
    async def test_initial_fields_in_start_attributes(self) -> None:
        """Fields passed to span() appear as OTel start attributes."""
        mock_otel_span = _make_mock_otel_span()
        mock_start = MagicMock(return_value=mock_otel_span)

        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch.object(tracing, "OTEL_ENDPOINT", "http://localhost:4317"),
            patch.object(tracing, "_start_otel_span", mock_start),
            patch("sys.stderr", io.StringIO()),
        ):
            async with span(EVENT_RETRIEVAL, tool="panda_doc_search",
                            backend="vector", hits=20):
                pass

        attrs = mock_start.call_args.kwargs["attributes"]
        assert attrs.get("backend") == "vector"
        assert attrs.get("hits") == 20

    @pytest.mark.asyncio
    async def test_set_fields_added_as_attributes_at_close(self) -> None:
        """Fields added via set() are set as OTel attributes when span closes."""
        mock_otel_span = _make_mock_otel_span()
        mock_start = MagicMock(return_value=mock_otel_span)

        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch.object(tracing, "OTEL_ENDPOINT", "http://localhost:4317"),
            patch.object(tracing, "_start_otel_span", mock_start),
            patch("sys.stderr", io.StringIO()),
        ):
            async with span(EVENT_SYNTHESIS, tool="bamboo_answer") as s:
                s.set(route="rag")

        mock_otel_span.set_attribute.assert_any_call("route", "rag")
        mock_otel_span.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_span_end_called_on_exception(self) -> None:
        """OTel span.end() is called in finally even when the body raises."""
        mock_otel_span = _make_mock_otel_span()
        mock_start = MagicMock(return_value=mock_otel_span)

        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch.object(tracing, "OTEL_ENDPOINT", "http://localhost:4317"),
            patch.object(tracing, "_start_otel_span", mock_start),
            patch("sys.stderr", io.StringIO()),
        ):
            with pytest.raises(ValueError):
                async with span(EVENT_TOOL_CALL, tool="bamboo_answer"):
                    raise ValueError("intentional")

        mock_otel_span.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_none_values_in_set_not_passed_as_attributes(self) -> None:
        """None values from set() are silently skipped in OTel attributes."""
        mock_otel_span = _make_mock_otel_span()
        mock_start = MagicMock(return_value=mock_otel_span)

        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch.object(tracing, "OTEL_ENDPOINT", "http://localhost:4317"),
            patch.object(tracing, "_start_otel_span", mock_start),
            patch("sys.stderr", io.StringIO()),
        ):
            async with span(EVENT_LLM_CALL, tool="bamboo_llm_answer") as s:
                s.set(input_tokens=None, output_tokens=50)

        # None must not be passed to set_attribute.
        for call_args in mock_otel_span.set_attribute.call_args_list:
            assert call_args[0][1] is not None
        mock_otel_span.set_attribute.assert_any_call("output_tokens", 50)


# ---------------------------------------------------------------------------
# Parent/child relationship via ContextVar
# ---------------------------------------------------------------------------


class TestOtelParentChild:
    """Nested spans must reference the outer span as their parent."""

    @pytest.mark.asyncio
    async def test_current_span_set_while_inside_outer(self) -> None:
        """_current_otel_span holds the outer OTel span while inside it."""
        mock_outer = _make_mock_otel_span()
        mock_start = MagicMock(return_value=mock_outer)

        captured: list[Any] = []

        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch.object(tracing, "OTEL_ENDPOINT", "http://localhost:4317"),
            patch.object(tracing, "_start_otel_span", mock_start),
            patch("sys.stderr", io.StringIO()),
        ):
            async with span(EVENT_TOOL_CALL, tool="bamboo_answer"):
                captured.append(tracing._current_otel_span.get())

        assert captured[0] is mock_outer

    @pytest.mark.asyncio
    async def test_inner_span_receives_outer_as_parent(self) -> None:
        """When spans are nested, the inner span is started with the outer as parent."""
        outer_otel = _make_mock_otel_span()
        inner_otel = _make_mock_otel_span()
        call_count = {"n": 0}

        def _start(name: str, attributes: dict[str, Any]) -> MagicMock:
            call_count["n"] += 1
            # The real _start_otel_span reads _current_otel_span from ContextVar.
            # Here we verify the ContextVar holds the outer span when the inner
            # span is being started.
            if call_count["n"] == 1:
                return outer_otel
            # By the time we start the inner span, the ContextVar must hold outer_otel.
            assert tracing._current_otel_span.get() is outer_otel
            return inner_otel

        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch.object(tracing, "OTEL_ENDPOINT", "http://localhost:4317"),
            patch.object(tracing, "_start_otel_span", _start),
            patch("sys.stderr", io.StringIO()),
        ):
            async with span(EVENT_TOOL_CALL, tool="bamboo_answer"):
                async with span(EVENT_GUARD, tool="topic_guard"):
                    pass

        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_context_var_restored_after_span(self) -> None:
        """_current_otel_span is restored to None after the outer span exits."""
        mock_otel_span = _make_mock_otel_span()
        mock_start = MagicMock(return_value=mock_otel_span)

        before = tracing._current_otel_span.get()

        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch.object(tracing, "OTEL_ENDPOINT", "http://localhost:4317"),
            patch.object(tracing, "_start_otel_span", mock_start),
            patch("sys.stderr", io.StringIO()),
        ):
            async with span(EVENT_TOOL_CALL, tool="bamboo_answer"):
                pass

        after = tracing._current_otel_span.get()
        assert before == after

    @pytest.mark.asyncio
    async def test_context_var_restored_after_exception(self) -> None:
        """_current_otel_span is restored even when the body raises."""
        mock_otel_span = _make_mock_otel_span()
        before = tracing._current_otel_span.get()

        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch.object(tracing, "OTEL_ENDPOINT", "http://localhost:4317"),
            patch.object(tracing, "_start_otel_span", MagicMock(return_value=mock_otel_span)),
            patch("sys.stderr", io.StringIO()),
        ):
            with pytest.raises(RuntimeError):
                async with span(EVENT_GUARD, tool="topic_guard"):
                    raise RuntimeError("boom")

        assert tracing._current_otel_span.get() == before


# ---------------------------------------------------------------------------
# NDJSON + OTel simultaneously
# ---------------------------------------------------------------------------


class TestOtelAndNdjsonTogether:
    """Both NDJSON and OTel outputs are active at the same time."""

    @pytest.mark.asyncio
    async def test_both_outputs_active(self) -> None:
        """NDJSON record is written and OTel span is started/ended together."""
        mock_otel_span = _make_mock_otel_span()
        mock_start = MagicMock(return_value=mock_otel_span)
        buf = io.StringIO()

        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch.object(tracing, "OTEL_ENDPOINT", "http://localhost:4317"),
            patch.object(tracing, "TRACE_FILE", None),
            patch.object(tracing, "_start_otel_span", mock_start),
            patch("sys.stderr", buf),
        ):
            async with span(EVENT_LLM_CALL, tool="bamboo_llm_answer",
                            provider="openai", model="gpt-4.1-mini") as s:
                s.set(input_tokens=100, output_tokens=50)

        # NDJSON record emitted.
        records = _ndjson_lines(buf)
        assert len(records) == 1
        rec = records[0]
        assert rec["event"] == EVENT_LLM_CALL
        assert rec["input_tokens"] == 100
        assert rec["output_tokens"] == 50

        # OTel span started and ended.
        mock_start.assert_called_once()
        mock_otel_span.end.assert_called_once()


# ---------------------------------------------------------------------------
# emit_sync — no OTel span
# ---------------------------------------------------------------------------


class TestEmitSyncNoOtel:
    """emit_sync() is NDJSON-only; it never starts an OTel span."""

    def test_emit_sync_does_not_call_start_otel_span(self) -> None:
        """emit_sync() never calls _start_otel_span regardless of OTel config."""
        mock_start = MagicMock()
        buf = io.StringIO()

        with (
            patch.object(tracing, "TRACING_ENABLED", True),
            patch.object(tracing, "OTEL_ENDPOINT", "http://localhost:4317"),
            patch.object(tracing, "TRACE_FILE", None),
            patch.object(tracing, "_start_otel_span", mock_start),
            patch("sys.stderr", buf),
        ):
            emit_sync(EVENT_TOOL_CALL, tool="bamboo_health", duration_ms=5.0)

        mock_start.assert_not_called()
        records = _ndjson_lines(buf)
        assert len(records) == 1
        assert records[0]["event"] == EVENT_TOOL_CALL
