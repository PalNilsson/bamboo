"""Structured request/response lifecycle tracing for the Bamboo MCP server.

Tracing is **opt-in**: set ``BAMBOO_TRACE=1`` to enable it.  When disabled
every public function is a no-op so there is zero overhead in production.

Output destinations
-------------------

NDJSON (stderr or file)
    Always active when ``BAMBOO_TRACE=1``.  When ``BAMBOO_TRACE_FILE`` is
    set, spans go to that file and **stderr is left clean** (required when
    the server runs as a Textual TUI subprocess).  Without a file, spans go
    to stderr::

        BAMBOO_TRACE=1 python -m bamboo.server 2>&1 | grep bamboo_trace | jq .

OpenTelemetry (optional)
    When ``BAMBOO_OTEL_ENDPOINT`` is set, spans are **also** exported via
    OTLP/gRPC to any compatible backend (Jaeger, Grafana Tempo, Honeycomb,
    Datadog, …).  The SDK is imported lazily — install only when needed::

        pip install -r requirements-otel.txt
        export BAMBOO_TRACE=1
        export BAMBOO_OTEL_ENDPOINT=http://localhost:4317
        python -m bamboo.server

    Spans form a proper parent/child tree: the ``tool_call`` span is the
    root; ``guard``, ``retrieval``, ``llm_call``, and ``synthesis`` are
    children.  The relationship is tracked via a ``contextvars.ContextVar``
    that asyncio propagates automatically across ``await`` boundaries — no
    changes are needed at call sites.

    Optional tuning::

        BAMBOO_OTEL_SERVICE_NAME   (default: bamboo)
        BAMBOO_OTEL_INSECURE       (default: 1 — set to 0 for TLS)

Event schema (all events share these top-level keys)::

    {
        "bamboo_trace": true,
        "event":        "<event_type>",
        "tool":         "<tool_name>",
        "ts":           "<iso8601>",
        "duration_ms":  <float>,
        ...             # event-specific extra fields
    }

Event types and extra fields
-----------------------------

``"tool_call"``   — ``args_keys`` (list[str])
``"guard"``       — ``allowed`` (bool), ``reason`` (str), ``llm_used`` (bool)
``"retrieval"``   — ``backend`` (str), ``hits`` (int)
``"llm_call"``    — ``provider`` (str), ``model`` (str),
                    ``input_tokens`` (int|None), ``output_tokens`` (int|None)
``"synthesis"``   — ``route`` (str)

Usage::

    from bamboo.tracing import EVENT_GUARD, span

    async with span(EVENT_GUARD, tool="topic_guard") as s:
        result = await check_topic(question)
        s.set(allowed=result.allowed, reason=result.reason, llm_used=result.llm_used)
"""
from __future__ import annotations

import contextvars
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

# ---------------------------------------------------------------------------
# Runtime configuration  (read once at import time)
# ---------------------------------------------------------------------------

#: Set ``BAMBOO_TRACE=1`` to enable all tracing output.
TRACING_ENABLED: bool = os.getenv("BAMBOO_TRACE", "0") == "1"

#: Write spans to this file instead of stderr (preferred for TUI).
TRACE_FILE: str | None = os.getenv("BAMBOO_TRACE_FILE") or None

#: OTLP/gRPC endpoint for OpenTelemetry export, e.g. ``http://localhost:4317``.
#: When unset, OTel export is disabled.
OTEL_ENDPOINT: str | None = os.getenv("BAMBOO_OTEL_ENDPOINT") or None

#: OTel service name reported to the backend.
OTEL_SERVICE_NAME: str = os.getenv("BAMBOO_OTEL_SERVICE_NAME", "bamboo")

#: Set to ``"0"`` to enable TLS on the OTLP exporter.
OTEL_INSECURE: bool = os.getenv("BAMBOO_OTEL_INSECURE", "1") == "1"

# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

EVENT_TOOL_CALL: str = "tool_call"
EVENT_GUARD: str = "guard"
EVENT_RETRIEVAL: str = "retrieval"
EVENT_LLM_CALL: str = "llm_call"
EVENT_SYNTHESIS: str = "synthesis"
EVENT_PLAN: str = "plan"

# ---------------------------------------------------------------------------
# OpenTelemetry — lazy initialisation
# ---------------------------------------------------------------------------

# The OTel tracer provider and tracer are initialised once on first use.
# All access is through the module-level helpers below.
_otel_tracer: Any = None   # opentelemetry.trace.Tracer | None
_otel_ready: bool = False  # True once initialisation has been attempted

# ContextVar holding the currently-active OTel span so child spans can
# attach to it as their parent.  asyncio copies ContextVar state across
# await boundaries automatically, giving us free parent propagation.
_current_otel_span: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "bamboo_current_otel_span", default=None
)


def _get_otel_tracer() -> Any:
    """Return the OTel tracer, initialising it on the first call.

    Imports are lazy so the ``opentelemetry-sdk`` package is only required
    when ``BAMBOO_OTEL_ENDPOINT`` is set.  Any initialisation failure is
    logged to stderr and returns ``None``; callers must treat ``None`` as
    "OTel disabled".

    Returns:
        A configured ``opentelemetry.trace.Tracer`` instance, or ``None``
        if OTel is not configured or the SDK is not installed.
    """
    global _otel_tracer, _otel_ready  # pylint: disable=global-statement
    if _otel_ready:
        return _otel_tracer
    _otel_ready = True  # attempt only once even if it fails

    if not OTEL_ENDPOINT:
        return None

    try:
        from opentelemetry import trace  # type: ignore  # pylint: disable=import-outside-toplevel
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore  # pylint: disable=import-outside-toplevel
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore  # pylint: disable=import-outside-toplevel
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME  # type: ignore  # pylint: disable=import-outside-toplevel
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # type: ignore  # pylint: disable=import-outside-toplevel

        resource = Resource(attributes={SERVICE_NAME: OTEL_SERVICE_NAME})
        provider = TracerProvider(resource=resource)

        exporter = OTLPSpanExporter(
            endpoint=OTEL_ENDPOINT,
            insecure=OTEL_INSECURE,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _otel_tracer = trace.get_tracer(OTEL_SERVICE_NAME)
        return _otel_tracer

    except ImportError:
        print(
            "[bamboo.tracing] BAMBOO_OTEL_ENDPOINT is set but the "
            "'opentelemetry-sdk' package is not installed.\n"
            "Run: pip install -r requirements-otel.txt",
            file=sys.stderr,
        )
        return None
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(
            f"[bamboo.tracing] OTel initialisation failed: {exc}",
            file=sys.stderr,
        )
        return None


def _start_otel_span(name: str, attributes: dict[str, Any]) -> Any:
    """Start an OTel span as a child of the current context span.

    Args:
        name: Span name shown in the trace backend.
        attributes: Key/value pairs attached to the span immediately.

    Returns:
        The started ``opentelemetry.trace.Span``, or ``None`` if OTel is
        not configured or the SDK is unavailable.
    """
    try:
        tracer = _get_otel_tracer()
        if tracer is None:
            return None

        from opentelemetry import context as otel_context, trace  # type: ignore  # pylint: disable=import-outside-toplevel

        # Attach the current parent span to the OTel context so the new
        # span is registered as its child.
        parent_span = _current_otel_span.get()
        ctx = (
            trace.set_span_in_context(parent_span)
            if parent_span is not None
            else otel_context.get_current()
        )

        safe_attrs: dict[str, Any] = {}
        for k, v in attributes.items():
            # OTel attributes must be scalar or sequence of scalars.
            if isinstance(v, (str, bool, int, float)):
                safe_attrs[k] = v
            elif isinstance(v, list):
                safe_attrs[k] = str(v)
            elif v is None:
                pass  # skip None values
            else:
                safe_attrs[k] = str(v)

        return tracer.start_span(name, context=ctx, attributes=safe_attrs)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _end_otel_span(otel_span: Any, extra: dict[str, Any]) -> None:
    """Set final attributes and end an OTel span.

    Args:
        otel_span: The span returned by :func:`_start_otel_span`.
        extra: Additional attributes set during the span's lifetime via
            :meth:`SpanContext.set`.
    """
    if otel_span is None:
        return
    try:
        for k, v in extra.items():
            if isinstance(v, (str, bool, int, float)):
                otel_span.set_attribute(k, v)
            elif v is not None:
                otel_span.set_attribute(k, str(v))
        otel_span.end()
    except Exception:  # pylint: disable=broad-exception-caught
        pass  # OTel failure must never affect the request path.


# ---------------------------------------------------------------------------
# NDJSON emission
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        ISO-8601 formatted UTC timestamp with microsecond precision.
    """
    return datetime.now(tz=timezone.utc).isoformat()


def _emit(record: dict[str, Any]) -> None:
    """Write a single trace record to stderr or a trace file.

    When ``BAMBOO_TRACE_FILE`` is set the record is written **only** to the
    file and stderr is left untouched.  This prevents raw JSON corrupting
    the Textual TUI display.

    The ``"bamboo_trace": true`` sentinel is injected automatically.  All
    I/O errors are silently swallowed so that a tracing failure can never
    interrupt the serving path.

    Args:
        record: Mapping of fields to include in the trace line.
    """
    try:
        record["bamboo_trace"] = True
        line = json.dumps(record, ensure_ascii=False, default=str)
        if TRACE_FILE:
            try:
                with open(TRACE_FILE, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        else:
            print(line, file=sys.stderr, flush=True)
    except Exception:  # pylint: disable=broad-exception-caught
        pass


# ---------------------------------------------------------------------------
# TUI helpers
# ---------------------------------------------------------------------------


def trace_file_position() -> int:
    """Return the current byte position at the end of the trace file.

    Call this **before** sending a request to the server; then pass the
    returned position to :func:`read_trace_spans_since` after the request
    returns to get only the spans produced by that request.

    Returns:
        Current file size in bytes, or 0 if the file does not yet exist or
        ``BAMBOO_TRACE_FILE`` is not configured.
    """
    path = TRACE_FILE
    if not path:
        return 0
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def read_trace_spans_since(position: int) -> list[dict[str, Any]]:
    """Read all trace spans appended to the trace file since *position*.

    Args:
        position: Byte offset returned by a prior call to
            :func:`trace_file_position`.

    Returns:
        List of parsed span dicts in emission order.  Lines that are not
        valid JSON or lack the ``"bamboo_trace"`` sentinel are silently
        skipped.
    """
    path = TRACE_FILE
    if not path:
        return []
    spans: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            fh.seek(position)
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    obj = json.loads(raw_line)
                    if isinstance(obj, dict) and obj.get("bamboo_trace"):
                        spans.append(obj)
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return spans


# ---------------------------------------------------------------------------
# SpanContext
# ---------------------------------------------------------------------------


class SpanContext:
    """Mutable context object yielded by :func:`span`.

    Callers may attach additional key/value pairs to the in-progress span at
    any time by calling :meth:`set`.  All attached fields are included in the
    emitted NDJSON record and set as OTel span attributes when the span
    closes.

    Attributes:
        event: Event type string (one of the ``EVENT_*`` constants).
        tool: Logical name of the tool or step being traced.
    """

    def __init__(self, event: str, tool: str) -> None:
        """Initialise a new span context.

        Args:
            event: Event type (one of the ``EVENT_*`` module constants).
            tool: Logical tool or step name.
        """
        self.event = event
        self.tool = tool
        self._extra: dict[str, Any] = {}
        self._start: float = time.monotonic()
        self._otel_span: Any = None  # set by span() after start

    def set(self, **kwargs: Any) -> None:
        """Attach extra fields to this span.

        The fields are included in the NDJSON record and set as OTel span
        attributes when the span closes.

        Args:
            **kwargs: Arbitrary key/value pairs.  Keys must be valid JSON
                field names.  OTel attribute type restrictions apply when
                OTel export is enabled (non-scalar values are stringified).
        """
        self._extra.update(kwargs)

    def _close(self) -> None:
        """Emit the completed span to all configured outputs."""
        duration_ms = (time.monotonic() - self._start) * 1000.0
        record: dict[str, Any] = {
            "event": self.event,
            "tool": self.tool,
            "ts": _utc_now(),
            "duration_ms": round(duration_ms, 3),
        }
        record.update(self._extra)
        _emit(record)
        _end_otel_span(self._otel_span, self._extra)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@asynccontextmanager
async def span(event: str, tool: str, **initial_fields: Any) -> AsyncIterator[SpanContext]:
    """Async context manager that times a step and emits a trace event on exit.

    When :data:`TRACING_ENABLED` is ``False`` this is a zero-overhead no-op:
    the context manager still yields a :class:`SpanContext` so callers never
    need to branch on the tracing flag, but nothing is ever emitted.

    When ``BAMBOO_OTEL_ENDPOINT`` is set, an OTel span is started and the
    ``_current_otel_span`` context variable is set so that any nested
    :func:`span` calls create child spans automatically.

    Example::

        async with span(EVENT_GUARD, tool="topic_guard") as s:
            result = await check_topic(question)
            s.set(allowed=result.allowed, reason=result.reason)

    Args:
        event: Event type (one of the ``EVENT_*`` module constants).
        tool: Logical tool or step name.
        **initial_fields: Optional key/value pairs to pre-attach to the span.

    Yields:
        SpanContext: Mutable context object the caller can annotate with
        :meth:`~SpanContext.set`.
    """
    ctx = SpanContext(event=event, tool=tool)
    if initial_fields:
        ctx.set(**initial_fields)

    # Start the OTel span (no-op if OTel is not configured).
    if TRACING_ENABLED and OTEL_ENDPOINT:
        initial_attrs: dict[str, Any] = {"bamboo.event": event, "bamboo.tool": tool}
        initial_attrs.update(initial_fields)
        ctx._otel_span = _start_otel_span(  # pylint: disable=protected-access
            name=f"{event}:{tool}",
            attributes=initial_attrs,
        )
        # Make this span the current parent for any nested spans.
        _token = _current_otel_span.set(ctx._otel_span)  # pylint: disable=protected-access
    else:
        _token = None

    try:
        yield ctx
    finally:
        if _token is not None:
            _current_otel_span.reset(_token)
        if TRACING_ENABLED:
            ctx._close()  # pylint: disable=protected-access


def emit_sync(event: str, tool: str, duration_ms: float = 0.0, **fields: Any) -> None:
    """Emit a one-shot trace record without a span context manager.

    Useful for synchronous leaf operations where timing is supplied
    externally.  Does not create an OTel span.

    Args:
        event: Event type string.
        tool: Logical tool or step name.
        duration_ms: Pre-computed duration in milliseconds.
        **fields: Additional fields to include in the record.
    """
    if not TRACING_ENABLED:
        return
    record: dict[str, Any] = {
        "event": event,
        "tool": tool,
        "ts": _utc_now(),
        "duration_ms": round(duration_ms, 3),
    }
    record.update(fields)
    _emit(record)


__all__ = [
    "TRACING_ENABLED",
    "TRACE_FILE",
    "OTEL_ENDPOINT",
    "OTEL_SERVICE_NAME",
    "EVENT_TOOL_CALL",
    "EVENT_GUARD",
    "EVENT_RETRIEVAL",
    "EVENT_LLM_CALL",
    "EVENT_SYNTHESIS",
    "EVENT_PLAN",
    "SpanContext",
    "span",
    "emit_sync",
    "trace_file_position",
    "read_trace_spans_since",
]
