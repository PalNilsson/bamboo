"""
AskPanDA MCP HTTP entrypoint (Streamable HTTP, ASGI).

This module exposes AskPanDA's MCP server over HTTP using the MCP Python SDK's
StreamableHTTPServerTransport. The transport in this MCP build requires:

- StreamableHTTPServerTransport(mcp_session_id=...)
- `async with transport.connect(): ...` must be entered before `handle_request(...)`
- `handle_request(scope, receive, send)` writes the HTTP response via ASGI `send`
  and returns None.

Important implementation detail:
The first HTTP request can arrive before the background connect task has entered
`transport.connect()`. To avoid a race, we maintain an asyncio.Event per session
that is set immediately after connect() is entered. Request handling awaits that
event before calling handle_request().

Run:
  uvicorn askpanda_mcp.entrypoints.http:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Awaitable, Callable, Dict, MutableMapping, Optional, Tuple
from urllib.parse import parse_qs

from mcp.server.streamable_http import StreamableHTTPServerTransport

from askpanda_mcp.core import create_server

# ASGI typing helpers
Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

server = create_server()

# Per-session state
_transports: Dict[str, StreamableHTTPServerTransport] = {}
_tasks: Dict[str, asyncio.Task[None]] = {}
_ready: Dict[str, asyncio.Event] = {}

_lock = asyncio.Lock()

# Header names clients may use for session id
_SESSION_HEADERS = (b"mcp-session-id", b"x-mcp-session-id")


async def _shutdown() -> None:
    """Shuts down per-session resources and shared clients.

    This is invoked from the ASGI lifespan shutdown event to ensure the service
    exits cleanly (important for Kubernetes SIGTERM handling). It:

    - Cancels all background connect/run tasks.
    - Clears per-session transport state.
    - Closes any shared LLM client manager if present on the server instance.

    The MCP StreamableHTTPServerTransport is held open by the background tasks
    via its `connect()` context manager. Cancelling those tasks is the primary
    mechanism to close transports gracefully.
    """
    # Cancel background tasks first (these own the transport.connect() context).
    tasks = list(_tasks.values())
    for task in tasks:
        task.cancel()

    # Await task completion to let connect() contexts exit.
    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            # Best-effort shutdown: ignore unexpected task errors.
            pass

    async with _lock:
        _tasks.clear()
        _transports.clear()
        _ready.clear()

    # Best-effort close of a shared LLM manager, if the server exposes one.
    llm_manager = getattr(server, "llm_manager", None)
    if llm_manager is not None and hasattr(llm_manager, "close_all"):
        try:
            await llm_manager.close_all()  # type: ignore[func-returns-value]
        except Exception:
            pass


def _get_session_id_from_scope(scope: Scope) -> Optional[str]:
    """Extract MCP session id from ASGI scope headers or query string.

    Args:
        scope: ASGI connection scope.

    Returns:
        The session id string, or None if not present.
    """
    headers = scope.get("headers") or []
    for k, v in headers:
        if k.lower() in _SESSION_HEADERS:
            try:
                return v.decode("utf-8").strip()
            except Exception:
                return None

    qs = scope.get("query_string") or b""
    if qs:
        params = parse_qs(qs.decode("utf-8"), keep_blank_values=True)
        sid = params.get("mcp_session_id", [None])[0]
        if sid:
            return str(sid).strip()

    return None


async def _run_session(session_id: str, transport: StreamableHTTPServerTransport, ready_evt: asyncio.Event) -> None:
    """Background task that keeps `transport.connect()` open and runs the MCP server.

    The MCP transport connect() context manager yields streams that are wired into
    `server.run(...)`. The `ready_evt` is set immediately after connect() is entered,
    guaranteeing that request handlers can call `handle_request` safely.

    Args:
        session_id: MCP session identifier.
        transport: Streamable HTTP transport for this session.
        ready_evt: Event set once connect() is active.
    """
    try:
        async with transport.connect() as streams:
            # We are now "connected" from the transport's perspective.
            ready_evt.set()

            # connect() may yield (read_stream, write_stream) or an object with attributes.
            read_stream: Any = None
            write_stream: Any = None

            if isinstance(streams, tuple) and len(streams) >= 2:
                read_stream, write_stream = streams[0], streams[1]
            else:
                if hasattr(streams, "read_stream") and hasattr(streams, "write_stream"):
                    read_stream = getattr(streams, "read_stream")
                    write_stream = getattr(streams, "write_stream")

            if read_stream is None or write_stream is None:
                raise TypeError(
                    "StreamableHTTPServerTransport.connect() did not yield usable streams. "
                    f"Got: {streams!r}"
                )

            await server.run(read_stream, write_stream, server.create_initialization_options())

    except asyncio.CancelledError:
        # Normal during shutdown / eviction
        raise
    finally:
        # Cleanup session state
        async with _lock:
            _transports.pop(session_id, None)
            _ready.pop(session_id, None)
            task = _tasks.pop(session_id, None)
            # Avoid self-cancel; task may already be completing.
            _ = task


async def _ensure_session(session_id: str) -> Tuple[StreamableHTTPServerTransport, asyncio.Event]:
    """Ensure transport/connect task exist for a given session id.

    Creates a transport and launches a background task that enters connect() and
    runs server.run(...). Returns the transport and the ready event.

    Args:
        session_id: MCP session id.

    Returns:
        (transport, ready_event)
    """
    async with _lock:
        transport = _transports.get(session_id)
        if transport is None:
            transport = StreamableHTTPServerTransport(mcp_session_id=session_id)
            _transports[session_id] = transport

        ready_evt = _ready.get(session_id)
        if ready_evt is None:
            ready_evt = asyncio.Event()
            _ready[session_id] = ready_evt

        if session_id not in _tasks:
            _tasks[session_id] = asyncio.create_task(_run_session(session_id, transport, ready_evt))

        return transport, ready_evt


async def _send_plain_text(send: Send, status: int, body: str) -> None:
    """Send a simple plain-text HTTP response.

    Args:
        send: ASGI send callable.
        status: HTTP status code.
        body: Response body text.
    """
    data = body.encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"text/plain; charset=utf-8")],
        }
    )
    await send({"type": "http.response.body", "body": data})


async def app(scope: Scope, receive: Receive, send: Send) -> None:
    """ASGI application entrypoint.

    Routes:
      - GET /healthz -> "ok"
      - * /mcp -> MCP Streamable HTTP handler

    Args:
        scope: ASGI scope.
        receive: ASGI receive callable.
        send: ASGI send callable.
    """
    scope_type = scope.get("type")

    # Support ASGI lifespan events so the service can shutdown cleanly in
    # process managers and Kubernetes.
    if scope_type == "lifespan":
        while True:
            message = await receive()
            msg_type = message.get("type")

            if msg_type == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif msg_type == "lifespan.shutdown":
                await _shutdown()
                await send({"type": "lifespan.shutdown.complete"})
                return

        # Unreachable, but keeps type checkers happy.
        return

    if scope_type != "http":
        # Only HTTP and lifespan are supported here.
        return

    path = scope.get("path", "")

    if path == "/healthz":
        await _send_plain_text(send, 200, "ok")
        return

    if path != "/mcp":
        await _send_plain_text(send, 404, "not found")
        return

    session_id = _get_session_id_from_scope(scope)
    if not session_id:
        session_id = str(uuid.uuid4())

        # Ensure the client learns this new session id: send in headers.
        # Note: transport.handle_request will generate the actual response; we can't
        # reliably inject headers there without a custom transport. For now, clients
        # can also operate without specifying a session id (server creates per-request),
        # but that reduces reuse. If you want guaranteed header propagation, we can
        # add a thin ASGI middleware that wraps send() to inject the header.
        #
        # Practically: the MCP client streamable_http_client manages session IDs itself.
        # So this branch is mainly for manual curl/debug.

    transport, ready_evt = await _ensure_session(session_id)

    # Wait until connect() is active to avoid the "No read stream writer available" race.
    # Keep timeout modest so failures are visible quickly.
    try:
        await asyncio.wait_for(ready_evt.wait(), timeout=5.0)
    except asyncio.TimeoutError as exc:
        raise RuntimeError("Timed out waiting for MCP transport.connect()") from exc

    # Let MCP handle the request/response fully via ASGI send().
    await transport.handle_request(scope, receive, send)
