"""PanDA MCP session establishment helper.

Reads environment variables to configure and connect a ``ClientSession``
to the external PanDA MCP server (streamable-HTTP or SSE transport).
The session is registered with :func:`~bamboo.tools._mcp_caller.get_mcp_caller`
under the name ``"panda"`` so that any Bamboo tool can call PanDA MCP tools
via ``MCPCaller.call("panda", tool_name, arguments)``.

Environment variables
---------------------
PANDA_MCP_BASE_URL
    Full base URL of the PanDA MCP HTTP endpoint,
    e.g. ``https://aipanda120.cern.ch:8443/mcp/``.
    If unset the session is skipped and a warning is logged.
PANDA_MCP_TOKEN
    Optional bearer token sent as ``Authorization: Bearer <token>``.
PANDA_MCP_ORIGIN
    Optional virtual-organisation name sent as ``Origin: <vo>``.
PANDA_MCP_USE_SSE
    Set to ``"1"``, ``"true"``, or ``"yes"`` to use the legacy SSE transport
    instead of streamable-HTTP.  Streamable-HTTP is the default.
PANDA_MCP_TLS_VERIFY
    Set to ``"0"`` or ``"false"`` to disable TLS certificate verification.
    **Use only for development/testing** — never in production.  The default
    is to verify certificates using the system CA store.
PANDA_MCP_CA_BUNDLE
    Path to a CA certificate bundle (PEM) to use for TLS verification,
    e.g. ``/etc/pki/tls/certs/ca-bundle.crt``.  When unset, the system
    default CA store is used (via ``ssl.create_default_context()``), which
    works on most Linux systems that have the CERN Grid CA installed as a
    system package.  Set this explicitly if the CERN CA is only available
    as a standalone PEM file.

Typical usage (inside an asyncio task at server startup)::

    shutdown_event = asyncio.Event()
    task = asyncio.create_task(
        run_panda_mcp_session(shutdown_event)
    )
    # … at shutdown …
    shutdown_event.set()
    await task
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import os
import ssl
from typing import Any

_logger = logging.getLogger(__name__)

#: Logical server name used with MCPCaller.register_session / call.
PANDA_MCP_SERVER_NAME: str = "panda"


def _build_config() -> dict[str, Any] | None:
    """Read PanDA MCP connection config from environment variables.

    Returns:
        Dict with keys ``url``, ``headers`` (dict), ``use_sse`` (bool),
        ``ssl_context`` (ssl.SSLContext or False), or ``None`` if
        ``PANDA_MCP_BASE_URL`` is not set.
    """
    base_url = os.environ.get("PANDA_MCP_BASE_URL", "").strip()
    if not base_url:
        return None

    headers: dict[str, str] = {}
    token = os.environ.get("PANDA_MCP_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    origin = os.environ.get("PANDA_MCP_ORIGIN", "").strip()
    if origin:
        headers["Origin"] = origin

    use_sse = os.environ.get("PANDA_MCP_USE_SSE", "").lower() in {"1", "true", "yes"}

    # TLS verification: False disables it entirely (dev/test only).
    tls_verify_raw = os.environ.get("PANDA_MCP_TLS_VERIFY", "1").lower()
    tls_verify = tls_verify_raw not in {"0", "false", "no"}

    # Build SSL context (or pass False to httpx to disable verification).
    ssl_value: ssl.SSLContext | bool
    if not tls_verify:
        _logger.warning(
            "PANDA_MCP_TLS_VERIFY=0 — TLS certificate verification is DISABLED. "
            "Use only for development/testing."
        )
        ssl_value = False
    else:
        ssl_value = ssl.create_default_context()
        ca_bundle = os.environ.get("PANDA_MCP_CA_BUNDLE", "").strip()
        if ca_bundle:
            ssl_value.load_verify_locations(cafile=ca_bundle)

    return {
        "url": base_url,
        "headers": headers or None,
        "use_sse": use_sse,
        "ssl_context": ssl_value,
    }


async def run_panda_mcp_session(shutdown_event: asyncio.Event) -> None:
    """Connect to the PanDA MCP server and keep the session alive until shutdown.

    This coroutine is intended to be run as a background ``asyncio.Task`` for
    the lifetime of the Bamboo process.  It:

    1. Reads connection config from environment variables via :func:`_build_config`.
    2. If no config is present, logs a warning and returns immediately.
    3. Establishes a ``ClientSession`` using the appropriate transport.
    4. Registers the session with the process-wide ``MCPCaller`` under the
       name :data:`PANDA_MCP_SERVER_NAME`.
    5. Waits for ``shutdown_event`` to be set, then exits (context managers
       clean up the transport automatically).

    Any connection error is caught and logged; the session is simply not
    registered in that case, and affected tools will return graceful errors.

    Args:
        shutdown_event: An ``asyncio.Event`` that is set when the server is
            shutting down.  The session is torn down when this event fires.
    """
    config = _build_config()
    if config is None:
        _logger.warning(
            "PANDA_MCP_BASE_URL is not set — PanDA MCP tools will return "
            "'server not connected' errors.  Set the env var to enable them."
        )
        return

    url: str = config["url"]
    headers: dict[str, str] | None = config["headers"]
    use_sse: bool = config["use_sse"]
    ssl_context: ssl.SSLContext | bool = config["ssl_context"]

    _logger.info(
        "Connecting to PanDA MCP server at %s (transport=%s)",
        url,
        "sse" if use_sse else "streamable-http",
    )

    try:
        if use_sse:
            await _run_sse_session(url, headers, ssl_context, shutdown_event)
        else:
            await _run_http_session(url, headers, ssl_context, shutdown_event)
    except asyncio.CancelledError:
        _logger.info("PanDA MCP session task cancelled — shutting down.")
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _logger.error(
            "PanDA MCP session failed — tools will be unavailable: %s", exc
        )


async def _run_http_session(
    url: str,
    headers: dict[str, str] | None,
    ssl_context: ssl.SSLContext | bool,
    shutdown_event: asyncio.Event,
) -> None:
    """Connect via streamable-HTTP transport and hold the session open.

    Args:
        url: PanDA MCP base URL.
        headers: Optional HTTP headers (auth, origin).
        ssl_context: SSL context for TLS verification, or False to disable.
        shutdown_event: Set when the process is shutting down.
    """
    from bamboo.tools._mcp_caller import get_mcp_caller  # type: ignore[import-untyped]

    import httpx
    from mcp.client.session import ClientSession

    try:
        mod = importlib.import_module("mcp.client.streamable_http")
        http_transport_fn = getattr(mod, "streamable_http_client")
    except (ImportError, AttributeError) as exc:
        _logger.error("streamable_http_client not available: %s", exc)
        return

    http_client = httpx.AsyncClient(
        headers=headers or {},
        timeout=httpx.Timeout(30.0),
        verify=ssl_context,
    )
    try:
        transport_cm = http_transport_fn(
            url,
            http_client=http_client,
            terminate_on_close=True,
        )
        async with transport_cm as (read_stream, write_stream, _get_sid):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                get_mcp_caller().register_session(PANDA_MCP_SERVER_NAME, session)
                _logger.info(
                    "PanDA MCP session registered (streamable-HTTP) at %s", url
                )
                await shutdown_event.wait()
                _logger.info("PanDA MCP session shutting down (streamable-HTTP).")
    finally:
        await http_client.aclose()


async def _run_sse_session(
    url: str,
    headers: dict[str, str] | None,
    ssl_context: ssl.SSLContext | bool,
    shutdown_event: asyncio.Event,
) -> None:
    """Connect via SSE transport and hold the session open.

    Args:
        url: PanDA MCP base URL.
        headers: Optional HTTP headers (auth, origin).
        ssl_context: SSL context for TLS verification, or False to disable.
        shutdown_event: Set when the process is shutting down.
    """
    from bamboo.tools._mcp_caller import get_mcp_caller  # type: ignore[import-untyped]

    import httpx

    try:
        from mcp.client.sse import sse_client  # type: ignore[import-untyped]
        from mcp.client.session import ClientSession
    except ImportError as exc:
        _logger.error("SSE client not available: %s", exc)
        return

    # Pass an explicit httpx client factory so we control the SSL context.
    # The mcp sse_client accepts httpx_client_factory as a keyword argument.
    def _make_client(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        """Return an AsyncClient with the configured SSL context and headers."""
        return httpx.AsyncClient(
            headers=headers or {},
            timeout=timeout or httpx.Timeout(30.0),
            verify=ssl_context,
        )

    async with sse_client(url, httpx_client_factory=_make_client) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            get_mcp_caller().register_session(PANDA_MCP_SERVER_NAME, session)
            _logger.info("PanDA MCP session registered (SSE) at %s", url)
            await shutdown_event.wait()
            _logger.info("PanDA MCP session shutting down (SSE).")


__all__ = [
    "PANDA_MCP_SERVER_NAME",
    "run_panda_mcp_session",
]
