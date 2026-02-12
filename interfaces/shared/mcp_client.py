"""
Shared MCP client helpers for AskPanDA interfaces.

This module provides a Streamlit-friendly synchronous wrapper around the MCP async
client, supporting both:

  - STDIO transport (dev): spawns a local MCP server subprocess
  - Streamable HTTP transport (prod): connects to an MCP server endpoint URL

Why a background event loop thread?
Streamlit runs scripts in a managed thread and may interrupt execution during reruns.
Running the MCP async session on a dedicated event-loop thread and calling it via
`run_coroutine_threadsafe` avoids cancellation issues and makes the UI stable.

Compatible with MCP streamable HTTP client signature:

  streamable_http_client(url, *, http_client=None, terminate_on_close=True)
    -> async generator yielding (read_stream, write_stream, get_session_id_callback)
"""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

import httpx

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# At runtime we import dynamically (some environments may not provide the helper).
# For static type checkers, import the symbol under TYPE_CHECKING so pyright can see it.
if TYPE_CHECKING:
    try:
        from mcp.client.streamable_http import streamable_http_client  # type: ignore
    except Exception:  # pylint: disable=broad-exception-caught  # pragma: no cover - only for static analysis
        streamable_http_client = None  # type: ignore

streamable_http_client: Any = None


TransportType = Literal["stdio", "http"]


@dataclass
class MCPServerConfig:
    """Configuration for connecting to an MCP server.

    Attributes:
        transport: "stdio" (spawn local server) or "http" (connect to HTTP endpoint).
        stdio_command: Executable for stdio server (typically sys.executable).
        stdio_args: Args for stdio server (e.g., ["-m", "bamboo.server"]).
        stdio_env: Optional environment overrides for stdio server.
        http_url: Streamable HTTP endpoint URL (e.g., "http://localhost:8000/mcp").
        http_headers: Optional headers (auth, etc.) for HTTP transport.
        terminate_on_close: If True, sends DELETE to terminate session on close.
        http_timeout_s: HTTP client timeout (seconds).
    """

    transport: TransportType = "stdio"

    # stdio options
    stdio_command: str = field(default_factory=lambda: sys.executable)
    stdio_args: list[str] = field(default_factory=lambda: ["-m", "bamboo.server"])
    stdio_env: dict[str, str] | None = None

    # http options
    http_url: str = "http://localhost:8000/mcp"
    http_headers: dict[str, str] | None = None
    terminate_on_close: bool = True
    http_timeout_s: float = 30.0


class MCPAsyncClient:
    """Async MCP client with stdio and HTTP transports."""

    def __init__(self, cfg: MCPServerConfig):
        """Initialize the async client.

        Args:
            cfg: Server connection configuration.
        """
        self.cfg = cfg
        self._session: ClientSession | None = None

        # Underlying transport context manager (stdio_client or streamable_http_client)
        self._transport_cm: Any = None

        # For HTTP: keep a configured AsyncClient if headers are needed
        self._http_client: httpx.AsyncClient | None = None

        # For debugging/observability
        self.http_session_id: str | None = None

    async def connect(self) -> "MCPAsyncClient":
        """Connect and initialize the MCP session.

        Returns:
            Self.

        Raises:
            RuntimeError: If initialization fails.
        """
        if self.cfg.transport == "stdio":
            params = StdioServerParameters(
                command=self.cfg.stdio_command,
                args=self.cfg.stdio_args,
                env=self.cfg.stdio_env,
            )
            self._transport_cm = stdio_client(params)
            read_stream, write_stream = await self._transport_cm.__aenter__()  # pylint: disable=unnecessary-dunder-call

        else:
            # Build HTTP client if headers are needed (auth, etc.)
            timeout = httpx.Timeout(self.cfg.http_timeout_s)
            self._http_client = httpx.AsyncClient(headers=self.cfg.http_headers, timeout=timeout)

            # Dynamically import the helper using importlib and getattr to avoid static attribute access checks
            import importlib  # pylint: disable=import-outside-toplevel
            try:
                _mod = importlib.import_module("mcp.client.streamable_http")
                func = getattr(_mod, "streamable_http_client", None)
            except Exception:  # pylint: disable=broad-exception-caught
                func = None

            if func is None:
                raise RuntimeError("streamable_http_client is not available in this environment")

            # streamable_http_client yields (read_stream, write_stream, get_session_id_callback)
            self._transport_cm = func(
                self.cfg.http_url,
                http_client=self._http_client,
                terminate_on_close=self.cfg.terminate_on_close,
            )
            read_stream, write_stream, get_session_id = await self._transport_cm.__aenter__()  # pylint: disable=unnecessary-dunder-call
            try:
                self.http_session_id = get_session_id()
            except Exception:  # pylint: disable=broad-exception-caught
                self.http_session_id = None

        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()  # pylint: disable=unnecessary-dunder-call

        # Initialize MCP session
        await self._session.initialize()
        return self

    async def aclose(self) -> None:
        """Close session and transport cleanly."""
        # Close session first
        if self._session is not None:
            try:
                await self._session.__aexit__(None, None, None)
            finally:
                self._session = None

        # Then close transport
        if self._transport_cm is not None:
            try:
                await self._transport_cm.__aexit__(None, None, None)
            finally:
                self._transport_cm = None

        # Then close HTTP client if used
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            finally:
                self._http_client = None

    async def list_tools(self) -> Any:
        """List tools from the MCP server."""
        if self._session is None:
            raise RuntimeError("MCP session not connected.")
        return await self._session.list_tools()

    async def list_prompts(self) -> Any:
        """List prompts from the MCP server."""
        if self._session is None:
            raise RuntimeError("MCP session not connected.")
        return await self._session.list_prompts()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call an MCP tool.

        Args:
            name: Tool name.
            arguments: Tool arguments JSON object.

        Returns:
            Tool call result.
        """
        if self._session is None:
            raise RuntimeError("MCP session not connected.")
        return await self._session.call_tool(name, arguments)


class MCPClientSync:
    """Synchronous wrapper around MCPAsyncClient for Streamlit and other sync UIs.

    Runs a dedicated asyncio event loop in a background thread and calls into it
    using `asyncio.run_coroutine_threadsafe`.
    """

    def __init__(self, cfg: MCPServerConfig):
        """Create and connect the MCP client synchronously.

        Args:
            cfg: Server connection configuration.

        Raises:
            RuntimeError: If connection/initialization fails.
        """
        self.cfg = cfg
        self._client = MCPAsyncClient(cfg)

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="mcp-client-loop", daemon=True)
        self._thread.start()

        # Connect on the loop thread
        self._run(self._client.connect())

    def _run_loop(self) -> None:
        """Event loop thread target."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run(self, coro: "asyncio.Future[Any] | Coroutine[Any, Any, Any]") -> Any:
        """Run a coroutine on the background loop and wait for the result.

        Args:
            coro: Coroutine to execute.

        Returns:
            Result of the coroutine.

        Raises:
            RuntimeError: If the coroutine fails.
        """
        # run_coroutine_threadsafe expects a coroutine; if a Future is passed
        # (unlikely in our usage), forward it as-is; otherwise use it directly.
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore[arg-type]
        try:
            return fut.result()
        except Exception as e:
            raise RuntimeError(str(e)) from e

    def close(self) -> None:
        """Close the MCP session and stop the background loop."""
        try:
            self._run(self._client.aclose())
        finally:
            if self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2.0)
            self._loop.close()

    def list_tools(self) -> Any:
        """List tools (sync)."""
        return self._run(self._client.list_tools())

    def list_prompts(self) -> Any:
        """List prompts (sync)."""
        return self._run(self._client.list_prompts())

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call tool (sync)."""
        return self._run(self._client.call_tool(name, arguments))

    @property
    def http_session_id(self) -> str | None:
        """Return HTTP MCP session id if using HTTP transport."""
        return self._client.http_session_id
