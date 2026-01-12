
"""
Shared MCP client helpers for AskPanDA interfaces.

This module intentionally keeps a small surface area:
- connect (stdio)
- list_tools
- call_tool
- close

It runs the MCP async client on a dedicated event loop thread so that
sync frameworks (Streamlit, etc.) can use it safely.
"""
from __future__ import annotations

import asyncio
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# MCP Python SDK imports (official)
# See client tutorial examples for these imports:
# https://modelcontextprotocol.info/docs/tutorials/building-a-client/  (also mirrored on modelcontextprotocol.io)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class _AsyncRunner:
    """Run async coroutines on a dedicated event loop in a daemon thread."""
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    def stop(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        # thread is daemon; no join required


@dataclass
class MCPServerConfig:
    command: str
    args: Sequence[str]
    env: Optional[Dict[str, str]] = None


class MCPStdioClient:
    """
    A small, synchronous wrapper around MCP ClientSession over stdio transport.

    Intended for UI code that runs in a sync context (e.g., Streamlit).
    """
    def __init__(self, cfg: MCPServerConfig) -> None:
        self.cfg = cfg
        self._runner = _AsyncRunner()
        self._exit_stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None

    def connect(self) -> None:
        self._runner.run(self._aconnect())

    async def _aconnect(self) -> None:
        if self._session is not None:
            return
        self._exit_stack = AsyncExitStack()
        params = StdioServerParameters(
            command=self.cfg.command,
            args=list(self.cfg.args),
            env=self.cfg.env,
        )
        stdio_transport = await self._exit_stack.enter_async_context(stdio_client(params))
        stdio, write = stdio_transport
        self._session = await self._exit_stack.enter_async_context(ClientSession(stdio, write))
        await self._session.initialize()

    def is_connected(self) -> bool:
        return self._session is not None

    def list_tools(self):
        return self._runner.run(self._alist_tools())

    async def _alist_tools(self):
        if not self._session:
            raise RuntimeError("MCP client not connected")
        resp = await self._session.list_tools()
        return resp.tools

    def list_prompts(self):
        return self._runner.run(self._alist_prompts())

    async def _alist_prompts(self):
        if not self._session:
            raise RuntimeError("MCP client not connected")
        resp = await self._session.list_prompts()
        return resp.prompts

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None):
        return self._runner.run(self._acall_tool(name, arguments or {}))

    async def _acall_tool(self, name: str, arguments: Dict[str, Any]):
        if not self._session:
            raise RuntimeError("MCP client not connected")
        return await self._session.call_tool(name, arguments)

    def close(self) -> None:
        self._runner.run(self._aclose())
        self._runner.stop()

    async def _aclose(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._exit_stack = None
        self._session = None
