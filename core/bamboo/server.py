"""
Bamboo MCP Server entry point. "Stdio server".

Uses the official MCP stdio transport.

Run:
  npx @modelcontextprotocol/inspector python3 -m bamboo.server
  python3 -m bamboo.server
"""

from __future__ import annotations

import asyncio

from mcp.server.stdio import stdio_server
from mcp.server import Server
from bamboo.core import create_server


async def main() -> None:
    """Run the Bamboo MCP stdio server.

    Bootstraps the MCP Server by creating the application via
    ``create_server()`` and serving it over the stdio transport returned by
    ``stdio_server()``.
    """
    app: Server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        try:
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options(),
            )
        except asyncio.CancelledError:
            # Expected on shutdown (e.g. when the client closes stdio).
            return
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Avoid a noisy traceback on Ctrl+C.
        pass
