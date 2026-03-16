"""Bamboo MCP Server entry point — stdio transport.

Uses the official MCP stdio transport.

Run:
  python3 -m bamboo.server
  npx @modelcontextprotocol/inspector python3 -m bamboo.server
"""

from __future__ import annotations

import asyncio
import sys

from anyio import BrokenResourceError
from mcp.server import Server
from mcp.server.stdio import stdio_server

from bamboo.config import Config
from bamboo.core import create_server


def _is_shutdown_exception(exc: BaseException) -> bool:
    """Return True if an exception represents a clean shutdown signal.

    Ctrl+C on Python 3.11+ can surface as an ``ExceptionGroup`` wrapping
    ``KeyboardInterrupt`` or ``BrokenResourceError`` (raised by anyio when
    the stdio streams are torn down mid-read). Both are expected on shutdown
    and should be swallowed silently.

    Args:
        exc: The exception to inspect.

    Returns:
        True if the exception (or all of its children, in the case of an
        ``ExceptionGroup``) represents a normal shutdown signal.
    """
    if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError, BrokenResourceError)):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return all(_is_shutdown_exception(e) for e in exc.exceptions)
    return False


async def main() -> None:
    """Run the Bamboo MCP stdio server.

    Prints a startup banner to stderr (keeping stdout clean for the MCP
    stdio protocol), then bootstraps the MCP Server via ``create_server()``
    and serves it over the stdio transport.
    """
    print(f"Bamboo MCP server v{Config.SERVER_VERSION} starting …", file=sys.stderr)

    app: Server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        try:
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options(),
            )
        except (asyncio.CancelledError, KeyboardInterrupt):
            return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
    except BaseException as exc:  # pylint: disable=broad-exception-caught
        if _is_shutdown_exception(exc):
            pass
        else:
            raise
