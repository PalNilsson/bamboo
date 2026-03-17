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

_SHUTDOWN_TYPES = (KeyboardInterrupt, asyncio.CancelledError, BrokenResourceError)


def _is_shutdown_exception(exc: BaseException) -> bool:
    """Return True if an exception represents a clean shutdown signal.

    Ctrl+C on Python 3.11+ can surface as an ``ExceptionGroup`` wrapping
    ``KeyboardInterrupt`` or ``BrokenResourceError`` (raised by anyio when
    the stdio streams are torn down mid-read). Both are expected on shutdown
    and should be swallowed silently.

    Avoids importing ``BaseExceptionGroup`` directly so the code works on
    Python 3.10 without the ``exceptiongroup`` backport and satisfies pyright
    on all supported versions.

    Args:
        exc: The exception to inspect.

    Returns:
        True if the exception (or all of its children, in the case of an
        ``ExceptionGroup``) represents a normal shutdown signal.
    """
    if isinstance(exc, _SHUTDOWN_TYPES):
        return True
    # BaseExceptionGroup is a builtin on 3.11+; check by type name to avoid
    # a conditional import that confuses pyright on older targets.
    if type(exc).__name__ in ("ExceptionGroup", "BaseExceptionGroup"):
        exceptions = getattr(exc, "exceptions", ())
        return all(_is_shutdown_exception(e) for e in exceptions)
    return False


async def main() -> None:
    """Run the Bamboo MCP stdio server.

    When the ``BAMBOO_QUIET`` environment variable is set to ``"1"`` (as the
    Textual TUI does when launching via stdio), the process-level stderr file
    descriptor is redirected to ``/dev/null`` before any output is written.
    This prevents startup banners, third-party library log lines, and any
    other stderr noise from leaking onto the terminal and corrupting the TUI
    display.  Tracing output uses ``BAMBOO_TRACE_FILE`` in this mode and is
    therefore unaffected.

    In normal (non-quiet) operation a startup banner is printed to stderr.
    """
    import os as _os

    if _os.getenv("BAMBOO_QUIET") == "1":
        # Redirect the underlying stderr file descriptor to /dev/null so that
        # *all* stderr output from this process (including third-party libs
        # that write directly to fd 2) is silently discarded.  We do this at
        # the fd level rather than just replacing sys.stderr so that C
        # extensions and the MCP SDK's own asyncio transports are also covered.
        try:
            _devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
            _os.dup2(_devnull_fd, 2)
            _os.close(_devnull_fd)
        except OSError:
            pass  # If the redirect fails, continue normally — better noisy than broken.
    else:
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
