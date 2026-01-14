"""
AskPanDA MCP Server entry point. "Stdio server".

Uses the official MCP stdio transport.

Run:
  npx @modelcontextprotocol/inspector python3 -m askpanda_mcp.server
  python3 -m askpanda_mcp.server
"""

from mcp.server.stdio import stdio_server
from askpanda_mcp.core import create_server

async def main():
    app = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
