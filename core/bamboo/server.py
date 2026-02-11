# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Authors
# - Paul Nilsson, paul.nilsson@cern.ch, 2026

"""
Bamboo MCP Server entry point. "Stdio server".

Uses the official MCP stdio transport.

Run:
  npx @modelcontextprotocol/inspector python3 -m bamboo.server
  python3 -m bamboo.server
"""

from __future__ import annotations

from mcp.server.stdio import stdio_server
from mcp.server import Server
from bamboo.core import create_server


async def main() -> None:
    """Run the Bamboo MCP stdio server.

    Bootstraps the MCP Server by creating the application via
    ``create_server()`` and serving it over the stdio transport returned by
    ``stdio_server()``.
    """
    # Note: stdio transport has no HTTP headers, so TokenAuth is only used by
    # HTTP-based transports. create_server() still initializes app.auth
    app: Server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
