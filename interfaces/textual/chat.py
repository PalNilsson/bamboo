#!/usr/bin/env python3
"""Textual TUI for Bamboo (formerly AskPanDA).

This interface provides a terminal UI for interacting with the Bamboo MCP server.

Key goals:
* Fast, non-blocking initial render (no awaits in on_mount).
* Single-task ownership of MCP connect/use/close to avoid AnyIO cancel-scope errors.
* Clean shutdown via /exit, /quit, Ctrl+D/Ctrl+Q.
* Inline mode by default for easy copy/paste in most terminals.
* Forward LLM defaults from environment variables.

Environment variables:
  - LLM_DEFAULT_PROVIDER
  - LLM_DEFAULT_MODEL
  - ASKPANDA_LLM_DEFAULT_PROVIDER (also forwarded)
  - ASKPANDA_LLM_DEFAULT_MODEL (also forwarded)
  - ASKPANDA_PLUGIN (default plugin id, e.g. "atlas")

Examples:
  python3 interfaces/textual/chat.py --transport stdio
  python3 interfaces/textual/chat.py --transport http --http-url http://localhost:8000/mcp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import os
import sys
import time
from typing import Any, Dict, List, Optional

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static

from interfaces.shared.mcp_client import MCPClientSync, MCPServerConfig

DEFAULT_PLUGIN = os.getenv("ASKPANDA_PLUGIN", "atlas")

ENV_LLM_DEFAULT_PROVIDER = "LLM_DEFAULT_PROVIDER"
ENV_LLM_DEFAULT_MODEL = "LLM_DEFAULT_MODEL"
ENV_LLM_DEFAULT_PROVIDER_ALT = "ASKPANDA_LLM_DEFAULT_PROVIDER"
ENV_LLM_DEFAULT_MODEL_ALT = "ASKPANDA_LLM_DEFAULT_MODEL"

ANSWER_TOOL_CANDIDATES: List[str] = ["bamboo_answer", "askpanda_answer", "bamboo_plan"]


_TASK_CMD_RE = re.compile(r"^/task\s+(\d{1,12})\s*$", re.IGNORECASE)

FALLBACK_BANNER = r"""
    _        _     ____            ____     _
   / \   ___| | __|  _ \ ___ _ __ |  _ \   / \
  / _ \ / __| |/ /  |_) /_` | '_ \| | | | / _ \
 / ___ \\__ \   <|  __/ (_| | | | | |_| |/ ___ \
/_/   \_\___/_|\_\_|   \__,_|_| |_|____//_/   \_\
""".strip("\n").splitlines()


def _now() -> str:
    """Return a short local timestamp for transcript headers.

    Returns:
        str: Local time formatted as ``HH:MM:SS``.
    """
    return time.strftime("%H:%M:%S")


def _pretty(obj: Any) -> str:
    """Serialize an object as pretty JSON when possible.

    Args:
        obj (Any): Value to serialize.

    Returns:
        str: JSON-formatted output, or ``str(obj)`` if serialization fails.
    """
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def _extract_text(result: Any) -> str:
    """Extract displayable text from an MCP tool result.

    Args:
        result: The raw tool result (SDK object, dict, string, etc.).

    Returns:
        A best-effort string suitable for rendering in the transcript.
    """
    if result is None:
        return ""

    if isinstance(result, str):
        return result.strip()

    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "\n".join([p for p in parts if p]).strip()
        for key in ("text", "output", "message", "result"):
            val = result.get(key)
            if isinstance(val, str):
                return val.strip()
        return _pretty(result)

    content2 = getattr(result, "content", None)
    if isinstance(content2, list):
        parts2: List[str] = []
        for item in content2:
            txt = getattr(item, "text", None)
            if isinstance(txt, str):
                parts2.append(txt)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts2.append(str(item.get("text", "")))
        return "\n".join([p for p in parts2 if p]).strip()

    return _pretty(result)


def _tool_names_from_list_tools(list_tools_result: Any) -> List[str]:
    """Extract tool names from an MCP ``list_tools`` response.

    Args:
        list_tools_result (Any): Raw MCP SDK object or dict response.

    Returns:
        List[str]: Tool names discovered in the response.
    """
    tools = getattr(list_tools_result, "tools", None)
    if isinstance(tools, list):
        names: List[str] = []
        for tool in tools:
            name = getattr(tool, "name", None)
            if isinstance(name, str):
                names.append(name)
            elif isinstance(tool, dict) and isinstance(tool.get("name"), str):
                names.append(tool["name"])
        return names

    if isinstance(list_tools_result, dict):
        tools2 = list_tools_result.get("tools")
        if isinstance(tools2, list):
            out: List[str] = []
            for tool in tools2:
                if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                    out.append(tool["name"])
            return out

    return []


class BambooTui(App):
    """Textual-based terminal UI for Bamboo."""

    ENABLE_MOUSE = False

    CSS = """
    Screen {
        height: 100%;
    }

    /* Inline apps must opt-in to a fixed height, otherwise they may render as 0 lines. */
    Screen:inline {
        height: 50;
        border: none;
    }

    #root {
        layout: vertical;
        height: 100%;
    }

    #banner {
        height: auto;
        padding: 1 2;
    }

    #transcript {
        height: 1fr;
        overflow-y: auto;
        border: round $surface;
    }

    #input_row {
        height: 3;
        padding: 0 2;
    }

    #input {
        height: 3;
    }

    """

    BINDINGS = [
        Binding("escape", "focus_input", "Focus input", show=False),
        Binding("ctrl+l", "clear", "Clear", show=True),
        Binding("ctrl+d", "quit", "Quit", show=False),
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("pageup", "scroll_up", show=False),
        Binding("pagedown", "scroll_down", show=False),

    ]

    def __init__(self, cfg: MCPServerConfig, plugin_id: str) -> None:
        """Initialize the app.

        Args:
            cfg (MCPServerConfig): MCP server config (stdio or HTTP).
            plugin_id (str): Active plugin namespace (e.g. "atlas").
        """
        super().__init__()
        self.cfg = cfg
        self.plugin_id = plugin_id

        # Sync wrapper; only used via asyncio.to_thread.
        self.mcp: MCPClientSync = MCPClientSync(cfg, connect_on_init=False)

        # One task owns the full MCP lifecycle.
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._mcp_task: Optional[asyncio.Task[None]] = None

        self.tool_names: List[str] = []
        self.answer_tool: Optional[str] = None

        self.banner_lines: List[str] = FALLBACK_BANNER[:]
        self.display_name: str = f"AskPanDA – {plugin_id.upper()}"
        self.help_text: str = "Enter to send • /help"

        self.debug_mode: bool = False
        self._mcp_ready: bool = False

        self.banner_widget: Optional[Static] = None
        self.transcript: Optional[RichLog] = None
        self.input_widget: Optional[Input] = None

    def compose(self) -> ComposeResult:
        """Build the Textual widget tree.

        Returns:
            ComposeResult: Stream of widgets to render.
        """
        yield Header(show_clock=True)

        with Container(id="root"):
            with Vertical(id="banner"):
                self.banner_widget = Static()
                yield self.banner_widget

            self.transcript = RichLog(id="transcript", wrap=True, markup=False)
            yield self.transcript

            with Container(id="input_row"):
                self.input_widget = Input(
                    id="input",
                    placeholder="AskPanDA > (Enter to send)",
                )
                yield self.input_widget

        yield Footer()

    async def on_mount(self) -> None:
        """Mount UI quickly and start MCP initialization in the background."""
        if self.banner_widget:
            self.banner_widget.can_focus = False
        if self.transcript:
            self.transcript.can_focus = False

        # Best-effort mouse disable (some terminals inject control sequences).
        try:
            driver = getattr(self, "_driver", None)
            if driver and hasattr(driver, "disable_mouse_support"):
                driver.disable_mouse_support()
        except Exception:
            pass

        self._render_banner_placeholder()
        self._write_system("Starting… initializing MCP…")

        # IMPORTANT: no awaits here; let Textual finish initial render.
        self._mcp_task = asyncio.create_task(self._mcp_main(), name="bamboo.mcp_main")
        self.action_focus_input()

    async def on_exit(self) -> None:
        """Request clean shutdown and wait for MCP task to finish."""
        self._shutdown_event.set()

        if self._mcp_task:
            try:
                await asyncio.wait_for(asyncio.shield(self._mcp_task), timeout=5)
            except asyncio.TimeoutError:
                # If the task is stuck, do a best-effort close without cancelling.
                try:
                    await self._mcp_close()
                except Exception:
                    pass
            except Exception:
                pass

    async def _mcp_main(self) -> None:
        """Own the MCP lifecycle in a single task."""
        try:
            await asyncio.wait_for(self._mcp_connect(), timeout=5)
            await asyncio.wait_for(self._refresh_tools(), timeout=20)
            self._detect_answer_tool()

            await asyncio.wait_for(self._load_banner(), timeout=20)
            self._render_banner()

            self._mcp_ready = True
            self._write_system(
                f"Connected via {self.cfg.transport}. Answer tool: {self.answer_tool or 'UNKNOWN'}"
            )

            await self._shutdown_event.wait()

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._write_error(f"Startup failed: {type(exc).__name__}: {exc}")
            self._write_system("Check ~/.textual/logs for details.")
            await self._shutdown_event.wait()
        finally:
            try:
                await self._mcp_close()
            except Exception:
                pass

    async def _to_thread(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a blocking callable in a worker thread.

        Args:
            func (Any): Callable to execute.
            *args (Any): Positional arguments for ``func``.
            **kwargs (Any): Keyword arguments for ``func``.

        Returns:
            Any: Value returned by ``func``.
        """
        return await asyncio.to_thread(func, *args, **kwargs)

    async def _mcp_connect(self) -> None:
        """Connect to the MCP server (best-effort across client implementations).

        The shared MCP client wrapper may evolve. This method avoids hard-coding a
        single entrypoint name (e.g., `connect`) by trying a small set of common
        lifecycle method names.

        Raises:
            AttributeError: If no known connect method exists.
            Exception: Propagates any connect failure from the underlying client.
        """
        # Prefer a dedicated connect() / open() / start() API if available.
        for name in ("connect", "open", "start", "acquire"):
            fn = getattr(self.mcp, name, None)
            if callable(fn):
                await self._to_thread(fn)
                return

        # As a fallback, support context-manager style wrappers.
        enter = getattr(self.mcp, "__enter__", None)
        if callable(enter):
            await self._to_thread(enter)
            return

        # This client wrapper connects lazily (e.g., on first list_tools/call_tool).
        # Treat "connect" as a no-op.
        return

    async def _mcp_close(self) -> None:
        """Close the MCP client connection using a best-effort strategy."""
        for name in ("close", "aclose", "stop", "shutdown", "release"):
            fn = getattr(self.mcp, name, None)
            if callable(fn):
                await self._to_thread(fn)
                return

        exit_fn = getattr(self.mcp, "__exit__", None)
        if callable(exit_fn):
            await self._to_thread(exit_fn, None, None, None)
            return

    def _render_banner_placeholder(self) -> None:
        """Render a fallback banner before plugin manifest data is loaded."""
        if not self.banner_widget:
            return
        banner_text = "\n".join(FALLBACK_BANNER)
        self.banner_widget.update(
            Panel(
                Text(banner_text, no_wrap=True),
                title=self.display_name,
                subtitle=f"plugin={self.plugin_id}",
                padding=(0, 1),
                expand=True,
            )
        )

    def action_scroll_up(self) -> None:
        """Scroll the transcript up by one page."""
        self.transcript.scroll_page_up()

    def action_scroll_down(self) -> None:
        """Scroll the transcript down by one page."""
        self.transcript.scroll_page_down()

    def action_focus_input(self) -> None:
        """Focus the input widget."""
        if self.input_widget:
            self.input_widget.focus()

    async def action_clear(self) -> None:
        """Clear the transcript."""
        if self.transcript:
            self.transcript.clear()
        self._write_system("Transcript cleared.")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter on the input field.

        Args:
            event (Input.Submitted): Submitted input event.
        """
        text = (event.value or "").strip()
        if self.input_widget:
            self.input_widget.value = ""

        if not text:
            self.action_focus_input()
            return

        if text.startswith("/"):
            await self._handle_command(text)
        else:
            await self._handle_question(text)

        self.action_focus_input()

    async def _refresh_tools(self) -> None:
        """Refresh the tool list from the MCP server."""
        res = await self._to_thread(self.mcp.list_tools)
        self.tool_names = sorted(_tool_names_from_list_tools(res))

    def _detect_answer_tool(self) -> None:
        """Select a preferred answer tool based on available tools."""
        for name in ANSWER_TOOL_CANDIDATES:
            if name in self.tool_names:
                self.answer_tool = name
                return
        for name in self.tool_names:
            if "answer" in name:
                self.answer_tool = name
                return
        self.answer_tool = None

    async def _load_banner(self) -> None:
        """Load banner/branding from <plugin>.ui_manifest if available."""
        tool = f"{self.plugin_id}.ui_manifest"
        if tool not in self.tool_names:
            self.banner_lines = FALLBACK_BANNER[:]
            self.display_name = f"AskPanDA – {self.plugin_id.upper()}"
            self.help_text = "Enter to send • /help"
            return

        try:
            res = await self._to_thread(self.mcp.call_tool, tool, {})
            txt = _extract_text(res)
            if self.debug_mode:
                self._write_tool(tool, {}, res)

            manifest = json.loads(txt) if txt else {}
            if isinstance(manifest, dict):
                self.display_name = str(manifest.get("display_name") or self.display_name)

                banner = manifest.get("banner")
                if isinstance(banner, list) and all(isinstance(x, str) for x in banner):
                    self.banner_lines = banner
                else:
                    self.banner_lines = FALLBACK_BANNER[:]

                help_text = manifest.get("help")
                if isinstance(help_text, str) and help_text.strip():
                    self.help_text = help_text.strip()
        except Exception:
            self.banner_lines = FALLBACK_BANNER[:]
            self.display_name = f"AskPanDA – {self.plugin_id.upper()}"
            self.help_text = "Enter to send • /help"

    def _render_banner(self) -> None:
        """Render the banner panel."""
        if not self.banner_widget:
            return
        banner_text = "\n".join(self.banner_lines)
        self.banner_widget.update(
            Panel(
                Text(banner_text, no_wrap=True),
                title=self.display_name,
                subtitle=f"plugin={self.plugin_id}",
                padding=(0, 1),
                expand=True,
            )
        )

    async def _handle_question(self, question: str) -> None:
        """Send a question to the answer tool and render the response.

        Args:
            question (str): User prompt text.
        """
        self._write_user(question)

        if not self._mcp_ready:
            self._write_system("Not connected yet. Please try again in a moment…")
            return

        if not self.answer_tool:
            self._write_error("No answer tool found. Try /tools to inspect server tools.")
            return

        args: Dict[str, Any] = {"question": question}

        provider = args.get("provider") or os.getenv("LLM_DEFAULT_PROVIDER") or "mistral"
        model = args.get("model") or os.getenv("LLM_DEFAULT_MODEL") or "..."
        if provider:
            args["provider"] = provider
            args["llm_provider"] = provider
        if model:
            args["model"] = model
            args["llm_model"] = model

        try:
            res = await self._to_thread(self.mcp.call_tool, self.answer_tool, args)
            if self.debug_mode:
                self._write_tool(self.answer_tool, args, res)

            out = _extract_text(res) or "*(No text output; enable /debug on to see raw result.)*"
            self._write_assistant(out)
        except Exception as exc:
            self._write_error(str(exc))

    async def _handle_command(self, cmdline: str) -> None:
        """Handle slash commands.

        Args:
            cmdline (str): Raw command line beginning with ``/``.
        """
        parts = cmdline.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]

        m_task = _TASK_CMD_RE.match(cmdline.strip())
        if m_task:
            task_id = m_task.group(1)
            await self._handle_question(
                f"Summarize the status of task {task_id} including dataset info."
            )
            return

        if cmd in ("/help", "/?"):
            self._write_system(
                "Commands:\n"
                "  /help                 Show this help\n"
                "  /tools                List tools exposed by the MCP server\n"
                "  /task <id>             Shorthand for: status of task <id>\n"
                "  /plugin <id>          Switch plugin (affects banner tool name)\n"
                "  /debug on|off         Toggle debug (shows tool calls + raw results)\n"
                "  /clear                Clear transcript\n"
                "  /exit, /quit          Exit the app\n"
                "\n"
                "Tip: This TUI runs inline by default so you can mouse-select/copy text.\n"
                "Use --no-inline to use the alternate screen."
            )
            return

        if cmd in ("/exit", "/quit"):
            self.exit()
            return

        if cmd == "/tools":
            if self._mcp_ready:
                await self._refresh_tools()
            if not self.tool_names:
                self._write_system("No tools returned by server.")
                return
            body = "\n".join(f"- `{n}`" for n in self.tool_names)
            self._write_panel(Markdown(body), title=f"{_now()}  tools")
            return

        if cmd == "/plugin":
            if not args:
                self._write_system(f"Current plugin: {self.plugin_id}")
                return
            self.plugin_id = args[0]
            if self._mcp_ready:
                await self._refresh_tools()
                self._detect_answer_tool()
                await self._load_banner()
                self._render_banner()
            else:
                self._render_banner_placeholder()
            self._write_system(f"Switched plugin to: {self.plugin_id}")
            return

        if cmd == "/debug":
            if args and args[0].lower() in ("on", "off"):
                self.debug_mode = args[0].lower() == "on"
            else:
                self.debug_mode = not self.debug_mode
            self._write_system(f"Debug {'ON' if self.debug_mode else 'OFF'}.")
            return

        if cmd == "/clear":
            await self.action_clear()
            return

        self._write_system(f"Unknown command: {cmdline} (try /help)")

    def _write_panel(self, renderable: Any, title: str, border_style: str = "dim") -> None:
        """Write a panel to the transcript.

        Args:
            renderable (Any): Rich renderable content.
            title (str): Panel title.
            border_style (str): Rich border style name.
        """
        if not self.transcript:
            return

        # Width of the transcript region (fallback to app width)
        width = None
        try:
            width = max(10, self.transcript.size.width)
        except Exception:
            try:
                width = max(10, self.size.width)
            except Exception:
                width = None

        self.transcript.write(
            Panel(
                renderable,
                title=title,
                border_style=border_style,
                expand=True,
                width=width,  # <-- this is the key
                padding=(0, 1),
            )
        )

        self.transcript.scroll_end(animate=False)
        if self.input_widget:
            self.input_widget.focus()

    def _write_panel_old(self, renderable: Any, title: str, border_style: str = "dim") -> None:
        """Write a panel using the legacy rendering path.

        Args:
            renderable (Any): Rich renderable content.
            title (str): Panel title.
            border_style (str): Rich border style name.
        """
        if not self.transcript:
            return
        self.transcript.write(Panel(renderable, title=title, border_style=border_style))
        self.transcript.scroll_end(animate=False)
        self.input_widget.focus()

        if self.transcript:
            self.transcript.scroll_end(animate=False)
        if self.input_widget:
            self.input_widget.focus()

        self.action_focus_input()

    def _write_system(self, msg: str) -> None:
        """Write a system message.

        Args:
            msg (str): Message text.
        """
        self._write_panel(Text(msg), title=f"{_now()}  system", border_style="dim")

    def _write_user(self, msg: str) -> None:
        """Write a user message.

        Args:
            msg (str): Message text.
        """
        self._write_panel(Text(msg), title=f"{_now()}  you", border_style="dim")

    def _write_assistant(self, msg: str) -> None:
        """Write an assistant message.

        Args:
            msg (str): Markdown message text.
        """
        self._write_panel(Markdown(msg), title=f"{_now()}  AskPanDA", border_style="dim")

    def _write_error(self, msg: str) -> None:
        """Write an error message.

        Args:
            msg (str): Error message text.
        """
        self._write_panel(Text(msg), title=f"{_now()}  error", border_style="red")

    def _write_tool(self, tool: str, args: Dict[str, Any], res: Any) -> None:
        """Write a debug tool call block.

        Args:
            tool (str): Tool name.
            args (Dict[str, Any]): Tool input arguments.
            res (Any): Raw tool result.
        """
        body = (
            f"**tool:** `{tool}`\n\n"
            f"**args:**\n```json\n{_pretty(args)}\n```\n\n"
            f"**result:**\n```text\n{_extract_text(res) or _pretty(res)}\n```"
        )
        self._write_panel(Markdown(body), title=f"{_now()}  tool", border_style="cyan")


def main() -> None:
    """Parse CLI arguments and run the Textual chat app."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", choices=["stdio", "http"], required=True)
    ap.add_argument("--http-url", default=os.getenv("MCP_URL", "http://localhost:8000/mcp"))
    ap.add_argument("--plugin", default=DEFAULT_PLUGIN)
    ap.add_argument("--terminate-on-close", action="store_true", default=True)

    inline_group = ap.add_mutually_exclusive_group()
    inline_group.add_argument("--inline", action="store_true", help="Run inline (default).")
    inline_group.add_argument("--no-inline", action="store_true", help="Run in alternate screen.")

    args = ap.parse_args()

    if args.transport == "http":
        cfg = MCPServerConfig(transport="http", http_url=args.http_url, terminate_on_close=args.terminate_on_close)
    else:
        cfg = MCPServerConfig(
            transport="stdio",
            stdio_command=sys.executable,
            stdio_args=["-m", "bamboo.server"],
            stdio_env=os.environ.copy()
        )

    app = BambooTui(cfg=cfg, plugin_id=args.plugin)
    inline = True if args.inline or not args.no_inline else False
    app.run(inline=inline, inline_no_clear=True, mouse=False)


if __name__ == "__main__":
    main()
