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
import tempfile
import time
from typing import Any, Dict, List, Optional

from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.events import Key
from textual.widgets import Footer, Header, Input, RichLog, Static

from interfaces.shared.mcp_client import MCPClientSync, MCPServerConfig

DEFAULT_PLUGIN = os.getenv("ASKPANDA_PLUGIN", "atlas")

# Maximum number of user+assistant turn *pairs* to keep in context.
# Each pair = 2 messages (1 user + 1 assistant), so 10 pairs = 20 messages.
_DEFAULT_HISTORY_TURNS = 10
try:
    _MAX_HISTORY_TURNS: int = int(os.getenv("BAMBOO_HISTORY_TURNS", str(_DEFAULT_HISTORY_TURNS)))
except ValueError:
    _MAX_HISTORY_TURNS = _DEFAULT_HISTORY_TURNS

ENV_LLM_DEFAULT_PROVIDER = "LLM_DEFAULT_PROVIDER"
ENV_LLM_DEFAULT_MODEL = "LLM_DEFAULT_MODEL"
ENV_LLM_DEFAULT_PROVIDER_ALT = "ASKPANDA_LLM_DEFAULT_PROVIDER"
ENV_LLM_DEFAULT_MODEL_ALT = "ASKPANDA_LLM_DEFAULT_MODEL"

ANSWER_TOOL_CANDIDATES: List[str] = ["bamboo_answer", "askpanda_answer", "bamboo_plan"]


_TASK_CMD_RE = re.compile(r"^/task\s+(\d{1,12})\s*$", re.IGNORECASE)
_JOB_CMD_RE = re.compile(r"^/job\s+(\d{1,12})\s*$", re.IGNORECASE)

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


def _span_detail(span_rec: Dict[str, Any]) -> str:
    """Format the most useful extra fields from a trace span as a short string.

    Args:
        span_rec: A parsed trace span dict from the NDJSON trace file.

    Returns:
        A compact human-readable summary of the span's event-specific fields,
        e.g. ``"route=rag"`` or ``"allowed=True reason=keyword_allow"``.
    """
    event = str(span_rec.get("event", ""))

    if event == "tool_call":
        keys = span_rec.get("args_keys", [])
        return f"args={keys}" if keys else ""

    if event == "guard":
        allowed = span_rec.get("allowed")
        reason = span_rec.get("reason", "")
        llm = span_rec.get("llm_used", False)
        llm_tag = " llm=yes" if llm else ""
        return f"allowed={allowed} reason={reason}{llm_tag}"

    if event == "retrieval":
        backend = span_rec.get("backend", "")
        hits = span_rec.get("hits")
        hits_str = str(hits) if hits is not None else "?"
        return f"backend={backend} hits={hits_str}"

    if event == "llm_call":
        provider = span_rec.get("provider", "")
        model = span_rec.get("model", "")
        inp = span_rec.get("input_tokens")
        out = span_rec.get("output_tokens")
        tokens = ""
        if inp is not None or out is not None:
            tokens = f" tokens={inp}→{out}"
        return f"{provider}/{model}{tokens}"

    if event == "synthesis":
        route = span_rec.get("route", "")
        return f"route={route}"

    # Unknown event — show all extra keys.
    skip = {"bamboo_trace", "event", "tool", "ts", "duration_ms"}
    extras = {k: v for k, v in span_rec.items() if k not in skip}
    return str(extras) if extras else ""


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

    #thinking {
        height: auto;
        padding: 0 2 0 3;
        display: none;
    }

    #thinking.active {
        display: block;
    }

    """

    BINDINGS = [
        Binding("escape", "focus_input", "Focus input", show=False),
        Binding("ctrl+l", "clear", "Clear", show=True),
        Binding("ctrl+d", "quit", "Quit", show=False),
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("pageup", "scroll_up", "Scroll up", show=True),
        Binding("pagedown", "scroll_down", "Scroll down", show=True),
        Binding("shift+pageup", "scroll_up", "Scroll up", show=False),
        Binding("shift+pagedown", "scroll_down", "Scroll down", show=False),
        Binding("ctrl+home", "scroll_home", "Scroll to top", show=False),
        Binding("ctrl+end", "scroll_end_action", "Scroll to bottom", show=False),
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
        self._history: List[Dict[str, str]] = []  # In-memory chat history for multi-turn context
        self._mcp_ready: bool = False
        self._last_raw_result: Any = None  # Most recent raw MCP tool result for /json
        self._last_task_id: Optional[int] = None  # Most recent task ID queried
        self._last_job_id: Optional[int] = None   # Most recent job ID queried
        self._command_history: List[str] = []     # Input history for arrow-up/down recall
        self._cmd_history_pos: int = -1           # Current position in _command_history (-1 = not browsing)
        self._thinking_task: Optional[Any] = None  # Textual Timer for animated thinking indicator
        self._last_spans: List[Dict[str, Any]] = []  # Trace spans for last request
        self._trace_file: str = os.path.join(
            tempfile.gettempdir(), f"bamboo_trace_{os.getpid()}.jsonl"
        )

        self.banner_widget: Optional[Static] = None
        self.transcript: Optional[RichLog] = None
        self.thinking_widget: Optional[Static] = None
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

            self.thinking_widget = Static("", id="thinking")
            yield self.thinking_widget

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

        self._render_banner_placeholder()
        self._write_system("Starting… initializing MCP…")

        # Disable mouse capture so the terminal can handle text selection normally.
        # Trackpad scrolling is handled via PageUp/PageDown bindings instead.
        try:
            driver = getattr(self, "_driver", None)
            if driver and hasattr(driver, "disable_mouse_support"):
                driver.disable_mouse_support()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

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
            # Fetch LLM info from the server-side health tool — calling
            # get_llm_info() directly here fails because the LLM selector
            # is not yet initialised at this point in the startup sequence.
            llm_info = ""
            try:
                health_res = await self._to_thread(
                    self.mcp.call_tool, "bamboo_health", {}
                )
                health_text = _extract_text(health_res)
                for line in health_text.splitlines():
                    if line.strip().startswith("- llm_info:"):
                        llm_info = line.split(":", 1)[1].strip()
                        break
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            llm_suffix = f"\n[LLM selected] {llm_info}" if llm_info and llm_info != "not configured" else ""
            self._write_system(
                f"Connected via {self.cfg.transport}. Answer tool: {self.answer_tool or 'UNKNOWN'}.{llm_suffix}"
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
        if self.transcript:
            self.transcript.scroll_page_up(animate=False)

    def action_scroll_down(self) -> None:
        """Scroll the transcript down by one page."""
        if self.transcript:
            self.transcript.scroll_page_down(animate=False)

    def action_scroll_home(self) -> None:
        """Scroll the transcript to the top."""
        if self.transcript:
            self.transcript.scroll_home(animate=False)

    def action_scroll_end_action(self) -> None:
        """Scroll the transcript to the bottom."""
        if self.transcript:
            self.transcript.scroll_end(animate=False)

    def action_focus_input(self) -> None:
        """Focus the input widget."""
        if self.input_widget:
            self.input_widget.focus()

    async def action_clear(self) -> None:
        """Clear the transcript, conversation history, and HTTP response cache."""
        if self.transcript:
            self.transcript.clear()
        self._history = []
        try:
            from askpanda_atlas._cache import clear as _cache_clear  # type: ignore[import]
            _cache_clear()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        self._write_system("Transcript cleared, context memory reset, and HTTP cache flushed.")

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

        # Append to command history, avoiding consecutive duplicates.
        if not self._command_history or self._command_history[-1] != text:
            self._command_history.append(text)
        self._cmd_history_pos = -1  # Reset browsing position after submit

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

    async def on_key(self, event: Key) -> None:
        """Handle arrow-up/down for command history recall in the input field.

        Arrow-up navigates backwards through :attr:`_command_history`;
        arrow-down navigates forwards.  Any other key resets the browsing
        position so the next up-press always starts from the most recent entry.

        Args:
            event: The key event from Textual.
        """
        if not self.input_widget or not self.input_widget.has_focus:
            return

        if not self._command_history:
            return

        if event.key == "up":
            event.stop()
            event.prevent_default()
            if self._cmd_history_pos == -1:
                # Start browsing from the most recent entry.
                self._cmd_history_pos = len(self._command_history) - 1
            elif self._cmd_history_pos > 0:
                self._cmd_history_pos -= 1
            self.input_widget.value = self._command_history[self._cmd_history_pos]
            # Move cursor to end of the restored text.
            self.input_widget.cursor_position = len(self.input_widget.value)

        elif event.key == "down":
            event.stop()
            event.prevent_default()
            if self._cmd_history_pos == -1:
                return  # Already at the empty prompt — nothing to do.
            if self._cmd_history_pos < len(self._command_history) - 1:
                self._cmd_history_pos += 1
                self.input_widget.value = self._command_history[self._cmd_history_pos]
            else:
                # Past the end — clear the input and stop browsing.
                self._cmd_history_pos = -1
                self.input_widget.value = ""
            self.input_widget.cursor_position = len(self.input_widget.value)

        else:
            # Any other key resets history browsing.
            self._cmd_history_pos = -1

    async def _handle_question(self, question: str) -> None:
        """Send a question to the answer tool and render the response.

        Appends the user turn to the in-memory history, sends the full history
        to the server on each call, then records the assistant reply.  History
        is capped at ``_MAX_HISTORY_TURNS`` user+assistant pairs so the context
        window stays manageable.

        Displays a "thinking" indicator while the server processes the request,
        then replaces it with the assistant response on completion.

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

        # Append the current user turn to history before sending.
        self._history.append({"role": "user", "content": question})

        args: Dict[str, Any] = {
            "question": question,
            "messages": list(self._history),
            "include_raw": self.debug_mode,
        }

        # Store task/job IDs if present so /json and /inspect can access them.
        try:
            import re as _re
            m = _re.search(r"\btask[:#/\-\s]+(\d{4,12})\b", question, _re.IGNORECASE)
            if m:
                self._last_task_id = int(m.group(1))
            m_job = _re.search(
                r"\b(?:job|pandaid|panda[\s_-]?id)[:#/\-\s]+(\d{4,12})\b",
                question, _re.IGNORECASE,
            )
            if m_job:
                self._last_job_id = int(m_job.group(1))
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        thinking: bool = self._write_thinking()
        try:
            _pre_pos = self._snapshot_trace_position()
            res = await self._to_thread(self.mcp.call_tool, self.answer_tool, args)
            self._last_spans = self._collect_spans(_pre_pos)
            self._last_raw_result = res  # Available via /json
            if self.debug_mode:
                self._write_tool(self.answer_tool, args, res)

            out = _extract_text(res) or "*(No text output; enable /debug on to see raw result.)*"
            self._replace_thinking(thinking, out)

            # Record the assistant reply and enforce the history cap.
            self._history.append({"role": "assistant", "content": out})
            self._cap_history()

        except Exception as exc:
            self._replace_thinking(thinking, None)
            self._write_error(str(exc))
            # Remove the user turn we optimistically appended on failure.
            if self._history and self._history[-1] == {"role": "user", "content": question}:
                self._history.pop()

    def _cap_history(self) -> None:
        """Trim ``_history`` to at most ``_MAX_HISTORY_TURNS`` user+assistant pairs.

        Each pair consists of one user message followed by one assistant
        message (2 entries).  The most recent turns are kept; older turns
        are discarded from the front of the list.
        """
        max_messages = _MAX_HISTORY_TURNS * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    async def _handle_command(self, cmdline: str) -> None:
        """Handle slash commands.

        Dispatches the ``/task <id>`` and ``/job <id>`` shorthands directly,
        then delegates all other commands to :meth:`_dispatch_slash_command`.

        Args:
            cmdline: Raw command line beginning with ``/``.
        """
        m_task = _TASK_CMD_RE.match(cmdline.strip())
        if m_task:
            task_id = m_task.group(1)
            await self._handle_question(
                f"Summarize the status of task {task_id} including dataset info."
            )
            return

        m_job = _JOB_CMD_RE.match(cmdline.strip())
        if m_job:
            job_id = m_job.group(1)
            await self._handle_question(
                f"Analyse the failure of job {job_id} and explain why it failed."
            )
            return

        await self._dispatch_slash_command(cmdline)

    async def _dispatch_slash_command(self, cmdline: str) -> None:
        """Dispatch a slash command to the appropriate handler.

        Split out from :meth:`_handle_command` to keep cyclomatic complexity
        under the project limit (max-complexity = 15).

        Args:
            cmdline (str): Raw command line beginning with ``/``.
        """
        parts = cmdline.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("/help", "/?"):
            self._cmd_help()
            return
        if cmd in ("/exit", "/quit"):
            self.exit()
            return
        if cmd == "/tools":
            await self._cmd_tools()
            return
        if cmd == "/tracing":
            self._handle_tracing_command()
            return
        if cmd == "/json":
            await self._handle_json_command()
            return
        if cmd == "/inspect":
            await self._handle_inspect_command()
            return
            self._handle_history_command()
            return
        if cmd == "/plugin":
            await self._cmd_plugin(args)
            return
        if cmd == "/debug":
            self._cmd_debug(args)
            return
        if cmd == "/clear":
            await self.action_clear()
            return
        self._write_system(f"Unknown command: {cmdline} (try /help)")

    def _cmd_help(self) -> None:
        """Display the slash-command reference."""
        self._write_system(
            "Commands:\n"
            "  /help                 Show this help\n"
            "  /tools                List tools exposed by the MCP server\n"
            "  /task <id>             Shorthand for: status of task <id>\n"
            "  /job <id>              Shorthand for: analyse failure of job <id>\n"
            "  /json                 Show verbatim BigPanDA API response (raw server JSON)\n"
            "  /inspect              Show structured evidence dict (job counts, sites, errors)\n"
            "  /tracing              Show timing + trace spans for last request\n"
            "  /history              Show turns currently held in context memory\n"
            "  /plugin <id>          Switch plugin (affects banner tool name)\n"
            "  /debug on|off         Toggle debug (shows tool calls + raw results)\n"
            "  /clear                Clear transcript, context memory, and HTTP cache\n"
            "  /exit, /quit          Exit the app\n"
            "\n"
            "Tip: Use PageUp/PageDown to scroll. To copy text, hold Option (macOS) or\n"
            "     Shift (Linux/Windows) while selecting with the mouse.\n"
            "Use --no-inline to use the alternate screen."
        )

    async def _cmd_tools(self) -> None:
        """Refresh and display the list of tools registered on the MCP server."""
        if self._mcp_ready:
            await self._refresh_tools()
        if not self.tool_names:
            self._write_system("No tools returned by server.")
            return
        body = "\n".join(f"- `{n}`" for n in self.tool_names)
        self._write_panel(Markdown(body), title=f"{_now()}  tools")

    async def _cmd_plugin(self, args: List[str]) -> None:
        """Switch the active plugin and reload banner and tools.

        Args:
            args (List[str]): Remaining command tokens; first element is the
                new plugin ID when present.
        """
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

    def _cmd_debug(self, args: List[str]) -> None:
        """Toggle or explicitly set debug mode.

        Args:
            args (List[str]): Remaining command tokens; first element may be
                ``"on"`` or ``"off"``.
        """
        if args and args[0].lower() in ("on", "off"):
            self.debug_mode = args[0].lower() == "on"
        else:
            self.debug_mode = not self.debug_mode
        self._write_system(f"Debug {'ON' if self.debug_mode else 'OFF'}.")

    def _handle_history_command(self) -> None:
        """Render the current in-context conversation turns as a Rich table.

        Shows each turn's role, a character count, and a truncated preview of
        the content.  Useful for debugging follow-up resolution and verifying
        that the history cap is working as expected.
        """
        if not self._history:
            turns_word = "turn" if _MAX_HISTORY_TURNS == 1 else "turns"
            self._write_system(
                f"No conversation history in context yet.\n"
                f"(Cap: {_MAX_HISTORY_TURNS} {turns_word} = {_MAX_HISTORY_TURNS * 2} messages. "
                f"Set BAMBOO_HISTORY_TURNS to change.)"
            )
            return

        table = Table(
            show_header=True,
            header_style="bold",
            box=None,
            pad_edge=False,
            collapse_padding=True,
        )
        table.add_column("#", style="dim", no_wrap=True)
        table.add_column("Role", style="cyan", no_wrap=True)
        table.add_column("Chars", justify="right", style="yellow", no_wrap=True)
        table.add_column("Preview", style="")

        for idx, msg in enumerate(self._history, start=1):
            role = str(msg.get("role", "?"))
            content = str(msg.get("content", ""))
            preview = content[:120].replace("\n", " ")
            if len(content) > 120:
                preview += "…"
            table.add_row(str(idx), role, str(len(content)), preview)

        pair_count = len(self._history) // 2
        remainder = len(self._history) % 2
        summary = f"{pair_count} complete pair(s)"
        if remainder:
            summary += " + 1 pending user turn"
        summary += f"  |  cap: {_MAX_HISTORY_TURNS} pairs ({_MAX_HISTORY_TURNS * 2} messages)"

        table.add_section()
        table.add_row("", "[bold]total[/bold]", f"[bold yellow]{len(self._history)}[/bold yellow]", summary)

        self._write_panel(table, title=f"{_now()}  context history", border_style="cyan")

    # ------------------------------------------------------------------
    # Tracing helpers
    # ------------------------------------------------------------------

    def _snapshot_trace_position(self) -> int:
        """Return the current end-of-file byte position of the trace file.

        Called immediately before each MCP tool invocation so that
        :meth:`_collect_spans` can read only the spans produced by that
        specific request.

        Returns:
            Current file size in bytes, or 0 if tracing is not configured.
        """
        try:
            return os.path.getsize(self._trace_file)
        except OSError:
            return 0

    def _collect_spans(self, position: int) -> List[Dict[str, Any]]:
        """Read trace spans written to the trace file since *position*.

        Args:
            position: Byte offset obtained from :meth:`_snapshot_trace_position`
                before the request was issued.

        Returns:
            List of parsed span dicts in emission order.  Non-JSON lines and
            lines without the ``"bamboo_trace"`` sentinel are silently skipped.
        """
        spans: List[Dict[str, Any]] = []
        try:
            with open(self._trace_file, "r", encoding="utf-8") as fh:
                fh.seek(position)
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                        if isinstance(obj, dict) and obj.get("bamboo_trace"):
                            spans.append(obj)
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass
        return spans

    def _handle_tracing_command(self) -> None:
        """Render a Rich table of trace spans for the most recent request.

        Displays event type, tool name, duration, and the most important
        event-specific fields (route, backend/hits, guard verdict, token
        counts) in a compact panel.  If no spans are available — because
        tracing is disabled or no request has been made yet — a helpful
        explanation is shown instead.
        """
        if not self._last_spans:
            if self.cfg.transport == "http":
                self._write_system(
                    "No trace spans available.\n\n"
                    "HTTP transport: set BAMBOO_TRACE=1 and BAMBOO_TRACE_FILE on the server "
                    "process, then tail the file manually:\n"
                    "  tail -f $BAMBOO_TRACE_FILE | grep bamboo_trace | jq ."
                )
            else:
                self._write_system(
                    "No trace spans yet — ask a question first, then type /tracing."
                )
            return

        # Build the summary table.
        table = Table(
            show_header=True,
            header_style="bold",
            box=None,
            pad_edge=False,
            collapse_padding=True,
        )
        table.add_column("Event", style="cyan", no_wrap=True)
        table.add_column("Tool", style="", no_wrap=True)
        table.add_column("ms", justify="right", style="yellow", no_wrap=True)
        table.add_column("Detail", style="dim")

        total_ms: float = 0.0
        for span_rec in self._last_spans:
            event = str(span_rec.get("event", ""))
            tool = str(span_rec.get("tool", ""))
            duration = float(span_rec.get("duration_ms", 0.0))
            if event == "tool_call":
                total_ms = duration  # outermost span — use as request total

            detail = _span_detail(span_rec)
            table.add_row(event, tool, f"{duration:.0f}", detail)

        # Footer row with total.
        table.add_section()
        table.add_row(
            "[bold]total[/bold]",
            "",
            f"[bold yellow]{total_ms:.0f}[/bold yellow]",
            "wall time for full tool_call span",
        )

        self._write_panel(table, title=f"{_now()}  tracing", border_style="cyan")

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

    def _write_thinking(self) -> bool:
        """Start the animated thinking indicator below the transcript.

        Uses :meth:`set_interval` (Textual's own timer mechanism) to cycle
        through ``Thinking.`` → ``Thinking..`` → ``Thinking...`` once per
        second, with the current wall-clock time updated on each frame.

        :meth:`set_interval` integrates with Textual's event loop directly,
        avoiding the timing issues that ``asyncio.ensure_future`` + ``asyncio.sleep``
        can cause when the loop is shared with blocking worker threads.

        Returns:
            True if the indicator was started, False if unavailable.
        """
        if not self.thinking_widget:
            return False
        self.thinking_widget.add_class("active")

        frames = ["Thinking.", "Thinking..", "Thinking..."]
        counter = [0]  # mutable cell so the closure can increment it

        def _tick() -> None:
            import datetime as _dt
            now = _dt.datetime.now().strftime("%H:%M:%S")
            if self.thinking_widget:
                self.thinking_widget.update(
                    Text(f"{now}  {frames[counter[0] % len(frames)]}", style="dim italic")
                )
            counter[0] += 1

        _tick()  # Show the first frame immediately; timer fires subsequent ones
        self._thinking_task = self.set_interval(1.0, _tick)
        return True

    def _replace_thinking(self, indicator: bool, answer: Optional[str]) -> None:
        """Cancel the animated thinking indicator and optionally write the answer.

        Args:
            indicator: Value returned by _write_thinking (True if it was
                started).
            answer: The assistant response text (Markdown), or None to
                simply hide the indicator without writing a response.
        """
        if self._thinking_task is not None:
            self._thinking_task.stop()
            self._thinking_task = None
        if self.thinking_widget:
            self.thinking_widget.remove_class("active")
            self.thinking_widget.update("")
        if answer is not None:
            self._write_assistant(answer)

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
        """Write a debug tool call block showing args and full raw result.

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

    async def _handle_json_command(self) -> None:
        """Display the verbatim BigPanDA API response for the last task or job.

        Calls ``bamboo_last_evidence`` with ``mode='raw'`` to retrieve the
        response from the server-side evidence store — no fresh HTTP request.
        Shows the full BigPanDA payload as returned by the API before any
        summarisation.  Use ``/inspect`` for the compact evidence dict instead.
        """
        if not self._mcp_ready:
            self._write_system("Not connected yet.")
            return
        try:
            res = await self._to_thread(
                self.mcp.call_tool, "bamboo_last_evidence", {"mode": "raw"},
            )
            text = _extract_text(res)
            if not text:
                self._write_system("No evidence stored yet — ask about a task or job first.")
                return
            parsed = json.loads(text)
            if "error" in parsed:
                self._write_system(parsed["error"])
                return
            payload = parsed.get("evidence", parsed)
            tool = parsed.get("tool", "")
            title = f"{_now()}  raw JSON — {tool}" if tool else f"{_now()}  raw JSON"
            self._write_panel(
                Markdown(f"```json\n{_pretty(payload)}\n```"),
                title=title,
                border_style="yellow",
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._write_error(f"Could not fetch raw JSON: {exc}")

    async def _handle_inspect_command(self) -> None:
        """Dump the structured evidence dict for the last task or job query.

        Calls ``bamboo_last_evidence`` with ``mode='evidence'`` to retrieve
        the compact evidence dict from the server-side store — the same data
        that was sent to the LLM: job counts, site breakdown, error tallies,
        sample job records, and the PanDA ID list.  No HTTP request is made.

        Use ``/json`` for the verbatim BigPanDA API response instead.
        """
        if not self._mcp_ready:
            self._write_system("Not connected yet.")
            return
        try:
            res = await self._to_thread(
                self.mcp.call_tool, "bamboo_last_evidence", {"mode": "evidence"},
            )
            text = _extract_text(res)
            if not text:
                self._write_system("No evidence stored yet — ask about a task or job first.")
                return
            parsed = json.loads(text)
            if "error" in parsed:
                self._write_system(parsed["error"])
                return
            evidence = parsed.get("evidence", parsed)
            tool = parsed.get("tool", "")
            title = f"{_now()}  evidence — {tool}" if tool else f"{_now()}  evidence"
            self._write_panel(
                Markdown(f"```json\n{_pretty(evidence)}\n```"),
                title=title,
                border_style="cyan",
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._write_error(f"Could not fetch evidence: {exc}")

    def _write_raw_json(self, res: Any) -> None:
        """Write the full raw BigPanDA JSON for the last tool result.

        Triggered by the ``/json`` command. Extracts the BigPanDA payload
        from the MCP result envelope, handling both dict and SDK object forms.

        Args:
            res (Any): Raw MCP tool result to display.
        """
        payload = self._extract_bigpanda_payload(res)
        if payload is None:
            self._write_system("Could not extract BigPanDA payload from last result.")
            return
        self._write_panel(
            Markdown(f"```json\n{_pretty(payload)}\n```"),
            title=f"{_now()}  raw JSON (BigPanDA)",
            border_style="cyan",
        )

    @staticmethod
    def _extract_bigpanda_payload(res: Any) -> Any:
        """Extract the raw BigPanDA payload from an MCP tool result.

        The MCP SDK wraps results in a ``CallToolResult`` object with a
        ``content`` list of ``TextContent`` items. The text itself is the
        LLM answer, which embeds structured data in a JSON prefix when the
        tool returns evidence. This method peels back all those layers.

        Args:
            res: Raw value returned by ``MCPClientSync.call_tool``.

        Returns:
            The BigPanDA payload dict, or ``None`` if it cannot be found.
        """
        # 1. Unwrap MCP SDK CallToolResult object → get text string.
        text: str | None = None
        content_attr = getattr(res, "content", None)
        if content_attr is not None:
            # SDK object: res.content is a list of TextContent items.
            items = content_attr if isinstance(content_attr, list) else [content_attr]
            for item in items:
                t = getattr(item, "text", None) or (
                    item.get("text") if isinstance(item, dict) else None
                )
                if isinstance(t, str) and t.strip():
                    text = t
                    break
        elif isinstance(res, list) and res:
            first = res[0]
            text = (
                first.get("text") if isinstance(first, dict)
                else getattr(first, "text", None)
            )
        elif isinstance(res, dict):
            text = res.get("text")
        elif isinstance(res, str):
            text = res

        if not isinstance(text, str):
            return None

        # 2. Try to parse the text as JSON — panda_task_status returns a
        # dict {"evidence": {"payload": {...}, ...}, "text": "..."} serialised
        # as JSON inside the TextContent.text field.
        text = text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                evidence = parsed.get("evidence")
                if isinstance(evidence, dict):
                    payload = evidence.get("payload")
                    if isinstance(payload, dict):
                        return payload
                    # No payload key — return the evidence dict itself.
                    return evidence
                return parsed
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        # 3. Text is plain (LLM answer) — payload not embedded.
        return None


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
        _trace_file = os.path.join(tempfile.gettempdir(), f"bamboo_trace_{os.getpid()}.jsonl")
        _stdio_env = os.environ.copy()
        _stdio_env["BAMBOO_TRACE"] = "1"
        _stdio_env["BAMBOO_TRACE_FILE"] = _trace_file
        _stdio_env["BAMBOO_QUIET"] = "1"  # redirect server stderr → /dev/null
        cfg = MCPServerConfig(
            transport="stdio",
            stdio_command=sys.executable,
            stdio_args=["-m", "bamboo.server"],
            stdio_env=_stdio_env,
        )

    app = BambooTui(cfg=cfg, plugin_id=args.plugin)
    inline = True if args.inline or not args.no_inline else False
    app.run(inline=inline, inline_no_clear=True, mouse=False)


if __name__ == "__main__":
    main()
