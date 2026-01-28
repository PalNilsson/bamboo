
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

"""
AskPanDA Streamlit Chat UI (MCP-first)

This Streamlit app connects to an AskPanDA MCP server using either:
  - STDIO transport (dev): spawns `python -m askpanda_mcp.server`
  - Streamable HTTP transport (prod): connects to `http://host:port/mcp`

Key Streamlit constraints handled here:
  - No widget calls inside cached functions.
  - The MCP client is wrapped in a synchronous interface for Streamlit.

Expected companion module:
  interfaces/shared/mcp_client.py

It must expose:
  - MCPServerConfig
  - MCPClientSync  (sync wrapper with methods: list_tools, list_prompts, call_tool, close)

Run:
  streamlit run interfaces/streamlit/chat.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import streamlit as st

# --- UI tweaks: pin chat input to bottom and neutralize focus styling ---
st.markdown(
        """
        <style>
  /* ChatGPT-like pinned input at the bottom, but avoid overlapping the sidebar */
  [data-testid="stChatInput"] {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 9999;
    background: var(--background-color, #fff);
    padding: 0.75rem 1rem 0.75rem 1rem;
    border-top: none !important; /* remove separator line */
    box-shadow: none !important;
  }

  /* When a sidebar is present, Streamlit typically reserves ~21rem on the left.
     This prevents the input from sliding underneath the left panel. */
  [data-testid="stChatInput"] {
    padding-left: calc(1rem + 21rem);
  }

  /* On smaller screens (or when sidebar overlays), don't reserve left space. */
  @media (max-width: 900px) {
    [data-testid="stChatInput"] {
      padding-left: 1rem;
    }
  }

  /* Give the main content room so it doesn't hide behind the fixed input */
  .block-container {
    padding-bottom: 6.5rem !important;
  }

  /* Aggressively neutralize focus styles to avoid red borders/rings */
  [data-testid="stChatInput"] *:focus,
  [data-testid="stChatInput"] *:focus-visible {
    outline: none !important;
    box-shadow: none !important;
  }

  /* Ensure the textarea border stays neutral even on focus */
  [data-testid="stChatInput"] textarea,
  [data-testid="stChatInput"] input {
    border-color: rgba(128,128,128,0.35) !important;
  }
  [data-testid="stChatInput"] textarea:focus,
  [data-testid="stChatInput"] textarea:focus-visible,
  [data-testid="stChatInput"] input:focus,
  [data-testid="stChatInput"] input:focus-visible {
    border-color: rgba(128,128,128,0.45) !important;
  }
</style>
        """,
        unsafe_allow_html=True,
    )
from interfaces.shared.mcp_client import MCPClientSync, MCPServerConfig


# -----------------------------
# Types
# -----------------------------
JSONDict = Dict[str, Any]
JSONList = List[Any]


@dataclass(frozen=True)
class UIConfig:
    """Configuration values gathered from Streamlit widgets.

    Attributes:
        transport: Connection transport ("stdio" or "http").
        stdio_command: Python executable to run the MCP server in stdio mode.
        stdio_args_json: JSON-encoded list of args for the stdio server command.
        stdio_env_json: JSON-encoded dict of environment vars for stdio.
        http_url: MCP streamable HTTP URL (e.g., http://localhost:8000/mcp).
    """

    transport: str
    stdio_command: str
    stdio_args_json: str
    stdio_env_json: str
    http_url: str


# -----------------------------
# Helpers (pure, no widgets)
# -----------------------------
def _safe_parse_json_list(value: str, fallback: Optional[List[str]] = None) -> List[str]:
    """Parse a JSON list of strings, returning a fallback on error.

    Args:
        value: JSON string that should decode to a list.
        fallback: Returned when parsing fails.

    Returns:
        A list of strings.
    """
    if fallback is None:
        fallback = ["-m", "askpanda_mcp.server"]
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            return fallback
        out: List[str] = []
        for item in parsed:
            out.append(str(item))
        return out
    except Exception:
        return fallback


def _safe_parse_json_dict(value: str) -> Optional[Dict[str, str]]:
    """Parse a JSON dict of string keys/values.

    Args:
        value: JSON string that should decode to a dict. Empty string => None.

    Returns:
        Dict[str, str] if valid, otherwise None.
    """
    if not value.strip():
        return None
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            return None
        out: Dict[str, str] = {}
        for k, v in parsed.items():
            out[str(k)] = str(v)
        return out
    except Exception:
        return None


def _extract_text_from_content(content_items: Any) -> str:
    """Extract human-readable text from MCP content items.

    MCP tool calls commonly return a list of content objects/dicts like:
      [{"type": "text", "text": "..."}]

    Args:
        content_items: Tool response.

    Returns:
        Concatenated text content.
    """
    if content_items is None:
        return ""

    # If response is an object with `.content`, try that.
    if hasattr(content_items, "content"):
        content_items = getattr(content_items, "content")

    parts: List[str] = []

    if isinstance(content_items, str):
        return content_items

    if isinstance(content_items, dict):
        # Single dict response
        if content_items.get("type") == "text":
            return str(content_items.get("text", ""))
        return json.dumps(content_items, indent=2)

    if isinstance(content_items, list):
        for item in content_items:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, indent=2))
                continue
            # Unknown object: try attributes
            if hasattr(item, "type") and getattr(item, "type") == "text" and hasattr(item, "text"):
                parts.append(str(getattr(item, "text")))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p.strip()])

    # Fallback
    try:
        return json.dumps(content_items, indent=2)
    except Exception:
        return str(content_items)


def _tool_names(tools_result: Any) -> List[str]:
    """Extract tool names from `session.list_tools()` results.

    Different MCP versions return different shapes:
      - list of Tool objects
      - dict-like tool definitions
      - object with `.tools` list

    Args:
        tools_result: The return value of MCP list_tools.

    Returns:
        Sorted list of tool names.
    """
    # unwrap `.tools` if present
    if hasattr(tools_result, "tools"):
        tools_result = getattr(tools_result, "tools")

    names: List[str] = []
    if tools_result is None:
        return names

    if isinstance(tools_result, list):
        for t in tools_result:
            if isinstance(t, dict) and "name" in t:
                names.append(str(t["name"]))
            elif hasattr(t, "name"):
                names.append(str(getattr(t, "name")))
            else:
                # last resort
                names.append(str(t))
    elif isinstance(tools_result, dict):
        # sometimes tools are returned as {"tools": [...]}
        inner = tools_result.get("tools")
        if isinstance(inner, list):
            return _tool_names(inner)
    return sorted(set(names))


def _prompt_names(prompts_result: Any) -> List[str]:
    """Extract prompt names from `session.list_prompts()` results.

    Args:
        prompts_result: The return value of MCP list_prompts.

    Returns:
        Sorted list of prompt names.
    """
    if hasattr(prompts_result, "prompts"):
        prompts_result = getattr(prompts_result, "prompts")

    names: List[str] = []
    if prompts_result is None:
        return names

    if isinstance(prompts_result, list):
        for p in prompts_result:
            if isinstance(p, dict) and "name" in p:
                names.append(str(p["name"]))
            elif hasattr(p, "name"):
                names.append(str(getattr(p, "name")))
            else:
                names.append(str(p))
    elif isinstance(prompts_result, dict):
        inner = prompts_result.get("prompts")
        if isinstance(inner, list):
            return _prompt_names(inner)
    return sorted(set(names))


def _guess_auto_tool(question: str, available_tools: Sequence[str]) -> Tuple[str, JSONDict]:
    """Pick a reasonable tool + arguments based on a question (light heuristic).

    This avoids requiring a server-side orchestration tool while you're bootstrapping.

    Args:
        question: User question.
        available_tools: Known tool names from the server.

    Returns:
        (tool_name, args)
    """
    q = question.lower().strip()

    # Prefer a single orchestration tool if you add it later.
    for candidate in ("askpanda_answer", "askpanda_chat", "askpanda_query"):
        if candidate in available_tools:
            return candidate, {"query": question}

    if "log" in q or "traceback" in q or "error" in q or "failed" in q:
        if "panda_log_analysis" in available_tools:
            return "panda_log_analysis", {"log_text": question}

    # crude task id extraction
    if "task" in q or "jedi" in q:
        if "panda_task_status" in available_tools:
            return "panda_task_status", {"task_id": question}

    if "queue" in q or "site" in q:
        if "panda_queue_info" in available_tools:
            return "panda_queue_info", {"site": question}

    if "panda_doc_search" in available_tools:
        return "panda_doc_search", {"query": question, "k": 5}

    # fallback: health
    if "askpanda_health" in available_tools:
        return "askpanda_health", {}

    # ultimate fallback: first available tool
    if available_tools:
        return available_tools[0], {}

    return "askpanda_health", {}


# -----------------------------
# Cached MCP client factory (NO WIDGETS INSIDE)
# -----------------------------
@st.cache_resource
def _get_mcp_client(cfg: UIConfig) -> MCPClientSync:
    """Create and cache an MCP client based on UIConfig.

    IMPORTANT: No Streamlit widgets may be called here.

    Args:
        cfg: UIConfig values (pure data).

    Returns:
        Connected MCPClientSync instance.
    """
    args = _safe_parse_json_list(cfg.stdio_args_json)
    env = _safe_parse_json_dict(cfg.stdio_env_json)

    if cfg.transport == "http":
        server_cfg = MCPServerConfig(
            transport="http",
            http_url=cfg.http_url,
        )
    else:
        server_cfg = MCPServerConfig(
            transport="stdio",
            stdio_command=cfg.stdio_command or sys.executable,
            stdio_args=args,
            stdio_env=env,
        )

    return MCPClientSync(server_cfg)


# -----------------------------
# Streamlit UI
# -----------------------------
def _sidebar_config() -> UIConfig:
    """Render sidebar widgets and return selected config.

    Returns:
        UIConfig with values from widgets.
    """
    st.sidebar.header("AskPanDA MCP Connection")

    transport = st.sidebar.selectbox("Transport", ["http", "stdio"], index=0)

    stdio_command = sys.executable
    stdio_args_json = st.sidebar.text_area(
        "STDIO args (JSON list)",
        value=json.dumps(["-m", "askpanda_mcp.server"]),
        height=70,
        help='Example: ["-m", "askpanda_mcp.server"]',
    )

    stdio_env_json = st.sidebar.text_area(
        "STDIO env (JSON object, optional)",
        value="",
        height=70,
        help='Example: {"PYTHONPATH": "/path/to/repo"}',
    )

    http_url = "http://localhost:8000/mcp"
    if transport == "http":
        http_url = st.sidebar.text_input("HTTP MCP URL", value=http_url)

    if st.sidebar.button("Reset MCP connection"):
        # Clear cached resources and rerun.
        st.cache_resource.clear()
        st.rerun()

    # End-to-end test / escape hatch: bypass tool routing and send the full chat
    # (including history) directly to the default LLM profile.
    if "bypass_routing" not in st.session_state:
        st.session_state["bypass_routing"] = True
    st.sidebar.toggle(
        "Bypass tool routing (send directly to default LLM)",
        key="bypass_routing",
        help="Useful for sanity-checking LLM configuration and as a future escape hatch.",
    )

    return UIConfig(
        transport=transport,
        stdio_command=stdio_command,
        stdio_args_json=stdio_args_json,
        stdio_env_json=stdio_env_json,
        http_url=http_url,
    )


def _render_connection_status(mcp: MCPClientSync) -> Tuple[List[str], List[str]]:
    """Fetch and display tools/prompts.

    Args:
        mcp: Connected MCP client.

    Returns:
        (tool_names, prompt_names)
    """
    with st.spinner("Fetching MCP capabilities..."):
        tools_result = mcp.list_tools()
        prompts_result = mcp.list_prompts()

    tool_names = _tool_names(tools_result)
    prompt_names = _prompt_names(prompts_result)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Tools")
        st.write(tool_names if tool_names else "No tools returned.")
    with c2:
        st.caption("Prompts")
        st.write(prompt_names if prompt_names else "No prompts returned.")

    return tool_names, prompt_names


def _manual_tool_panel(mcp: MCPClientSync, tool_names: Sequence[str]) -> None:
    """Render a manual tool invocation panel.

    Args:
        mcp: MCP client.
        tool_names: Available tool names.
    """
    st.subheader("Manual tool call")

    if not tool_names:
        st.info("No tools available to call.")
        return

    selected = st.selectbox("Tool", list(tool_names))
    args_text = st.text_area("Arguments (JSON object)", value="{}", height=120)

    if st.button("Call tool"):
        try:
            args = json.loads(args_text) if args_text.strip() else {}
            if not isinstance(args, dict):
                raise ValueError("Arguments must be a JSON object.")
        except Exception as e:
            st.error(f"Invalid JSON arguments: {e}")
            return

        try:
            result = mcp.call_tool(selected, args)
            st.code(_extract_text_from_content(result), language="text")
        except Exception as e:
            st.error(f"Tool call failed: {e}")


def _chat_panel(mcp: MCPClientSync, tool_names: Sequence[str]) -> None:
    """Renders the main chat UI.

    This implementation keeps the chat input fixed at the bottom by:
    - recording the submitted user message and triggering a rerun
    - generating the assistant response on the *next* run before rendering history

    Args:
        mcp: Connected MCP client (sync wrapper).
        tool_names: List of available tool names.
    """
    st.subheader("Chat")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # type: ignore[assignment]

    # If we have a pending assistant response, generate it first so it appears
    # in the history above the input box (ChatGPT-style).
    if st.session_state.get("_pending_assistant", False):
        st.session_state["_pending_assistant"] = False

        with st.spinner("Thinking…"):
            try:
                # Always route via the server-side orchestration tool.
                tool = "askpanda_answer"
                last_user = next(
                    (m["content"] for m in reversed(st.session_state["messages"]) if m.get("role") == "user"),
                    "",
                )
                result = mcp.call_tool(
                    tool,
                    {
                        "question": last_user,
                        "messages": st.session_state["messages"],
                        "bypass_routing": st.session_state.get("bypass_routing", False),
                    },
                )

                answer = _extract_text_from_content(result)
                if not answer.strip():
                    answer = "(Tool returned no text content.)"
            except Exception as e:  # noqa: BLE001
                answer = f"Tool call failed: {e}"

        st.session_state["messages"].append({"role": "assistant", "content": answer})  # type: ignore[index]
        # Rerun once more so the assistant message is rendered as part of history
        # and the input stays at the bottom.
        st.rerun()

    # Show chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input should be the last UI element so it stays at the bottom.
    question = st.chat_input("Ask AskPanDA…")

    if question:
        st.session_state["messages"].append({"role": "user", "content": question})  # type: ignore[index]
        st.session_state["_pending_assistant"] = True
        st.rerun()


def main() -> None:
    """Main Streamlit entrypoint."""
    st.set_page_config(page_title="AskPanDA (MCP)", layout="wide")

    # Style tweaks: remove the red focus border on the chat input.
    st.title("AskPanDA (MCP-first)")

    cfg = _sidebar_config()

    # Build (cached) MCP client – widgets are already done above
    try:
        mcp = _get_mcp_client(cfg)
    except Exception as e:
        st.error(f"Failed to create MCP client: {e}")
        st.stop()

    # Show tools/prompts
    try:
        tool_names, _prompt_names_list = _render_connection_status(mcp)
    except Exception as e:
        st.error(f"Failed to list tools/prompts: {e}")
        st.stop()

    # Tabs: Chat + Manual tool calls
    tab_chat, tab_tools = st.tabs(["Chat", "Tools"])
    with tab_chat:
        _chat_panel(mcp, tool_names)
    with tab_tools:
        _manual_tool_panel(mcp, tool_names)


if __name__ == "__main__":
    main()