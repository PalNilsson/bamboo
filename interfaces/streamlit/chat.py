
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
AskPanDA Streamlit UI (MCP-first)

This UI talks directly to the AskPanDA MCP server (stdio transport) instead of
invoking the old clients.selection CLI.

Run:
  streamlit run interfaces/streamlit/chat.py

Notes:
- For dev convenience, this UI can spawn the MCP server process via stdio using:
    <python> -m askpanda_mcp.server
- For production / multi-client use, consider adding Streamable HTTP transport
  to your MCP server and connecting via HTTP instead.
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from typing import Any, Dict, Optional

import streamlit as st

from interfaces.shared.mcp_client import MCPServerConfig, MCPStdioClient


# -------------------- MCP helpers -------------------- #

@st.cache_resource
def _get_mcp_client(command: str, args_json: str, env_json: str) -> MCPStdioClient:
    """
    Cache a connected MCP client across Streamlit reruns.

    IMPORTANT: This keeps the stdio server process + session alive for the duration
    of the Streamlit session, which is what we want for chat UX.
    """
    try:
        args = json.loads(args_json)
        if not isinstance(args, list):
            raise ValueError("args must be a JSON list")
    except Exception:
        args = ["-m", "askpanda_mcp.server"]

    try:
        env = json.loads(env_json) if env_json.strip() else None
        if env is not None and not isinstance(env, dict):
            raise ValueError("env must be a JSON object")
    except Exception:
        env = None

    cfg = MCPServerConfig(command=command, args=args, env=env)
    client = MCPStdioClient(cfg)
    client.connect()
    return client


def _tool_names(tools) -> set[str]:
    names = set()
    for t in tools or []:
        n = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)
        if n:
            names.add(n)
    return names


def _render_call_tool_result(result: Any) -> str:
    """
    Convert MCP CallToolResult into a markdown string, robustly across SDK versions.
    """
    # Typical: result.content is list of content items with .type/.text (or dicts).
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")

    if not content:
        # Some SDK variants return list directly
        if isinstance(result, list):
            content = result

    if not content:
        return str(result)

    parts: list[str] = []
    for item in content:
        t = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
        if t == "text":
            txt = getattr(item, "text", None) or (item.get("text") if isinstance(item, dict) else "")
            parts.append(txt)
        elif t == "json":
            obj = getattr(item, "json", None) or (item.get("json") if isinstance(item, dict) else None)
            parts.append(f"```json\n{json.dumps(obj, indent=2, sort_keys=True)}\n```")
        else:
            # fallback
            parts.append(f"```json\n{json.dumps(item if isinstance(item, dict) else item.__dict__, indent=2, default=str)}\n```")

    return "\n\n".join([p for p in parts if p])


def _extract_task_id(text: str) -> Optional[int]:
    # Very loose heuristic: PanDA/JEDI task IDs are typically long integers.
    m = re.search(r"\b(\d{6,})\b", text)
    return int(m.group(1)) if m else None


def _auto_answer(client: MCPStdioClient, question: str, session_id: str, model: str) -> str:
    """
    Minimal "auto" orchestration for early bootstrapping.

    Preference order:
      1) call askpanda_answer if server provides it (recommended server-side orchestration)
      2) call askpanda_chat if provided
      3) fallback: doc search (+ task status if a task id is detected)
    """
    tools = client.list_tools()
    names = _tool_names(tools)

    if "askpanda_answer" in names:
        res = client.call_tool("askpanda_answer", {"question": question, "session_id": session_id, "model": model})
        return _render_call_tool_result(res)

    if "askpanda_chat" in names:
        res = client.call_tool("askpanda_chat", {"message": question, "session_id": session_id, "model": model})
        return _render_call_tool_result(res)

    # fallback: use basic tools
    chunks: list[str] = []
    if "panda_doc_search" in names:
        res = client.call_tool("panda_doc_search", {"query": question, "k": 5})
        chunks.append("### Doc search\n" + _render_call_tool_result(res))

    task_id = _extract_task_id(question)
    if task_id and "panda_task_status" in names:
        res = client.call_tool("panda_task_status", {"task_id": task_id})
        chunks.append(f"### Task status ({task_id})\n" + _render_call_tool_result(res))

    if not chunks:
        return (
            "I can reach the MCP server, but I don't see an orchestration tool like "
            "`askpanda_answer` and I can't find fallback tools like `panda_doc_search`. "
            "Try adding an `askpanda_answer` tool server-side, or enable the dummy tools."
        )

    return "\n\n".join(chunks)


# ---------------- Chat state helpers ---------------- #

def create_new_chat(title: str | None = None) -> Dict[str, Any]:
    """Create a new chat object with its own session_id and greeting."""
    chat_id = uuid.uuid4().hex
    session_id = f"streamlit-{chat_id[:8]}"
    return {
        "id": chat_id,
        "title": title or "New chat",
        "session_id": session_id,
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "Hello! I'm AskPanDA (Streamlit). "
                    "This UI talks to the AskPanDA MCP server and can call tools."
                ),
            }
        ],
        "last_error": None,
    }


def update_chat_title_from_prompt(chat: Dict[str, Any], prompt: str) -> None:
    """Auto-set chat title from first user prompt."""
    if chat.get("title") in ("New chat", None) and prompt.strip():
        chat["title"] = prompt.strip()[:48] + ("…" if len(prompt.strip()) > 48 else "")


# ---------------- Streamlit app ---------------- #

st.set_page_config(page_title="AskPanDA (MCP)", page_icon="🐼", layout="wide")

# Inject CSS for a centered main column and chat-like input styling
st.markdown(
    """
<style>
/* ChatGPT-like centered main column */
section[data-testid="stMain"] > div {
    max-width: 900px;
    margin: 0 auto;
}

/* Input container spacing */
div[data-testid="stChatInput"] {
    padding-top: 12px;
    padding-bottom: 24px;
}

/* Chat input wrapper */
div[data-testid="stChatInput"] > div {
    border-radius: 18px !important;
    border: 1px solid rgba(0, 0, 0, 0.2);
    background: #f2f2f2 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# Initialize chats on first run
if "chats" not in st.session_state:
    st.session_state.chats = [create_new_chat()]
    st.session_state.current_chat_index = 0

chats = st.session_state.chats
current_chat = chats[st.session_state.current_chat_index]

# ---------------- Sidebar ---------------- #
with st.sidebar:
    st.title("🐼 AskPanDA (MCP)")
    st.caption("Streamlit interface calling MCP tools")

    if st.button("🆕 New chat", use_container_width=True):
        st.session_state.chats.append(create_new_chat())
        st.session_state.current_chat_index = len(st.session_state.chats) - 1
        st.rerun()

    st.markdown("---")
    st.subheader("Connection")

    # Default to current interpreter for convenience
    default_command = sys.executable or "python3"
    command = st.text_input("Server command", value=default_command)

    args_json = st.text_input(
        "Server args (JSON list)",
        value='["-m", "askpanda_mcp.server"]',
        help='Example: ["-m", "askpanda_mcp.server"]',
    )

    env_json = st.text_area(
        "Extra env (JSON object, optional)",
        value="",
        height=80,
        help='Example: {"ASKPANDA_ENABLE_REAL_PANDA":"0"}',
    )

    # Allow clearing cache to force reconnect
    if st.button("🔌 Reconnect", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.subheader("Chat settings")
    model = st.text_input("Model (passed to server tool if supported)", value="gemini")
    current_chat["session_id"] = st.text_input(
        "Session ID",
        value=current_chat["session_id"],
        help="Passed to tools that support session-based context/memory.",
        key=f"session_id_{current_chat['id']}",
    )

    mode = st.radio("Mode", options=["Auto", "Manual tool call"], index=0)

# Connect (cached)
try:
    mcp = _get_mcp_client(command, args_json, env_json)
    tools = mcp.list_tools()
    prompts = mcp.list_prompts()
except Exception as e:
    st.error(f"Failed to connect to MCP server: {e}")
    st.stop()

# --------------- Top bar: capabilities --------------- #
with st.expander("Server capabilities", expanded=False):
    st.write("Tools:", sorted(list(_tool_names(tools))))
    st.write("Prompts:", [getattr(p, "name", None) or p.get("name") for p in (prompts or [])])

# ---------------- Render conversation ---------------- #
for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- Input and tool execution ---------------- #
prompt = st.chat_input("Ask PanDA…")

if prompt:
    # Add user message
    current_chat["messages"].append({"role": "user", "content": prompt})
    update_chat_title_from_prompt(current_chat, prompt)

    # Assistant response
    with st.chat_message("assistant"):
        try:
            if mode == "Manual tool call":
                tool_name = st.selectbox("Tool", options=sorted(list(_tool_names(tools))))
                args_text = st.text_area("Arguments (JSON)", value="{}", height=120)
                args = json.loads(args_text) if args_text.strip() else {}
                with st.spinner(f"Calling {tool_name}…"):
                    res = mcp.call_tool(tool_name, args)
                answer_md = _render_call_tool_result(res)
            else:
                with st.spinner("Thinking / calling tools…"):
                    answer_md = _auto_answer(
                        client=mcp,
                        question=prompt.strip(),
                        session_id=current_chat["session_id"],
                        model=model.strip(),
                    )

            st.markdown(answer_md)
            current_chat["messages"].append({"role": "assistant", "content": answer_md})
            current_chat["last_error"] = None
        except Exception as e:
            err = f"❌ Error calling MCP tools: `{e}`"
            st.markdown(err)
            current_chat["messages"].append({"role": "assistant", "content": err})
            current_chat["last_error"] = str(e)

    st.rerun()
