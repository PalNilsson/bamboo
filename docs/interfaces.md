# Interfaces

Bamboo provides multiple user interfaces built on top of the same MCP server.
All interfaces are thin clients. Tool orchestration, routing, and LLM selection
are handled server-side by the Bamboo MCP server.

---

# 1. Streamlit Web UI

The Streamlit interface provides a browser-based chat experience.

## Run (stdio)

```bash
streamlit run interfaces/streamlit/chat.py --transport stdio
```

## Run (HTTP)

```bash
streamlit run interfaces/streamlit/chat.py --transport http --http-url http://localhost:8000/mcp
```

### Characteristics

- Browser-based UI
- Good for demos and collaborative workflows
- Easy to expose over HTTP
- Stable and robust for normal use

---

# 2. Textual Terminal UI

The Textual interface provides a Copilot-style terminal experience.

It supports two transport modes:

- `stdio` (local MCP server subprocess)
- `http` (remote or local HTTP MCP endpoint)

## Run (stdio)

```bash
python interfaces/textual/chat.py --transport stdio --no-inline
```

## Run (HTTP)

```bash
python interfaces/textual/chat.py --transport http --http-url http://localhost:8000/mcp --no-inline
```

---

## Inline vs No-Inline (Textual)

The Textual TUI supports two rendering modes.

### --no-inline (Alternate Screen) — Recommended

Uses the terminal’s alternate screen buffer (like `vim`, `less`).

**Advantages:**

- Full terminal control
- Proper resizing
- Reliable scrolling
- Clean exit restores previous shell screen
- Most stable mode

Recommended for production terminal usage.

---

### --inline (Copy-Friendly Mode)

Renders inside the normal terminal scrollback.

**Advantages:**

- Easier mouse selection and copy/paste
- UI remains visible in shell history after exit

**Trade-offs:**

- Uses fixed inline height
- More sensitive to other processes writing to stdout
- Slightly less robust than alternate screen mode

Example:

```bash
python interfaces/textual/chat.py --transport stdio --inline
```

Optional: control inline height

```bash
export BAMBOO_TUI_INLINE_HEIGHT=60
```

---

## Slash Commands (Textual TUI)

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/tools` | List tools registered on the server |
| `/task <id>` | Shorthand for "summarise task \<id\>" |
| `/json` | Show raw BigPanDA JSON for the last task query |
| `/tracing` | Show timing and trace spans for the last request |
| `/history` | Show turns currently held in context memory |
| `/debug on\|off` | Toggle verbose tool call output |
| `/clear` | Clear transcript **and reset context memory** |
| `/exit` | Quit |

---

## Context Memory (Multi-Turn Chat)

The Textual TUI maintains an in-memory conversation history that is sent
to the server on every question.  This enables follow-up questions such as:

- *"Tell me more about the brokerage part."* (after a RAG answer)
- *"What about the failed jobs?"* (after a task query)
- *"Is that the same error as last time?"* (after a log analysis)

### How it works

- Each user question and its assistant reply are appended to `_history`.
- The full history is included as `messages` in every `bamboo_answer` call.
- The server extracts prior turns and injects them between the system prompt
  and the synthesised user message, so the LLM sees the conversation context.
- History uses **raw question/answer text**, not the synthesised prompts that
  embed task JSON or RAG excerpts — keeping token counts low.

### History cap

History is capped at `BAMBOO_HISTORY_TURNS` user+assistant pairs (default 10).
Older turns are discarded from the front of the list when the cap is reached.

```bash
export BAMBOO_HISTORY_TURNS=5   # keep only the last 5 question+answer pairs
```

### Resetting history

- `/clear` clears the transcript **and** resets the history.
- Restarting the TUI always starts with empty history (no persistence).

### Inspecting history

```
/history
```

Displays a table of the turns currently in context — role, character count,
and a truncated preview.  Useful for debugging follow-up resolution.

---

Both Streamlit and Textual interfaces use:

```
interfaces/shared/mcp_client.py
```

Responsibilities:

- Support stdio and HTTP transports
- Manage connection lifecycle (lazy connect for TUI stability)
- Inherit environment variables (LLM configuration, plugin selection)
- Isolate subprocess output to avoid terminal corruption

---

# Architecture Overview

```
User Interface (Streamlit / Textual)
        ↓
Shared MCP Client
        ↓
Bamboo MCP Server
        ↓
Plugins + Tools (atlas, etc.)
        ↓
LLM Provider (OpenAI, Mistral, etc.)
```

All user interfaces are thin clients.
Server-side logic handles routing, planning, tool selection, and LLM execution.
