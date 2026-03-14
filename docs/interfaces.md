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

# 3. Shared MCP Client

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
