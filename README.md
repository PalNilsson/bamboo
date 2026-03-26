# Bamboo

**Bamboo** is a lightweight MCP-based runtime with a plugin architecture for
AI-assisted scientific tools, primarily targeting PanDA/ATLAS workflows.

LLMs are used for *summarisation and explanation*, not as sources of truth.
Structured evidence is always returned alongside natural-language answers.

> **Status (March 2026):** core infrastructure is stable; plugins and
> documentation are still being expanded. The current focus is orchestration
> using tool families and planning for complex multi-step prompts.

---

## Quick start

### 1. Create a virtual environment

```bash
python3 -m venv ~/Development/venv-bamboo
source ~/Development/venv-bamboo/bin/activate
```

### 2. Install the packages

```bash
# Core MCP server — required
pip install -r requirements.txt
pip install -e ./core

# ATLAS / PanDA plugin — required for AskPanDA workflows
pip install -e ./packages/askpanda_atlas

# Root package — required for the TUI and Streamlit UI
pip install -e .

# TUI interface
pip install -r requirements-textual.txt
```

Install **one** LLM provider (Mistral is the default):

```bash
pip install -r requirements-mistral.txt    # Mistral (default)
pip install -r requirements-openai.txt     # OpenAI / OpenAI-compatible
pip install -r requirements-anthropic.txt  # Anthropic
pip install -r requirements-gemini.txt     # Google Gemini
```

See [`docs/developer.md`](docs/developer.md) for the full list of optional
feature packages (RAG, tracing, Streamlit UI, etc.).

### 3. Configure environment

```bash
cp bamboo_env_example.sh bamboo_env.sh
# Edit bamboo_env.sh: set your API key and preferred provider/model
source bamboo_env.sh
```

The minimum you need to set:

```bash
export LLM_DEFAULT_PROVIDER="mistral"          # or openai, anthropic, gemini
export LLM_DEFAULT_MODEL="mistral-large-latest"
export MISTRAL_API_KEY="your-key-here"         # whichever provider you chose
```

### 4. Launch the TUI

```bash
# Alternate screen (recommended)
python interfaces/textual/chat.py --transport stdio --no-inline

# Inline — stays in terminal scrollback, easier copy/paste
python interfaces/textual/chat.py --transport stdio --inline
```

Type any question and press Enter.

---

## TUI slash commands

| Command | What it does |
|---|---|
| `/help` | Show all commands |
| `/task <id>` | Shorthand for "summarise task \<id\>" |
| `/job <id>` | Shorthand for "analyse failure of job \<id\>" |
| `/tracing` | Show timing and trace spans for the last request |
| `/costs` | Show estimated LLM token cost for the last request |
| `/json` | Show raw BigPanDA JSON for the last query |
| `/inspect` | Show compact evidence dict (what the LLM saw) for the last query |
| `/history` | Show turns currently held in context memory |
| `/fastpath on\|off` | Toggle deterministic fast-path routing (off → use LLM planner) |
| `/debug on\|off` | Toggle verbose tool call output |
| `/tools` | List tools registered on the server |
| `/clear` | Clear transcript, context memory, and HTTP cache |
| `/exit` | Quit |

`PageUp`/`PageDown` to scroll · `Ctrl+Q` to quit ·
Hold **Option** (macOS) or **Shift** (Linux/Windows) to select text with the mouse.

See [`docs/question-cheatsheet.md`](docs/question-cheatsheet.md) for ready-to-paste test questions.

---

## Key ideas

- **Tool-first** — tools are authoritative; LLMs only summarise their output
- **Plugin architecture** — experiment-specific logic lives in plugins, not in core
- **Narrow waist** — every tool returns `list[MCPContent]`; the MCP wire format is JSON-RPC 2.0
- **Context memory** — multi-turn chat history is maintained in the client and threaded into every LLM call
- **Deterministic routing** — `bamboo_answer` selects tools by regex, not by LLM

---

## Inspecting the server

```bash
# List available tools
python -m bamboo tools list

# Start the MCP server directly (stdio)
python -m bamboo.server

# Interactive inspection via MCP Inspector
npx @modelcontextprotocol/inspector python3 -m bamboo.server
```

---

## Documentation

| Doc | Contents |
|---|---|
| [`docs/developer.md`](docs/developer.md) | Full setup, editable installs, testing, linting |
| [`docs/mcp.md`](docs/mcp.md) | MCP protocol, tool contracts, LLM roles, orchestration |
| [`docs/interfaces.md`](docs/interfaces.md) | TUI, Streamlit UI, HTTP transport, context memory |
| [`docs/plugins.md`](docs/plugins.md) | Writing and registering plugins |
| [`docs/jobs-database.md`](docs/jobs-database.md) | Live PanDA jobs DB queries — schema, examples, guard rules, routing |
| [`docs/rag.md`](docs/rag.md) | RAG pipeline (ChromaDB + BM25) |
| [`docs/tracing.md`](docs/tracing.md) | Structured tracing and OpenTelemetry |
| [`docs/security.md`](docs/security.md) | Authentication and token management |
| [`docs/question-cheatsheet.md`](docs/question-cheatsheet.md) | Ready-to-paste test questions for every tool and routing path |

---

## Plugins

| Package | Status | Description |
|---|---|---|
| `askpanda_atlas` | Active | ATLAS / PanDA workflows |
| `askpanda_verarubin` | Planned | Vera Rubin Observatory |
| `askpanda_epic` | Planned | EPIC / EIC experiment |
| `cgsim` | Planned | SimGrid-based workflows (non-PanDA) |
