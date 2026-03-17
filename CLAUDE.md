# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Bamboo** is a lightweight MCP-based runtime with a plugin architecture for AI-assisted scientific tools, primarily targeting ATLAS/PanDA workflows. LLMs are used for summarization and explanation, not as sources of truth. Structured evidence is always returned alongside natural-language answers.

## Development Setup

```bash
# Install core and the ATLAS plugin in editable mode
pip install -e ./core
pip install -e ./packages/askpanda_atlas

# Install dev dependencies
pip install -r requirements-dev.txt

# Install UI dependencies (Streamlit)
pip install -r requirements-ui.txt

# Install Textual TUI dependencies
pip install -r requirements-textual.txt

# Configure environment (copy and fill in API keys)
cp bamboo_env_example.sh bamboo_env.sh
source bamboo_env.sh
```

## Commands

```bash
# Run the MCP stdio server
python -m bamboo.server
# or
python -m bamboo server

# Inspect via MCP Inspector
npx @modelcontextprotocol/inspector python3 -m bamboo.server

# List available tools
python -m bamboo tools list

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_task_status.py

# Run a single test
pytest tests/test_task_status.py::test_name

# Linting
flake8 .
pylint core/ interfaces/ packages/
pyright .

# Pre-commit checks (runs flake8 + circular import detection)
pre-commit run --all-files

# Launch Streamlit UI
streamlit run interfaces/streamlit/chat.py

# Launch Textual TUI
python -m interfaces.textual.chat
# or
textual run interfaces/textual/chat.py
```

## Architecture

### Core (`core/bamboo/`)

- **`core.py`** — `create_server()` builds the MCP `Server` instance. Registers all built-in tools and loads plugin tools via Python entry points (`bamboo.tools`). Implements all MCP handlers (`list_tools`, `call_tool`, `list_prompts`, `get_prompt`).
- **`server.py`** — Stdio transport entry point. Run with `python -m bamboo.server`.
- **`config.py`** — Frozen `Config` dataclass driven entirely by environment variables (prefixed `ASKPANDA_*`, with fallback to unprefixed variants). Also reads `[tool.askpanda]` from `pyproject.toml`.
- **`tools/bamboo_answer.py`** — Primary orchestration tool. Extracts task IDs from questions, calls `panda_task_status_tool` for structured evidence, then calls `bamboo_llm_answer_tool` with a structured prompt. For general knowledge questions (no ID detected) it runs vector search (`panda_doc_search`) and BM25 keyword search (`panda_doc_bm25`) **concurrently** via `asyncio.gather`, merges results, and passes them to the LLM as grounding context. This is the main entry point for ATLAS queries.
- **`llm/`** — Multi-LLM abstraction layer. `factory.py` creates provider clients; `manager.py` routes by profile (default/fast/reasoning); `selector.py` picks profiles; `registry.py` and `config_loader.py` manage profiles. Providers: `anthropic`, `openai`, `gemini`, `mistral`, `openai_compat`.
- **`tracing.py`** — Structured request/response lifecycle tracing (opt-in via `BAMBOO_TRACE=1`). Emits NDJSON spans to stderr or a file (`BAMBOO_TRACE_FILE`). File and stderr are mutually exclusive — when the file is set, stderr is left clean (required for TUI compatibility). See `docs/tracing.md`.

### Plugin System

Tools are registered via `pyproject.toml` entry points under `bamboo.tools`:

```toml
[project.entry-points."bamboo.tools"]
"atlas.task_status" = "askpanda_atlas.task_status:panda_task_status_tool"
"atlas.ui_manifest" = "askpanda_atlas.ui_manifest:atlas_ui_manifest_tool"
```

Each plugin is a separate installable package under `packages/`. Core discovers plugins at startup via `importlib.metadata.entry_points`.

### Interfaces (`interfaces/`)

- **`shared/mcp_client.py`** — `MCPAsyncClient` (async) and `MCPClientSync` (thread-safe sync wrapper for Streamlit). Supports both stdio (spawns a subprocess) and HTTP (streamable HTTP) transports.
- **`streamlit/chat.py`** — Browser-based chat UI. Uses `st.cache_resource` for the MCP client. Sidebar for transport config. Routes questions to `bamboo_answer` or `askpanda_answer` tools based on heuristics.
- **`textual/chat.py`** — Terminal TUI. Loads plugin banners via `atlas.ui_manifest` entry point. Commands: `/help`, `/tools`, `/task <id>`, `/plugin`, `/debug`, `/clear`, `/exit`.

## Key Conventions

- **LLM profiles**: `default`, `fast`, `reasoning` — selected per task type. Profile → provider/model mapping is set via env vars.
- **Tool results**: Tools return dicts with an `evidence` key containing structured data. The `bamboo_answer` orchestrator always checks for `not_found`, `http_status`, `error`, `non_json` fields to detect upstream failures.
- **Entry point name format**: `"namespace.tool_name"` (e.g., `"atlas.task_status"`).
- **Tests**: `conftest.py` adds `core/` and `packages/askpanda_atlas/` to `sys.path` so tests run without editable installs.

## Environment Variables

See `bamboo_env_example.sh` for the full list. Key variables:

| Variable | Purpose |
|---|---|
| `ASKPANDA_ENABLE_REAL_PANDA` | `1` to use real BigPanDA API |
| `ASKPANDA_ENABLE_REAL_LLM` | `1` to use real LLM calls |
| `LLM_DEFAULT_PROVIDER` | `openai`, `anthropic`, `mistral`, `gemini`, `openai_compat` |
| `LLM_DEFAULT_MODEL` | Model string for the default profile |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `MISTRAL_API_KEY` / `GEMINI_API_KEY` | Provider API keys |
| `ASKPANDA_OPENAI_COMPAT_BASE_URL` | For vLLM/Ollama/custom OpenAI-compatible endpoints |
| `PANDA_BASE_URL` | BigPanDA API base URL (default: `https://bigpanda.cern.ch`) |
| `BAMBOO_TRACE` | `1` to enable structured tracing (default: off) |
| `BAMBOO_TRACE_FILE` | Write trace spans to this file instead of stderr |
