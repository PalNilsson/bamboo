# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Bamboo** is a lightweight MCP-based runtime with a plugin architecture for AI-assisted scientific tools, primarily targeting ATLAS/PanDA workflows. LLMs are used for summarisation and explanation, not as sources of truth. Structured evidence is always fetched from BigPanDA, cached, and passed to the LLM; the raw API payload is kept available for inspection.

## Development Setup

```bash
# Install core and the ATLAS plugin in editable mode
pip install -e ./core
pip install -e ./packages/askpanda_atlas

# Install dev dependencies
pip install -r requirements-dev.txt

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

# Inspect via MCP Inspector
npx @modelcontextprotocol/inspector python3 -m bamboo.server

# List available tools
python -m bamboo tools list

# Run all tests (405 passing; 10 pre-existing failures in test_bamboo_executor.py and test_narrow_waist.py)
pytest tests/

# Run a single test file
pytest tests/test_task_status_impl.py

# Linting
flake8 .
pydocstyle .
pyright .

# Pre-commit checks (runs flake8, pydocstyle, circular import detection)
pre-commit run --all-files

# Launch Textual TUI
python -m interfaces.textual.chat
```

## Architecture

### Core (`core/bamboo/`)

- **`core.py`** — Builds the MCP `Server` instance. Registers all built-in tools (see TOOLS registry below) and loads plugin tools via Python entry points (`bamboo.tools`). Implements all MCP handlers (`list_tools`, `call_tool`, `list_prompts`, `get_prompt`).
- **`server.py`** — Stdio transport entry point.
- **`config.py`** — Frozen `Config` dataclass driven entirely by environment variables (prefixed `ASKPANDA_*`).
- **`tracing.py`** — Structured request/response lifecycle tracing (opt-in via `BAMBOO_TRACE=1`). Emits NDJSON spans to stderr or a file (`BAMBOO_TRACE_FILE`). File and stderr are mutually exclusive — when the file is set, stderr is left clean (required for TUI compatibility). See `docs/tracing.md`.
- **`llm/`** — Multi-LLM abstraction layer. `factory.py` creates provider clients; `manager.py` routes by profile (`default`/`fast`/`reasoning`); `selector.py` picks profiles. Providers: `anthropic`, `openai`, `gemini`, `mistral`, `openai_compat`.

### TOOLS Registry (`core/bamboo/core.py`)

```python
TOOLS = {
    "bamboo_health":        bamboo_health_tool,        # Server status + LLM info
    "bamboo_llm_answer":    bamboo_llm_answer_tool,    # Raw LLM passthrough
    "bamboo_answer":        bamboo_answer_tool,        # Primary orchestration entry point
    "bamboo_plan":          bamboo_plan_tool,          # LLM-driven planner
    "bamboo_last_evidence": bamboo_last_evidence_tool, # Retrieve last tool evidence for TUI
    "panda_doc_search":     panda_doc_search_tool,     # Vector RAG
    "panda_doc_bm25":       panda_doc_bm25_tool,       # BM25 keyword search
    "panda_queue_info":     panda_queue_info_tool,
    "panda_task_status":    panda_task_status_tool,
    "panda_job_status":     panda_job_status_tool,
    "panda_log_analysis":   panda_log_analysis_tool,
    "panda_pilot_status":   panda_pilot_status_tool,
}
```

### Routing (`core/bamboo/tools/bamboo_answer.py`)

`BambooAnswerTool._route()` is the main routing method. Helpers extracted to keep cyclomatic complexity ≤ 15:

- **`_is_greeting(text)`** / **`_is_ack(text)`** — social intercept returning canned responses before the topic guard runs.
- **`_bypass_response(question, history)`** — delegates directly to `bamboo_llm_answer_tool` when `bypass_routing=True`.
- **`_run_topic_guard(question, history)`** — runs the topic guard and handles content-free followups (`"Tell me more"`). Returns `(rag_query, blocked)`.
- **`_resolve_contextual_ids(question, task_id, job_id, history)`** — fills in `task_id`/`job_id` from history for contextual follow-ups when the current question has none of its own. Two detection paths:
  1. **Explicit back-reference**: pronouns/demonstratives (`"those"`, `"them"`, `"that task"`, `"it"`) — always scans history.
  2. **Implicit short follow-up**: ≤ 10 words + status-specific domain term (`"failed"`, `"finished"`, `"running"`, etc.) — only applies the found ID if history actually contains one.
- **`_build_deterministic_plan(rag_query, task_id, job_id)`** — fast-path routing without an LLM call: job ID + analysis keywords → `panda_log_analysis`; job ID → `panda_job_status`; task ID → `panda_task_status`; no IDs → RAG retrieval.

Routing order in `_route()`:
1. `bypass_routing` → direct LLM passthrough
2. Social intercept (greeting / ack)
3. Topic guard + content-free followup reformulation
4. ID extraction from question, then contextual ID resolution from history
5. Deterministic fast-path plan
6. LLM planner fallback (`bamboo_plan` with `execute=True`)

### Execution (`core/bamboo/tools/bamboo_executor.py`)

`execute_plan()` iterates the plan's tool calls, calls each tool, and synthesises a grounded LLM answer. Key behaviour:

- After each successful tool call, the full evidence dict (including `raw_payload`) is stored in `_last_evidence_store[tool_name]`. `raw_payload` is **stripped before synthesis** so the LLM only sees the compact evidence fields.
- `BambooLastEvidenceTool` (`bamboo_last_evidence`) exposes the store via MCP. Accepts `mode="evidence"` (compact dict, `raw_payload` excluded) or `mode="raw"` (verbatim BigPanDA API response). Used by the TUI `/inspect` and `/json` commands.

### LLM Passthrough (`core/bamboo/tools/llm_passthrough.py`)

Returns clean `resp.text` — no debug prefix in the response body. `get_llm_info()` helper exposes `"provider=<p> model=<m>"` for the TUI startup banner; called server-side (via `bamboo_health`) so the LLM selector is guaranteed to be initialised.

### Health (`core/bamboo/tools/health.py`)

`bamboo_health` includes `- llm_info: provider=<p> model=<m>` in its response text. The TUI parses this line at startup to display the LLM selection once.

### Plugin System

Tools are registered via `pyproject.toml` entry points under `bamboo.tools`:

```toml
[project.entry-points."bamboo.tools"]
"atlas.task_status" = "askpanda_atlas.task_status:panda_task_status_tool"
"atlas.ui_manifest" = "askpanda_atlas.ui_manifest:atlas_ui_manifest_tool"
```

Each plugin is a separate installable package under `packages/`. Core discovers plugins at startup via `importlib.metadata.entry_points`. The `TOOLS` dict keys (e.g. `"panda_task_status"`) are internal identifiers; MCP-exposed names come from `get_definition()["name"]`.

### ATLAS Plugin (`packages/askpanda_atlas/`)

#### HTTP Cache (`_cache.py`)

Thread-safe in-process TTL cache keyed on URL strings. Protects a `dict[str, (expiry_float, value)]` with a `threading.Lock`.

- `METADATA_TTL = 60.0` seconds — task and job metadata.
- `LOG_TTL = math.inf` — log files are immutable once written; cached forever.
- `cached_fetch_jsonish(url, timeout=30, ttl=METADATA_TTL)` — wraps `_fallback_http.fetch_jsonish`. Returns a 4-tuple `(status_code, content_type, body_text, parsed_json_or_none)`.
- `cached_fetch_log(url, timeout=60)` — wraps `requests.get`. Returns log text or `None` (404). `None` results are also cached (infinite TTL) to prevent re-requesting missing logs.
- `invalidate(url)`, `clear()`, `stats()` — cache management utilities. `clear()` is called by the TUI `/clear` command to flush HTTP state alongside context memory.

#### Task Status (`task_status_impl.py`)

Endpoint: `GET /jobs/?jeditaskid={task_id}&json` (richer than the task endpoint — returns full job list).

Public surface:
- `fetch_and_analyse(base_url, task_id) -> dict` — fetches via `cached_fetch_jsonish`, builds compact evidence with `build_evidence(PandaTaskData(payload))`, attaches `raw_payload` for the evidence store.
- `PandaTaskStatusTool` / `get_definition()` / `panda_task_status_tool` singleton.

All `bamboo.tools.base` imports are deferred inside `call()` — never at module level — so the module is importable without bamboo installed (required by the fallback shim).

#### Schema (`panda_task_schema.py`)

- `PandaJob` — wraps a raw job dict. Exposes typed properties (`pandaid`, `jobstatus`, `computingsite`, `piloterrorcode`, `jeditaskid`, etc.).
- `PandaTaskData` — wraps the full jobs-endpoint payload. Exposes `jobs: list[PandaJob]`, `selectionsummary`, `errsByCount`.
- `build_evidence(task)` — produces the compact evidence dict sent to the LLM: `jobs_by_status`, `jobs_by_site`, `jobs_by_piloterrorcode`, `failed_jobs_sample` (≤ 20), `finished_jobs_sample` (≤ 5), `task_summary`, and `pandaid_list` (inlined only when ≤ 500 jobs).

#### Log Analysis (`log_analysis_impl.py`)

- `_fetch_metadata` uses `cached_fetch_jsonish` (60 s TTL).
- `_fetch_log_text` uses `cached_fetch_log` (infinite TTL).

### Textual TUI (`interfaces/textual/chat.py`)

#### Slash Commands

| Command | Description |
|---|---|
| `/help` | Show command reference |
| `/tools` | List MCP tools |
| `/task <id>` | Shorthand: status of task `<id>` |
| `/job <id>` | Shorthand: analyse failure of job `<id>` |
| `/json` | Verbatim BigPanDA API response for last query (yellow panel) |
| `/inspect` | Compact evidence dict for last query — what the LLM saw (cyan panel) |
| `/tracing` | Timing and trace spans for last request |
| `/history` | Conversation turns in context memory |
| `/plugin <id>` | Switch active plugin |
| `/debug on\|off` | Toggle tool call + raw result display |
| `/clear` | Clear transcript, context memory, and HTTP cache |
| `/exit`, `/quit` | Exit |

Arrow-up/down in the input field navigates command history (consecutive duplicates are deduplicated).

#### LLM Info Display

At startup, the TUI calls `bamboo_health` via MCP and parses the `- llm_info:` line from the response. Displayed once in the connected system message:

```
Connected via stdio. Answer tool: bamboo_answer.
[LLM selected] provider=mistral model=mistral-large-latest
```

#### Thinking Animation

`_write_thinking()` uses `self.set_interval(1.0, _tick)` (Textual-native timer) to cycle `Thinking.` → `Thinking..` → `Thinking...` with the current wall-clock time (`datetime.datetime.now().strftime("%H:%M:%S")`) updated each frame. Cancelled via `timer.stop()` in `_replace_thinking()`.

`asyncio.ensure_future` + `asyncio.sleep` was deliberately avoided — it caused 60× speed issues when the Textual event loop competed with blocking MCP worker threads.

## Key Conventions

### Code Quality

All code must pass:
- `flake8` — max complexity 15 (`max-complexity = 15`), max line length 120.
- `pydocstyle` — Google-style docstrings. First line must be imperative mood (D401). Magic methods need docstrings (D105).
- `pyright` — strict type checking. Use `Sequence[Any]` (covariant) instead of `list[Message]` for function parameters that are only read, not mutated.

### Docstrings

Every public function, method, and class must have a Google-style docstring. The first line must be imperative: write `"Return ..."` not `"Returns ..."`, `"Fetch ..."` not `"Fetches ..."`.

### Tool Implementation Pattern

Each tool module must export:
- `get_definition() -> dict` — MCP tool definition.
- A tool class with `get_definition()` and `async call(arguments: dict) -> list[MCPContent]`.
- A module-level singleton (two blank lines before it — E305).

`bamboo.tools.base` imports (`text_content`, `MCPContent`) must be **deferred inside `call()`**, never at module level, so the module remains importable without bamboo installed.

### Tool Results

`call()` never raises — all errors are returned as `text_content(json.dumps({"error": "..."}))`. The evidence dict always includes `task_id` / `job_id` and `endpoint` fields. `raw_payload` is stored in the evidence dict but stripped before LLM synthesis by `bamboo_executor.execute_plan()`.

### HTTP Cache Mock Pattern

All tests that mock HTTP calls must patch `"askpanda_atlas._cache.cached_fetch_jsonish"` (not `_fallback_http.fetch_jsonish`). Mock signatures must include `timeout`:

```python
def _ok_response(raw): return (200, "application/json", json.dumps(raw), raw)
def _err_response(status=404): return (status, "text/html", "", None)

# In tests:
with patch("askpanda_atlas._cache.cached_fetch_jsonish",
           lambda url, timeout=30: _ok_response(raw)):
    ...
```

### LLM Responses

`llm_passthrough.py` returns clean `resp.text` — no debug prefix. The `[LLM selected]` line appears only once per session in the TUI startup system message, sourced via `bamboo_health`.

## Environment Variables

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
| `BAMBOO_HISTORY_TURNS` | Max conversation turns held in context (default: 10) |
