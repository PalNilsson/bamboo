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
    "bamboo_health":           bamboo_health_tool,        # Server status + LLM info
    "bamboo_llm_answer":       bamboo_llm_answer_tool,    # Raw LLM passthrough
    "bamboo_answer":           bamboo_answer_tool,        # Primary orchestration entry point
    "bamboo_plan":             bamboo_plan_tool,          # LLM-driven planner
    "bamboo_last_evidence":    bamboo_last_evidence_tool, # Retrieve last tool evidence for TUI
    "panda_doc_search":        panda_doc_search_tool,     # Vector RAG
    "panda_doc_bm25":          panda_doc_bm25_tool,       # BM25 keyword search
    "panda_queue_info":        panda_queue_info_tool,
    "panda_task_status":       panda_task_status_tool,
    "panda_job_status":        panda_job_status_tool,
    "panda_log_analysis":      panda_log_analysis_tool,
    "panda_jobs_query":        panda_jobs_query_tool,     # NL→SQL against ingestion DuckDB
    "panda_harvester_workers": panda_harvester_workers_tool, # Harvester pilot/worker stats
    "panda_server_health":     panda_server_health_tool,  # PanDA server liveness via PanDA MCP
}
# panda_jobs_query, panda_harvester_workers, and panda_server_health are
# conditionally added after the dict if askpanda_atlas is installed.
```

### Routing (`core/bamboo/tools/bamboo_answer.py`)

`BambooAnswerTool._route()` is the main routing method. Helpers extracted to keep cyclomatic complexity ≤ 15:

- **`_is_greeting(text)`** / **`_is_ack(text)`** — social intercept returning canned responses before the topic guard runs.
- **`_bypass_response(question, history)`** — delegates directly to `bamboo_llm_answer_tool` when `bypass_routing=True`.
- **`_run_topic_guard(question, history)`** — runs the topic guard and handles content-free followups (`"Tell me more"`). Returns `(rag_query, blocked)`.
- **`_resolve_contextual_ids(question, task_id, job_id, history)`** — fills in `task_id`/`job_id` from history for contextual follow-ups when the current question has none of its own. Two detection paths:
  1. **Explicit back-reference**: pronouns/demonstratives (`"those"`, `"them"`, `"that task"`, `"it"`) — always scans history.
  2. **Implicit short follow-up**: ≤ 10 words + status-specific domain term (`"failed"`, `"finished"`, `"running"`, etc.) — only applies the found ID if history actually contains one.
- **`_build_deterministic_plan(rag_query, task_id, job_id)`** — fast-path routing without an LLM call: job ID + analysis keywords → `panda_log_analysis`; job ID → `panda_job_status`; task ID → `panda_task_status`; pilot/Harvester signals (no IDs) → `panda_harvester_workers`; jobs DB signals (no IDs) → `panda_jobs_query`; no IDs → RAG retrieval. The pilot rule runs before the jobs DB rule because the word "pilot" can co-occur with jobs DB signal phrases.
- **`_is_pilot_question(question)`** — returns `True` when the question contains pilot/Harvester signal phrases (`"pilot"`, `"pilots"`, `"harvester worker"`, `"nworkers"`, `"pilot count"`, etc.) regardless of site or time expression. Used by the pilot fast-path intercept to bypass the topic guard for clearly on-topic pilot queries.
- **`_is_jobs_db_question(question)`** — returns `True` when the question contains site/status signal phrases (`"failed at"`, `"top errors"`, `"each status"`, `"which queues"`, `"last updated"`, etc.) and no `"task"` keyword. Used by the fast-path intercept to skip the topic guard for clearly on-topic DB queries.
- **`_is_site_health_question(question)`** — returns `True` when the question contains signals from **both** `_PILOT_SIGNALS` and either `_JOBS_DB_SPECIFIC_SIGNALS` or any bare occurrence of the word `"job"` / `"jobs"` (matched with word boundaries via `re.search(r"\bjobs?\b", q)`). The word-boundary check handles natural phrasing like `"pilot and job failure rates"` without false-positives on `"how many pilots?"`. Questions with a `"task"` keyword are excluded.
- **`_JOBS_DB_SPECIFIC_SIGNALS`** — the job-specific subset of `_JOBS_DB_SIGNALS` used by `_is_site_health_question`. Excludes generic counting phrases (`"how many"`, `"count"`) and status-at phrases (`"ran at"`, `"failed at"`) that can appear in pure pilot questions.
- **`_extract_site_from_question(question)`** — extracts a computing site name using two strategies: (1) a contextual regex matching tokens after `at/for/from/site/queue` that filters out stop-words and requires the token to look like an ATLAS site (has digit, separator char, or is all-uppercase); (2) a short fallback list for sites used without a preposition. Handles arbitrary site names like MWT2, SLAC-SCS, CERN\_PROD, SWT2\_CPB without requiring them to be pre-registered.
- **`_extract_time_window_from_question(question)`** — translates natural-language temporal expressions into explicit ISO-8601 `(from_dt, to_dt)` pairs (UTC). Handles: `"last/past N hours/minutes/days"`, `"yesterday"`, `"since yesterday"`, `"today"`, explicit `"between ISO and ISO"` ranges. Returns `None` for `"right now"` / `"currently"` / no temporal expression, in which case the tool defaults to the last hour.

The **site-health fast-path** (in `_run_fast_path_intercepts`) builds a two-tool plan passing `site=<SITE>` to `panda_harvester_workers` **and** `queue=<SITE>` to `panda_jobs_query`. Both arguments are required for the tools to scope their queries to the same site; omitting `queue=` causes `panda_jobs_query` to query globally and return incorrect (all-site) statistics.

`bamboo_answer` accepts a `bypass_fast_path: bool` argument (default `False`). When `True`, both `_run_fast_path_intercepts()` and `_build_deterministic_plan()` are skipped — the question falls through to the topic guard and LLM planner. Exposed via the TUI `/fastpath off` command for testing planner routing on questions that would normally be short-circuited.
- **`_is_jobs_db_question(question)`** — returns `True` when the question contains site/status signal phrases (`"failed at"`, `"top errors"`, `"each status"`, `"which queues"`, `"last updated"`, etc.) and no `"task"` keyword. Used by the fast-path intercept to skip the topic guard for clearly on-topic DB queries.
- **`QUERYABLE_DATABASES`** — registry of queryable databases (`{"jobs": "...", ...}`). Currently one entry; add `"cric"` when CRIC integration lands. When more than one entry is present, `_resolve_target_database()` is called to detect ambiguous questions; `_build_clarification_response()` prompts the user to specify which database they mean.

Routing order in `_route()`:
1. `bypass_routing` → direct LLM passthrough
2. Social intercept (greeting / ack)
3. **Fast-path intercepts** (`_run_fast_path_intercepts`) — contextual ID resolution first, then:
   - **PanDA server health** — liveness/health question (no IDs) → `panda_server_health`. Checked first, before site-health, so "is PanDA alive?" is never mistaken for a job or site question.
   - **Site-health** — both pilot AND job-specific signals present → `panda_harvester_workers` + `panda_jobs_query` in one plan. Checked before individual pilot/jobs to prevent the pilot check from firing alone.
   - **Pilot-only** — pilot signals, no job signals → `panda_harvester_workers`.
   - **Jobs DB** — jobs DB signals → `panda_jobs_query`. All four bypass the topic guard (~3s saved).
4. Topic guard + content-free followup reformulation
5. ID extraction from question, then contextual ID resolution from history
6. Deterministic fast-path plan
7. LLM planner fallback (`bamboo_plan` with `execute=True`)

### Execution (`core/bamboo/tools/bamboo_executor.py`)

`execute_plan()` iterates the plan's tool calls, calls each tool, and synthesises a grounded LLM answer. Key behaviour:

- After each successful tool call, the full evidence dict (including `raw_payload`) is stored in `_last_evidence_store[tool_name]`. `raw_payload` is **stripped before synthesis** so the LLM only sees the compact evidence fields.
- `BambooLastEvidenceTool` (`bamboo_last_evidence`) exposes the store via MCP. Accepts `mode="evidence"` (compact dict, `raw_payload` excluded) or `mode="raw"` (verbatim BigPanDA API response). Used by the TUI `/inspect` and `/json` commands.
- `_pick_synthesis_prompt(tool_names)` selects the system prompt for synthesis based on which tools ran: `panda_log_analysis` → log diagnostic; `panda_job_status` → job summary; `panda_task_status` → task summary; `panda_server_health` → PanDA liveness prompt; `panda_harvester_workers` + `panda_jobs_query` together → site-health prompt (two labelled evidence sources); `panda_harvester_workers` alone → pilot stats prompt; `panda_jobs_query` alone → jobs DB prompt; `panda_doc_*` → RAG documentation; other → generic.

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
"atlas.jobs_query"  = "askpanda_atlas.jobs_query:panda_jobs_query_tool"
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

#### Jobs Query (`jobs_query_impl.py`, `jobs_query_schema.py`)

NL→SQL tool that queries the local ingestion DuckDB file written by the `askpanda-ingestion-agent`. See [`docs/jobs-database.md`](../docs/jobs-database.md) for full documentation.

Pipeline: schema context (cached, 1 h TTL) → LLM SQL generation (temperature 0.0, max 512 tokens) → fence-strip → cannot-answer detection → AST guard (`sqlglot`, DuckDB dialect) → synchronous DuckDB execution → evidence dict.

Key design decisions:
- **AST guard** (`validate_and_guard`) enforces 7 rules: single statement, SELECT-only root, no DDL/DML/DCL/TCL anywhere in the tree, no system tables, no unknown tables (allow-list: `jobs`, `selectionsummary`, `errors_by_count`), LIMIT injection. Implemented with `sqlglot` AST inspection — never regex.
- **Synchronous execution** — DuckDB queries run on the event loop thread (2-15 ms typical). `asyncio.to_thread` was deliberately removed after macOS threading conflicts were diagnosed.
- **Datetime serialisation** — `_serialise_row()` converts `datetime`, `date`, and `Decimal` values to JSON-safe types before the evidence dict is serialised.
- **No bamboo-core import at module level** — `bamboo.llm.*` imports are deferred inside `_call_llm_for_sql()` so the module is importable without bamboo installed.
- `PANDA_DUCKDB_PATH` env var (default: `jobs.duckdb`) controls the database path.

Schema context and SQL generation prompt live in `jobs_query_schema.py`. The prompt includes six few-shot examples covering freshness queries, counts, group-by, errors, cross-queue ranking, and job listing.

#### Harvester Workers (`harvester_worker_impl.py`)

Fetches live Harvester pilot/worker counts from the BigPanDA Harvester API.

Endpoint: `GET /harvester/getworkerstats/?lastupdate_from=<ISO>&lastupdate_to=<ISO>[&computingsite=<SITE>]`

The API returns a JSON array of records, each describing `nworkers` for a specific combination of `computingsite`, `harvesterid`, `jobtype`, `resourcetype`, and `status`. Workers and pilots are synonymous in this context.

Public surface:
- `fetch_worker_stats(base_url, from_dt, to_dt, site) -> dict` — synchronous; call via `asyncio.to_thread()`. Fetches via `cached_fetch_jsonish` (30 s TTL — shorter than metadata TTL because pilot counts change frequently), extracts records from the `{"_data": [...]}` wrapper that `fetch_jsonish` produces for top-level JSON arrays, aggregates, and attaches `raw_payload`.
- `PandaHarvesterWorkersTool` / `get_definition()` / `panda_harvester_workers_tool` singleton.

Evidence structure (what the LLM sees — `raw_payload` is stripped before synthesis):
- `nworkers_total` — grand total across all statuses.
- `nworkers_by_status`, `nworkers_by_jobtype`, `nworkers_by_resourcetype`, `nworkers_by_site` — flat one-dimensional breakdowns, sorted by count descending.
- `pivot` — list of `{status, jobtype, resourcetype, nworkers}` dicts sorted by `nworkers` descending, summed across all harvester instances and sites. Bounded by `len(statuses) × len(jobtypes) × len(resourcetypes)` (≤ ~54 rows in practice). Allows the LLM to answer any combination of these three dimensions — including three-way slices like "running MCORE managed pilots" — without raw records.
- `total_records`, `from_dt`, `to_dt`, `site_filter`, `error`.

Site is excluded from the pivot deliberately: scoped queries should use `site_filter` on the API call, making the pivot already site-specific. For all-sites queries, `nworkers_by_site` covers per-site totals.

Key design decisions:
- **`asyncio.to_thread` for HTTP** — unlike DuckDB (which must run on the event loop thread), HTTP fetches are safe with the thread pool on all platforms.
- **30 s cache TTL** — pilot counts change much more frequently than task/job metadata; the standard 60 s TTL would give stale answers to "right now" questions.
- **Time window defaults to last hour** — `_default_window()` is called when `from_dt`/`to_dt` are absent from `arguments`, so the tool always has a valid query window.
- **Temporal extraction happens at the router, not the tool** — `_extract_time_window_from_question()` in `bamboo_answer.py` translates natural-language time expressions (`"since yesterday"`, `"last 6 hours"`, `"today"`, explicit ISO ranges) into `from_dt`/`to_dt` before the fast-path plan is built. The tool itself is stateless with respect to time.

Entry point registration in `pyproject.toml`:
```toml
"atlas.harvester_workers" = "askpanda_atlas.harvester_worker:panda_harvester_workers_tool"
```

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
| `/fastpath on\|off` | Toggle deterministic fast-path routing. `off` bypasses all fast-path intercepts and `_build_deterministic_plan`, forcing the LLM planner to handle all routing. Useful for testing planner coverage on questions that would normally be short-circuited. Default: `on`. |
| `/costs` | Show estimated LLM token cost for the last request, broken down by model call. Reads `input_tokens`/`output_tokens` from `llm_call` trace spans and prices them from `_MODEL_COST_PER_MTOK`. Requires `BAMBOO_TRACE=1`. |
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

#### Cost Estimation (`/costs`)

`_estimate_cost(spans)` reads every `llm_call` span from `_last_spans`, looks up per-token pricing from `_MODEL_COST_PER_MTOK` (a `Dict[str, tuple]` of `(input_rate_per_Mtok, output_rate_per_Mtok)` in USD), and returns a breakdown dict. Unknown models fall back to `_DEFAULT_COST_PER_MTOK` ($1.00/$3.00 per Mtok) and are flagged in the output.

The pricing table covers Mistral, Anthropic, OpenAI, and Google models. To add a new model, extend `_MODEL_COST_PER_MTOK` in `interfaces/textual/chat.py` — no restart needed as the dict is read at command time.

**Important**: the planner (`bamboo_plan`) previously emitted no trace spans, so its token usage was invisible to `/costs`. It now emits an `llm_call` span with `tool="bamboo_plan"`, so `/fastpath off` queries correctly show two LLM call rows — the planning call and the synthesis call.

### PanDA MCP Integration (`packages/askpanda_atlas/`)

#### Session Wiring (`panda_mcp_session.py`)

Bamboo connects to the external PanDA MCP server at startup and holds the session open for the process lifetime. The session is registered with the process-wide `MCPCaller` under the name `"panda"`.

**Startup:**
- `core/bamboo/server.py` (stdio): `asyncio.create_task(run_panda_mcp_session(shutdown_event))` before entering `stdio_server()`.
- `core/bamboo/entrypoints/http.py` (ASGI): `_startup_panda_mcp()` is called from the `lifespan.startup` handler; teardown is wired into `_shutdown()`.

If `PANDA_MCP_BASE_URL` is not set, the session helper logs a warning and exits immediately — tools that need it return a graceful `"server not connected"` error rather than crashing.

Transport selection: `streamable_http_client` by default; `sse_client` when `PANDA_MCP_USE_SSE=1`.

Authentication is passed as HTTP headers (`Authorization: Bearer <token>` and `Origin: <vo>`), matching the PanDA Server authentication model described in the PandaMCP documentation.

#### PanDA Server Health (`panda_server_health.py`)

Calls the `is_alive` tool on the `"panda"` MCP session. The first (and currently only) tool that delegates to the PanDA MCP server.

Evidence structure:
- `is_alive: bool` — true if the server is alive.
- `raw_response: str | None` — first 500 chars of the raw response.
- `error: str | None` — null on success.

The `_parse_alive(raw)` helper handles plain-string (`"True"` / `"false"`) and JSON (`{"alive": true}`) responses.

**Mocking pattern** (important — differs from other tools):

Because `get_mcp_caller` is imported **inside `call()` at runtime** (required by the no-bamboo-core-at-module-level rule), it cannot be patched via `monkeypatch.setattr("askpanda_atlas.panda_server_health.get_mcp_caller", ...)`. Instead, patch the process-wide singleton directly:

```python
monkeypatch.setattr(
    "bamboo.tools._mcp_caller._mcp_caller",
    _make_caller(text="True"),
)
```

The same pattern applies to any future PanDA MCP tool that uses deferred imports.

#### Adding More PanDA MCP Tools

The session wiring is complete — adding a new PanDA MCP tool only requires steps 3–5 from the original plan:

1. Create `packages/askpanda_atlas/askpanda_atlas/panda_mcp_<toolname>.py` following `panda_server_health.py` exactly.
2. Use `_SERVER = "panda"` and `_TOOL = "<actual_tool_name_on_server>"`.
3. Register in `pyproject.toml` entry points, `core.py` TOOLS dict, `bamboo_answer.py` fast-path, `bamboo_executor.py` synthesis prompt, `planner.py` routing rule.
4. Add to `test_narrow_waist.py` TOOLS and `_STUB_ARGS`.
5. Write `tests/test_panda_mcp_<toolname>.py`.

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
| `PANDA_MCP_BASE_URL` | Full URL of the PanDA MCP HTTP endpoint, e.g. `http://pandaserver01.sdcc.bnl.gov:25080/mcp/`. If unset, PanDA MCP tools return a graceful "server not connected" error. |
| `PANDA_MCP_TOKEN` | Optional bearer token sent as `Authorization: Bearer <token>` to the PanDA MCP server |
| `PANDA_MCP_ORIGIN` | Optional VO name sent as `Origin: <vo>` (e.g. `atlas`) to the PanDA MCP server |
| `PANDA_MCP_USE_SSE` | `1`/`true`/`yes` to use the legacy SSE transport; default is streamable-HTTP |