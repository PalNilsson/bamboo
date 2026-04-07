# `panda_harvester_workers` — Harvester Worker Statistics Tool

**Plugin:** `askpanda_atlas`
**Entry point:** `atlas.harvester_workers`
**Source files:** `harvester_worker.py`, `harvester_worker_impl.py`

---

## Overview

`panda_harvester_workers` fetches live **Harvester pilot/worker counts** from the BigPanDA
Harvester API. It answers natural-language questions about pilot activity at ATLAS computing
sites — how many pilots are running, idle, submitted, or failed — for any time window and
optional site filter.

"Workers" and "pilots" are synonymous throughout: each record in the Harvester API describes
how many pilot processes (`nworkers`) were active for a given combination of computing site,
Harvester instance, job type, resource type, and pilot status.

---

## API Endpoint

```
GET /harvester/getworkerstats/?lastupdate_from=<ISO>&lastupdate_to=<ISO>[&computingsite=<SITE>]
```

**Base URL:** `PANDA_BASE_URL` env var (default: `https://bigpanda.cern.ch`)

The endpoint returns a JSON **array** of flat records. `fetch_jsonish` (in `_cache.py`) wraps
top-level JSON arrays as `{"_data": [...]}` — `_extract_records()` handles this transparently.

### Record shape

| Field | Type | Description |
|---|---|---|
| `nworkers` | int | Pilot count for this combination |
| `status` | str | Pilot status: `running`, `submitted`, `idle`, `failed`, `cancelled`, etc. |
| `jobtype` | str | Job type: `managed`, `user`, `test`, etc. |
| `resourcetype` | str | Resource type: `SCORE`, `MCORE`, etc. |
| `computingsite` | str | ATLAS computing site name, e.g. `BNL`, `CERN`, `AGLT2` |
| `harvesterid` | str | Harvester instance identifier |

---

## File Structure

```
askpanda_atlas/
├── harvester_worker.py          # Thin entry-point wrapper; re-exports impl symbols
└── harvester_worker_impl.py     # Full implementation (URL building, HTTP, aggregation, tool class)

tests/
└── test_harvester_worker.py     # Full test suite (~50 tests across unit/integration layers)
```

### `harvester_worker.py`

Thin delegation module — imports and re-exports the three public symbols from
`harvester_worker_impl`. If `requests` is not installed, the `ImportError` is logged and
re-raised; Bamboo catches this at plugin load time and skips registration of the tool.

### `harvester_worker_impl.py`

Contains the full implementation. No `bamboo.*` imports at module level — all core imports are
deferred inside `call()` and helpers so the module can be imported (and tested) without bamboo
core installed.

---

## Public Surface

| Symbol | Kind | Description |
|---|---|---|
| `get_definition()` | function | Returns the MCP tool definition dict |
| `PandaHarvesterWorkersTool` | class | MCP tool class with `call()` |
| `panda_harvester_workers_tool` | singleton | Module-level instance registered via entry point |
| `build_harvester_url(...)` | function | Builds the full API URL (testable in isolation) |
| `fetch_worker_stats(...)` | function | Synchronous HTTP fetch + aggregation (no bamboo dep) |

---

## Tool Input Schema

```json
{
  "question": "string (required) — original user question for LLM synthesis",
  "site":     "string (optional) — computing site filter, e.g. 'BNL', 'CERN', 'AGLT2'",
  "from_dt":  "string (optional) — ISO-8601 start, e.g. '2026-03-01T00:00:00'",
  "to_dt":    "string (optional) — ISO-8601 end,   e.g. '2026-03-01T06:00:00'"
}
```

When `from_dt`/`to_dt` are omitted the tool defaults to the **last hour (UTC)**, so questions
like *"How many pilots are running at BNL right now?"* work without requiring the caller to
provide timestamps. Partial overrides are supported: if only one bound is specified the other
defaults to the computed window endpoint.

---

## Evidence Structure

`call()` returns a JSON-serialised `{"evidence": {...}}` dict.
`raw_payload` (the unmodified record list) is attached for `/json` inspection via
`bamboo_last_evidence` but is stripped before LLM synthesis by `bamboo_executor`.

| Key | Type | Description |
|---|---|---|
| `nworkers_total` | int | Grand total across all records in the window |
| `nworkers_by_status` | dict[str, int] | Pilots per status, sorted by count descending |
| `nworkers_by_jobtype` | dict[str, int] | Pilots per job type, sorted by count descending |
| `nworkers_by_resourcetype` | dict[str, int] | Pilots per resource type, sorted descending |
| `nworkers_by_site` | dict[str, int] | Pilots per computing site, sorted descending |
| `pivot` | list[dict] | `{status, jobtype, resourcetype, nworkers}` rows sorted descending |
| `total_records` | int | Raw record count from the API |
| `from_dt` | str | Actual lower bound used for the fetch |
| `to_dt` | str | Actual upper bound used for the fetch |
| `site_filter` | str \| None | Site filter applied, or `None` for all sites |
| `endpoint` | str \| None | Full URL fetched (useful for debugging) |
| `raw_payload` | list \| None | Verbatim API records (stripped before LLM synthesis) |
| `error` | str \| None | Human-readable error message, or `None` on success |

### The pivot table

The `pivot` list allows the LLM to answer any **two- or three-way combination** of status,
job type, and resource type without receiving raw records. For example:

- *"How many MCORE pilots are running?"* → filter `resourcetype=MCORE, status=running`
- *"Running managed MCORE pilots"* → filter all three dimensions

The pivot is bounded by `len(statuses) × len(jobtypes) × len(resourcetypes)` — typically
≤ 54 rows in practice.

Computing site is intentionally **excluded** from the pivot. Scoped queries should pass `site`
as a tool argument so the API call is already site-specific. For all-sites queries,
`nworkers_by_site` provides per-site totals.

---

## Routing and Fast-Path Integration

`panda_harvester_workers` is invoked by `bamboo_answer.py` via three paths:

### 1. Pilot-only fast path
`_is_pilot_question(question)` returns `True` when the question contains pilot/Harvester
signal phrases (`"pilot"`, `"pilots"`, `"harvester worker"`, `"nworkers"`, `"pilot count"`,
etc.) with no job-specific signals. The topic guard is bypassed; a single-tool plan is built
calling `panda_harvester_workers` with the resolved site and time window.

### 2. Site-health fast path
`_is_site_health_question(question)` fires when both pilot signals **and** job-specific
signals are present. A two-tool plan is built passing `site=<SITE>` to
`panda_harvester_workers` **and** `queue=<SITE>` to `panda_jobs_query`. Both arguments must
be set to scope both queries to the same site.

### 3. LLM planner fallback
Questions that reach the planner include `panda_harvester_workers` in the available tool set.
The planner selects it for pilot/worker queries when the fast-path heuristics do not fire.

### Temporal argument extraction

`_extract_time_window_from_question()` in `bamboo_answer.py` translates natural-language time
expressions into `from_dt`/`to_dt` ISO strings **before** the plan is built. The tool itself
is stateless with respect to time — it accepts whatever window the router provides, or falls
back to the last-hour default.

Examples of supported phrasings:

| User phrase | `from_dt` → `to_dt` |
|---|---|
| *"right now"* / omitted | `now - 1h` → `now` |
| *"last 6 hours"* | `now - 6h` → `now` |
| *"today"* | start of UTC day → `now` |
| *"since yesterday"* | `yesterday 00:00` → `now` |
| *"2026-03-01 to 2026-03-02"* | `2026-03-01T00:00:00` → `2026-03-02T00:00:00` |

---

## Synthesis Prompt Selection

`_pick_synthesis_prompt(tool_names)` in `bamboo_answer.py` selects the LLM synthesis prompt
based on which tools ran:

| Tools that ran | Synthesis prompt |
|---|---|
| `panda_harvester_workers` only | Pilot stats prompt |
| `panda_harvester_workers` + `panda_jobs_query` | Site-health prompt (two labelled evidence sources) |

---

## Caching

HTTP responses are cached via `cached_fetch_jsonish` from `_cache.py`.

**TTL: 30 seconds** — deliberately shorter than the 60 s TTL used for task/job metadata.
Pilot counts change frequently; the shorter TTL ensures *"right now"* answers are fresh
without hammering the API on repeated questions.

---

## Error Handling

`call()` never raises. All exceptions are caught, logged at `ERROR` level, and converted to a
structured `_error_evidence(...)` dict with `error` populated and all counters zeroed. The
caller receives a graceful JSON response with a human-readable message rather than a stack
trace.

HTTP errors (non-2xx responses) and non-JSON responses raise `RuntimeError` inside
`fetch_worker_stats`, which is caught by `call()`.

---

## Design Decisions

**`asyncio.to_thread` for HTTP** — unlike DuckDB queries (which must run on the event loop
thread due to macOS threading issues), blocking HTTP fetches are safe in the thread pool on
all platforms. `fetch_worker_stats` is synchronous and wrapped in `asyncio.to_thread()` inside
`call()`.

**Deferred bamboo-core imports** — `from bamboo.tools.base import text_content` is deferred
inside `call()`. This keeps the module importable and fully testable without bamboo core
installed.

**No module-level I/O** — the singleton `panda_harvester_workers_tool` is instantiated at
import time (it only calls `get_definition()`) but performs no HTTP requests until `call()` is
invoked.

**`_extract_records` wrapping tolerance** — the helper handles the `{"_data": [...]}` wrapper
produced by `fetch_jsonish` for top-level JSON arrays, plus a fallback that walks dict values
for the first non-empty list. This guards against minor Harvester API shape changes without
requiring a code update.

---

## Entry Point Registration

```toml
# packages/askpanda_atlas/pyproject.toml
[project.entry-points."bamboo.tools"]
"atlas.harvester_workers" = "askpanda_atlas.harvester_worker:panda_harvester_workers_tool"
```

---

## Example Questions

```
How many pilots are running at BNL right now?
Show pilot counts at CERN for the last 6 hours.
Were there Harvester workers at AGLT2 yesterday?
How many MCORE pilots are running across all sites?
What is the pilot failure rate at BNL?
Show running vs submitted pilots at SLAC.
```

---

## Testing

Tests live in `packages/askpanda_atlas/tests/test_harvester_worker.py`. All HTTP calls are
mocked via `unittest.mock.patch` on `askpanda_atlas._cache.cached_fetch_jsonish`.

Test coverage includes:

- `_safe_int` — type coercion edge cases (None, string, bad string)
- `_default_window` — UTC window construction
- `build_harvester_url` — URL formatting with and without site filter
- `_extract_records` — `_data` unwrapping, dict-value fallback, empty/malformed payloads
- `_aggregate_evidence` — flat breakdowns, pivot table construction, sorting
- `_error_evidence` — structured error dict shape
- `fetch_worker_stats` — HTTP success, HTTP error, non-JSON response
- `PandaHarvesterWorkersTool.call()` — full async path, default window, site filter, partial
  time override, exception handling

Run with:

```bash
cd packages/askpanda_atlas
pytest tests/test_harvester_worker.py -v
```
