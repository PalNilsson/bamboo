# Bamboo Tracing

Bamboo includes a structured tracing system that records the request/response
lifecycle as it passes through the pipeline — topic guard, document retrieval,
LLM calls, and synthesis — without touching the MCP stdio channel.

Tracing is **opt-in and off by default**. There is zero overhead when it is
not enabled.

---

## Quick start

### Command line

```bash
BAMBOO_TRACE=1 python -m bamboo.server
```

Trace events are written to **stderr** as newline-delimited JSON. To filter
and pretty-print them:

```bash
BAMBOO_TRACE=1 python -m bamboo.server 2>&1 \
  | grep bamboo_trace \
  | jq '{event: .event, tool: .tool, ms: .duration_ms}'
```

### Textual TUI

Nothing extra to configure. When you launch the TUI in stdio mode, tracing
is enabled automatically and the server writes spans to a temporary file:

```
/tmp/bamboo_trace_<pid>.jsonl
```

After asking any question, type `/tracing` to see a summary table of the
spans produced by that request.

---

## Output destinations

File and stderr are **mutually exclusive** — `BAMBOO_TRACE_FILE` takes
priority. This is intentional: the server subprocess's stderr is inherited
by the parent terminal process, so writing JSON to it while the Textual TUI
is running would corrupt the display.

### stderr (CLI / direct server use)

When `BAMBOO_TRACE_FILE` is **not** set, spans are written to stderr:

```bash
BAMBOO_TRACE=1 python -m bamboo.server 2>&1 \
  | grep bamboo_trace \
  | jq '{event: .event, tool: .tool, ms: .duration_ms}'
```

### BAMBOO_TRACE_FILE (TUI / log aggregation)

When `BAMBOO_TRACE_FILE` is set, spans are written **only** to the file and
stderr is left clean:

```bash
export BAMBOO_TRACE=1
export BAMBOO_TRACE_FILE=/tmp/bamboo_trace.jsonl
python -m bamboo.server
```

The file is opened in append mode on each write, so it survives server
restarts and multiple server processes can safely share the same file.

---

## Environment variables

| Variable | Values | Description |
|---|---|---|
| `BAMBOO_TRACE` | `0` (default), `1` | Enable trace output. |
| `BAMBOO_TRACE_FILE` | file path | Write spans to this file instead of stderr. |
| `BAMBOO_OTEL_ENDPOINT` | URL | OTLP/gRPC endpoint. Enables OTel export. |
| `BAMBOO_OTEL_SERVICE_NAME` | string | Service name in the OTel backend (default: `bamboo`). |
| `BAMBOO_OTEL_INSECURE` | `1` (default), `0` | Use plaintext gRPC; set to `0` for TLS. |

All variables are read once at import time. Set them before starting the
server process.

---

## Output format

Every span is emitted as a single JSON object on one line. All spans share
these fields:

```json
{
  "bamboo_trace": true,
  "event":        "synthesis",
  "tool":         "bamboo_answer",
  "ts":           "2026-03-17T14:32:01.123456+00:00",
  "duration_ms":  681.4
}
```

| Field | Type | Description |
|---|---|---|
| `bamboo_trace` | `true` | Always present. Use as a grep anchor. |
| `event` | string | Event type (see below). |
| `tool` | string | Logical tool or step name. |
| `ts` | string | UTC timestamp, ISO-8601 with microsecond precision. |
| `duration_ms` | float | Wall time for this span in milliseconds. |

---

## Event types

### `tool_call`

Outermost span. Emitted by `core.call_tool` and covers the entire request,
including all nested spans.

Extra fields:

| Field | Type | Description |
|---|---|---|
| `args_keys` | list[str] | Argument key names (not values) passed to the tool. |

Example:
```json
{
  "event": "tool_call", "tool": "bamboo_answer",
  "duration_ms": 823.0, "args_keys": ["question", "include_raw"]
}
```

---

### `guard`

Topic-guard verdict from `topic_guard.check_topic`. Always the first nested
span; emitted even when the guard allows the question through.

Extra fields:

| Field | Type | Description |
|---|---|---|
| `allowed` | bool | Whether the question was permitted. |
| `reason` | string | How the verdict was reached (see below). |
| `llm_used` | bool | Whether the LLM classifier was invoked. |

`reason` values:

| Value | Meaning |
|---|---|
| `keyword_allow` | Matched an allow-list term; no LLM cost. |
| `keyword_deny` | Matched a deny-list term; no LLM cost. |
| `llm_allow` | Ambiguous question permitted by the LLM classifier. |
| `llm_deny` | Ambiguous question rejected by the LLM classifier. |
| `llm_error_allow` | LLM classifier failed; question allowed (fail-open). |

Example:
```json
{
  "event": "guard", "tool": "topic_guard",
  "duration_ms": 0.1,
  "allowed": true, "reason": "keyword_allow", "llm_used": false
}
```

---

### `retrieval`

One document-retrieval step. For general knowledge questions, two retrieval
spans are emitted concurrently (vector and BM25).

Extra fields:

| Field | Type | Description |
|---|---|---|
| `backend` | string | `"vector"` (ChromaDB) or `"bm25"` (BM25Okapi). |
| `hits` | int | Non-empty result lines returned. `-1` if the call raised an exception. |

Example:
```json
{
  "event": "retrieval", "tool": "panda_doc_search",
  "duration_ms": 45.2, "backend": "vector", "hits": 38
}
```

---

### `llm_call`

A single `client.generate()` call inside `llm_passthrough`. Token counts are
populated when the provider returns usage information.

Extra fields:

| Field | Type | Description |
|---|---|---|
| `provider` | string | LLM provider id, e.g. `"openai"`, `"anthropic"`. |
| `model` | string | Model string, e.g. `"gpt-4.1-mini"`. |
| `input_tokens` | int \| null | Prompt tokens, if reported by the provider. |
| `output_tokens` | int \| null | Completion tokens, if reported by the provider. |

Example:
```json
{
  "event": "llm_call", "tool": "bamboo_llm_answer",
  "duration_ms": 680.0,
  "provider": "openai", "model": "gpt-4.1-mini",
  "input_tokens": 3412, "output_tokens": 290
}
```

---

### `synthesis`

The LLM synthesis step inside `bamboo_answer`, wrapping an `llm_call`.
Identifies which routing branch was taken.

Extra fields:

| Field | Type | Description |
|---|---|---|
| `route` | string | Routing branch (see below). |

`route` values:

| Value | Triggered by |
|---|---|
| `rag` | General knowledge question (no task/job ID found). |
| `task` | Task ID extracted from the question. |
| `job` | Job ID extracted, no analysis keywords. |
| `log_analysis` | Job ID extracted with failure/analysis keywords. |
| `bypass` | `bypass_routing=true` flag set by the caller. |

Example:
```json
{
  "event": "synthesis", "tool": "bamboo_answer",
  "duration_ms": 681.4, "route": "rag"
}
```

---

## A complete request trace

A typical general knowledge question (RAG route) produces these spans in
order:

```
tool_call   bamboo_answer        823 ms   args=['question']
  guard       topic_guard          0 ms   allowed=True reason=keyword_allow
  retrieval   panda_doc_search    45 ms   backend=vector hits=38
  retrieval   panda_doc_bm25      12 ms   backend=bm25 hits=19
  llm_call    bamboo_llm_answer  680 ms   openai/gpt-4.1-mini tokens=3412→290
  synthesis   bamboo_answer      681 ms   route=rag
```

Note that `synthesis` and `llm_call` overlap: `synthesis` is the outer span
and includes `llm_call` inside it.

A task status question (no retrieval) looks like this:

```
tool_call   bamboo_answer        540 ms   args=['question']
  guard       topic_guard          0 ms   allowed=True reason=keyword_allow
  synthesis   bamboo_answer      530 ms   route=task
    llm_call  bamboo_llm_answer  528 ms   openai/gpt-4.1-mini tokens=2100→180
```

---

## TUI `/tracing` command

In the Textual TUI, type `/tracing` after any question to display the spans
for that request in a formatted table:

```
╭──── 14:32:01  tracing ──────────────────────────────────────────────────╮
│ Event       Tool                 ms   Detail                             │
│ tool_call   bamboo_answer       823   args=['question']                  │
│ guard       topic_guard           0   allowed=True reason=keyword_allow  │
│ retrieval   panda_doc_search     45   backend=vector hits=38             │
│ retrieval   panda_doc_bm25       12   backend=bm25 hits=19               │
│ llm_call    bamboo_llm_answer   680   openai/gpt-4.1-mini tokens=3412→290│
│ synthesis   bamboo_answer       681   route=rag                          │
│ ─────────────────────────────────────────────────────────────────────── │
│ total                           823   wall time for full tool_call span  │
╰────────────────────────────────────────────────────────────────────────╯
```

The TUI automatically enables `BAMBOO_TRACE=1` and sets `BAMBOO_TRACE_FILE`
for the server subprocess — no manual configuration needed when using stdio
transport.

For HTTP transport the server process is managed externally. Set the env vars
on that process and tail the file directly:

```bash
# on the server host
BAMBOO_TRACE=1 BAMBOO_TRACE_FILE=/tmp/bamboo_trace.jsonl python -m bamboo.server
tail -f /tmp/bamboo_trace.jsonl | grep bamboo_trace | jq .
```

---

## Writing to a trace file

Set `BAMBOO_TRACE_FILE` to write all spans to a file instead of stderr.
This is the recommended mode for log aggregation, post-request analysis,
feeding into an external monitoring system, or any time the server runs
as a subprocess (such as under the Textual TUI).

```bash
export BAMBOO_TRACE=1
export BAMBOO_TRACE_FILE=/var/log/bamboo/trace.jsonl
python -m bamboo.server
```

To read only the spans produced by a single request from the file you can
snapshot the file size before the request and seek to that position
afterwards. The `bamboo.tracing` module exposes helpers for this:

```python
from bamboo.tracing import trace_file_position, read_trace_spans_since

pos = trace_file_position()        # snapshot before request
# ... issue request ...
spans = read_trace_spans_since(pos) # read only the new spans
```

---

## jq recipes

Filter by event type:
```bash
cat trace.jsonl | jq 'select(.event == "llm_call")'
```

Summarise token usage across all requests:
```bash
cat trace.jsonl \
  | jq 'select(.event == "llm_call") | {model, input_tokens, output_tokens}' \
  | jq -s 'group_by(.model)[] | {model: .[0].model, calls: length,
      total_in: map(.input_tokens // 0) | add,
      total_out: map(.output_tokens // 0) | add}'
```

Find slow requests (> 2 seconds):
```bash
cat trace.jsonl \
  | jq 'select(.event == "tool_call" and .duration_ms > 2000)'
```

Show guard denials:
```bash
cat trace.jsonl \
  | jq 'select(.event == "guard" and .allowed == false)'
```

Show retrieval hit counts per request:
```bash
cat trace.jsonl \
  | jq 'select(.event == "retrieval") | "\(.ts) \(.backend) hits=\(.hits)"' -r
```

---

## Implementation notes

The tracing system lives in `core/bamboo/tracing.py`. Integration points:

| File | What is traced |
|---|---|
| `core/bamboo/core.py` | `tool_call` span around every `call_tool` invocation |
| `core/bamboo/tools/bamboo_answer.py` | `guard`, `retrieval`, and `synthesis` spans |
| `core/bamboo/tools/llm_passthrough.py` | `llm_call` span with token counts |

All spans use the `span()` async context manager:

```python
from bamboo.tracing import EVENT_GUARD, span

async with span(EVENT_GUARD, tool="topic_guard") as s:
    result = await check_topic(question)
    s.set(allowed=result.allowed, reason=result.reason, llm_used=result.llm_used)
```

Key design decisions:

- **Opt-in**: `BAMBOO_TRACE=1` required. All functions are no-ops when disabled.
- **stderr-safe**: output goes to stderr, never stdout, so the MCP stdio protocol is unaffected.
- **Fail-open**: any I/O error in the tracing path is silently swallowed and never propagates to the request path.
- **No dependencies**: the module uses only the Python standard library.

---

## OpenTelemetry export

When `BAMBOO_OTEL_ENDPOINT` is set, spans are exported via OTLP/gRPC to any
compatible backend **in addition to** the existing NDJSON output. Install the
optional dependency first:

```bash
pip install -r requirements-otel.txt
```

Then configure and start the server:

```bash
export BAMBOO_TRACE=1
export BAMBOO_OTEL_ENDPOINT=http://localhost:4317
python -m bamboo.server
```

### Trace structure

Spans form a proper parent/child tree. The `tool_call` span is the root;
`guard`, `retrieval`, `llm_call`, and `synthesis` are children:

```
tool_call: bamboo_answer  (root)
├── guard: topic_guard
├── retrieval: panda_doc_search
├── retrieval: panda_doc_bm25
├── llm_call: bamboo_llm_answer
└── synthesis: bamboo_answer
```

The parent/child relationship is tracked via a Python `contextvars.ContextVar`
that asyncio propagates across `await` boundaries automatically — no changes
are needed at call sites.

OTel span names follow the pattern `event:tool`, e.g. `guard:topic_guard`.
All event-specific fields (route, hits, token counts, etc.) are attached as
span attributes prefixed with `bamboo.*` for the initial fields, and set
directly for fields added via `SpanContext.set()`.

### Backends

Any OTLP-compatible backend works. Quick-start examples:

**Jaeger (local, all-in-one):**
```bash
docker run -d --name jaeger \
  -p 4317:4317 -p 16686:16686 \
  jaegertracing/all-in-one:latest
# then: open http://localhost:16686
```

**Grafana Tempo + Grafana:**
```bash
# Use the docker-compose from grafana/tempo on GitHub
# BAMBOO_OTEL_ENDPOINT=http://localhost:4317
```

**Honeycomb / Datadog / other SaaS:**
```bash
export BAMBOO_OTEL_ENDPOINT=https://api.honeycomb.io:443
export BAMBOO_OTEL_INSECURE=0   # TLS required for SaaS endpoints
# Set the API key via the exporter's own env vars per vendor docs
```

### TLS

For SaaS endpoints or secured backends, disable insecure mode:

```bash
export BAMBOO_OTEL_INSECURE=0
```

### Service name

The service name appears as the top-level resource in your trace backend:

```bash
export BAMBOO_OTEL_SERVICE_NAME=bamboo-production
```

### OTel + NDJSON together

Both outputs are active simultaneously when both `BAMBOO_TRACE=1` and
`BAMBOO_OTEL_ENDPOINT` are set. Use `BAMBOO_TRACE_FILE` for the TUI or log
aggregation while OTel sends to the backend:

```bash
export BAMBOO_TRACE=1
export BAMBOO_TRACE_FILE=/var/log/bamboo/trace.jsonl
export BAMBOO_OTEL_ENDPOINT=http://localhost:4317
python -m bamboo.server
```

### Missing SDK

If `BAMBOO_OTEL_ENDPOINT` is set but `opentelemetry-sdk` is not installed,
Bamboo logs a warning to stderr on startup and continues without OTel export.
NDJSON tracing is unaffected.
