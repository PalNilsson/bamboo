# MCP workflow, tool orchestration, and LLM usage

## What MCP provides

Bamboo runs an MCP server that publishes a catalog of tools and executes tool
calls over JSON-RPC 2.0.  Two transports are supported:

- **stdio** (default, preferred) — the server is spawned as a subprocess; the
  TUI and MCP Inspector use this mode.
- **Streamable HTTP** — the server listens on an HTTP endpoint; the Streamlit
  UI and multi-user deployments use this mode.

## How LLMs are used

LLMs play three distinct roles in Bamboo, each with a different purpose and
trigger condition.  Critically, **LLMs do not select tools** — tool selection
is deterministic.

| Role | When | LLM profile | Source |
|---|---|---|---|
| **Topic classification** | Only when keyword matching is ambiguous | fast | `bamboo.tools.topic_guard` |
| **Answer synthesis** | Always, after every tool call | default | `bamboo_answer._call_llm()` |
| **Plan generation** | Only when `bamboo_plan` is called explicitly | default | `bamboo.tools.planner` |

### Topic classification

A two-stage guard runs before any tool or LLM call.  Stage 1 is a fast
keyword allow/deny check.  Stage 2 — the LLM classifier — fires only when
keywords cannot reach a confident verdict (e.g. a follow-up question like
"why?").  If the LLM itself fails, the guard fails open (allows the question
through) to avoid blocking legitimate requests.

### Answer synthesis

This is the primary LLM use and runs on every request that passes the topic
guard.  After the appropriate tool has been called and returned structured
evidence, the LLM receives:

- A route-specific system prompt (different rules for RAG, task, job, and log
  analysis routes)
- Prior conversation history (raw question/answer pairs from `messages`)
- A synthesised user message containing the original question plus the tool
  evidence

The LLM's job is purely presentational: it summarises and explains evidence
but does not select tools, invent facts, or call further tools.

### Plan generation

`bamboo_plan` is an optional standalone tool for clients that need LLM-backed
multi-step planning.  It is not called by `bamboo_answer` — it is a separate
MCP entry point.  See the [bamboo_plan](#bamboo_plan-llm-backed-planner)
section below.

---

## Tool discovery

Tools are discovered via Python entry points in the `bamboo.tools` group (with
an optional legacy fallback to `askpanda.tools`).

Example entry point:

```toml
[project.entry-points."bamboo.tools"]
"atlas.task_status" = "askpanda_atlas.task_status:panda_task_status_tool"
```

Built-in tools are also registered directly in `bamboo.core.TOOLS`.

## Tool execution contract

Each tool provides:

- `get_definition() -> dict` — MCP tool definition including `name`,
  `description`, and `inputSchema` (JSON Schema with
  `"additionalProperties": false`).
- `async call(arguments: dict) -> list[MCPContent]` — returns a non-empty
  list of MCP content dicts, always with at least `{"type": "text", "text": "..."}`.

Tools that carry structured evidence (task status, job status, log analysis)
JSON-serialise their payload into the `text` field:

```python
return text_content(json.dumps({"evidence": {...}, "text": "Summary..."}))
```

Callers that need the raw evidence parse it back with
`json.loads(result[0]["text"])`.  The `bamboo_answer` tool does this
automatically via `_unpack_tool_result()`.

**Tools must never raise.** All error conditions are returned as
`text_content(error_message)` so the MCP client always receives a well-formed
response.

## Server-side argument validation

Before dispatching to a tool, `core.py`'s `call_tool` handler runs
`_validate_arguments()` against the tool's `inputSchema`.  This checks:

- `anyOf` satisfaction (e.g. `question` OR `messages` required)
- Required fields present and non-null
- No unknown keys when `additionalProperties: false`

Validation failures return a descriptive `text_content` error rather than
raising, so clients always receive a response.

## Orchestration model

Bamboo supports two complementary orchestration styles:

1. **Client-driven**: an MCP client reads `/tools/list` and lets a host LLM
   choose tools (standard MCP pattern).
2. **Server-driven (`bamboo_answer`)**: the `bamboo_answer` tool routes
   requests itself using deterministic rules, a two-stage topic guard, and
   LLM synthesis.  The `bamboo_plan` tool is available as a separate
   entry point for clients that want LLM-backed planning.

### bamboo_answer routing (server-driven)

`bamboo_answer` is the primary entry point for both UIs.  It accepts
`question` (string), `messages` (full chat history for multi-turn context),
and `bypass_routing` (flag to skip routing and go directly to the LLM).

**Step 1 — Argument validation** (in `core.py`, before dispatch)

The `inputSchema` is checked against the supplied arguments.

**Step 2 — History extraction**

Prior user/assistant turns are extracted from `messages`, stripping system
messages and the current question so they can be injected into later LLM calls.

**Step 3 — Topic guard** (`bamboo.tools.topic_guard`)

A two-stage guard runs before any tool or LLM call.  See
[Topic classification](#topic-classification) above for detail.  If the
question is rejected, a polite rejection message is returned immediately and
no downstream calls are made.

**Step 4 — Deterministic routing** (inside `bamboo_answer._route()`)

Regex patterns extract `task_id` and `job_id` from the question text.
The combination of IDs and keywords determines the route:

| Condition | Route | Tool called |
|---|---|---|
| job_id + analysis keywords | log analysis | `panda_log_analysis` |
| job_id only | job status | `panda_job_status` |
| task_id | task status | `panda_task_status` |
| no ID | RAG (vector + BM25, concurrent) | `panda_doc_search` + `panda_doc_bm25` |

There is no LLM involvement in this step — routing is fully deterministic
from the question text.

**Step 5 — LLM synthesis** (`bamboo_llm_answer` / `_call_llm`)

See [Answer synthesis](#answer-synthesis) above.

### bamboo_plan (LLM-backed planner)

`bamboo_plan` is a standalone MCP tool that an orchestrating client can call
when deterministic routing is insufficient.  It accepts a question, optional
hints, and an optional namespace list, and returns a validated JSON plan.

The planner uses two messages:

**System prompt:**

- You are a tool planner for an MCP server.
- Output MUST be a single JSON object matching the plan schema.
- Do not wrap output in Markdown.
- Only propose tools present in the provided tool catalog.

**User message:**

```json
{
  "question": "...",
  "hints": {"task_id": 123456},
  "tools": [
    {"name": "panda_task_status", "description": "...", "inputSchema": {}}
  ]
}
```

The planner extracts the first JSON object from the model output, validates it
against the `Plan` Pydantic model, and performs at most one repair attempt.
If both attempts fail, a structured error is returned (not raised).

The authoritative plan schema:

```python
from bamboo.tools.planner import get_plan_json_schema
schema = get_plan_json_schema()
```

## Context memory (multi-turn chat)

Both UIs maintain an in-memory conversation history (capped at
`BAMBOO_HISTORY_TURNS` user+assistant pairs, default 10).  On each question
the full history is sent as `messages` to `bamboo_answer`.  The server
extracts prior turns and injects them between the system prompt and the
synthesised user message.  The server is stateless — history lives in the
client.

## Sequence diagram

<img src="images/mcp_sequence_diagram.png" alt="Diagram" width="100%" />
