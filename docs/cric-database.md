# CRIC Queuedata Database (`cric_query`)

This document describes the `cric_query` tool, which answers natural-language
questions about ATLAS computing queues by translating questions into SQL and
executing them against the local CRIC queuedata database.

---

## Overview

The `cric_agent` (in `askpanda-atlas-agents`) periodically downloads queue
configuration and status data from the CRIC REST API and writes it to a local
[DuckDB](https://duckdb.org) file (`cric.duckdb` by default).  `cric_query`
reads that file and answers questions like:

- *"Which queues are not online?"*
- *"Which queues are using the objectstore copytool?"*
- *"Is the BNL-PTEST queue active?"*
- *"What is the status of all queues at BNL?"*
- *"How many queues are online?"*
- *"Which MCORE queues are online at BNL-ATLAS?"*
- *"When was the CRIC database last updated?"*

The tool uses the configured LLM to generate a SQL query, validates it through
a strict AST-based guard, executes it read-only against the DuckDB file, and
returns the results.

---

## Data freshness

The `cric_agent` fetches from:

```
https://atlas-cric.cern.ch/api/atlas/pandaqueue/query/?json&preset=schedconf.all
```

This endpoint returns the full ATLAS queuedata snapshot.  The agent writes a
`harvest_utc` / `last_modified` timestamp on every row.

To check data freshness:

```
When was the CRIC database last updated?
```

This generates:

```sql
SELECT MAX(last_modified) AS last_updated, COUNT(*) AS queue_count
FROM queuedata
LIMIT 1
```

---

## Database schema

One table is exposed to the LLM: `queuedata`.  All other tables (internal
DuckDB system tables, `information_schema`, the `snapshots` audit table) are
blocked by the guard.

### `queuedata`

One row per PanDA queue.  Rows are upserted on each ingestion cycle.

Key columns for natural-language queries:

| Column | Type | Description |
|---|---|---|
| `queue` | VARCHAR | PanDA queue name (primary key), e.g. `'BNL_ATLAS_1'` |
| `name` | VARCHAR | Human-readable queue nickname |
| `status` | VARCHAR | Queue status: `online`, `offline`, `test`, `brokeroff` |
| `state` | VARCHAR | Internal CRIC API field — always `'ACTIVE'` for all rows; **do not filter on this** |
| `type` | VARCHAR | Queue type: `production`, `analysis`, `unified`, etc. |
| `atlas_site` | VARCHAR | ATLAS site name, e.g. `'BNL-ATLAS'`, `'CERN-PROD'`, `'AGLT2'` |
| `site` | VARCHAR | Site identifier (may differ from `atlas_site`) |
| `cloud` | VARCHAR | PanDA cloud region, e.g. `'US'`, `'CERN'`, `'DE'` |
| `country` | VARCHAR | Country where the queue is hosted |
| `copytools` | VARCHAR | JSON array of primary copy tools, e.g. `'["rucio"]'` |
| `acopytools` | VARCHAR | JSON array of archive copy tools |
| `corecount` | INTEGER | CPU cores per job slot |
| `maxtime` | INTEGER | Maximum wall-clock time per job (seconds) |
| `maxrss` | INTEGER | Maximum memory per job (MB) |
| `pledgedcpu` | INTEGER | Pledged CPU capacity (cores) |
| `resource_type` | VARCHAR | Resource type: `SCORE`, `MCORE`, `HCORE` |
| `tier` | VARCHAR | WLCG tier: `Tier1`, `Tier2`, etc. |
| `last_modified` | TIMESTAMP | UTC timestamp of the last CRIC API update |

> **Important:** The `state` column is always `'ACTIVE'` for all rows — it
> reflects an internal CRIC API field and is not useful for filtering.  Use
> `status` instead: `'online'`, `'offline'`, `'test'`, `'brokeroff'`.

> **Site name format:** `atlas_site` values use the full CRIC site name, which
> typically includes a suffix: `'BNL-ATLAS'` (not `'BNL'`), `'CERN-PROD'`
> (not `'CERN'`), `'CERN-T0'`, `'CERN-P1'`, etc.  For user queries by site
> name, the SQL generation prompt instructs the LLM to use
> `atlas_site ILIKE '%BNL%'` rather than exact equality to match all
> BNL-related sites in one query.

> **Copytool format:** `copytools` and `acopytools` are JSON arrays stored as
> VARCHAR, e.g. `'["rucio"]'` or `'["rucio","gfalcopy"]'`.  Filter them with
> `copytools LIKE '%rucio%'`.  For negation use
> `(copytools IS NULL OR copytools NOT LIKE '%rucio%')` to handle NULL values
> correctly.

---

## Example questions and generated SQL

**Which queues are not online?**

```sql
SELECT atlas_site, status, COUNT(*) AS n, STRING_AGG(queue, ', ') AS queues
FROM queuedata
WHERE status != 'online'
GROUP BY atlas_site, status
ORDER BY atlas_site
LIMIT 500
```

**How many queues are in each status?**

```sql
SELECT status, COUNT(*) AS n
FROM queuedata
GROUP BY status
ORDER BY n DESC
LIMIT 500
```

**Which queues at BNL are active?**

```sql
SELECT queue, status, type
FROM queuedata
WHERE atlas_site ILIKE '%BNL%' AND status = 'online'
ORDER BY queue
LIMIT 50
```

**Which queues use the objectstore copytool?**

```sql
SELECT atlas_site, COUNT(*) AS n, STRING_AGG(queue, ', ') AS queues
FROM queuedata
WHERE copytools LIKE '%objectstore%'
GROUP BY atlas_site
ORDER BY atlas_site
LIMIT 500
```

**Which queues are NOT using the rucio copytool?**

```sql
SELECT atlas_site, COUNT(*) AS n, STRING_AGG(queue, ', ') AS queues
FROM queuedata
WHERE (copytools IS NULL OR copytools NOT LIKE '%rucio%')
GROUP BY atlas_site
ORDER BY atlas_site
LIMIT 500
```

**What copytools are in use?**

```sql
SELECT json_extract_string(copytools, '$[0]') AS copytool, COUNT(*) AS n
FROM queuedata
GROUP BY copytool
ORDER BY n DESC
LIMIT 500
```

**What sites are available?**

```sql
SELECT DISTINCT atlas_site, COUNT(*) AS n
FROM queuedata
GROUP BY atlas_site
ORDER BY atlas_site
LIMIT 500
```

**Which MCORE queues are online at BNL?**

```sql
SELECT queue, status, corecount
FROM queuedata
WHERE atlas_site ILIKE '%BNL%'
  AND resource_type = 'MCORE'
  AND status = 'online'
LIMIT 50
```

---

## Security: the AST guard

Every SQL string generated by the LLM passes through `validate_and_guard()`
in `cric_query_schema.py` before it touches DuckDB.  The guard uses
[sqlglot](https://github.com/tobymao/sqlglot) to parse the SQL into an AST
and applies seven rules:

| Rule | What it blocks |
|---|---|
| Single statement | Stacked statements (`SELECT 1; DROP TABLE queuedata`) |
| SELECT-only root | INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE, GRANT, REVOKE, COMMIT, ROLLBACK |
| No forbidden nodes anywhere | DDL/DML/DCL/TCL at any depth of the AST, including subqueries |
| No system tables | `information_schema`, `pg_catalog`, `sqlite_master`, `duckdb_*`, `pg_*` |
| No system functions | `duckdb_tables()`, `duckdb_columns()`, and similar internal functions |
| Table allow-list | Any table not in `{queuedata}` — including `jobs`, `snapshots`, etc. |
| LIMIT injection | Queries without LIMIT get one injected; GROUP BY queries get `LIMIT 500`, raw per-queue queries get `LIMIT 50` |

> **GROUP BY limit behaviour:** The guard distinguishes aggregation queries
> (which have a GROUP BY clause) from raw row-per-queue queries.  Aggregation
> results are bounded by the number of distinct groups (at most a few hundred
> sites/states/copytools), so they receive a higher automatic cap (`LIMIT 500`)
> to ensure no groups are truncated.  If the LLM writes a lower limit on a
> GROUP BY query (e.g. `LIMIT 50`), the guard raises it to `LIMIT 500`
> automatically.  Raw per-queue SELECT queries keep the lower cap (`LIMIT 50`)
> to keep the synthesis prompt small.

The DuckDB connection is always opened with `read_only=True` as a backstop.

---

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `CRIC_DUCKDB_PATH` | `cric.duckdb` | Path to the DuckDB file written by the `cric_agent` |
| `CRIC_QUERY_MAX_ROWS` | `50` | Maximum rows for raw per-queue queries |

Set these in `bamboo_env.sh`:

```bash
export CRIC_DUCKDB_PATH="${HOME}/.askpanda/cric.duckdb"
```

---

## Routing

`bamboo_answer` detects CRIC questions via `_is_cric_question()`, which uses
two strategies:

1. **Signal phrase match** — any phrase in `_CRIC_SIGNALS` appears in the
   question.  Signal phrases include: `"cric"`, `"copytool"`, `"brokeroff"`,
   `"maxwalltime"`, `"queue online"`, `"queue offline"`, `"active queues"`,
   `"online queues"`, `"offline queues"`, `"panda queues"`, etc.

2. **Queue + status combo** — the question contains a queue-reference word
   (`"queue"` or `"queues"`) AND a queue-status word (`"active"`, `"online"`,
   `"offline"`, `"brokeroff"`).  This catches patterns like *"Which queues at
   BNL are active?"* where the status word appears after a site name.

CRIC questions bypass the topic guard (they are self-evidently on-topic) and
route directly to `cric_query` via `_build_deterministic_plan()`.  CRIC
signals take priority over jobs DB signals when both fire simultaneously —
a question about "queues" is almost always a CRIC question, not a PanDA jobs
question.

**Contextual follow-ups:** after a CRIC response, short status-check
follow-ups like *"Is BNL-PTEST active?"* are recognised as CRIC follow-ups
even without explicit CRIC vocabulary.  This is detected by
`_last_tool_was_cric()` (scans the last assistant turn for CRIC vocabulary)
and `_is_cric_followup()` (matches short queue-status question patterns).

**Clarification replies:** if the router triggers a disambiguation prompt
("which database did you mean?") and the user replies with just `"cric"`,
the original question is recovered from history and passed to `cric_query`.

---

## Multi-database disambiguation

Both `cric_query` and `panda_jobs_query` answer questions about queues and
sites, but from different perspectives.  `cric_query` knows about queue
*configuration and status* (copytools, CPU limits, online/offline state);
`panda_jobs_query` knows about live *job statistics* at each queue (failure
counts, error codes, running jobs).

When a question is genuinely ambiguous — for example, *"which queues have the
most failed jobs?"* — `bamboo_answer` detects that both databases match and
returns a clarification response:

> *"I can query multiple databases.  Which one did you mean?*
> *jobs — PanDA jobs database (computing site job statistics, error counts)*
> *cric — CRIC (Computing Resource Information Catalogue — queues, sites, copytools)"*

Most queue questions that contain words like `"copytool"`, `"online"`,
`"offline"`, `"active"`, `"MCORE"`, or `"maxwalltime"` are unambiguously
CRIC and route directly without prompting.

---

## Known limitations

- **No job counts:** `nqueued` and `nrunning` columns are not present in the
  current `cric_agent` schema.  Questions about running job counts at a site
  should use `panda_jobs_query` instead.
- **Copytool arrays:** `copytools` is a JSON array, not a plain string.
  `LIKE '%rucio%'` works correctly for most cases, but a queue with
  `copytools = '["gfalcopy","rucio"]'` will match both `%rucio%` and
  `%gfalcopy%`.
- **State column:** `state` is always `'ACTIVE'` — it mirrors the CRIC API's
  internal field and cannot be used to detect offline or test queues.
  Always use `status` for queue availability filtering.
