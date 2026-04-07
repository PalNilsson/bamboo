"""Schema allow-list, SQL guard, and LLM context builder for the CRIC query tool.

This module is the single security gate between LLM-generated SQL and the
DuckDB execution engine for CRIC (Computing Resource Information Catalogue)
queuedata.  It has **zero bamboo-core dependency** so it can be imported
freely by tests, stubs, and the tool implementation alike.

Guard pipeline
--------------
Every SQL string produced by the LLM passes through :func:`validate_and_guard`
before it is executed.  The guard:

1. Parses the SQL into an AST via ``sqlglot`` (DuckDB dialect).
2. Rejects any input that is not a single SELECT statement.
3. Rejects DDL, DML, DCL, and TCL at any depth of the AST.
4. Rejects references to system/internal tables.
5. Rejects table references not in :data:`ALLOWED_TABLES`.
6. Injects a ``LIMIT`` clause if the query omits one, capped at
   :data:`MAX_ROWS`.

Schema context
--------------
:func:`build_schema_context` returns a compact multi-line string describing
the CRIC database schema.  The string is intended for inclusion in LLM system
prompts.  Results are cached in-process with a configurable TTL
(:data:`SCHEMA_CONTEXT_TTL`) because the schema never changes between
ingestion cycles.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import sqlglot
import sqlglot.expressions as exp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Tables exposed to the LLM and permitted in generated queries.
#: CRIC v1 exposes only the single ``queuedata`` table produced by cric_agent.
ALLOWED_TABLES: frozenset[str] = frozenset({"queuedata"})

#: Maximum rows returned per query.  There are ~230 ATLAS queues total.
#: Aggregation queries (GROUP BY) return few rows regardless of this cap.
#: Raw queue-per-row queries are capped here to keep synthesis prompts
#: manageable (~50 rows * ~28 tok/row ≈ 1400 tokens).
MAX_ROWS: int = 50

#: Higher cap applied when the query contains a GROUP BY clause.
#: Aggregations produce one row per distinct group, not one row per queue,
#: so the full result set is always small and must not be truncated.
MAX_ROWS_AGGREGATION: int = 500

#: Row cap for the dedicated ``list_all_queues`` fast path, which bypasses the
#: NL→SQL pipeline entirely.  ATLAS has ~500 queues today; 2000 gives ample
#: headroom without risking unbounded memory use.
MAX_ROWS_FULL_LIST: int = 2000

#: Hard timeout applied to every query execution (seconds).
QUERY_TIMEOUT_SECS: int = 10

#: TTL for the cached schema context string (seconds).  One hour is
#: sufficient — the schema does not change between ingestion cycles.
SCHEMA_CONTEXT_TTL: float = 3600.0

# System / internal table name prefixes and exact names to reject.
_SYSTEM_TABLE_NAMES: frozenset[str] = frozenset(
    {
        "information_schema",
        "pg_catalog",
        "sys",
        "sqlite_master",
        "sqlite_temp_master",
    }
)
_SYSTEM_TABLE_PREFIXES: tuple[str, ...] = ("duckdb_", "pg_", "sqlite_")

# AST node types that are forbidden anywhere in the statement tree.
_FORBIDDEN_NODE_TYPES: tuple[type[Any], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Grant,
    exp.Revoke,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Command,   # covers EXECUTE, CALL, and other commands
)

# ---------------------------------------------------------------------------
# GuardResult
# ---------------------------------------------------------------------------


@dataclass
class GuardResult:
    """Outcome of :func:`validate_and_guard`.

    Attributes:
        passed: ``True`` if the SQL passed all checks and is safe to execute.
        sanitised_sql: The SQL string with a ``LIMIT`` injected (if absent),
            ready for execution.  ``None`` when ``passed`` is ``False``.
        rejection_reason: Human-readable explanation of why the SQL was
            rejected.  ``None`` when ``passed`` is ``True``.
        triggered_rule: Short identifier for the rule that triggered
            rejection.  ``None`` when ``passed`` is ``True``.
    """

    passed: bool
    sanitised_sql: str | None = field(default=None)
    rejection_reason: str | None = field(default=None)
    triggered_rule: str | None = field(default=None)


# ---------------------------------------------------------------------------
# Guard implementation
# ---------------------------------------------------------------------------


def _is_system_name(name: str) -> bool:
    """Return ``True`` if *name* matches a known system table prefix or name.

    Args:
        name: Lowercase identifier to check (table name, db qualifier, etc.).

    Returns:
        ``True`` if the name refers to an internal or system object.
    """
    if not name:
        return False
    if name in _SYSTEM_TABLE_NAMES:
        return True
    return any(name.startswith(pfx) for pfx in _SYSTEM_TABLE_PREFIXES)


def _check_system_tables(
    ast: exp.Select,
    cte_aliases: frozenset[str],
) -> GuardResult | None:
    """Check AST for references to system tables or internal functions.

    Inspects all ``Table`` nodes and ``Anonymous`` function calls in *ast*.
    Returns a :class:`GuardResult` rejection if a system reference is found,
    or ``None`` if all references are clean.

    Args:
        ast: Parsed SELECT expression to inspect.
        cte_aliases: Set of CTE-defined alias names to skip (they are not
            external table references).

    Returns:
        A failed :class:`GuardResult` if a system reference is detected,
        otherwise ``None``.
    """
    for table_node in ast.find_all(exp.Table):
        name = (table_node.name or "").lower()
        if name in cte_aliases:
            continue
        db_node = table_node.args.get("db")
        db = (db_node.name if db_node is not None else "").lower()
        catalog_node = table_node.args.get("catalog")
        catalog = (catalog_node.name if catalog_node is not None else "").lower()
        for part in (name, db, catalog):
            if _is_system_name(part):
                return GuardResult(
                    passed=False,
                    rejection_reason=f"Reference to system table '{part}' is not permitted.",
                    triggered_rule="system_table",
                )

    for func_node in ast.find_all(exp.Anonymous):
        func_name = (func_node.name or "").lower()
        if any(func_name.startswith(pfx) for pfx in _SYSTEM_TABLE_PREFIXES):
            return GuardResult(
                passed=False,
                rejection_reason=f"Reference to system function '{func_name}' is not permitted.",
                triggered_rule="system_table",
            )
    return None


def _check_table_allowlist(
    ast: exp.Select,
    cte_aliases: frozenset[str],
) -> GuardResult | None:
    """Check that every table reference in *ast* is in :data:`ALLOWED_TABLES`.

    Args:
        ast: Parsed SELECT expression to inspect.
        cte_aliases: Set of CTE-defined alias names to skip.

    Returns:
        A failed :class:`GuardResult` if an unknown table is found,
        otherwise ``None``.
    """
    for table_node in ast.find_all(exp.Table):
        name = (table_node.name or "").lower()
        if name in cte_aliases:
            continue
        if name and name not in ALLOWED_TABLES:
            return GuardResult(
                passed=False,
                rejection_reason=(
                    f"Table '{name}' is not in the list of permitted tables: "
                    f"{sorted(ALLOWED_TABLES)}."
                ),
                triggered_rule="unknown_table",
            )
    return None


def validate_and_guard(sql: str) -> GuardResult:
    """Validate SQL syntax and enforce the read-only allow-list guard.

    Parses *sql* into an AST using the DuckDB dialect and evaluates it
    against the configured guard rules.  Returns a structured result
    indicating whether the SQL is safe to execute and, if not, which rule
    triggered rejection.

    The guard never attempts to fix or rewrite malformed SQL — it either
    passes the statement (optionally injecting a ``LIMIT``) or rejects it.

    Args:
        sql: Raw SQL string as generated by the LLM.  May be malformed or
            contain adversarial constructs.

    Returns:
        :class:`GuardResult` with ``passed=True`` and ``sanitised_sql`` set
        on success, or ``passed=False`` with ``rejection_reason`` and
        ``triggered_rule`` set on failure.
    """
    # --- Rule 1: parse must succeed -----------------------------------------
    try:
        statements = sqlglot.parse(sql, dialect="duckdb", error_level=sqlglot.ErrorLevel.RAISE)
    except sqlglot.ParseError as exc:
        return GuardResult(
            passed=False,
            rejection_reason=f"SQL could not be parsed: {exc}",
            triggered_rule="parse_error",
        )

    # --- Rule 2: exactly one statement --------------------------------------
    if len(statements) != 1:
        return GuardResult(
            passed=False,
            rejection_reason=(
                f"Expected exactly one SQL statement, got {len(statements)}. "
                "Stacked statements are not permitted."
            ),
            triggered_rule="multiple_statements",
        )

    ast = statements[0]
    if ast is None:
        return GuardResult(
            passed=False,
            rejection_reason="Empty SQL statement.",
            triggered_rule="empty_statement",
        )

    # --- Rule 3: top-level must be SELECT -----------------------------------
    if not isinstance(ast, exp.Select):
        node_type = type(ast).__name__
        return GuardResult(
            passed=False,
            rejection_reason=(
                f"Only SELECT statements are permitted; got {node_type}."
            ),
            triggered_rule="non_select_root",
        )

    # --- Rule 4: no forbidden node types anywhere in the tree ---------------
    for node in ast.walk():
        if isinstance(node, _FORBIDDEN_NODE_TYPES):
            node_type = type(node).__name__
            return GuardResult(
                passed=False,
                rejection_reason=(
                    f"Forbidden SQL construct '{node_type}' found in query."
                ),
                triggered_rule="forbidden_construct",
            )

    # --- Rules 5 & 5b: no system / internal tables or functions ------------
    cte_aliases: frozenset[str] = frozenset(
        cte.alias.lower() for cte in ast.find_all(exp.CTE)
    )
    system_rejection = _check_system_tables(ast, cte_aliases)
    if system_rejection is not None:
        return system_rejection

    # --- Rule 6: table allow-list -------------------------------------------
    allowlist_rejection = _check_table_allowlist(ast, cte_aliases)
    if allowlist_rejection is not None:
        return allowlist_rejection

    # --- Rule 7: LIMIT injection / correction ------------------------------
    # For aggregation queries (GROUP BY) use the higher cap so no groups
    # are truncated.  The LLM sometimes writes LIMIT 50 even on GROUP BY
    # queries — raise that to MAX_ROWS_AGGREGATION so the full result is
    # returned.  For raw-row queries use MAX_ROWS (50) to keep synthesis small.
    is_aggregation = _has_group_by(ast)
    effective_max = MAX_ROWS_AGGREGATION if is_aggregation else MAX_ROWS

    existing_limit = ast.find(exp.Limit)
    if existing_limit is None:
        sanitised_sql = _inject_limit_if_absent(ast, effective_max)
    elif is_aggregation:
        # Raise an LLM-supplied limit that is too low for an aggregation.
        try:
            llm_limit = int(existing_limit.expression.sql())
        except Exception:  # noqa: BLE001
            llm_limit = 0
        if llm_limit < effective_max:
            ast = ast.limit(effective_max)
        sanitised_sql = ast.sql(dialect="duckdb")
    else:
        sanitised_sql = ast.sql(dialect="duckdb")

    return GuardResult(passed=True, sanitised_sql=sanitised_sql)


def _inject_limit_if_absent(ast: exp.Select, max_rows: int) -> str:
    """Return the SQL string, injecting LIMIT *max_rows* if not already present.

    Modifies the AST in-place (sqlglot nodes are mutable) and renders it
    back to a SQL string using the DuckDB dialect.

    Args:
        ast: A parsed SELECT expression.
        max_rows: The LIMIT value to inject if none is present.

    Returns:
        SQL string with a LIMIT clause guaranteed to be present.
    """
    if ast.find(exp.Limit) is None:
        ast = ast.limit(max_rows)
    return ast.sql(dialect="duckdb")


def _has_group_by(ast: exp.Select) -> bool:
    """Return True if *ast* contains a GROUP BY clause at the top level.

    Args:
        ast: Parsed SELECT expression.

    Returns:
        True when a GROUP BY clause is present.
    """
    return ast.find(exp.Group) is not None


# ---------------------------------------------------------------------------
# Schema context cache
# ---------------------------------------------------------------------------

_ctx_lock: threading.Lock = threading.Lock()
# key: tuple of sorted table names (as a string); value: (expiry, context_str)
_ctx_cache: dict[str, tuple[float, str]] = {}


def build_schema_context(tables: list[str] | None = None) -> str:
    """Return a compact schema summary suitable for inclusion in an LLM prompt.

    Results are cached in-process for :data:`SCHEMA_CONTEXT_TTL` seconds.
    The schema is stable between ingestion cycles so a long TTL is safe.

    Args:
        tables: Optional list of table names to include.  Defaults to the
            single CRIC data table: ``queues``.

    Returns:
        A multi-line string describing the schema, ready to paste into an
        LLM system prompt.
    """
    if tables is None:
        tables = ["queuedata"]

    cache_key = ",".join(sorted(tables))
    now = time.monotonic()

    with _ctx_lock:
        entry = _ctx_cache.get(cache_key)

    if entry is not None:
        expiry, ctx = entry
        if expiry == math.inf or now < expiry:
            return ctx

    # Cache miss — build the context string.
    ctx = _build_context_uncached(tables)
    expiry = now + SCHEMA_CONTEXT_TTL

    with _ctx_lock:
        _ctx_cache[cache_key] = (expiry, ctx)

    return ctx


def _build_context_uncached(tables: list[str]) -> str:
    """Build the schema context string without consulting the cache.

    Uses a hand-built summary of the CRIC ``queuedata`` table based on the
    actual schema produced by the cric_agent.  The column list reflects the
    CRIC REST API response fields as stored by the agent.

    Args:
        tables: List of table names to include in the context.

    Returns:
        Multi-line schema context string.
    """
    _FALLBACK: dict[str, list[tuple[str, str, str]]] = {
        "queuedata": [
            ("queue", "VARCHAR", "PanDA queue name (primary key), e.g. 'BNL_ATLAS_1'."),
            ("name", "VARCHAR", "Human-readable queue nickname (may differ from queue)."),
            ("state", "VARCHAR", "Queue state (CRIC API, UPPERCASE):"
                      " 'ACTIVE'=online, 'BROKEROFF', 'OFFLINE', 'TEST'."),
            ("status", "VARCHAR", "Alternate status field — may also use CRIC API values."),
            ("type", "VARCHAR", "Queue type: 'production', 'analysis', 'unified', etc."),
            ("atlas_site", "VARCHAR", "ATLAS site name (e.g. 'BNL', 'CERN', 'AGLT2')."),
            ("site", "VARCHAR", "Site identifier (may differ from atlas_site)."),
            ("panda_site", "VARCHAR", "PanDA site name."),
            ("cloud", "VARCHAR", "PanDA cloud region (e.g. 'US', 'CERN', 'DE')."),
            ("country", "VARCHAR", "Country where the queue is hosted."),
            ("copytools", "VARCHAR", "JSON array of primary copy tools, e.g. '[\"rucio\"]'."),
            ("acopytools", "VARCHAR", "JSON array of archive copy tools, e.g. '[\"rucio\"]'."),
            ("corecount", "INTEGER", "CPU cores allocated per job slot."),
            ("maxtime", "INTEGER", "Maximum wall-clock time per job (seconds). Equiv. maxwalltime."),
            ("maxrss", "INTEGER", "Maximum memory per job (MB). Equiv. maxmemory."),
            ("pledgedcpu", "INTEGER", "Pledged CPU capacity (cores)."),
            ("tier", "VARCHAR", "WLCG tier level (e.g. 'Tier1', 'Tier2')."),
            ("tier_level", "INTEGER", "Numeric tier level (1, 2, 3)."),
            ("resource_type", "VARCHAR", "Resource type (e.g. 'MCORE', 'SCORE', 'HCORE')."),
            ("vo_name", "VARCHAR", "Virtual organisation (almost always 'atlas')."),
            ("last_modified", "TIMESTAMP", "UTC timestamp when this queue record was last updated."),
            ("state_update", "TIMESTAMP", "UTC timestamp of the last state change."),
        ],
    }

    lines: list[str] = []
    for table in tables:
        cols = _FALLBACK.get(table, [])
        lines.append(f"Table: {table}")
        for col_name, col_type, desc in cols:
            lines.append(f"  {col_name:<30} {col_type:<10}  {desc}")
        lines.append("")
    return "\n".join(lines)


def invalidate_schema_cache() -> None:
    """Evict all cached schema context strings.

    Intended for tests and situations where the schema has changed and a
    fresh context string is required immediately.
    """
    with _ctx_lock:
        _ctx_cache.clear()


# ---------------------------------------------------------------------------
# LLM prompt builder
# ---------------------------------------------------------------------------

#: Sentinel returned by the LLM when it cannot generate SQL.
CANNOT_ANSWER_SENTINEL: str = "CANNOT_ANSWER"

#: Instructions injected into the LLM system prompt for SQL generation.
SQL_GENERATION_SYSTEM_TEMPLATE: str = """\
You are a read-only SQL assistant for a CRIC (Computing Resource Information
Catalogue) queuedata database (DuckDB dialect).

{schema_context}

Rules:
- Return ONLY a single SELECT statement. No explanation, no markdown, no fences.
- Do not use INSERT, UPDATE, DELETE, DROP, CREATE, or any DDL or DML.
- Do not reference information_schema or any system tables.
- Do not use semicolons.
- The table is named 'queuedata' (not 'queues').
- Use the 'status' column for queue filtering (NOT 'state' — state is always 'ACTIVE').
  Status values are lowercase: 'online', 'offline', 'test', 'brokeroff'.
- The site column is 'atlas_site'. Site names often include suffixes, e.g. 'BNL-ATLAS'
  (not 'BNL'), 'CERN-PROD' (not 'CERN'). For user-provided site names, always use
  ILIKE '%<site>%' rather than exact equality, e.g. atlas_site ILIKE '%BNL%'.
- 'copytools' and 'acopytools' are JSON arrays stored as VARCHAR, e.g. '["rucio"]'.
  Use copytools LIKE '%rucio%' for containment checks.
  For NULL safety when negating: WHERE (copytools IS NULL OR copytools NOT LIKE '%rucio%')
- Use last_modified for freshness checks.
- For questions asking "which queues are not X" across all sites, always use
  GROUP BY atlas_site, status to return a complete summary grouped by site —
  never use a raw per-queue SELECT for these broad questions or data will be
  truncated. Only use raw per-queue SELECT when scoped to a specific site.
- If the question cannot be answered by a SQL query against the available tables,
  reply with exactly: CANNOT_ANSWER

Example queries:
- "Which queues are not online?" →
  SELECT atlas_site, status, COUNT(*) AS n, STRING_AGG(queue, ', ') AS queues
  FROM queuedata WHERE status != 'online'
  GROUP BY atlas_site, status ORDER BY atlas_site LIMIT 500
- "How many queues are in each state?" →
  SELECT status, COUNT(*) AS n FROM queuedata GROUP BY status ORDER BY n DESC LIMIT 500
- "Which queues are using the rucio copytool?" →
  SELECT atlas_site, COUNT(*) AS n FROM queuedata
  WHERE copytools LIKE '%rucio%' GROUP BY atlas_site ORDER BY n DESC LIMIT 500
- "Which queues are NOT using the rucio copytool?" →
  SELECT atlas_site, COUNT(*) AS n, STRING_AGG(queue, ', ') AS queues
  FROM queuedata WHERE (copytools IS NULL OR copytools NOT LIKE '%rucio%')
  GROUP BY atlas_site ORDER BY atlas_site LIMIT 500
- "Is the BNL queue online?" →
  SELECT queue, state FROM queuedata WHERE atlas_site ILIKE '%BNL%' ORDER BY queue LIMIT 50
- "What is the status of all queues at CERN?" →
  SELECT queue, state, type FROM queuedata WHERE atlas_site ILIKE '%CERN%' ORDER BY queue LIMIT 50
- "How many queues are online?" →
  SELECT COUNT(*) AS n FROM queuedata WHERE status = 'online' LIMIT 1
- "Which queues use gfalcopy?" →
  SELECT atlas_site, COUNT(*) AS n, STRING_AGG(queue, ', ') AS queues
  FROM queuedata WHERE copytools LIKE '%gfalcopy%'
  GROUP BY atlas_site ORDER BY atlas_site LIMIT 500
- "What copytools are in use?" →
  SELECT json_extract_string(copytools, '$[0]') AS copytool, COUNT(*) AS n
  FROM queuedata GROUP BY copytool ORDER BY n DESC LIMIT 500
- "When was the CRIC database last updated?" →
  SELECT MAX(last_modified) AS last_updated, COUNT(*) AS queue_count FROM queuedata LIMIT 1
- "What sites are available?" →
  SELECT DISTINCT atlas_site, COUNT(*) AS n FROM queuedata GROUP BY atlas_site ORDER BY atlas_site LIMIT 500
- "Show queues at BNL" →
  SELECT queue, status, type FROM queuedata WHERE atlas_site ILIKE '%BNL%' ORDER BY queue LIMIT 50
- "Which MCORE queues are online at BNL?" →
  SELECT queue, state, corecount FROM queuedata
  WHERE atlas_site ILIKE '%BNL%' AND resource_type = 'MCORE' AND status = 'online' LIMIT 50
- "How many CPU cores are pledged at CERN?" →
  SELECT SUM(pledgedcpu) AS total_pledged FROM queuedata WHERE atlas_site ILIKE '%CERN%' LIMIT 1
- "Summary of queue states across all sites" →
  SELECT status, COUNT(*) AS n FROM queuedata GROUP BY status ORDER BY n DESC LIMIT 500
"""


def build_sql_prompt(
    question: str,
    schema_context: str | None = None,
) -> list[dict[str, Any]]:
    """Build the LLM message list for SQL generation.

    Args:
        question: Natural-language question from the user.
        schema_context: Pre-built schema context string.  If ``None``,
            :func:`build_schema_context` is called to generate it.

    Returns:
        A list of ``{"role": str, "content": str}`` dicts suitable for
        passing to any Bamboo LLM provider.
    """
    if schema_context is None:
        schema_context = build_schema_context()

    system_content = SQL_GENERATION_SYSTEM_TEMPLATE.format(
        schema_context=schema_context,
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]
