"""Implementation of ``cric_query`` — natural-language to SQL for the CRIC DB.

Translates a natural-language question into a DuckDB SQL query, validates the
generated SQL through the AST guard in :mod:`askpanda_atlas.cric_query_schema`,
executes it against the read-only CRIC DuckDB file, and returns a compact
evidence dict structured for LLM synthesis.

Public surface (mirrors ``jobs_query_impl`` pattern exactly):

- ``get_definition()``         — MCP tool definition dict
- ``CricQueryTool``            — MCP tool class with ``get_definition()`` and
  async ``call()``
- ``cric_query_tool``          — singleton instance

Design rules (per Bamboo architecture):

* All ``bamboo.tools.base`` and ``bamboo.llm.*`` imports are **deferred inside
  ``call()`` and helpers** — never at module level.  This keeps every pure
  helper importable without bamboo installed.
* The LLM call is async (``await client.generate(...)``); DuckDB execution
  runs synchronously on the event loop thread — queries are fast (2-15ms)
  and this avoids thread-pool conflicts with DuckDB on macOS.
* ``call()`` never raises — errors are returned as ``text_content`` payloads.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

try:
    import duckdb
except ImportError as _duckdb_import_error:  # pragma: no cover
    raise ImportError(
        "cric_query requires the 'duckdb' package. "
        "Install it with: pip install duckdb"
    ) from _duckdb_import_error

logger = logging.getLogger(__name__)


def _serialise_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a DuckDB result row to a JSON-safe dict.

    DuckDB returns Python ``datetime``, ``date``, and ``Decimal`` objects
    for the corresponding column types.  These are not JSON-serialisable by
    default, so they are converted to strings here before the evidence dict
    is passed to ``json.dumps``.

    Args:
        row: Raw row dict from DuckDB with Python-typed values.

    Returns:
        Row dict with all values converted to JSON-safe types.
    """
    import datetime
    import decimal

    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, (datetime.datetime, datetime.date, datetime.time)):
            out[k] = v.isoformat()
        elif isinstance(v, decimal.Decimal):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _db_mtime(duckdb_path: str) -> str | None:
    """Return the database file's last-modified time as a compact UTC string.

    Used to populate ``db_last_modified`` in evidence dicts so the TUI can
    display a "Database last updated" footnote without an extra SQL query.

    Args:
        duckdb_path: Filesystem path to the DuckDB file.

    Returns:
        ISO-style UTC timestamp string (e.g. ``"2026-04-07 10:31 UTC"``), or
        ``None`` when the path is ``":memory:"``, empty, or the file is absent.
    """
    import datetime  # deferred — only called at query time
    if not duckdb_path or duckdb_path == ":memory:":
        return None
    try:
        mtime = os.path.getmtime(duckdb_path)
        return datetime.datetime.fromtimestamp(
            mtime, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")
    except OSError:
        return None


def _execute_query(
    duckdb_path: str,
    sql: str,
    timeout_secs: int,
    max_rows: int,
) -> dict[str, Any]:
    """Open the database, execute *sql*, and return rows plus metadata.

    Opens a fresh read-only connection on the calling thread (required for
    DuckDB thread safety — connections must not be shared across threads),
    executes the query with a timeout, caps results at *max_rows*, and closes
    the connection before returning.

    Args:
        duckdb_path: Filesystem path to the DuckDB file, or ``":memory:"``.
        sql: Validated SQL string ready for execution.
        timeout_secs: Maximum execution time in seconds.
        max_rows: Maximum rows to return.

    Returns:
        Dict with keys ``columns``, ``rows``, ``row_count``, ``truncated``,
        and ``execution_time_ms``.

    Raises:
        Exception: Any DuckDB error is re-raised so the caller can wrap it
            in a structured error response.
    """
    read_only = duckdb_path != ":memory:"
    conn = duckdb.connect(duckdb_path, read_only=read_only)
    try:
        try:
            conn.execute(f"SET statement_timeout='{timeout_secs}s'")
        except duckdb.Error:
            pass

        t0 = time.monotonic()
        result = conn.execute(sql)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        columns: list[str] = [d[0] for d in (result.description or [])]
        raw_rows = result.fetchmany(max_rows + 1)
        truncated = len(raw_rows) > max_rows
        rows = [_serialise_row(dict(zip(columns, row))) for row in raw_rows[:max_rows]]

        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
            "execution_time_ms": round(elapsed_ms, 2),
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# SQL string helpers (synchronous, no I/O)
# ---------------------------------------------------------------------------


def _strip_sql_fences(raw: str) -> str:
    """Remove markdown code fences from *raw*, returning the inner SQL.

    Handles both ````sql ... ```` and plain ```` ``` ... ```` wrappers.
    Leading and trailing whitespace is stripped from the result.

    Args:
        raw: Raw string as returned by the LLM.

    Returns:
        SQL string with code fences removed.
    """
    text = raw.strip()
    for fence_open in ("```sql", "```SQL", "```"):
        if text.startswith(fence_open):
            text = text[len(fence_open):]
            if text.endswith("```"):
                text = text[:-3]
            break
    return text.strip()


def _looks_like_cannot_answer(text: str) -> bool:
    """Return ``True`` when the LLM signals it cannot produce SQL.

    Args:
        text: Stripped LLM reply text.

    Returns:
        ``True`` if the reply matches the cannot-answer sentinel or a
        plausible natural-language refusal.
    """
    from askpanda_atlas.cric_query_schema import CANNOT_ANSWER_SENTINEL

    if text.upper() == CANNOT_ANSWER_SENTINEL:
        return True
    lower = text.lower()
    refusal_phrases = (
        "i cannot", "i can't", "i don't know", "i do not know",
        "unable to", "cannot generate", "cannot answer", "not possible",
        "no sql", "i'm sorry",
    )
    return any(phrase in lower for phrase in refusal_phrases)


# ---------------------------------------------------------------------------
# List-all-queues fast path (no LLM, no SQL generation)
# ---------------------------------------------------------------------------

#: Signal phrases that indicate the user wants *all* queues without a site or
#: status filter.  Matching is case-insensitive substring search.  Short,
#: Regex that matches the core "list/show/get all queues" intent.
#: Handles natural variations like "show me all the PanDA queues in CRIC",
#: "list all queues", "all panda queues", "give me a list of all queues", etc.
#: Trailing "in/from CRIC" or "in the CRIC database" are intentionally allowed
#: because they name CRIC as the data source, not as a site filter.
_LIST_ALL_RE: re.Pattern[str] = re.compile(
    r"""
    (?:
        (?:list|show|get|give\s+me|display)
        (?:\s+(?:me|us))?
        (?:\s+a\s+list\s+of)?
        \s+all
        (?:\s+the)?
        (?:\s+(?:panda|pandda|atlas))?
        \s+queues
    |
        all
        (?:\s+the)?
        (?:\s+(?:panda|pandda|atlas))?
        \s+queues
    |
        every\s+queue
    |
        (?:full|complete)\s+queue\s+list
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

#: Veto regex: if any of these patterns appear alongside a list-all signal,
#: the question is scoped (site or status filter) and must go through the
#: NL-to-SQL pipeline instead of the fast path.
_LIST_ALL_VETO_RE: re.Pattern[str] = re.compile(
    r"""
    \b(?:
        at\s+\S
        | for\s+\S
        | using\b
        | with\b
        | that\b
        | where\b
        | which\b
        | \bonline\b
        | \boffline\b
        | \bbrokeroff\b
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

#: SQL executed by the list-all fast path.  Columns chosen to be informative
#: yet compact; ordered by site then queue for human readability.
_LIST_ALL_SQL: str = (
    "SELECT queue, atlas_site, status, type "
    "FROM queuedata "
    "ORDER BY atlas_site, queue"
)


def _is_list_all_queues(question: str) -> bool:
    """Return ``True`` when *question* is an unscoped request for all queues.

    Detects a wide range of natural phrasings -- "list all queues",
    "show me all the PanDA queues in CRIC", "all panda queues", etc. --
    while vetoing scoped requests like "list all queues at BNL" or
    "all online queues", which must flow through the NL-to-SQL pipeline.

    The detection uses two compiled regexes rather than a fixed phrase list
    so that natural word-order variations ("show me all the ...") are covered
    without enumerating every possible phrasing.

    Args:
        question: Natural-language question from the user (already stripped).

    Returns:
        ``True`` if the question is an unscoped list-all request;
        ``False`` otherwise.
    """
    if not _LIST_ALL_RE.search(question):
        return False
    return not bool(_LIST_ALL_VETO_RE.search(question))


def list_all_queues(duckdb_path: str) -> dict[str, Any]:
    """Return all queues in the CRIC database without LLM involvement.

    Executes a fixed, pre-validated SELECT directly against the DuckDB file,
    bypassing the LLM SQL-generation pipeline entirely.  This avoids the
    ``MAX_ROWS=50`` fetch cap that applies to NL→SQL queries and ensures the
    user always receives the full queue inventory.

    The result structure mirrors the evidence dict produced by
    :func:`fetch_and_analyse` so the Bamboo executor and synthesis prompt
    work unchanged.

    Args:
        duckdb_path: Filesystem path to the DuckDB file, or ``\":memory:\"``
            for tests.

    Returns:
        Evidence dictionary with keys: ``question``, ``sql``, ``columns``,
        ``rows``, ``row_count``, ``truncated``, ``execution_time_ms``,
        ``db_path``, ``error``, ``guard_rejection``, ``raw_payload``.
    """
    from askpanda_atlas.cric_query_schema import MAX_ROWS_FULL_LIST  # deferred

    question = "List all PanDA queues in CRIC"

    if duckdb_path != ":memory:" and not os.path.exists(duckdb_path):
        logger.warning("list_all_queues: database file not found: %s", duckdb_path)
        return _db_unavailable_evidence(question, duckdb_path)

    try:
        exec_result = _execute_query(
            duckdb_path, _LIST_ALL_SQL, timeout_secs=10, max_rows=MAX_ROWS_FULL_LIST
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("list_all_queues: execution error: %s", exc)
        return _execution_error_evidence(
            question=question,
            sql=_LIST_ALL_SQL,
            duckdb_path=duckdb_path,
            detail=str(exc),
        )

    logger.debug(
        "list_all_queues: returned %d rows (truncated=%s) in %.1f ms",
        exec_result["row_count"],
        exec_result["truncated"],
        exec_result["execution_time_ms"],
    )

    return {
        "question": question,
        "sql": _LIST_ALL_SQL,
        "columns": exec_result["columns"],
        "rows": exec_result["rows"],
        "row_count": exec_result["row_count"],
        "truncated": exec_result["truncated"],
        "execution_time_ms": exec_result["execution_time_ms"],
        "db_path": duckdb_path,
        "db_last_modified": _db_mtime(duckdb_path),
        "error": None,
        "guard_rejection": None,
        "raw_payload": None,
    }


# ---------------------------------------------------------------------------
# Async LLM call
# ---------------------------------------------------------------------------


async def _call_llm_for_sql(
    question: str,
    schema_context: str,
) -> str:
    """Call the configured Bamboo LLM and return its raw reply.

    Uses the ``"synthesize"`` task profile (default model) at temperature 0.0
    with a tight token cap to minimise hallucination.

    Args:
        question: Natural-language question from the user.
        schema_context: Pre-built schema context string for the prompt.

    Returns:
        Raw reply string from the LLM (may contain SQL, fences, or a refusal).

    Raises:
        RuntimeError: If the LLM manager or selector is not initialised.
    """
    from bamboo.llm.runtime import get_llm_manager, get_llm_selector  # deferred
    from bamboo.llm.types import GenerateParams, Message  # deferred
    from askpanda_atlas.cric_query_schema import build_sql_prompt

    selector = get_llm_selector()
    manager = get_llm_manager()

    registry = getattr(selector, "registry", None)
    if registry is None:
        raise RuntimeError("LLM selector does not expose a registry.")

    default_profile = getattr(selector, "default_profile", "default")
    model_spec = registry.get(default_profile)
    client = await manager.get_client(model_spec)

    messages_raw = build_sql_prompt(question, schema_context)
    messages: list[Message] = [
        {"role": m["role"], "content": m["content"]}
        for m in messages_raw
    ]

    resp = await client.generate(
        messages=messages,
        params=GenerateParams(temperature=0.0, max_tokens=512),
    )
    return resp.text


# ---------------------------------------------------------------------------
# Main pipeline (async)
# ---------------------------------------------------------------------------


async def fetch_and_analyse(
    question: str,
    duckdb_path: str,
) -> dict[str, Any]:
    """Translate *question* to SQL, guard it, execute it, and return evidence.

    This is the end-to-end pipeline:

    1. Build schema context (cached).
    2. Call the LLM (async) to generate SQL.
    3. Strip markdown fences.
    4. Detect "cannot answer" replies.
    5. Run :func:`~askpanda_atlas.cric_query_schema.validate_and_guard`.
    6. Execute the sanitised SQL synchronously on the event loop thread.
    7. Build and return the evidence dict.

    Every failure mode produces a structured evidence dict (never an
    exception) so the Bamboo executor always receives a usable payload.

    Args:
        question: Natural-language question from the user.
        duckdb_path: Filesystem path to the DuckDB file (or ``":memory:"``
            for tests).

    Returns:
        Evidence dictionary with keys: ``question``, ``sql``, ``columns``,
        ``rows``, ``row_count``, ``truncated``, ``execution_time_ms``,
        ``db_path``, ``error``, ``guard_rejection``, ``raw_payload``.
    """
    from askpanda_atlas.cric_query_schema import (
        MAX_ROWS,
        MAX_ROWS_AGGREGATION,
        QUERY_TIMEOUT_SECS,
        build_schema_context,
        validate_and_guard,
        _has_group_by as _sql_has_group_by,
    )

    schema_context = build_schema_context()

    # --- Stage 2-4: async LLM SQL generation --------------------------------
    try:
        raw_reply = await _call_llm_for_sql(question, schema_context)
    except Exception as exc:  # noqa: BLE001
        logger.exception("cric_query: LLM call failed")
        return _execution_error_evidence(
            question=question,
            sql=None,
            duckdb_path=duckdb_path,
            detail=f"LLM call failed: {exc}",
        )

    raw_sql = _strip_sql_fences(raw_reply)
    if not raw_sql or _looks_like_cannot_answer(raw_sql):
        logger.debug("cric_query: LLM declined to generate SQL: %r", raw_reply[:120])
        return _unable_to_answer_evidence(question, duckdb_path)

    logger.debug("cric_query: generated SQL: %s", raw_sql)

    # --- Stage 5: guard -----------------------------------------------------
    guard = validate_and_guard(raw_sql)
    if not guard.passed:
        logger.warning(
            "cric_query: guard rejected SQL (rule=%s): %s",
            guard.triggered_rule,
            raw_sql,
        )
        return _guard_rejected_evidence(
            question=question,
            raw_sql=raw_sql,
            reason=guard.rejection_reason or "Unknown guard violation.",
            rule=guard.triggered_rule or "unknown",
            duckdb_path=duckdb_path,
        )

    sanitised_sql: str = guard.sanitised_sql  # type: ignore[assignment]

    # --- Stage 6: execute synchronously ------------------------------------
    # DuckDB queries against the CRIC ingestion database are fast (2-15ms)
    # so running them directly on the event loop thread is safe and avoids
    # asyncio.to_thread thread-pool issues with DuckDB on macOS.

    # Choose the Python-side fetch cap to match the SQL LIMIT.
    # Aggregation queries (GROUP BY) use MAX_ROWS_AGGREGATION so the
    # fetchmany cap never truncates results that the guard already allowed.
    import sqlglot as _sqlglot  # deferred to avoid module-level cost
    import sqlglot.expressions as _exp  # deferred
    try:
        _stmts = _sqlglot.parse(sanitised_sql, dialect="duckdb")
        _first = _stmts[0] if _stmts else None
        _is_agg = (
            isinstance(_first, _exp.Select) and _sql_has_group_by(_first)
        )
    except Exception:  # noqa: BLE001
        _is_agg = False
    fetch_cap = MAX_ROWS_AGGREGATION if _is_agg else MAX_ROWS

    # Pre-flight: verify the file exists before attempting to open it.
    if duckdb_path != ":memory:" and not os.path.exists(duckdb_path):
        logger.warning("cric_query: database file not found: %s", duckdb_path)
        return _db_unavailable_evidence(question, duckdb_path)

    try:
        exec_result = _execute_query(
            duckdb_path, sanitised_sql, QUERY_TIMEOUT_SECS, fetch_cap
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("cric_query: execution error: %s", exc)
        exc_str = str(exc).lower()
        if "does not exist" in exc_str or "no such file" in exc_str or "cannot open" in exc_str:
            return _db_unavailable_evidence(question, duckdb_path)
        # Try to surface available table names to help diagnose schema mismatches.
        available_tables = _probe_table_names(duckdb_path)
        return _execution_error_evidence(
            question=question,
            sql=sanitised_sql,
            duckdb_path=duckdb_path,
            detail=str(exc),
            available_tables=available_tables,
        )

    logger.debug(
        "cric_query: query returned %d rows (truncated=%s) in %.1f ms",
        exec_result["row_count"],
        exec_result["truncated"],
        exec_result["execution_time_ms"],
    )

    return {
        "question": question,
        "sql": sanitised_sql,
        "columns": exec_result["columns"],
        "rows": exec_result["rows"],
        "row_count": exec_result["row_count"],
        "truncated": exec_result["truncated"],
        "execution_time_ms": exec_result["execution_time_ms"],
        "db_path": duckdb_path,
        "error": None,
        "guard_rejection": None,
        "raw_payload": None,
    }


# ---------------------------------------------------------------------------
# Structured error constructors
# ---------------------------------------------------------------------------


def _probe_table_names(duckdb_path: str) -> list[str]:
    """Return the list of table names in the DuckDB file, or an empty list on error.

    Used as a diagnostic aid when a query fails due to a missing or
    mis-named table.  The result is included in the error evidence so
    operators can see what tables the cric_agent actually created.

    Args:
        duckdb_path: Filesystem path to the DuckDB file.

    Returns:
        List of table name strings, or an empty list if the probe fails.
    """
    try:
        read_only = duckdb_path != ":memory:"
        conn = duckdb.connect(duckdb_path, read_only=read_only)
        try:
            result = conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name"
            )
            return [row[0] for row in result.fetchall()]
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return []


def _unable_to_answer_evidence(question: str, duckdb_path: str) -> dict[str, Any]:
    """Return a structured evidence dict for the 'LLM cannot answer' case.

    Args:
        question: The original user question.
        duckdb_path: Path to the DuckDB file.

    Returns:
        Evidence dict with a user-safe error message.
    """
    return {
        "question": question,
        "sql": None,
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
        "execution_time_ms": 0.0,
        "db_path": duckdb_path,
        "error": (
            "I wasn't able to translate that question into a CRIC database query. "
            "Try rephrasing — for example, ask about a specific site, queue status, "
            "or copytool."
        ),
        "guard_rejection": None,
        "raw_payload": None,
    }


def _db_unavailable_evidence(question: str, duckdb_path: str) -> dict[str, Any]:
    """Return a structured evidence dict when the CRIC DuckDB file is absent.

    Gives a more actionable message than the generic execution error: the
    database file is created by the ``cric_agent`` ingestion process, so the
    user or operator needs to run that agent first.

    Args:
        question: The original user question.
        duckdb_path: Path where the DuckDB file was expected.

    Returns:
        Evidence dict with a user-safe error message explaining the situation.
    """
    return {
        "question": question,
        "sql": None,
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
        "execution_time_ms": 0.0,
        "db_path": duckdb_path,
        "error": (
            f"The CRIC database has not been populated yet "
            f"(expected at: {duckdb_path}). "
            "Run the cric_agent ingestion script to fetch the latest "
            "CRIC queuedata and create the database file."
        ),
        "guard_rejection": None,
        "raw_payload": None,
    }


def _guard_rejected_evidence(
    question: str,
    raw_sql: str,
    reason: str,
    rule: str,
    duckdb_path: str,
) -> dict[str, Any]:
    """Return a structured evidence dict for a guard rejection.

    Args:
        question: The original user question.
        raw_sql: The SQL string that triggered the rejection.
        reason: Human-readable rejection reason.
        rule: Short rule identifier that triggered rejection.
        duckdb_path: Path to the DuckDB file.

    Returns:
        Evidence dict with a user-safe error message and guard details.
    """
    return {
        "question": question,
        "sql": raw_sql,
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
        "execution_time_ms": 0.0,
        "db_path": duckdb_path,
        "error": "That query isn't permitted by this tool. Only read-only lookups are supported.",
        "guard_rejection": f"[{rule}] {reason}",
        "raw_payload": None,
    }


def _execution_error_evidence(
    question: str,
    sql: str | None,
    duckdb_path: str,
    detail: str = "",
    available_tables: list[str] | None = None,
) -> dict[str, Any]:
    """Return a structured evidence dict for a query execution error.

    The raw database error is intentionally not included in the user-facing
    message to avoid leaking schema information.  However, when the
    ``available_tables`` probe succeeds, the table list is included so
    operators can see what the cric_agent actually created.

    Args:
        question: The original user question.
        sql: The SQL that was attempted (or None if the error preceded execution).
        duckdb_path: Path to the DuckDB file.
        detail: Optional internal detail logged but not shown to the user.
        available_tables: Optional list of table names found in the DB,
            used to surface schema-mismatch diagnostics.

    Returns:
        Evidence dict with a user-safe error message.
    """
    if detail:
        logger.debug("cric_query execution error detail: %s", detail)
    if available_tables is not None:
        logger.warning(
            "cric_query: tables found in DB: %s (expected: queues)",
            available_tables or "(none)",
        )
    error_msg = (
        "The CRIC query could not be executed. "
        "Try a more specific question or check that the CRIC database is available."
    )
    if available_tables is not None and "queues" not in available_tables:
        found = ", ".join(available_tables) if available_tables else "(none)"
        error_msg = (
            f"The CRIC database at {duckdb_path} does not contain a 'queues' table. "
            f"Tables found: {found}. "
            "Check that the cric_agent wrote to the correct database file and schema."
        )
    return {
        "question": question,
        "sql": sql,
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
        "execution_time_ms": 0.0,
        "db_path": duckdb_path,
        "error": error_msg,
        "guard_rejection": None,
        "raw_payload": None,
    }


# ---------------------------------------------------------------------------
# MCP tool definition
# ---------------------------------------------------------------------------


def get_definition() -> dict[str, Any]:
    """Return the MCP tool definition for ``cric_query``.

    Returns:
        Tool definition dict compatible with MCP discovery.
    """
    return {
        "name": "cric_query",
        "description": (
            "Answer natural-language questions about ATLAS computing queues "
            "by querying the local CRIC (Computing Resource Information "
            "Catalogue) DuckDB database. "
            "Use this tool when the user asks about queue status, copytools, "
            "site resources, or queue configuration — for example: "
            "'Which queues are using the rucio copytool?', "
            "'Is the BNL queue online?', "
            "'What is the status of all queues at CERN?', "
            "'Which sites have more than 1000 running jobs?'. "
            "The database reflects the latest CRIC snapshot fetched by the "
            "ingestion agent."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "Natural-language question about ATLAS computing queues, "
                        "e.g. 'Which queues at BNL are online?'"
                    ),
                },
                "site": {
                    "type": "string",
                    "description": (
                        "Optional: ATLAS site name to scope the query "
                        "(e.g. 'BNL', 'CERN').  If supplied, it is appended "
                        "to the question so the LLM uses it as a filter hint."
                    ),
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    }


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------


class CricQueryTool:
    """MCP tool that answers NL questions about CRIC queues via DuckDB SQL.

    Translates the user's natural-language question into a single SELECT
    statement, validates it through the AST guard, executes it against the
    read-only CRIC ingestion database, and returns a compact evidence dict
    structured for LLM synthesis.
    """

    def __init__(self) -> None:
        """Initialise with the cached tool definition."""
        self._def: dict[str, Any] = get_definition()

    def get_definition(self) -> dict[str, Any]:
        """Return the MCP tool definition.

        Returns:
            Tool definition dictionary.
        """
        return self._def

    async def call(self, arguments: dict[str, Any]) -> list[Any]:
        """Translate the question to SQL, execute it, and return evidence.

        The LLM call is awaited directly; the blocking DuckDB execution runs
        synchronously on the event loop thread (avoids DuckDB macOS issues).
        ``bamboo.tools.base`` is imported inside this method (deferred) so the
        rest of this module remains importable when bamboo core is not installed.

        Args:
            arguments: Dict with required ``"question"`` (str) and optional
                ``"site"`` (str).

        Returns:
            One-element MCP content list containing the JSON-serialised
            evidence dict, or an error payload if anything goes wrong.
        """
        from bamboo.tools.base import text_content  # deferred — see module docstring

        question: str = arguments.get("question", "").strip()
        if not question:
            return text_content(json.dumps({
                "evidence": {"error": "question argument is required."},
            }))

        if len(question) > 2000:
            return text_content(json.dumps({
                "evidence": {
                    "error": "Question is too long (max 2000 characters). Please be more concise.",
                },
            }))

        site: str | None = arguments.get("site")
        if site:
            question = f"{question} (focus on site: {site})"

        duckdb_path: str = os.environ.get("CRIC_DUCKDB_PATH", "cric.duckdb")

        # Fast path: unscoped "list all queues" requests bypass the LLM entirely.
        # The NL→SQL pipeline caps non-aggregation fetches at MAX_ROWS (50), which
        # would truncate a full-inventory response.  list_all_queues() uses a
        # fixed, pre-validated SQL with MAX_ROWS_FULL_LIST (2000) as the fetch cap.
        if not site and _is_list_all_queues(question):
            logger.debug("cric_query: list-all fast path triggered for %r", question)
            try:
                evidence = list_all_queues(duckdb_path)
                return text_content(json.dumps({"evidence": evidence}))
            except Exception as exc:  # noqa: BLE001
                logger.exception("list_all_queues fast path failed")
                return text_content(json.dumps({
                    "evidence": {"question": question, "error": repr(exc)},
                }))

        try:
            evidence = await fetch_and_analyse(question, duckdb_path)
            return text_content(json.dumps({"evidence": evidence}))
        except Exception as exc:  # noqa: BLE001
            logger.exception("cric_query tool call failed")
            return text_content(json.dumps({
                "evidence": {
                    "question": question,
                    "error": repr(exc),
                },
            }))


cric_query_tool = CricQueryTool()

__all__ = [
    "CricQueryTool",
    "fetch_and_analyse",
    "get_definition",
    "list_all_queues",
    "_is_list_all_queues",
    "cric_query_tool",
]
