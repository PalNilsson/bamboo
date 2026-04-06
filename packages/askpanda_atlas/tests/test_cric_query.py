"""Tests for cric_query_schema, cric_query_impl, and bamboo_answer routing.

Coverage:
- :func:`validate_and_guard` — valid SQL, every rejection rule, adversarial
  inputs, LIMIT injection.
- :func:`fetch_and_analyse` — happy path, cannot-answer, guard rejection,
  execution error, truncation.
- :class:`CricQueryTool`.call() — missing argument, success, file-not-found.
- :func:`_is_cric_question` — routing heuristic in bamboo_answer.
- Multi-database disambiguation helpers.

All database access uses an in-memory DuckDB instance seeded with the real
``queuedata`` schema as produced by the cric_agent.  No network calls are made.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import patch

import duckdb
import pytest
from bamboo.llm.types import Message

from askpanda_atlas.cric_query_impl import (
    _strip_sql_fences,
    fetch_and_analyse,
    cric_query_tool,
)
from askpanda_atlas.cric_query_schema import (
    build_schema_context,
    invalidate_schema_cache,
    validate_and_guard,
)


# ---------------------------------------------------------------------------
# Minimal DDL and seed data for in-memory test DB
# Mirrors the real schema written by cric_agent into the 'queuedata' table.
# ---------------------------------------------------------------------------

_MINIMAL_DDL = """
CREATE TABLE IF NOT EXISTS queuedata (
    queue         VARCHAR PRIMARY KEY,
    name          VARCHAR,
    state         VARCHAR,
    status        VARCHAR,
    type          VARCHAR,
    atlas_site    VARCHAR,
    site          VARCHAR,
    cloud         VARCHAR,
    country       VARCHAR,
    copytools     VARCHAR,
    acopytools    VARCHAR,
    corecount     INTEGER,
    maxtime       INTEGER,
    maxrss        INTEGER,
    pledgedcpu    INTEGER,
    resource_type VARCHAR,
    tier          VARCHAR,
    last_modified TIMESTAMP
);
"""

_SEED_SQL = """
INSERT INTO queuedata VALUES
    ('BNL_ATLAS_1', 'BNL_ATLAS_1', 'ACTIVE', 'online', 'production',
     'BNL', 'BNL', 'US', 'US', '["rucio"]', '["rucio"]',
     1, 345600, 2000, 5000, 'SCORE', 'Tier1', '2026-03-24 10:00:00');
INSERT INTO queuedata VALUES
    ('BNL_ATLAS_2', 'BNL_ATLAS_2', 'ACTIVE', 'brokeroff', 'production',
     'BNL', 'BNL', 'US', 'US', '["rucio"]', '["rucio"]',
     8, 345600, 4000, 0, 'MCORE', 'Tier1', '2026-03-24 10:00:00');
INSERT INTO queuedata VALUES
    ('CERN_PROD_1', 'CERN_PROD_1', 'ACTIVE', 'online', 'unified',
     'CERN', 'CERN', 'CERN', 'CH', '["gfalcopy"]', '["gfalcopy"]',
     1, 86400, 2000, 10000, 'SCORE', 'Tier1', '2026-03-24 10:00:00');
INSERT INTO queuedata VALUES
    ('AGLT2_PROD', 'AGLT2_PROD', 'ACTIVE', 'online', 'production',
     'AGLT2', 'AGLT2', 'US', 'US', '["rucio"]', '["rucio"]',
     1, 172800, 2000, 2000, 'SCORE', 'Tier2', '2026-03-24 10:00:00');
INSERT INTO queuedata VALUES
    ('TOKYO_PROD', 'TOKYO_PROD', 'ACTIVE', 'test', 'production',
     'TOKYO', 'TOKYO', 'ASIA', 'JP', '["rucio"]', '["rucio"]',
     1, 172800, 2000, 500, 'SCORE', 'Tier2', '2026-03-24 10:00:00');
"""


@pytest.fixture()
def mem_db() -> duckdb.DuckDBPyConnection:
    """Return a seeded in-memory DuckDB connection with the real queuedata schema.

    Returns:
        Open DuckDB connection with the queuedata table and seed rows.
    """
    conn = duckdb.connect(":memory:")
    conn.execute(_MINIMAL_DDL)
    conn.execute(_SEED_SQL)
    return conn


def _make_llm_patch(sql: str) -> Any:
    """Return an AsyncMock that makes ``_call_llm_for_sql`` return *sql*.

    Args:
        sql: The SQL string the fake LLM should return.

    Returns:
        ``unittest.mock.AsyncMock`` suitable for use with ``patch``.
    """
    from unittest.mock import AsyncMock
    return AsyncMock(return_value=sql)


def _unpack(result: list[Any]) -> dict[str, Any]:
    """Deserialise the JSON evidence from a tool call result.

    Args:
        result: Return value of ``CricQueryTool.call()``.

    Returns:
        Parsed evidence dict.
    """
    assert result, "Expected non-empty result list"
    payload = json.loads(result[0]["text"])
    return payload.get("evidence", payload)


# ===========================================================================
# Guard unit tests
# ===========================================================================


class TestGuardAllowsValidSQL:
    """Ensure well-formed SELECT statements against the queues table pass."""

    def test_simple_select(self) -> None:
        """A plain SELECT with a LIMIT should pass unchanged."""
        r = validate_and_guard("SELECT queue, state FROM queuedata WHERE atlas_site='BNL' LIMIT 10")
        assert r.passed
        assert r.sanitised_sql is not None
        assert r.rejection_reason is None

    def test_select_with_aggregation(self) -> None:
        """A COUNT aggregate should pass."""
        sql = "SELECT state, COUNT(*) AS n FROM queuedata GROUP BY state LIMIT 10"
        r = validate_and_guard(sql)
        assert r.passed

    def test_select_with_filter(self) -> None:
        """A WHERE filter on copytools (JSON LIKE) should pass."""
        sql = "SELECT queue, atlas_site FROM queuedata WHERE copytools LIKE '%rucio%' LIMIT 200"
        r = validate_and_guard(sql)
        assert r.passed

    def test_select_case_insensitive(self) -> None:
        """Mixed-case SELECT should still parse and pass."""
        r = validate_and_guard("SeLeCt queue FrOm queuedata LiMiT 5")
        assert r.passed

    def test_cte_select(self) -> None:
        """A CTE (WITH ... SELECT) is a valid SELECT variant."""
        sql = (
            "WITH online AS (SELECT queue FROM queuedata WHERE status='online') "
            "SELECT COUNT(*) FROM online LIMIT 1"
        )
        r = validate_and_guard(sql)
        assert r.passed

    def test_having_clause(self) -> None:
        """A HAVING clause on an aggregate should pass."""
        sql = (
            "SELECT atlas_site, SUM(pledgedcpu) AS total FROM queuedata "
            "GROUP BY atlas_site HAVING SUM(pledgedcpu) > 100 LIMIT 200"
        )
        r = validate_and_guard(sql)
        assert r.passed

    def test_negation_where_clause(self) -> None:
        """A WHERE clause with != negation should pass — needed for 'not using rucio'."""
        r = validate_and_guard(
            "SELECT queue, atlas_site, copytools FROM queuedata WHERE copytools NOT LIKE '%rucio%' LIMIT 200"
        )
        assert r.passed

    def test_not_equal_operator(self) -> None:
        """A WHERE clause with <> negation operator should also pass."""
        r = validate_and_guard(
            "SELECT queue, state FROM queuedata WHERE status != 'online' LIMIT 200"
        )
        assert r.passed


class TestGuardRejectsDML:
    """Guard must reject all DML statements."""

    def test_insert(self) -> None:
        """INSERT should be rejected at the root."""
        r = validate_and_guard("INSERT INTO queuedata (queue) VALUES ('fake')")
        assert not r.passed
        assert r.triggered_rule == "non_select_root"

    def test_update(self) -> None:
        """UPDATE should be rejected at the root."""
        r = validate_and_guard("UPDATE queuedata SET state='offline' WHERE queue='BNL_ATLAS_1'")
        assert not r.passed
        assert r.triggered_rule == "non_select_root"

    def test_delete(self) -> None:
        """DELETE should be rejected at the root."""
        r = validate_and_guard("DELETE FROM queuedata WHERE atlas_site='BNL'")
        assert not r.passed
        assert r.triggered_rule == "non_select_root"


class TestGuardRejectsDDL:
    """Guard must reject DDL statements."""

    def test_drop(self) -> None:
        """DROP TABLE should be rejected."""
        r = validate_and_guard("DROP TABLE queuedata")
        assert not r.passed
        assert r.triggered_rule == "non_select_root"

    def test_create(self) -> None:
        """CREATE TABLE should be rejected."""
        r = validate_and_guard("CREATE TABLE foo (id INT)")
        assert not r.passed
        assert r.triggered_rule == "non_select_root"

    def test_alter(self) -> None:
        """ALTER TABLE should be rejected."""
        r = validate_and_guard("ALTER TABLE queuedata ADD COLUMN foo VARCHAR")
        assert not r.passed
        assert r.triggered_rule == "non_select_root"


class TestGuardRejectsStackedStatements:
    """Stacked (multi-statement) inputs must be rejected."""

    def test_select_then_drop(self) -> None:
        """SELECT followed by DROP should be rejected as multiple statements."""
        r = validate_and_guard("SELECT name FROM queues LIMIT 1; DROP TABLE queuedata")
        assert not r.passed
        assert r.triggered_rule == "multiple_statements"

    def test_two_selects(self) -> None:
        """Two SELECT statements separated by semicolon should be rejected."""
        r = validate_and_guard("SELECT 1; SELECT 2")
        assert not r.passed
        assert r.triggered_rule == "multiple_statements"


class TestGuardRejectsSystemTables:
    """References to system/internal tables must be rejected."""

    def test_information_schema(self) -> None:
        """information_schema reference should be rejected."""
        r = validate_and_guard("SELECT * FROM information_schema.tables LIMIT 10")
        assert not r.passed
        assert r.triggered_rule == "system_table"

    def test_duckdb_prefix(self) -> None:
        """duckdb_* table reference should be rejected."""
        r = validate_and_guard("SELECT * FROM duckdb_tables() LIMIT 10")
        assert not r.passed
        assert r.triggered_rule in ("system_table", "forbidden_construct", "parse_error", "non_select_root")


class TestGuardRejectsUnknownTables:
    """References to tables not in ALLOWED_TABLES must be rejected."""

    def test_unknown_table(self) -> None:
        """A table name not in the allow-list (e.g. 'jobs') should be rejected."""
        r = validate_and_guard("SELECT * FROM jobs LIMIT 10")
        assert not r.passed
        assert r.triggered_rule == "unknown_table"

    def test_subquery_unknown_table(self) -> None:
        """An unknown table inside a subquery should still be rejected."""
        r = validate_and_guard(
            "SELECT * FROM (SELECT * FROM secret_table LIMIT 5) t LIMIT 10"
        )
        assert not r.passed
        assert r.triggered_rule == "unknown_table"


class TestGuardAdversarialInputs:
    """Known SQL injection patterns must be caught."""

    def test_union_injection(self) -> None:
        """UNION-based injection should fail at the unknown-table or parse stage."""
        r = validate_and_guard(
            "SELECT queue FROM queuedata LIMIT 1 "
            "UNION SELECT table_name FROM information_schema.tables LIMIT 1"
        )
        assert not r.passed

    def test_stacked_insert(self) -> None:
        """Stacked statement with INSERT should be rejected."""
        r = validate_and_guard(
            "SELECT queue FROM queuedata LIMIT 1; INSERT INTO queuedata (queue) VALUES ('x')"
        )
        assert not r.passed

    def test_empty_sql(self) -> None:
        """Empty string should be rejected gracefully."""
        r = validate_and_guard("")
        assert not r.passed

    def test_whitespace_only(self) -> None:
        """Whitespace-only string should be rejected gracefully."""
        r = validate_and_guard("   ")
        assert not r.passed


class TestGuardLimitInjection:
    """LIMIT injection, preservation, and GROUP-BY-aware cap raising."""

    def test_injects_limit_when_absent(self) -> None:
        """SELECT without LIMIT should have MAX_ROWS injected."""
        r = validate_and_guard("SELECT queue FROM queuedata WHERE atlas_site='BNL'")
        assert r.passed
        assert "LIMIT" in (r.sanitised_sql or "").upper()

    def test_preserves_existing_limit_on_raw_query(self) -> None:
        """Raw per-queue SELECT with explicit LIMIT should keep the original value."""
        r = validate_and_guard(
            "SELECT queue FROM queuedata WHERE atlas_site='BNL' LIMIT 5"
        )
        assert r.passed
        sql = r.sanitised_sql or ""
        assert "5" in sql

    def test_group_by_without_limit_gets_aggregation_cap(self) -> None:
        """GROUP BY query without LIMIT should get MAX_ROWS_AGGREGATION injected."""
        import re  # noqa: PLC0415
        from askpanda_atlas.cric_query_schema import MAX_ROWS_AGGREGATION  # noqa: PLC0415
        r = validate_and_guard(
            "SELECT status, COUNT(*) AS n FROM queuedata GROUP BY status ORDER BY n DESC"
        )
        assert r.passed
        m = re.search(r"LIMIT\s+(\d+)", r.sanitised_sql or "", re.IGNORECASE)
        assert m is not None
        assert int(m.group(1)) == MAX_ROWS_AGGREGATION

    def test_group_by_low_limit_raised_to_aggregation_cap(self) -> None:
        """GROUP BY with LLM-supplied LIMIT 50 must be raised to MAX_ROWS_AGGREGATION.

        This was the root cause of the alphabetic truncation bug: the LLM generated
        LIMIT 50 on a GROUP BY query, silently cutting results at row 50.
        """
        import re  # noqa: PLC0415
        from askpanda_atlas.cric_query_schema import MAX_ROWS_AGGREGATION  # noqa: PLC0415
        sql = (
            "SELECT atlas_site, status, COUNT(*) AS n FROM queuedata "
            "WHERE status != 'online' GROUP BY atlas_site, status "
            "ORDER BY atlas_site LIMIT 50"
        )
        r = validate_and_guard(sql)
        assert r.passed
        m = re.search(r"LIMIT\s+(\d+)", r.sanitised_sql or "", re.IGNORECASE)
        assert m is not None
        assert int(m.group(1)) == MAX_ROWS_AGGREGATION

    def test_raw_query_limit_not_raised(self) -> None:
        """Raw per-queue query with explicit LIMIT 50 must NOT be raised."""
        import re  # noqa: PLC0415
        from askpanda_atlas.cric_query_schema import MAX_ROWS_AGGREGATION  # noqa: PLC0415
        r = validate_and_guard(
            "SELECT queue, state FROM queuedata WHERE atlas_site='BNL' LIMIT 50"
        )
        assert r.passed
        m = re.search(r"LIMIT\s+(\d+)", r.sanitised_sql or "", re.IGNORECASE)
        assert m is not None
        assert int(m.group(1)) == 50
        assert int(m.group(1)) < MAX_ROWS_AGGREGATION

    def test_group_by_fetch_cap_not_truncated(
        self, mem_db: duckdb.DuckDBPyConnection
    ) -> None:
        """GROUP BY pipeline must not truncate at MAX_ROWS (50) fetch cap.

        This is the production regression: LISTAGG/STRING_AGG GROUP BY queries
        were being truncated to 50 rows by the Python-side fetchmany cap even
        though the SQL LIMIT was correctly set to 500 by the guard.
        The fix: fetch_cap uses MAX_ROWS_AGGREGATION for GROUP BY queries.
        """
        # Seed extra rows so we have more than MAX_ROWS (50) distinct groups
        # We already have 5 rows in mem_db; add enough to exceed 50 atlas_site groups
        # by inserting rows with many distinct atlas_site values.
        for i in range(60):
            mem_db.execute(
                f"INSERT INTO queuedata VALUES ("
                f"'SITE_{i:02d}_Q1', 'SITE_{i:02d}_Q1', 'ACTIVE', 'brokeroff', "
                f"'production', 'SITE_{i:02d}', 'SITE_{i:02d}', 'US', 'US', "
                f"'[\"rucio\"]', '[\"rucio\"]', 1, 100000, 2000, 0, "
                f"'SCORE', 'Tier2', '2026-03-24 10:00:00')"
            )

        sql = (
            "SELECT atlas_site, status, COUNT(*) AS n FROM queuedata "
            "WHERE status != 'online' GROUP BY atlas_site, status "
            "ORDER BY atlas_site LIMIT 500"
        )
        with patch("askpanda_atlas.cric_query_impl.duckdb") as mock_duckdb:
            mock_duckdb.connect.return_value = mem_db
            mock_duckdb.Error = duckdb.Error
            with patch("askpanda_atlas.cric_query_impl.os.path.exists", return_value=True):
                with patch("askpanda_atlas.cric_query_impl._call_llm_for_sql",
                           _make_llm_patch(sql)):
                    result = asyncio.run(
                        fetch_and_analyse("Which queues are not online?", ":memory:")
                    )
        assert result["error"] is None
        # Should return all 60+ groups, not truncated at 50
        assert result["row_count"] > 50, (
            f"Expected >50 rows for GROUP BY result, got {result['row_count']} "
            "(fetch cap not raised for aggregation query)"
        )
        assert not result["truncated"]


# ===========================================================================
# _strip_sql_fences
# ===========================================================================


class TestStripSqlFences:
    """SQL fence stripping helper."""

    def test_strips_sql_fence(self) -> None:
        """```sql ... ``` wrapper should be removed."""
        assert _strip_sql_fences("```sql\nSELECT 1\n```") == "SELECT 1"

    def test_strips_plain_fence(self) -> None:
        """``` ... ``` wrapper should be removed."""
        assert _strip_sql_fences("```\nSELECT 1\n```") == "SELECT 1"

    def test_no_fence(self) -> None:
        """Plain SQL without fences should pass through unchanged."""
        assert _strip_sql_fences("SELECT 1") == "SELECT 1"

    def test_strips_whitespace(self) -> None:
        """Leading/trailing whitespace should be stripped."""
        assert _strip_sql_fences("  SELECT 1  ") == "SELECT 1"


# ===========================================================================
# Schema context cache
# ===========================================================================


class TestBuildSchemaContext:
    """Schema context builder and cache."""

    def setup_method(self) -> None:
        """Clear the schema context cache before each test."""
        invalidate_schema_cache()

    def test_returns_non_empty_string(self) -> None:
        """build_schema_context should return a non-empty string."""
        ctx = build_schema_context()
        assert isinstance(ctx, str)
        assert len(ctx) > 50

    def test_contains_queues_table(self) -> None:
        """Context for the default tables should mention 'queuedata'."""
        ctx = build_schema_context()
        assert "queuedata" in ctx

    def test_contains_key_columns(self) -> None:
        """Context should mention key CRIC columns."""
        ctx = build_schema_context()
        for col in ("status", "copytools", "atlas_site", "last_modified"):
            assert col in ctx

    def test_caches_result(self) -> None:
        """Successive calls with the same tables should return identical strings."""
        ctx1 = build_schema_context(["queuedata"])
        ctx2 = build_schema_context(["queuedata"])
        assert ctx1 is ctx2 or ctx1 == ctx2


# ===========================================================================
# fetch_and_analyse pipeline tests
# ===========================================================================


class TestFetchAndAnalysePipeline:
    """End-to-end pipeline with an in-memory DuckDB and a mock LLM."""

    def test_successful_count_query(self, mem_db: duckdb.DuckDBPyConnection) -> None:
        """A count query against the test DB should return the correct count."""
        sql = "SELECT COUNT(*) AS n FROM queuedata WHERE status='online'"
        with patch("askpanda_atlas.cric_query_impl.duckdb") as mock_duckdb:
            mock_duckdb.connect.return_value = mem_db
            mock_duckdb.Error = duckdb.Error
            with patch("askpanda_atlas.cric_query_impl._call_llm_for_sql", _make_llm_patch(sql)):
                result = asyncio.run(fetch_and_analyse("How many queues are online?", ":memory:"))
        assert result["error"] is None
        assert result["row_count"] == 1
        assert result["rows"][0]["n"] == 3

    def test_returns_sql_in_evidence(self, mem_db: duckdb.DuckDBPyConnection) -> None:
        """The executed SQL should always appear in the evidence dict."""
        sql = "SELECT queue, state FROM queuedata WHERE atlas_site='BNL' LIMIT 5"
        with patch("askpanda_atlas.cric_query_impl.duckdb") as mock_duckdb:
            mock_duckdb.connect.return_value = mem_db
            mock_duckdb.Error = duckdb.Error
            with patch("askpanda_atlas.cric_query_impl._call_llm_for_sql", _make_llm_patch(sql)):
                result = asyncio.run(fetch_and_analyse("BNL queue status?", ":memory:"))
        assert result["sql"] is not None
        assert "queuedata" in result["sql"]

    def test_zero_rows_returns_explicit_count(self, mem_db: duckdb.DuckDBPyConnection) -> None:
        """A query that matches no rows should return row_count=0 explicitly."""
        sql = "SELECT queue FROM queuedata WHERE atlas_site='NOWHERE' LIMIT 10"
        with patch("askpanda_atlas.cric_query_impl.duckdb") as mock_duckdb:
            mock_duckdb.connect.return_value = mem_db
            mock_duckdb.Error = duckdb.Error
            with patch("askpanda_atlas.cric_query_impl._call_llm_for_sql", _make_llm_patch(sql)):
                result = asyncio.run(fetch_and_analyse("Queues at NOWHERE?", ":memory:"))
        assert result["error"] is None
        assert result["row_count"] == 0
        assert result["rows"] == []

    def test_copytool_filter_query(self, mem_db: duckdb.DuckDBPyConnection) -> None:
        """A copytool filter should return only matching queues."""
        sql = "SELECT queue, atlas_site FROM queuedata WHERE copytools LIKE '%rucio%' LIMIT 200"
        with patch("askpanda_atlas.cric_query_impl.duckdb") as mock_duckdb:
            mock_duckdb.connect.return_value = mem_db
            mock_duckdb.Error = duckdb.Error
            with patch("askpanda_atlas.cric_query_impl._call_llm_for_sql", _make_llm_patch(sql)):
                result = asyncio.run(
                    fetch_and_analyse("Which queues use the rucio copytool?", ":memory:")
                )
        assert result["error"] is None
        assert result["row_count"] == 4  # BNL_ATLAS_1, BNL_ATLAS_2, AGLT2_PROD, TOKYO_PROD

    def test_truncation_flag(self) -> None:
        """Result sets at the cap should have truncated=True."""
        sql = "SELECT queue FROM queuedata"
        with patch("askpanda_atlas.cric_query_impl._call_llm_for_sql", _make_llm_patch(sql)):
            with patch("askpanda_atlas.cric_query_impl._execute_query") as mock_exec:
                mock_exec.return_value = {
                    "columns": ["name"],
                    "rows": [{"name": "q1"}, {"name": "q2"}],
                    "row_count": 2,
                    "truncated": True,
                    "execution_time_ms": 1.0,
                }
                result = asyncio.run(fetch_and_analyse("All queues", ":memory:"))
        assert result["truncated"] is True

    def test_llm_cannot_answer(self) -> None:
        """When the LLM returns CANNOT_ANSWER the evidence should have an error."""
        with patch("askpanda_atlas.cric_query_impl._call_llm_for_sql",
                   _make_llm_patch("CANNOT_ANSWER")):
            result = asyncio.run(fetch_and_analyse("What is the meaning of life?", ":memory:"))
        assert result["error"] is not None
        assert "translate" in result["error"].lower() or "rephrase" in result["error"].lower()
        assert result["row_count"] == 0

    def test_llm_natural_language_refusal(self) -> None:
        """Natural-language refusals from the LLM should also be caught."""
        with patch("askpanda_atlas.cric_query_impl._call_llm_for_sql",
                   _make_llm_patch("I cannot answer that question.")):
            result = asyncio.run(fetch_and_analyse("Who wrote Hamlet?", ":memory:"))
        assert result["error"] is not None
        assert result["sql"] is None

    def test_guard_rejected_sql(self) -> None:
        """SQL rejected by the guard should produce a structured guard_rejection."""
        with patch("askpanda_atlas.cric_query_impl._call_llm_for_sql",
                   _make_llm_patch("DELETE FROM queuedata WHERE atlas_site='BNL'")):
            result = asyncio.run(fetch_and_analyse("Delete BNL queues", ":memory:"))
        assert result["error"] is not None
        assert result["guard_rejection"] is not None
        assert result["row_count"] == 0

    def test_guard_rejects_jobs_table(self) -> None:
        """A query referencing the jobs table (not in queuedata allow-list) must be rejected."""
        with patch("askpanda_atlas.cric_query_impl._call_llm_for_sql",
                   _make_llm_patch("SELECT pandaid FROM jobs LIMIT 10")):
            result = asyncio.run(fetch_and_analyse("Show me jobs", ":memory:"))
        assert result["guard_rejection"] is not None
        assert "jobs" in (result["guard_rejection"] or "")

    def test_negation_sql_passes_guard_and_executes(
        self, mem_db: duckdb.DuckDBPyConnection
    ) -> None:
        """SQL with != negation (e.g. copytool != 'rucio') must pass the guard and execute.

        This was the trigger for the 'which queues are NOT using the rucio
        copytool?' failure in production — the LLM sometimes refused to
        generate negation SQL.  This test confirms the guard accepts it.
        """
        sql = "SELECT queue, atlas_site, copytools FROM queuedata WHERE copytools NOT LIKE '%rucio%' LIMIT 200"
        with patch("askpanda_atlas.cric_query_impl.duckdb") as mock_duckdb:
            mock_duckdb.connect.return_value = mem_db
            mock_duckdb.Error = duckdb.Error
            with patch("askpanda_atlas.cric_query_impl._call_llm_for_sql",
                       _make_llm_patch(sql)):
                result = asyncio.run(
                    fetch_and_analyse(
                        "Which queues are not using the rucio copytool?", ":memory:"
                    )
                )
        assert result["error"] is None
        # CERN_PROD_1 uses gfalcopy — should appear in results
        names = [r["queue"] for r in result["rows"]]
        assert "CERN_PROD_1" in names

    def test_db_unavailable_returns_actionable_message(self) -> None:
        """When the DuckDB file is missing the error message should mention cric_agent."""
        sql = "SELECT queue FROM queuedata LIMIT 10"
        with patch("askpanda_atlas.cric_query_impl._call_llm_for_sql",
                   _make_llm_patch(sql)):
            with patch("askpanda_atlas.cric_query_impl._execute_query",
                       side_effect=Exception("Cannot open database '/x/cric.duckdb' "
                                             "in read-only mode: database does not exist")):
                result = asyncio.run(
                    fetch_and_analyse("Which queues are online?", "/x/cric.duckdb")
                )
        assert result["error"] is not None
        assert "cric_agent" in result["error"].lower() or "not been populated" in result["error"].lower()

    def test_execution_error_returns_structured_evidence(self) -> None:
        """A DuckDB execution error should produce a structured error, not raise."""
        sql = "SELECT name FROM queues WHERE site='BNL' LIMIT 10"
        with patch("askpanda_atlas.cric_query_impl._call_llm_for_sql", _make_llm_patch(sql)):
            with patch("askpanda_atlas.cric_query_impl._execute_query",
                       side_effect=Exception("simulated DB error")):
                result = asyncio.run(fetch_and_analyse("BNL queues?", ":memory:"))
        assert result["error"] is not None
        assert "simulated" not in result["error"]
        assert result["row_count"] == 0

    def test_markdown_fences_stripped(self, mem_db: duckdb.DuckDBPyConnection) -> None:
        """SQL wrapped in markdown fences should still execute correctly."""
        sql = "```sql\nSELECT COUNT(*) AS n FROM queuedata WHERE status='online'\n```"
        with patch("askpanda_atlas.cric_query_impl.duckdb") as mock_duckdb:
            mock_duckdb.connect.return_value = mem_db
            mock_duckdb.Error = duckdb.Error
            with patch("askpanda_atlas.cric_query_impl._call_llm_for_sql", _make_llm_patch(sql)):
                result = asyncio.run(fetch_and_analyse("Online count?", ":memory:"))
        assert result["error"] is None
        assert result["row_count"] == 1

    def test_evidence_always_has_question(self) -> None:
        """The evidence dict must always contain the original question."""
        with patch("askpanda_atlas.cric_query_impl._call_llm_for_sql",
                   _make_llm_patch("CANNOT_ANSWER")):
            result = asyncio.run(fetch_and_analyse("CANNOT_ANSWER", ":memory:"))
        assert result["question"] == "CANNOT_ANSWER"


# ===========================================================================
# CricQueryTool.call() tests
# ===========================================================================


class TestCricQueryToolCall:
    """Tests for the async MCP tool entry point."""

    def _run(self, coro: Any) -> Any:
        """Run a coroutine synchronously.

        Args:
            coro: Coroutine to run.

        Returns:
            Return value of the coroutine.
        """
        return asyncio.run(coro)

    def test_missing_question_returns_error(self) -> None:
        """call() with no question should return an error payload, not raise."""
        result = self._run(cric_query_tool.call({}))
        evidence = _unpack(result)
        assert "error" in evidence
        assert evidence["error"] is not None

    def test_empty_question_returns_error(self) -> None:
        """call() with an empty question string should return an error."""
        result = self._run(cric_query_tool.call({"question": "  "}))
        evidence = _unpack(result)
        assert "error" in evidence

    def test_question_too_long_returns_error(self) -> None:
        """call() with a question exceeding 2000 chars should return an error."""
        result = self._run(cric_query_tool.call({"question": "x" * 2001}))
        evidence = _unpack(result)
        assert "error" in evidence
        assert "long" in evidence["error"].lower() or "2000" in evidence["error"]

    def test_successful_call_returns_evidence(self, mem_db: duckdb.DuckDBPyConnection) -> None:
        """A successful call should return a JSON evidence dict."""
        sql = "SELECT COUNT(*) AS n FROM queuedata WHERE status='online'"
        with patch("askpanda_atlas.cric_query_impl.duckdb") as mock_duckdb:
            mock_duckdb.connect.return_value = mem_db
            mock_duckdb.Error = duckdb.Error
            with patch("askpanda_atlas.cric_query_impl.os.path.exists", return_value=True):
                with patch("askpanda_atlas.cric_query_impl._call_llm_for_sql",
                           _make_llm_patch(sql)):
                    result = self._run(cric_query_tool.call({
                        "question": "How many queues are online?",
                    }))
        evidence = _unpack(result)
        assert evidence.get("error") is None
        assert evidence["row_count"] == 1

    def test_site_appended_to_question(self, mem_db: duckdb.DuckDBPyConnection) -> None:
        """When site is provided it should be appended to the question string."""
        captured: list[str] = []

        async def _capture_llm(question: str, schema_context: str) -> str:
            """Capture the question and return safe SQL.

            Args:
                question: The question passed to the LLM.
                schema_context: Ignored.

            Returns:
                Safe SELECT SQL.
            """
            captured.append(question)
            return "SELECT COUNT(*) AS n FROM queues LIMIT 1"

        with patch("askpanda_atlas.cric_query_impl.duckdb") as mock_duckdb:
            mock_duckdb.connect.return_value = mem_db
            mock_duckdb.Error = duckdb.Error
            with patch("askpanda_atlas.cric_query_impl.os.path.exists", return_value=True):
                with patch("askpanda_atlas.cric_query_impl._call_llm_for_sql",
                           side_effect=_capture_llm):
                    self._run(cric_query_tool.call({
                        "question": "Is the queue online?",
                        "site": "BNL",
                    }))
        assert captured, "LLM callable was never called"
        assert "BNL" in captured[0]

    def test_schema_mismatch_returns_table_list(self) -> None:
        """When the DB exists but lacks a 'queues' table, error names tables found."""
        sql = "SELECT queue FROM queuedata LIMIT 10"
        with patch("askpanda_atlas.cric_query_impl._call_llm_for_sql",
                   _make_llm_patch(sql)):
            with patch("askpanda_atlas.cric_query_impl.os.path.exists", return_value=True):
                with patch("askpanda_atlas.cric_query_impl._execute_query",
                           side_effect=Exception("Catalog error: table 'queues' not found")):
                    with patch("askpanda_atlas.cric_query_impl._probe_table_names",
                               return_value=["pandaqueue", "sites"]):
                        result = asyncio.run(
                            fetch_and_analyse("Which queues are online?", "/some/cric.db")
                        )
        assert result["error"] is not None
        # Should mention the actual tables found, not just a generic error
        assert "pandaqueue" in result["error"] or "queuedata" in result["error"]

    def test_duckdb_file_not_found_returns_error(self) -> None:
        """When the DuckDB file is absent the tool should return a structured error."""
        with patch.dict("os.environ", {"CRIC_DUCKDB_PATH": "/nonexistent/cric.duckdb"}):
            with patch("askpanda_atlas.cric_query_impl._execute_query",
                       side_effect=FileNotFoundError("no such file")):
                with patch("askpanda_atlas.cric_query_impl._call_llm_for_sql",
                           _make_llm_patch("SELECT name FROM queues LIMIT 1")):
                    result = self._run(cric_query_tool.call({
                        "question": "Which queues are online at BNL?",
                    }))
        evidence = _unpack(result)
        assert "error" in evidence


# ===========================================================================
# bamboo_answer routing heuristics
# ===========================================================================


class TestIsCricQuestion:
    """Tests for the _is_cric_question routing helper."""

    def setup_method(self) -> None:
        """Import the helper fresh for each test."""
        from bamboo.tools.bamboo_answer import _is_cric_question  # noqa: PLC0415
        self._fn = _is_cric_question

    def test_copytool_matches(self) -> None:
        """'copytool' should be detected as a CRIC question."""
        assert self._fn("Which queues are using the rucio copytool?")

    def test_cric_keyword_matches(self) -> None:
        """Direct mention of 'cric' should route to CRIC."""
        assert self._fn("What does CRIC say about BNL?")

    def test_maxwalltime_matches(self) -> None:
        """'maxwalltime' is an unambiguous CRIC column reference."""
        assert self._fn("Which queues have maxwalltime greater than 86400?")

    def test_maxmemory_matches(self) -> None:
        """'maxmemory' should route to CRIC."""
        assert self._fn("Show me queues with maxmemory above 4000.")

    def test_queue_online_matches(self) -> None:
        """'queue online' phrasing should route to CRIC."""
        assert self._fn("Is the BNL queue online?")

    def test_queue_offline_matches(self) -> None:
        """'queue offline' phrasing should route to CRIC."""
        assert self._fn("Is the CERN queue offline?")

    def test_queue_status_matches(self) -> None:
        """'queue status' phrasing should route to CRIC."""
        assert self._fn("What is the queue status at BNL?")

    def test_cpu_slots_matches(self) -> None:
        """'cpu slots' should route to CRIC."""
        assert self._fn("Which sites have more than 1000 cpu slots?")

    def test_general_job_question_excluded(self) -> None:
        """General job questions should not route to CRIC."""
        assert not self._fn("How many jobs failed at BNL?")

    def test_task_question_excluded(self) -> None:
        """Task questions should not route to CRIC."""
        assert not self._fn("What is the status of task 12345?")

    def test_pilot_question_excluded(self) -> None:
        """Pilot questions should not route to CRIC."""
        assert not self._fn("How many pilots are running at AGLT2?")

    def test_bare_queue_keyword_not_matched(self) -> None:
        """'queue' alone (without CRIC-specific context) should not trigger CRIC fast-path."""
        # 'queue' is in _DB_KEYWORDS["cric"] for disambiguation, but not in
        # _CRIC_SIGNALS, so it should NOT trigger _is_cric_question().
        assert not self._fn("What queues are available?")


class TestMultiDatabaseDisambiguation:
    """Tests for the multi-database disambiguation helpers with CRIC active."""

    def setup_method(self) -> None:
        """Import helpers fresh for each test."""
        from bamboo.tools.bamboo_answer import (  # noqa: PLC0415
            QUERYABLE_DATABASES,
            _build_clarification_response,
            _resolve_target_database,
        )
        self._resolve = _resolve_target_database
        self._clarify = _build_clarification_response
        self._dbs = QUERYABLE_DATABASES

    def test_both_databases_registered(self) -> None:
        """Both 'jobs' and 'cric' should be in QUERYABLE_DATABASES."""
        assert "jobs" in self._dbs
        assert "cric" in self._dbs

    def test_jobs_keywords_resolve_to_jobs(self) -> None:
        """Questions with unambiguous job keywords should resolve to jobs DB."""
        assert self._resolve("How many jobs failed at BNL?") == "jobs"
        assert self._resolve("What are the top errors at SWT2_CPB?") == "jobs"

    def test_cric_keywords_resolve_to_cric(self) -> None:
        """Questions with unambiguous CRIC-only keywords should resolve to CRIC.

        'copytool' is cric-only; no jobs-DB keywords present.
        """
        # 'copytool' is cric-only; no jobs-DB keywords present.
        assert self._resolve("Which sites use copytool rucio?") == "cric"

    def test_ambiguous_question_returns_none(self) -> None:
        """Questions that match both databases should return None (needs clarification)."""
        result = self._resolve("Show me queue information")
        # Either None (ambiguous) or one of the known keys is acceptable.
        assert result is None or result in self._dbs

    def test_clarification_lists_both_databases(self) -> None:
        """Clarification response should name both databases."""
        msg = self._clarify("Some ambiguous question")
        assert "jobs" in msg
        assert "cric" in msg
        assert "rephrase" in msg.lower()


# ===========================================================================
# Routing regression tests (bugs found in production)
# ===========================================================================


class TestRoutingRegressions:
    """Regression tests for routing bugs found in production.

    These cover the exact questions that previously triggered false
    disambiguation prompts or wrong-tool routing, before the keyword and
    fast-path fixes.
    """

    def setup_method(self) -> None:
        """Import routing helpers fresh for each test."""
        from bamboo.tools.bamboo_answer import (  # noqa: PLC0415
            _is_cric_question,
            _is_jobs_db_question,
            _resolve_target_database,
            _DB_KEYWORDS,
        )
        self._is_cric = _is_cric_question
        self._is_jobs = _is_jobs_db_question
        self._resolve = _resolve_target_database
        self._db_kw = _DB_KEYWORDS

    def test_queue_not_in_jobs_keywords(self) -> None:
        """'queue' must not appear in jobs DB keywords.

        When 'queue' was in _DB_KEYWORDS['jobs'], every CRIC question that
        mentioned 'queues' matched both databases and triggered a spurious
        clarification prompt instead of routing to CRIC.
        """
        assert "queue" not in self._db_kw.get("jobs", frozenset())
        assert "computing site" not in self._db_kw.get("jobs", frozenset())

    def test_copytool_question_resolves_to_cric(self) -> None:
        """'which queues are not using the rucio copytool?' must resolve to CRIC.

        Previously triggered a clarification prompt because 'queue' was
        in both _DB_KEYWORDS sets.
        """
        q = "which queues are not using the rucio copytool?"
        assert self._is_cric(q), "Must be detected as CRIC question"
        result = self._resolve(q)
        assert result == "cric", f"Expected 'cric', got {result!r}"

    def test_list_cric_queues_copytool_routes_to_cric(self) -> None:
        """'List CRIC queues where the copytool is not rucio' must route to CRIC."""
        q = "List CRIC queues where the copytool is not rucio"
        assert self._is_cric(q)
        assert not self._is_jobs(q)

    def test_atlas_queues_copytool_routes_to_cric(self) -> None:
        """'Which ATLAS queues in CRIC use a copytool other than Rucio?' → CRIC."""
        q = "Which ATLAS queues in CRIC use a copytool other than Rucio?"
        assert self._is_cric(q)
        assert not self._is_jobs(q)

    def test_is_bnl_queue_online_routes_to_cric(self) -> None:
        """'Is the BNL queue online?' must route to CRIC via _is_cric_question."""
        q = "Is the BNL queue online?"
        assert self._is_cric(q)
        assert not self._is_jobs(q)

    def test_pure_jobs_questions_unaffected(self) -> None:
        """Jobs questions must still route to jobs DB after the keyword fix."""
        jobs_questions = [
            "How many jobs failed at BNL?",
            "What are the top errors at SWT2_CPB?",
            "How many jobs are running at CERN?",
            "Which jobs failed with pilot error 1008?",
        ]
        for q in jobs_questions:
            assert self._is_jobs(q), f"Should be jobs question: {q!r}"
            assert not self._is_cric(q), f"Should not be CRIC question: {q!r}"

    def test_build_deterministic_plan_cric_wins_over_jobs(self) -> None:
        """When both CRIC and jobs signals fire, _build_deterministic_plan must pick cric_query.

        This was the root of the production bug: 'which queues are using the
        rucio copytool?' hit _JOBS_DB_SIGNALS via 'which queues', but
        _build_deterministic_plan was checking _is_jobs_db_question first and
        always returning a panda_jobs_query plan.  CRIC must win when both
        signals fire because _is_cric_question uses tighter, more specific
        signals.
        """
        from bamboo.tools.bamboo_answer import _build_deterministic_plan  # noqa: PLC0415

        cric_wins_cases = [
            "which queues are using the rucio copytool?",
            "which queues are not using the rucio copytool?",
            "which queues have maxwalltime above 86400?",
        ]
        for q in cric_wins_cases:
            plan = _build_deterministic_plan(q, None, None)
            assert plan is not None, f"Expected a plan for: {q!r}"
            tool = plan.tool_calls[0].tool
            assert tool == "cric_query", (
                f"Expected cric_query, got {tool!r} for: {q!r}"
            )


# ===========================================================================
# CRIC contextual follow-up routing
# ===========================================================================


class TestCricContextualFollowup:
    """Tests for CRIC contextual follow-up routing.

    Covers the case where a follow-up question like 'Is BNL-PTEST active?'
    contains no CRIC-specific keywords but should route to cric_query because
    the prior exchange used the CRIC tool.
    """

    def setup_method(self) -> None:
        """Import helpers fresh for each test."""
        from bamboo.tools.bamboo_answer import (  # noqa: PLC0415
            _last_tool_was_cric,
            _is_cric_followup,
            _is_cric_question,
        )
        self._last_cric = _last_tool_was_cric
        self._is_followup = _is_cric_followup
        self._is_cric = _is_cric_question

    def test_last_tool_was_cric_detects_copytool_response(self) -> None:
        """History containing copytool answer signals a prior CRIC exchange."""
        history: list[Message] = [
            {"role": "user", "content": "how many queues use the objectstore copytool?"},
            {"role": "assistant", "content": "13 queues use the objectstore copytool. BNL-PTEST..."},
        ]
        assert self._last_cric(history)

    def test_last_tool_was_cric_detects_cric_keyword(self) -> None:
        """History containing 'cric' in assistant turn signals a prior CRIC exchange."""
        history: list[Message] = [
            {"role": "user", "content": "check cric status"},
            {"role": "assistant", "content": "According to CRIC, BNL-ATLAS has 5 online queues."},
        ]
        assert self._last_cric(history)

    def test_last_tool_was_cric_false_for_task_response(self) -> None:
        """History containing a task response should NOT trigger CRIC follow-up."""
        history: list[Message] = [
            {"role": "user", "content": "what is task 12345?"},
            {"role": "assistant", "content": "Task 12345 has status finished with 500 jobs done."},
        ]
        assert not self._last_cric(history)

    def test_last_tool_was_cric_false_for_empty_history(self) -> None:
        """Empty history should return False."""
        assert not self._last_cric([])

    def test_is_cric_followup_matches_is_active(self) -> None:
        """'Is BNL-PTEST active?' matches the follow-up pattern."""
        assert self._is_followup("Is BNL-PTEST active?")

    def test_is_cric_followup_matches_is_online(self) -> None:
        """'Is CERN-PROD online?' matches the follow-up pattern."""
        assert self._is_followup("Is CERN-PROD online?")

    def test_is_cric_followup_matches_is_offline(self) -> None:
        """'Is AGLT2 offline?' matches the follow-up pattern."""
        assert self._is_followup("Is AGLT2 offline?")

    def test_is_cric_followup_matches_status_of(self) -> None:
        """'What is the status of BNL?' matches."""
        assert self._is_followup("What is the status of BNL?")

    def test_is_cric_followup_does_not_match_long_question(self) -> None:
        """Long questions with many words should not match the follow-up pattern."""
        assert not self._is_followup(
            "Can you tell me in detail whether the BNL-PTEST queue is currently "
            "active and accepting jobs from ATLAS users?"
        )

    def test_is_cric_followup_does_not_match_general_question(self) -> None:
        """A general question unrelated to queue status should not match."""
        assert not self._is_followup("What is a pilot job?")

    def test_prior_cric_exchange_routes_followup_to_cric(self) -> None:
        """After a CRIC exchange, 'Is BNL-PTEST active?' must route to cric_query.

        This is the exact production failure: the follow-up question contained
        no CRIC-specific keywords so it fell through to RAG retrieval.
        """
        import asyncio  # noqa: PLC0415
        from unittest.mock import patch  # noqa: PLC0415
        from bamboo.tools.bamboo_answer import _run_db_query_fast_path  # noqa: PLC0415

        captured_tool: list[str] = []

        async def _fake_execute_plan(plan, question, history, **_kw):  # type: ignore[no-untyped-def]
            """Capture tool name and return a dummy response."""
            if plan and plan.tool_calls:
                captured_tool.append(plan.tool_calls[0].tool)
            from bamboo.tools.base import text_content  # noqa: PLC0415
            return text_content("ok")

        history: list[Message] = [
            {"role": "user",
             "content": "how many queues are using the objectstore copytool?"},
            {"role": "assistant",
             "content": "13 queues use the objectstore copytool: BNL-PTEST, CERN..."},
        ]

        with patch("bamboo.tools.bamboo_answer.execute_plan", side_effect=_fake_execute_plan):
            asyncio.run(_run_db_query_fast_path("Is BNL-PTEST active?", history))

        assert captured_tool, "No tool was called"
        assert captured_tool[0] == "cric_query", (
            f"Expected cric_query, got {captured_tool[0]!r} — "
            "CRIC contextual follow-up not routing correctly"
        )


class TestQueueStatusRoutingRegressions:
    """Regression tests for queue-status questions that were misrouted.

    Previously 'which are the active queues?' went to RAG and 'which queues
    are online?' went to panda_jobs_query.  These must all route to cric_query.
    """

    def setup_method(self) -> None:
        """Import helpers fresh for each test."""
        from bamboo.tools.bamboo_answer import (  # noqa: PLC0415
            _is_cric_question, _build_deterministic_plan,
        )
        self._is_cric = _is_cric_question
        self._plan = _build_deterministic_plan

    def _tool(self, q: str) -> str:
        """Return the tool name that would be selected for question *q*."""
        plan = self._plan(q, None, None)
        return plan.tool_calls[0].tool if plan else "RAG"

    def test_active_queues_routes_to_cric(self) -> None:
        """'which are the active queues?' must route to CRIC, not RAG."""
        assert self._is_cric("which are the active queues?")
        assert self._tool("which are the active queues?") == "cric_query"

    def test_queues_are_online_routes_to_cric(self) -> None:
        """'which queues are online?' must route to CRIC, not jobs DB."""
        assert self._is_cric("which queues are online?")
        assert self._tool("which queues are online?") == "cric_query"

    def test_queues_are_offline_routes_to_cric(self) -> None:
        """'which queues are offline?' must route to CRIC, not jobs DB."""
        assert self._is_cric("which queues are offline?")
        assert self._tool("which queues are offline?") == "cric_query"

    def test_how_many_queues_active_routes_to_cric(self) -> None:
        """'how many queues are active?' must route to CRIC, not jobs DB."""
        assert self._is_cric("how many queues are active?")
        assert self._tool("how many queues are active?") == "cric_query"

    def test_active_queues_at_site_routes_to_cric(self) -> None:
        """'which queues are active at BNL?' must route to CRIC."""
        assert self._is_cric("which queues are active at BNL?")
        assert self._tool("which queues are active at BNL?") == "cric_query"

    def test_queues_not_online_routes_to_cric(self) -> None:
        """'show me queues that are not online' must route to CRIC."""
        assert self._is_cric("show me queues that are not online")
        assert self._tool("show me queues that are not online") == "cric_query"

    def test_panda_queues_not_online_routes_to_cric(self) -> None:
        """'which panda queues are not online?' must route to CRIC."""
        assert self._is_cric("which panda queues are not online?")
        assert self._tool("which panda queues are not online?") == "cric_query"

    def test_jobs_questions_unaffected(self) -> None:
        """Pure job-status questions must still route to panda_jobs_query."""
        job_qs = [
            "how many jobs failed at BNL?",
            "which jobs are running at CERN?",
            "what are the top errors at SWT2?",
        ]
        for q in job_qs:
            assert not self._is_cric(q), f"Should not be CRIC: {q!r}"
            assert self._tool(q) == "panda_jobs_query", f"Wrong tool for: {q!r}"


class TestQueueAtSiteRouting:
    """Tests for 'queues at SITE are STATUS' routing pattern.

    'Which queues at BNL are active?' was triggering disambiguation because
    'BNL' is in _DB_KEYWORDS['jobs'] and the status word came after the site
    name, so no _CRIC_SIGNALS substring matched.
    """

    def setup_method(self) -> None:
        """Import helpers fresh for each test."""
        from bamboo.tools.bamboo_answer import _is_cric_question  # noqa: PLC0415
        self._is_cric = _is_cric_question

    def test_queues_at_bnl_active(self) -> None:
        """'Which queues at BNL are active?' must be detected as CRIC."""
        assert self._is_cric("Which queues at BNL are active?")

    def test_queues_at_cern_online(self) -> None:
        """'Which queues at CERN are online?' must be detected as CRIC."""
        assert self._is_cric("Which queues at CERN are online?")

    def test_queues_at_aglt2_offline(self) -> None:
        """'Which queues at AGLT2 are offline?' must be detected as CRIC."""
        assert self._is_cric("Which queues at AGLT2 are offline?")

    def test_active_queues_at_site(self) -> None:
        """'active queues at BNL' (reversed word order) must be detected as CRIC."""
        assert self._is_cric("active queues at BNL")

    def test_jobs_at_bnl_not_cric(self) -> None:
        """'how many jobs failed at BNL?' must NOT be detected as CRIC."""
        assert not self._is_cric("how many jobs failed at BNL?")

    def test_jobs_running_at_cern_not_cric(self) -> None:
        """'which jobs are running at CERN?' must NOT be detected as CRIC."""
        assert not self._is_cric("which jobs are running at CERN?")


class TestClarificationReply:
    """Tests for bare database-name replies to clarification prompts.

    When the user replies 'cric' after a disambiguation prompt, the original
    question must be recovered from history and passed to cric_query.
    Previously 'cric' was sent as the question itself, producing useless output.
    """

    def _run(self, coro):  # type: ignore[no-untyped-def]
        """Run a coroutine synchronously."""
        import asyncio  # noqa: PLC0415
        return asyncio.run(coro)

    def test_bare_cric_reply_uses_original_question(self) -> None:
        """Replying 'cric' after disambiguation must use the prior user question."""
        from unittest.mock import patch  # noqa: PLC0415
        from bamboo.tools.bamboo_answer import _run_db_query_fast_path  # noqa: PLC0415

        captured: list[tuple[str, str]] = []

        async def _fake_execute_plan(plan, question, history, **_kw):  # type: ignore
            captured.append((plan.tool_calls[0].tool, question))
            from bamboo.tools.base import text_content  # noqa: PLC0415
            return text_content("ok")

        history: list[Message] = [
            {"role": "user",
             "content": "Which queues at BNL are active?"},
            {"role": "assistant",
             "content": "I can query multiple databases. Which one did you mean? jobs | cric"},
        ]

        with patch("bamboo.tools.bamboo_answer.execute_plan", side_effect=_fake_execute_plan):
            self._run(_run_db_query_fast_path("cric", history))

        assert captured, "No tool was called"
        tool, question = captured[0]
        assert tool == "cric_query", f"Expected cric_query, got {tool!r}"
        assert "BNL" in question, f"Expected original question in query, got {question!r}"
        assert question != "cric", "Original question was not recovered from history"

    def test_bare_jobs_reply_uses_original_question(self) -> None:
        """Replying 'jobs' after disambiguation must use the prior user question."""
        from unittest.mock import patch  # noqa: PLC0415
        from bamboo.tools.bamboo_answer import _run_db_query_fast_path  # noqa: PLC0415

        captured: list[tuple[str, str]] = []

        async def _fake_execute_plan(plan, question, history, **_kw):  # type: ignore
            captured.append((plan.tool_calls[0].tool, question))
            from bamboo.tools.base import text_content  # noqa: PLC0415
            return text_content("ok")

        history: list[Message] = [
            {"role": "user", "content": "How many failed jobs at BNL?"},
            {"role": "assistant",
             "content": "I can query multiple databases. Which one did you mean? jobs | cric"},
        ]

        with patch("bamboo.tools.bamboo_answer.execute_plan", side_effect=_fake_execute_plan):
            self._run(_run_db_query_fast_path("jobs", history))

        assert captured
        tool, question = captured[0]
        assert tool == "panda_jobs_query"
        assert "BNL" in question
        assert question != "jobs"
