"""Tests for ``askpanda_atlas.chart_utils``.

Covers all public functions with empty data, single-status, multi-status,
pivot table rendering, width clamping, mode dispatching, and metadata
label formatting.
"""
from __future__ import annotations

from typing import Any

import pytest

from askpanda_atlas.chart_utils import (
    _format_window,
    _status_summary,
    ascii_pivot_table,
    ascii_status_bar,
    render_chart,
)


# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------


def _make_evidence(
    by_status: dict[str, int] | None = None,
    pivot: list[dict[str, Any]] | None = None,
    from_dt: str = "2026-03-25T10:00:00",
    to_dt: str = "2026-03-25T11:00:00",
    site_filter: str | None = None,
    nworkers_total: int | None = None,
) -> dict[str, Any]:
    """Build a minimal Harvester evidence dict for testing.

    Args:
        by_status: Mapping of status → count.
        pivot: List of ``{status, jobtype, resourcetype, nworkers}`` dicts.
        from_dt: ISO-8601 lower bound.
        to_dt: ISO-8601 upper bound.
        site_filter: Computing site name or ``None``.
        nworkers_total: Grand total; computed from ``by_status`` when ``None``.

    Returns:
        Evidence dict matching the shape produced by
        ``harvester_worker_impl._aggregate_evidence``.
    """
    by_status = by_status or {}
    total = nworkers_total if nworkers_total is not None else sum(by_status.values())
    return {
        "nworkers_by_status": by_status,
        "nworkers_total": total,
        "pivot": pivot or [],
        "from_dt": from_dt,
        "to_dt": to_dt,
        "site_filter": site_filter,
        "error": None,
    }


def _make_pivot_row(
    status: str = "running",
    jobtype: str = "managed",
    resourcetype: str = "SCORE",
    nworkers: int = 100,
) -> dict[str, Any]:
    """Return a minimal pivot-table row dict.

    Args:
        status: Worker status string.
        jobtype: Job type string.
        resourcetype: Resource type string.
        nworkers: Worker count.

    Returns:
        Dict with ``status``, ``jobtype``, ``resourcetype``, ``nworkers``.
    """
    return {
        "status": status,
        "jobtype": jobtype,
        "resourcetype": resourcetype,
        "nworkers": nworkers,
    }


# ---------------------------------------------------------------------------
# _status_summary
# ---------------------------------------------------------------------------


class TestStatusSummary:
    """Unit tests for :func:`_status_summary`."""

    def test_empty_evidence_returns_empty(self) -> None:
        """Empty nworkers_by_status yields an empty list."""
        assert _status_summary({}) == []

    def test_missing_key_returns_empty(self) -> None:
        """Missing key is treated as empty."""
        assert _status_summary({"other_key": 5}) == []

    def test_single_entry(self) -> None:
        """Single status is returned as a one-element list."""
        ev = _make_evidence(by_status={"running": 42})
        assert _status_summary(ev) == [("running", 42)]

    def test_sorted_descending(self) -> None:
        """Pairs are returned sorted by count descending."""
        ev = _make_evidence(by_status={"failed": 5, "running": 100, "submitted": 20})
        result = _status_summary(ev)
        counts = [c for _, c in result]
        assert counts == sorted(counts, reverse=True)

    def test_preserves_all_entries(self) -> None:
        """All status entries are returned."""
        statuses = {"running": 9821, "finished": 3042, "failed": 117, "submitted": 23}
        ev = _make_evidence(by_status=statuses)
        assert len(_status_summary(ev)) == 4


# ---------------------------------------------------------------------------
# _format_window
# ---------------------------------------------------------------------------


class TestFormatWindow:
    """Unit tests for :func:`_format_window`."""

    def test_full_window_no_site(self) -> None:
        """Both timestamps present, no site filter."""
        ev = _make_evidence(from_dt="2026-03-25T10:00:00", to_dt="2026-03-25T11:00:00")
        label = _format_window(ev)
        assert "2026-03-25T10:00:00" in label
        assert "2026-03-25T11:00:00" in label
        assert "site" not in label

    def test_site_appended(self) -> None:
        """Site filter is appended when present."""
        ev = _make_evidence(site_filter="BNL")
        label = _format_window(ev)
        assert "BNL" in label
        assert "site:" in label

    def test_missing_timestamps(self) -> None:
        """Missing timestamps fall back to 'window unknown'."""
        label = _format_window({"site_filter": None})
        assert label == "window unknown"

    def test_truncates_to_seconds(self) -> None:
        """Timestamps are truncated to 19 chars (no microseconds)."""
        ev = _make_evidence(
            from_dt="2026-03-25T10:00:00.123456",
            to_dt="2026-03-25T11:00:00.000Z",
        )
        label = _format_window(ev)
        assert ".123456" not in label
        assert ".000" not in label


# ---------------------------------------------------------------------------
# ascii_status_bar
# ---------------------------------------------------------------------------


class TestAsciiStatusBar:
    """Unit tests for :func:`ascii_status_bar`."""

    def test_empty_evidence_returns_no_data_msg(self) -> None:
        """Empty evidence returns the standard no-data message."""
        result = ascii_status_bar({})
        assert "No Harvester" in result

    def test_contains_header(self) -> None:
        """Output contains the section header."""
        ev = _make_evidence(by_status={"running": 100})
        assert "Harvester workers by status" in ascii_status_bar(ev)

    def test_contains_all_statuses(self) -> None:
        """Every status in evidence appears in the output."""
        statuses = {"running": 500, "failed": 20, "submitted": 10}
        ev = _make_evidence(by_status=statuses)
        out = ascii_status_bar(ev)
        for s in statuses:
            assert s in out

    def test_bar_characters_present(self) -> None:
        """Output contains at least one bar character for a non-zero count."""
        ev = _make_evidence(by_status={"running": 50})
        assert "█" in ascii_status_bar(ev)

    def test_total_displayed(self) -> None:
        """Grand total is included in the output."""
        ev = _make_evidence(by_status={"running": 100, "failed": 50})
        out = ascii_status_bar(ev)
        assert "150" in out

    def test_zero_count_no_bar_chars(self) -> None:
        """A zero-count status shows no bar characters on its row."""
        ev = _make_evidence(by_status={"running": 100, "cancelled": 0})
        out = ascii_status_bar(ev)
        lines = {ln.split()[0]: ln for ln in out.splitlines() if ln.strip()}
        cancelled_line = lines.get("cancelled", "")
        assert "█" not in cancelled_line

    def test_bar_width_clamped_to_minimum(self) -> None:
        """bar_width below 10 is clamped to 10 without error."""
        ev = _make_evidence(by_status={"running": 100})
        out = ascii_status_bar(ev, bar_width=1)
        assert "running" in out

    def test_site_filter_in_output(self) -> None:
        """Site filter from evidence appears in the window label line."""
        ev = _make_evidence(by_status={"running": 100}, site_filter="CERN")
        assert "CERN" in ascii_status_bar(ev)

    def test_counts_formatted_with_thousands(self) -> None:
        """Large counts are formatted with thousands separators."""
        ev = _make_evidence(by_status={"running": 9821})
        assert "9,821" in ascii_status_bar(ev)

    def test_single_status_full_bar(self) -> None:
        """With a single status the bar occupies the full bar_width."""
        ev = _make_evidence(by_status={"running": 100})
        out = ascii_status_bar(ev, bar_width=20)
        bar_line = [ln for ln in out.splitlines() if "running" in ln][0]
        assert "█" * 20 in bar_line


# ---------------------------------------------------------------------------
# ascii_pivot_table
# ---------------------------------------------------------------------------


class TestAsciiPivotTable:
    """Unit tests for :func:`ascii_pivot_table`."""

    def test_empty_pivot_returns_no_data_msg(self) -> None:
        """Empty pivot yields the standard no-data message."""
        ev = _make_evidence(pivot=[])
        assert "No Harvester" in ascii_pivot_table(ev)

    def test_header_row_present(self) -> None:
        """Output contains Status / Jobtype / Resource / Workers headers."""
        ev = _make_evidence(pivot=[_make_pivot_row()])
        out = ascii_pivot_table(ev)
        assert "Status" in out
        assert "Jobtype" in out
        assert "Resource" in out
        assert "Workers" in out

    def test_data_row_present(self) -> None:
        """A pivot row's fields appear in the table output."""
        row = _make_pivot_row(status="failed", jobtype="user", resourcetype="MCORE", nworkers=42)
        ev = _make_evidence(pivot=[row])
        out = ascii_pivot_table(ev)
        assert "failed" in out
        assert "user" in out
        assert "MCORE" in out
        assert "42" in out

    def test_top_n_limits_rows(self) -> None:
        """top_n argument limits displayed rows."""
        rows = [_make_pivot_row(status=f"s{i}", nworkers=100 - i) for i in range(20)]
        ev = _make_evidence(pivot=rows)
        out = ascii_pivot_table(ev, top_n=5)
        # Only 5 data rows after the separator
        sep_seen = False
        data_rows = 0
        for ln in out.splitlines():
            if set(ln.strip()) <= {"-", " "} and ln.strip():
                sep_seen = True
                continue
            if sep_seen and ln.strip():
                data_rows += 1
        assert data_rows == 5

    def test_top_n_zero_shows_all(self) -> None:
        """top_n=0 shows all rows."""
        rows = [_make_pivot_row(status=f"s{i}", nworkers=100 - i) for i in range(10)]
        ev = _make_evidence(pivot=rows)
        out = ascii_pivot_table(ev, top_n=0)
        for i in range(10):
            assert f"s{i}" in out

    def test_thousands_separator_in_workers(self) -> None:
        """Large worker counts are formatted with thousands separators."""
        row = _make_pivot_row(nworkers=12345)
        ev = _make_evidence(pivot=[row])
        assert "12,345" in ascii_pivot_table(ev)

    def test_contains_window_label(self) -> None:
        """Time-window label appears in the table."""
        ev = _make_evidence(
            pivot=[_make_pivot_row()],
            from_dt="2026-03-25T10:00:00",
            to_dt="2026-03-25T11:00:00",
        )
        out = ascii_pivot_table(ev)
        assert "2026-03-25" in out


# ---------------------------------------------------------------------------
# render_chart
# ---------------------------------------------------------------------------


class TestRenderChart:
    """Unit tests for :func:`render_chart`."""

    def test_invalid_mode_raises(self) -> None:
        """Unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            render_chart({}, mode="radar")

    def test_auto_prefers_status_bar(self) -> None:
        """auto mode uses status bar when nworkers_by_status is populated."""
        ev = _make_evidence(by_status={"running": 100})
        out = render_chart(ev, mode="auto")
        assert "Harvester workers by status" in out

    def test_auto_falls_back_to_pivot(self) -> None:
        """auto mode falls back to pivot when nworkers_by_status is empty."""
        ev = _make_evidence(pivot=[_make_pivot_row()], by_status={})
        out = render_chart(ev, mode="auto")
        assert "breakdown" in out

    def test_auto_empty_evidence_no_data_msg(self) -> None:
        """auto mode with no data returns the no-data message."""
        out = render_chart({}, mode="auto")
        assert "No Harvester" in out

    def test_explicit_status_mode(self) -> None:
        """mode='status' always calls ascii_status_bar."""
        ev = _make_evidence(by_status={"running": 50})
        out = render_chart(ev, mode="status")
        assert "Harvester workers by status" in out

    def test_explicit_pivot_mode(self) -> None:
        """mode='pivot' always calls ascii_pivot_table."""
        ev = _make_evidence(pivot=[_make_pivot_row()])
        out = render_chart(ev, mode="pivot")
        assert "breakdown" in out

    def test_width_passed_to_status_bar(self) -> None:
        """width argument controls max bar length in status mode."""
        ev = _make_evidence(by_status={"running": 100})
        # With width=10 the bar should have at most 10 block chars.
        out = render_chart(ev, width=10, mode="status")
        running_line = [ln for ln in out.splitlines() if "running" in ln][0]
        bar_segment = running_line.replace("running", "").strip()
        bar_chars = bar_segment.split()[0] if bar_segment else ""
        assert len(bar_chars) <= 10

    def test_mode_defaults_to_auto(self) -> None:
        """Default mode is 'auto'."""
        ev = _make_evidence(by_status={"running": 10})
        # Should not raise; should produce status bar output.
        out = render_chart(ev)
        assert "Harvester workers by status" in out
