"""ASCII chart utilities for Harvester worker/pilot evidence.

Renders compact ASCII charts from the structured evidence dict produced by
``panda_harvester_workers_tool``.  Designed for use in the Textual TUI
(``/chart`` command) and as a fallback for terminal-only environments.

All functions are pure (no I/O, no bamboo dependency) so they can be
imported and tested independently of the rest of the plugin.

Public surface
--------------
- ``ascii_status_bar(evidence, bar_width)``  — horizontal bar chart of
  worker counts per status.
- ``ascii_pivot_table(evidence, top_n)``     — tabular view of the pivot
  breakdown by status × jobtype × resourcetype.
- ``render_chart(evidence, width, mode)``    — dispatcher that picks the
  best chart type and returns a ready-to-display string.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_BAR_CHAR: str = "█"
_EMPTY_MSG: str = "No Harvester worker data available."


def _status_summary(evidence: dict[str, Any]) -> list[tuple[str, int]]:
    """Extract ``(status, count)`` pairs from an evidence dict.

    Returns pairs sorted by count descending, which mirrors the order
    already guaranteed by ``_aggregate_evidence`` in
    ``harvester_worker_impl``.

    Args:
        evidence: Evidence dict as stored by ``bamboo_executor`` —
            must contain a ``nworkers_by_status`` mapping.

    Returns:
        List of ``(status, count)`` tuples, descending by count.
        Empty list when the key is absent or the mapping is empty.
    """
    raw: dict[str, int] = evidence.get("nworkers_by_status") or {}
    return sorted(raw.items(), key=lambda kv: kv[1], reverse=True)


def _format_window(evidence: dict[str, Any]) -> str:
    """Build a compact time-window label from evidence metadata.

    Args:
        evidence: Evidence dict containing optional ``from_dt``, ``to_dt``,
            and ``site_filter`` keys.

    Returns:
        A single-line string such as
        ``"2026-03-25T10:00:00 → 2026-03-25T11:00:00  site: BNL"`` or
        ``"window unknown"`` when the timestamps are absent.
    """
    from_dt: str = evidence.get("from_dt") or ""
    to_dt: str = evidence.get("to_dt") or ""
    site: str | None = evidence.get("site_filter")

    if not from_dt or not to_dt:
        label = "window unknown"
    else:
        # Trim microseconds / timezone suffix for readability.
        label = f"{from_dt[:19]} \u2192 {to_dt[:19]}"

    if site:
        label += f"  site: {site}"
    return label


# ---------------------------------------------------------------------------
# Public chart functions
# ---------------------------------------------------------------------------


def ascii_status_bar(
    evidence: dict[str, Any],
    bar_width: int = 40,
) -> str:
    """Render a horizontal ASCII bar chart of worker counts per status.

    Each row shows one status with a proportional bar of ``█`` characters
    and the raw count.  The longest bar occupies exactly ``bar_width``
    characters; all others are scaled proportionally.

    Args:
        evidence: Evidence dict from ``panda_harvester_workers_tool``.
            Must contain ``nworkers_by_status`` (mapping of status → int).
        bar_width: Maximum number of ``█`` characters for the longest bar.
            Clamped to a minimum of 10.

    Returns:
        Multi-line string suitable for ``print()`` or a Textual
        ``Text``/``Static`` widget.  Returns a short "no data" message
        when ``nworkers_by_status`` is absent or empty.

    Example output::

        Harvester workers by status
        2026-03-25T10:00:00 → 2026-03-25T11:00:00  site: BNL
        Total: 12 405

        running   ████████████████████████████████████████  9 821
        finished  ████████████████                          3 042
        failed    █                                           117
        submitted                                              23
        cancelled                                               2
    """
    pairs = _status_summary(evidence)
    if not pairs:
        return _EMPTY_MSG

    bar_width = max(10, bar_width)
    total: int = evidence.get("nworkers_total") or sum(c for _, c in pairs)
    window_label = _format_window(evidence)

    max_count = pairs[0][1]  # already sorted descending
    max_label_len = max(len(s) for s, _ in pairs)

    lines: list[str] = [
        "Harvester workers by status",
        window_label,
        f"Total: {total:,}",
        "",
    ]

    for status, count in pairs:
        label = status.ljust(max_label_len)
        bar_len = round((count / max_count) * bar_width) if max_count > 0 else 0
        bar = (_BAR_CHAR * bar_len).ljust(bar_width)
        lines.append(f"{label}  {bar}  {count:,}")

    return "\n".join(lines)


def ascii_pivot_table(
    evidence: dict[str, Any],
    top_n: int = 15,
) -> str:
    """Render the top rows of the pivot table as a fixed-width ASCII table.

    The pivot table in the evidence dict breaks counts down by
    ``(status, jobtype, resourcetype)``.  This function renders the top
    ``top_n`` rows sorted by ``nworkers`` descending.

    Args:
        evidence: Evidence dict from ``panda_harvester_workers_tool``.
            Must contain a ``pivot`` key (list of dicts with
            ``status``, ``jobtype``, ``resourcetype``, ``nworkers``).
        top_n: Maximum number of rows to display.  Pass ``0`` to show all.

    Returns:
        Multi-line string with a header row, a separator, and one data
        row per pivot entry.  Returns a short "no data" message when
        ``pivot`` is absent or empty.

    Example output::

        Harvester pilot breakdown (top 15)
        2026-03-25T10:00:00 → 2026-03-25T11:00:00

        Status     Jobtype   Resource   Workers
        ---------  --------  ---------  -------
        running    managed   SCORE        7 412
        running    managed   MCORE        2 409
        submitted  managed   SCORE          843
        finished   managed   SCORE          312
    """
    pivot: list[dict[str, Any]] = evidence.get("pivot") or []
    if not pivot:
        return _EMPTY_MSG

    rows = pivot[:top_n] if top_n > 0 else pivot
    window_label = _format_window(evidence)
    shown = len(rows)
    total_rows = len(pivot)

    # Column widths — at least as wide as the header.
    col_status = max(len("Status"), max(len(str(r.get("status", ""))) for r in rows))
    col_jobtype = max(len("Jobtype"), max(len(str(r.get("jobtype", ""))) for r in rows))
    col_restype = max(len("Resource"), max(len(str(r.get("resourcetype", ""))) for r in rows))
    col_workers = max(len("Workers"), max(len(f"{r.get('nworkers', 0):,}") for r in rows))

    def _row(s: str, j: str, r: str, n: str) -> str:
        return (
            s.ljust(col_status) + "  " +
            j.ljust(col_jobtype) + "  " +
            r.ljust(col_restype) + "  " +
            n.rjust(col_workers)
        )

    sep = (
        "-" * col_status + "  " +
        "-" * col_jobtype + "  " +
        "-" * col_restype + "  " +
        "-" * col_workers
    )

    header_note = f"top {shown}" if shown < total_rows else f"all {shown}"
    lines: list[str] = [
        f"Harvester pilot breakdown ({header_note})",
        window_label,
        "",
        _row("Status", "Jobtype", "Resource", "Workers"),
        sep,
    ]

    for r in rows:
        lines.append(_row(
            str(r.get("status", "?")),
            str(r.get("jobtype", "?")),
            str(r.get("resourcetype", "?")),
            f"{r.get('nworkers', 0):,}",
        ))

    return "\n".join(lines)


def ascii_timeseries(
    buckets: list[dict[str, Any]],
    status: str = "running",
    width: int = 60,
    height: int = 15,
) -> str:
    """Render a time-series bucket list as a compact ASCII bar chart.

    Each column represents one time bucket; column height is proportional
    to the worker count.  A simple y-axis with min/max/mid labels and an
    x-axis with first/last timestamps are included.

    Adapted from ``ascii_histogram`` in ``opensearch_monitor.py``.

    Args:
        buckets: List of ``{"timestamp": str, "count": int}`` dicts as
            returned by ``fetch_timeseries``.  Expected in ascending
            timestamp order.
        status: Status label shown in the chart header.
        width: Maximum number of character columns available for bars.
        height: Number of character rows for the plot area.

    Returns:
        Multi-line string suitable for a Textual ``Text`` widget.
        Returns the standard no-data message when ``buckets`` is empty
        or contains fewer than two entries.

    Example output::

        Harvester workers – running  (min: 0  max: 569  mean: 169)
        569 ┤  █
            │  █                   █
            │  █  █           █    █
          0 ┤  █  █           █    █
              14:18                          15:15
    """
    if len(buckets) < 2:
        return _EMPTY_MSG

    counts = [b["count"] for b in buckets]
    max_count = max(counts)
    min_count = min(counts)
    mean_count = sum(counts) / len(counts)

    # Scale buckets to fill the available width.
    # When there are fewer buckets than columns, each bucket occupies
    # multiple characters so the chart fills the panel.
    # When there are more buckets than columns, downsample instead.
    n = len(counts)
    if n > width:
        # Downsample: map width columns onto n buckets.
        step = n / width
        counts_display = [counts[int(i * step)] for i in range(width)]
        col_w = 1
    else:
        # Upsample: each bucket gets at least one column, spread evenly.
        col_w = max(1, width // n)
        counts_display = counts

    # Build the grid: height rows × (len(counts_display) * col_w) cols.
    total_cols = len(counts_display) * col_w
    rows: list[list[str]] = [[" "] * total_cols for _ in range(height)]
    for bucket_idx, c in enumerate(counts_display):
        bar_height = round((c / max_count) * height) if max_count > 0 else 0
        for row in range(height - bar_height, height):
            for char_idx in range(col_w):
                col = bucket_idx * col_w + char_idx
                rows[row][col] = _BAR_CHAR

    def _value_to_row(v: int) -> int:
        """Map a worker-count value to the topmost filled row for that bar.

        Args:
            v: Worker count value to convert.

        Returns:
            Zero-based row index (0 = top of grid).
        """
        bar_h = round((v / max_count) * height) if max_count > 0 else 0
        bar_h = max(1, bar_h)
        return height - bar_h

    tick_values: dict[int, str] = {
        _value_to_row(max_count): str(max_count),
        _value_to_row(min_count): str(min_count),
    }
    mid_count = (max_count + min_count) // 2
    mid_row = _value_to_row(mid_count)
    if mid_row not in tick_values:
        tick_values[mid_row] = str(mid_count)

    label_w = max(len(v) for v in tick_values.values())

    header = (
        f"Harvester workers reporting \u2013 {status}  "
        f"(min: {min_count:,}  max: {max_count:,}  mean: {mean_count:,.0f})"
        f"  [update events per bucket, not total active pilots]"
    )

    lines: list[str] = [header]
    for row_idx, row in enumerate(rows):
        if row_idx in tick_values:
            prefix = tick_values[row_idx].rjust(label_w) + " \u2524"
        else:
            prefix = " " * (label_w + 1) + "\u2502"
        lines.append(prefix + "".join(row))

    # X-axis: first and last bucket timestamps (HH:MM only).
    first_ts = buckets[0]["timestamp"][11:16]
    last_ts = buckets[-1]["timestamp"][11:16]
    pad = total_cols - len(first_ts) - len(last_ts)
    x_axis = " " * (label_w + 2) + first_ts + " " * max(1, pad) + last_ts
    lines.append(x_axis)

    return "\n".join(lines)


def render_chart(
    evidence: dict[str, Any],
    width: int = 60,
    mode: str = "auto",
) -> str:
    """Render the best available ASCII chart for a Harvester evidence dict.

    Dispatcher function: selects the chart type based on ``mode`` and the
    contents of ``evidence``.

    Modes:

    - ``"status"``  — always render the status bar chart.
    - ``"pivot"``   — always render the pivot table.
    - ``"timeseries"`` — render time-series bar chart from ``buckets`` list.
    - ``"auto"``    — render status bar if ``nworkers_by_status`` is present
      and non-empty; fall back to pivot table; fall back to the no-data
      message.

    Args:
        evidence: Evidence dict from ``panda_harvester_workers_tool``.
        width: Available character width for the bar chart's bar segment.
            The actual rendered width will be slightly wider due to labels
            and counts.  Ignored for pivot-table mode.
        mode: One of ``"auto"``, ``"status"``, ``"pivot"``.

    Returns:
        Rendered multi-line string ready for display.

    Raises:
        ValueError: If ``mode`` is not one of the accepted values.
            Accepted values: ``"auto"``, ``"status"``, ``"pivot"``,
            ``"timeseries"``.
    """
    valid_modes = {"auto", "status", "pivot", "timeseries"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)!r}, got {mode!r}")

    if mode == "status":
        return ascii_status_bar(evidence, bar_width=width)
    if mode == "pivot":
        return ascii_pivot_table(evidence)
    if mode == "timeseries":
        buckets = evidence.get("buckets") or []
        status = str(evidence.get("status") or "running")
        return ascii_timeseries(buckets, status=status, width=width)

    # mode == "auto"
    if evidence.get("nworkers_by_status"):
        return ascii_status_bar(evidence, bar_width=width)
    if evidence.get("pivot"):
        return ascii_pivot_table(evidence)
    return _EMPTY_MSG
