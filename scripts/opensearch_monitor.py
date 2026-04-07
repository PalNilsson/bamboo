"""PanDA Harvester worker monitoring utilities.

This module provides functions to query OpenSearch for PanDA Harvester worker
metrics, explore worker status distributions, and visualise the results as
matplotlib figures (for Streamlit) or ASCII charts (for Textual / terminal).

Typical usage::

    client = create_client()
    data   = fetch_worker_counts(client, status="running")
    fig    = plot_worker_counts(data, status="running")

    summary = fetch_status_summary(client)
    all_ts  = fetch_worker_counts_by_status(client)
    details = fetch_failure_details(client)
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from opensearch_dsl import Search
from opensearchpy import OpenSearch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_INDEX: str = "atlas_harvesterworkers-*"
_CA_CERTS: str = "/etc/pki/tls/certs/CERN-bundle.pem"
_OS_HOST: str = "https://os-atlas.cern.ch/os"
_OS_USER: str = "pilot-monitor-agent"

# Known Harvester worker statuses (non-exhaustive – extend as needed).
KNOWN_STATUSES: list[str] = [
    "running",
    "submitted",
    "finished",
    "failed",
    "cancelled",
    "missed",
]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def create_client() -> OpenSearch:
    """Create and return an authenticated OpenSearch client.

    The password is read from the ``ASKPANDA_OPENSEARCH`` environment variable.

    Returns:
        An authenticated :class:`opensearchpy.OpenSearch` client instance.

    Raises:
        SystemExit: If the ``ASKPANDA_OPENSEARCH`` environment variable is not set.
    """
    password = os.environ.get("ASKPANDA_OPENSEARCH", "")
    if not password:
        print("ERROR: environment variable ASKPANDA_OPENSEARCH is not set.", file=sys.stderr)
        sys.exit(1)

    return OpenSearch(
        hosts=[_OS_HOST],
        http_auth=(_OS_USER, password),
        use_ssl=True,
        verify_certs=True,
        ca_certs=_CA_CERTS,
    )


# ---------------------------------------------------------------------------
# Core query helpers
# ---------------------------------------------------------------------------


def _time_range(hours: int = 1) -> tuple[datetime, datetime]:
    """Return a (from, now) UTC datetime pair covering the last *hours* hours.

    Args:
        hours: Number of hours to look back from now.

    Returns:
        A tuple ``(time_from, time_now)`` as UTC :class:`datetime` objects.
    """
    time_now = datetime.utcnow()
    time_from = time_now - timedelta(hours=hours)
    return time_from, time_now


def fetch_worker_counts(
    client: OpenSearch,
    status: str = "running",
    hours: int = 1,
    interval: str = "1m",
    index: str = _DEFAULT_INDEX,
) -> list[dict[str, Any]]:
    """Fetch per-minute worker counts for a single status over a time window.

    Args:
        client: An authenticated OpenSearch client.
        status: Harvester worker status to filter on (e.g. ``"running"``,
            ``"failed"``).
        hours: How many hours back from now to query.
        interval: Histogram bucket width understood by OpenSearch
            (e.g. ``"1m"``, ``"5m"``).
        index: OpenSearch index pattern to query.

    Returns:
        A list of dicts, each with keys ``"timestamp"`` (ISO-8601 string)
        and ``"count"`` (int).

    Example::

        [
            {"timestamp": "2026-03-19T16:11:00.000Z", "count": 158},
            {"timestamp": "2026-03-19T16:12:00.000Z", "count": 843},
        ]
    """
    time_from, time_now = _time_range(hours)

    s = (
        Search(using=client, index=index)
        .filter("range", **{"@timestamp": {"gte": time_from, "lte": time_now}})
        .filter("term", **{"status.keyword": status})
        .extra(size=0)
    )
    s.aggs.bucket(
        "counts_over_time",
        "date_histogram",
        field="@timestamp",
        fixed_interval=interval,
    )

    response = s.execute()
    return [
        {"timestamp": b.key_as_string, "count": b.doc_count}
        for b in response.aggregations.counts_over_time.buckets
    ]


def fetch_status_summary(
    client: OpenSearch,
    hours: int = 1,
    index: str = _DEFAULT_INDEX,
) -> list[dict[str, Any]]:
    """Fetch total worker counts grouped by status over a time window.

    Useful for a quick overview of how many workers are in each state.

    Args:
        client: An authenticated OpenSearch client.
        hours: How many hours back from now to query.
        index: OpenSearch index pattern to query.

    Returns:
        A list of dicts sorted by count descending, each with keys
        ``"status"`` (str) and ``"count"`` (int).

    Example::

        [
            {"status": "running",  "count": 9821},
            {"status": "finished", "count": 3042},
            {"status": "failed",   "count":  117},
        ]
    """
    time_from, time_now = _time_range(hours)

    s = (
        Search(using=client, index=index)
        .filter("range", **{"@timestamp": {"gte": time_from, "lte": time_now}})
        .extra(size=0)
    )
    s.aggs.bucket("by_status", "terms", field="status.keyword", size=50)

    response = s.execute()
    return sorted(
        [
            {"status": b.key, "count": b.doc_count}
            for b in response.aggregations.by_status.buckets
        ],
        key=lambda x: x["count"],
        reverse=True,
    )


def fetch_worker_counts_by_status(
    client: OpenSearch,
    statuses: list[str] | None = None,
    hours: int = 1,
    interval: str = "1m",
    index: str = _DEFAULT_INDEX,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch per-minute worker counts for multiple statuses in one query.

    Uses a single OpenSearch request with a ``filters`` aggregation so that
    the time-range scan happens only once.

    Args:
        client: An authenticated OpenSearch client.
        statuses: List of status strings to retrieve. Defaults to
            :data:`KNOWN_STATUSES`.
        hours: How many hours back from now to query.
        interval: Histogram bucket width (e.g. ``"1m"``).
        index: OpenSearch index pattern to query.

    Returns:
        A dict mapping each status string to a list of
        ``{"timestamp": str, "count": int}`` dicts (same format as
        :func:`fetch_worker_counts`).
    """
    if statuses is None:
        statuses = KNOWN_STATUSES

    time_from, time_now = _time_range(hours)

    s = (
        Search(using=client, index=index)
        .filter("range", **{"@timestamp": {"gte": time_from, "lte": time_now}})
        .extra(size=0)
    )

    # One date_histogram per status via a filters + nested date_histogram.
    s.aggs.bucket("by_status", "terms", field="status.keyword", size=len(statuses)).bucket(
        "counts_over_time",
        "date_histogram",
        field="@timestamp",
        fixed_interval=interval,
    )

    response = s.execute()

    result: dict[str, list[dict[str, Any]]] = {st: [] for st in statuses}
    for status_bucket in response.aggregations.by_status.buckets:
        st = status_bucket.key
        if st in result:
            result[st] = [
                {"timestamp": b.key_as_string, "count": b.doc_count}
                for b in status_bucket.counts_over_time.buckets
            ]
    return result


def fetch_failure_details(
    client: OpenSearch,
    hours: int = 1,
    top_n: int = 10,
    index: str = _DEFAULT_INDEX,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch breakdowns of failed workers by compute site and error code.

    Aggregates failed workers over the time window, grouped first by
    ``computingSite`` and then by ``errorCode`` so that operators can quickly
    identify the noisiest failure sources.

    Args:
        client: An authenticated OpenSearch client.
        hours: How many hours back from now to query.
        top_n: Number of top sites / error codes to return per bucket.
        index: OpenSearch index pattern to query.

    Returns:
        A dict with two keys:

        - ``"by_site"``: list of ``{"site": str, "count": int}`` dicts.
        - ``"by_error_code"``: list of ``{"error_code": str, "count": int}``
          dicts.

        Both lists are sorted by count descending.

    Example::

        {
            "by_site": [
                {"site": "CERN-PROD", "count": 42},
            ],
            "by_error_code": [
                {"error_code": "1105", "count": 19},
            ],
        }
    """
    time_from, time_now = _time_range(hours)

    s = (
        Search(using=client, index=index)
        .filter("range", **{"@timestamp": {"gte": time_from, "lte": time_now}})
        .filter("term", **{"status.keyword": "failed"})
        .extra(size=0)
    )
    s.aggs.bucket("by_site", "terms", field="computingSite.keyword", size=top_n)
    s.aggs.bucket("by_error_code", "terms", field="errorCode.keyword", size=top_n)

    response = s.execute()

    by_site = sorted(
        [
            {"site": b.key, "count": b.doc_count}
            for b in response.aggregations.by_site.buckets
        ],
        key=lambda x: x["count"],
        reverse=True,
    )
    by_error_code = sorted(
        [
            {"error_code": str(b.key), "count": b.doc_count}
            for b in response.aggregations.by_error_code.buckets
        ],
        key=lambda x: x["count"],
        reverse=True,
    )
    return {"by_site": by_site, "by_error_code": by_error_code}


# ---------------------------------------------------------------------------
# Visualisation – matplotlib (Streamlit / notebook)
# ---------------------------------------------------------------------------


def _parse_timestamps(data: list[dict[str, Any]]) -> tuple[list[datetime], list[int]]:
    """Parse a time-series list into parallel datetime and count lists.

    Args:
        data: List of ``{"timestamp": str, "count": int}`` dicts as returned
            by :func:`fetch_worker_counts`.

    Returns:
        A tuple ``(timestamps, counts)`` where *timestamps* is a list of
        :class:`datetime` objects and *counts* is a list of ints.
    """
    timestamps = [
        datetime.strptime(d["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ") for d in data
    ]
    counts = [d["count"] for d in data]
    return timestamps, counts


def plot_worker_counts(
    data: list[dict[str, Any]],
    status: str = "running",
    title: str | None = None,
    figsize: tuple[float, float] = (12, 4),
) -> plt.Figure:
    """Plot per-minute worker counts as a filled time-series figure.

    Args:
        data: Time-series data as returned by :func:`fetch_worker_counts`.
        status: Status label used in the y-axis and default title.
        title: Optional custom title; auto-generated if ``None``.
        figsize: Matplotlib figure size ``(width, height)`` in inches.

    Returns:
        A :class:`matplotlib.figure.Figure` ready for ``st.pyplot()`` or
        ``fig.savefig()``.
    """
    timestamps, counts = _parse_timestamps(data)

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(timestamps, counts, alpha=0.3)
    ax.plot(timestamps, counts, linewidth=1.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(f"{status} workers")
    ax.set_title(title or f"Harvester workers – status: {status}")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_multi_status(
    data: dict[str, list[dict[str, Any]]],
    title: str = "Harvester workers by status",
    figsize: tuple[float, float] = (12, 6),
) -> plt.Figure:
    """Plot per-minute worker counts for multiple statuses on the same axes.

    Args:
        data: Dict mapping status strings to time-series lists, as returned
            by :func:`fetch_worker_counts_by_status`.
        title: Figure title.
        figsize: Matplotlib figure size ``(width, height)`` in inches.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for status, series in data.items():
        if not series:
            continue
        timestamps, counts = _parse_timestamps(series)
        ax.plot(timestamps, counts, label=status, linewidth=1.5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Worker count")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_status_bar(
    summary: list[dict[str, Any]],
    title: str = "Worker counts by status",
    figsize: tuple[float, float] = (8, 4),
) -> plt.Figure:
    """Plot a horizontal bar chart of total worker counts per status.

    Args:
        summary: Data as returned by :func:`fetch_status_summary`.
        title: Figure title.
        figsize: Matplotlib figure size ``(width, height)`` in inches.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    statuses = [d["status"] for d in reversed(summary)]
    counts = [d["count"] for d in reversed(summary)]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(statuses, counts)
    ax.bar_label(bars, padding=4)
    ax.set_xlabel("Total worker count")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Visualisation – ASCII (Textual / terminal)
# ---------------------------------------------------------------------------


def ascii_histogram(
    data: list[dict[str, Any]],
    status: str = "running",
    width: int = 60,
    height: int = 15,
) -> str:
    """Render a time-series as a compact ASCII bar chart.

    Each column represents one time bucket; column height is proportional to
    the worker count.  A simple y-axis with min/max labels is included.

    Args:
        data: Time-series data as returned by :func:`fetch_worker_counts`.
        status: Status label shown in the chart header.
        width: Maximum number of character columns available for bars.
        height: Number of character rows for the plot area.

    Returns:
        A multi-line string suitable for ``print()`` or a Textual
        :class:`~textual.widgets.Static` widget.

    Example output::

        Harvester workers – running  (min: 5  max: 843  mean: 175)
        843 ┤
            │  █
            │  █
            │  █
            │  █
          5 ┤  █                                                █
              16:11                                          17:08
    """
    if not data:
        return f"No data for status='{status}'"

    counts = [d["count"] for d in data]
    max_count = max(counts)
    min_count = min(counts)
    mean_count = sum(counts) / len(counts)

    # Downsample if there are more buckets than available columns.
    n = len(counts)
    if n > width:
        step = n / width
        counts_display = [counts[int(i * step)] for i in range(width)]
    else:
        counts_display = counts

    # Build the grid (height rows × len(counts_display) cols).
    rows: list[list[str]] = [[" "] * len(counts_display) for _ in range(height)]
    for col, c in enumerate(counts_display):
        bar_height = round((c / max_count) * height) if max_count > 0 else 0
        for row in range(height - bar_height, height):
            rows[row][col] = "█"

    # Build y-axis tick labels: map a value to the row index where that
    # value's bar top sits.  row = height - round((v / max_count) * height),
    # clamped to [0, height-1].
    def _value_to_row(v: int) -> int:
        """Convert a worker-count value to the first filled row for that bar.

        Uses the same ``round()`` as the bar-drawing loop so that the label
        always sits exactly on the top edge of the bar for value *v*.

        Args:
            v: Worker count value to map.

        Returns:
            Zero-based row index (0 = top of grid).  The returned row is the
            first (topmost) row that would be filled for a bar of height *v*.
        """
        bar_height = round((v / max_count) * height) if max_count > 0 else 0
        bar_height = max(1, bar_height)  # at least one row so label is visible
        return height - bar_height

    # Always show max and min; add a mid-point tick if the gap is large enough.
    tick_values: dict[int, str] = {
        _value_to_row(max_count): str(max_count),
        _value_to_row(min_count): str(min_count),
    }
    mid_count = (max_count + min_count) // 2
    mid_row = _value_to_row(mid_count)
    # Only add mid tick if it doesn't collide with max/min rows.
    if mid_row not in tick_values:
        tick_values[mid_row] = str(mid_count)

    label_w = max(len(v) for v in tick_values.values())

    # Compose output lines.
    header = (
        f"Harvester workers – {status}  "
        f"(min: {min_count}  max: {max_count}  mean: {mean_count:.0f})"
    )

    lines: list[str] = [header]
    for row_idx, row in enumerate(rows):
        if row_idx in tick_values:
            prefix = tick_values[row_idx].rjust(label_w) + " ┤"
        else:
            prefix = " " * (label_w + 1) + "│"
        lines.append(prefix + "".join(row))

    # X-axis timestamps (first and last).
    first_ts = data[0]["timestamp"][11:16]   # "HH:MM"
    last_ts = data[-1]["timestamp"][11:16]
    x_axis = " " * (label_w + 2) + first_ts + " " * (len(counts_display) - len(first_ts) - len(last_ts)) + last_ts
    lines.append(x_axis)

    return "\n".join(lines)


def ascii_status_summary(summary: list[dict[str, Any]], bar_width: int = 40) -> str:
    """Render a status summary as an ASCII horizontal bar chart.

    Args:
        summary: Data as returned by :func:`fetch_status_summary`.
        bar_width: Maximum number of ``█`` characters for the longest bar.

    Returns:
        A multi-line string.

    Example output::

        Worker counts by status
        running  ████████████████████████████████████████  9821
        finished ████████████████                          3042
        failed   █                                          117
    """
    if not summary:
        return "No data."

    max_count = max(d["count"] for d in summary)
    max_label = max(len(d["status"]) for d in summary)

    lines = ["Worker counts by status"]
    for d in summary:
        label = d["status"].ljust(max_label)
        bar_len = round((d["count"] / max_count) * bar_width) if max_count > 0 else 0
        bar = ("█" * bar_len).ljust(bar_width)
        lines.append(f"{label} {bar}  {d['count']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run a quick exploration of Harvester worker data and print summaries.

    Connects to OpenSearch, fetches a status summary and running-worker
    time-series for the last hour, then prints ASCII charts to stdout.
    """
    client = create_client()

    print("=" * 70)
    print("Status summary (last 1 h)")
    print("=" * 70)
    summary = fetch_status_summary(client, hours=1)
    print(ascii_status_summary(summary))

    print()
    print("=" * 70)
    print("Running workers – time series (last 1 h)")
    print("=" * 70)
    running_data = fetch_worker_counts(client, status="running", hours=1)
    print(ascii_histogram(running_data, status="running"))

    if any(d["status"] == "failed" for d in summary):
        print()
        print("=" * 70)
        print("Failed workers – breakdown (last 1 h)")
        print("=" * 70)
        failures = fetch_failure_details(client, hours=1)
        print("Top sites:")
        for row in failures["by_site"]:
            print(f"  {row['site']:<30} {row['count']}")
        print("Top error codes:")
        for row in failures["by_error_code"]:
            print(f"  {row['error_code']:<30} {row['count']}")


if __name__ == "__main__":
    main()
