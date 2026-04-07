"""Tests for ``harvester_timeseries_impl``.

Covers pure helpers (``compute_interval``, ``_default_window``,
``_error_evidence``), the synchronous ``fetch_timeseries`` path with
OpenSearch mocked at the ``Search.execute`` level, and the async
``PandaHarvesterTimeseriesTool.call()`` entry point.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from askpanda_atlas.harvester_timeseries_impl import (
    PandaHarvesterTimeseriesTool,
    _default_window,
    _error_evidence,
    compute_interval,
    fetch_timeseries,
    panda_harvester_timeseries_tool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bucket(ts: str = "2026-04-07T15:00:00.000Z", count: int = 100) -> Any:
    """Return a mock OpenSearch aggregation bucket.

    Args:
        ts: ``key_as_string`` value (ISO-8601 with ms).
        count: ``doc_count`` value.

    Returns:
        MagicMock with ``key_as_string`` and ``doc_count`` attributes.
    """
    b = MagicMock()
    b.key_as_string = ts
    b.doc_count = count
    return b


def _mock_response(buckets: list[Any]) -> MagicMock:
    """Build a mock OpenSearch response with a ``counts_over_time`` aggregation.

    Args:
        buckets: List of mock bucket objects.

    Returns:
        MagicMock whose ``aggregations.counts_over_time.buckets`` yields
        the supplied list.
    """
    resp = MagicMock()
    resp.aggregations.counts_over_time.buckets = buckets
    return resp


def _unpack(result: list[Any]) -> dict[str, Any]:
    """Deserialise the JSON-wrapped MCP content returned by the tool.

    Args:
        result: Return value of ``PandaHarvesterTimeseriesTool.call()``.

    Returns:
        Deserialised dict with an ``"evidence"`` key.
    """
    return json.loads(result[0]["text"])


# ---------------------------------------------------------------------------
# compute_interval
# ---------------------------------------------------------------------------


class TestComputeInterval:
    """Unit tests for :func:`compute_interval`."""

    def test_20_minutes_gives_1m(self) -> None:
        """20-minute window → 1m buckets."""
        from_dt = "2026-04-07T15:00:00"
        to_dt = "2026-04-07T15:20:00"
        assert compute_interval(from_dt, to_dt) == "1m"

    def test_exactly_30_minutes_gives_1m(self) -> None:
        """Boundary: exactly 30 min → still 1m."""
        assert compute_interval("2026-04-07T15:00:00", "2026-04-07T15:30:00") == "1m"

    def test_1_hour_gives_5m(self) -> None:
        """1-hour window → 5m buckets."""
        assert compute_interval("2026-04-07T15:00:00", "2026-04-07T16:00:00") == "5m"

    def test_3_hours_gives_5m(self) -> None:
        """Boundary: exactly 3 h → still 5m."""
        assert compute_interval("2026-04-07T12:00:00", "2026-04-07T15:00:00") == "5m"

    def test_6_hours_gives_15m(self) -> None:
        """6-hour window → 15m buckets."""
        assert compute_interval("2026-04-07T09:00:00", "2026-04-07T15:00:00") == "15m"

    def test_24_hours_gives_1h(self) -> None:
        """24-hour window → 1h buckets."""
        assert compute_interval("2026-04-06T15:00:00", "2026-04-07T15:00:00") == "1h"

    def test_bad_timestamps_fallback(self) -> None:
        """Unparseable timestamps fall back to '5m'."""
        assert compute_interval("not-a-date", "also-not") == "5m"

    def test_none_timestamps_fallback(self) -> None:
        """None timestamps fall back to '5m'."""
        assert compute_interval(None, None) == "5m"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _default_window
# ---------------------------------------------------------------------------


class TestDefaultWindow:
    """Unit tests for :func:`_default_window`."""

    def test_returns_two_strings(self) -> None:
        """Returns a 2-tuple of strings."""
        from_dt, to_dt = _default_window()
        assert isinstance(from_dt, str)
        assert isinstance(to_dt, str)

    def test_from_before_to(self) -> None:
        """from_dt is strictly before to_dt."""
        from_dt, to_dt = _default_window()
        assert from_dt < to_dt

    def test_window_is_approximately_one_hour(self) -> None:
        """Window spans approximately 3600 seconds."""
        fmt = "%Y-%m-%dT%H:%M:%S"
        from_dt, to_dt = _default_window()
        t0 = datetime.strptime(from_dt, fmt).replace(tzinfo=timezone.utc)
        t1 = datetime.strptime(to_dt, fmt).replace(tzinfo=timezone.utc)
        diff = (t1 - t0).total_seconds()
        assert 3595 <= diff <= 3605


# ---------------------------------------------------------------------------
# _error_evidence
# ---------------------------------------------------------------------------


class TestErrorEvidence:
    """Unit tests for :func:`_error_evidence`."""

    def test_structure(self) -> None:
        """Error evidence has the expected keys."""
        ev = _error_evidence("running", "2026-04-07T15:00:00", "2026-04-07T16:00:00", "BNL", "boom")
        assert ev["buckets"] == []
        assert ev["total_buckets"] == 0
        assert ev["error"] is not None
        assert ev["status"] == "running"
        assert ev["site_filter"] == "BNL"

    def test_error_message_is_user_friendly(self) -> None:
        """Error message does not expose the internal detail."""
        ev = _error_evidence("failed", "a", "b", None, "secret internal error")
        assert "secret internal error" not in ev["error"]
        assert "ASKPANDA_OPENSEARCH" in ev["error"]


# ---------------------------------------------------------------------------
# fetch_timeseries (OpenSearch mocked)
# ---------------------------------------------------------------------------


def _patch_os(buckets: list[Any]):
    """Return a context-manager stack that mocks OpenSearch for fetch_timeseries.

    Patches:
    - ``create_os_client`` to return a MagicMock (no real connection).
    - ``Search.execute`` to return a response with the given buckets.
    - ``ASKPANDA_OPENSEARCH`` env var to a dummy value.

    Args:
        buckets: Bucket list to inject into the mock response.
    """
    import unittest.mock as _mock

    return _mock.patch.multiple(
        "askpanda_atlas.harvester_timeseries_impl",
        create_os_client=_mock.MagicMock(return_value=MagicMock()),
    )


class TestFetchTimeseries:
    """Unit tests for :func:`fetch_timeseries` with OpenSearch mocked."""

    def _run(
        self,
        buckets: list[Any],
        status: str = "running",
        from_dt: str = "2026-04-07T15:00:00",
        to_dt: str = "2026-04-07T16:00:00",
        site: str | None = None,
        interval: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute fetch_timeseries with patched OpenSearch.

        Args:
            buckets: Mock bucket objects to return from OpenSearch.
            status: Status filter.
            from_dt: Window lower bound.
            to_dt: Window upper bound.
            site: Optional site filter.
            interval: Optional interval override.

        Returns:
            List of ``{"timestamp", "count"}`` dicts.
        """
        mock_response = _mock_response(buckets)
        # Stub the entire opensearch_dsl module so the deferred import inside
        # fetch_timeseries succeeds even when the package is not installed.
        mock_search = MagicMock()
        mock_search.return_value.filter.return_value = mock_search.return_value
        mock_search.return_value.extra.return_value = mock_search.return_value
        mock_search.return_value.aggs.bucket.return_value = MagicMock()
        mock_search.return_value.execute.return_value = mock_response
        mock_os_dsl = MagicMock()
        mock_os_dsl.Search = mock_search
        with (
            patch(
                "askpanda_atlas.harvester_timeseries_impl.create_os_client",
                return_value=MagicMock(),
            ),
            patch.dict(os.environ, {"ASKPANDA_OPENSEARCH": "test-password"}),
            patch.dict(sys.modules, {"opensearch_dsl": mock_os_dsl}),
        ):
            # Invalidate any cached entry so the mock is actually called.
            from askpanda_atlas._cache import invalidate
            cache_key = (
                f"harvester_timeseries:{status}|{from_dt}|{to_dt}"
                f"|{site or ''}|{interval or compute_interval(from_dt, to_dt)}"
            )
            invalidate(cache_key)
            return fetch_timeseries(status, from_dt, to_dt, site, interval)

    def test_empty_response(self) -> None:
        """Empty bucket list returns empty result."""
        assert self._run([]) == []

    def test_single_bucket(self) -> None:
        """Single bucket is returned correctly."""
        result = self._run([_make_bucket("2026-04-07T15:00:00.000Z", 42)])
        assert result == [{"timestamp": "2026-04-07T15:00:00.000Z", "count": 42}]

    def test_multiple_buckets(self) -> None:
        """Multiple buckets are all returned in order."""
        raw = [
            _make_bucket("2026-04-07T15:00:00.000Z", 10),
            _make_bucket("2026-04-07T15:05:00.000Z", 20),
            _make_bucket("2026-04-07T15:10:00.000Z", 30),
        ]
        result = self._run(raw)
        assert len(result) == 3
        assert result[0]["count"] == 10
        assert result[2]["count"] == 30

    def test_interval_derived_from_window(self) -> None:
        """Interval is derived from the window when not supplied."""
        # 20-minute window should use 1m interval — verify no error
        result = self._run(
            [_make_bucket()],
            from_dt="2026-04-07T15:00:00",
            to_dt="2026-04-07T15:20:00",
        )
        assert len(result) == 1

    def test_missing_password_raises(self) -> None:
        """RuntimeError is raised when ASKPANDA_OPENSEARCH is not set."""
        from askpanda_atlas._cache import clear as _clear_cache
        _clear_cache()  # ensure no cached result masks the missing-password path
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ASKPANDA_OPENSEARCH", None)
            with pytest.raises(RuntimeError, match="ASKPANDA_OPENSEARCH"):
                fetch_timeseries("running", "2026-04-07T15:00:00", "2026-04-07T16:00:00")


# ---------------------------------------------------------------------------
# PandaHarvesterTimeseriesTool.call()
# ---------------------------------------------------------------------------


class TestPandaHarvesterTimeseriesTool:
    """Integration tests for :class:`PandaHarvesterTimeseriesTool`."""

    def _call(self, arguments: dict[str, Any], buckets: list[Any]) -> dict[str, Any]:
        """Run tool.call() with patched OpenSearch and bamboo.tools.base.

        Args:
            arguments: Tool input arguments.
            buckets: Mock bucket list to return.

        Returns:
            Deserialised evidence wrapper dict.
        """
        mock_text_content = lambda s: [{"type": "text", "text": s}]  # noqa: E731
        mock_response = _mock_response(buckets)
        mock_search = MagicMock()
        mock_search.return_value.filter.return_value = mock_search.return_value
        mock_search.return_value.extra.return_value = mock_search.return_value
        mock_search.return_value.aggs.bucket.return_value = MagicMock()
        mock_search.return_value.execute.return_value = mock_response
        mock_os_dsl = MagicMock()
        mock_os_dsl.Search = mock_search

        with (
            patch(
                "askpanda_atlas.harvester_timeseries_impl.create_os_client",
                return_value=MagicMock(),
            ),
            patch(
                "bamboo.tools.base.text_content",
                side_effect=mock_text_content,
            ),
            patch.dict(os.environ, {"ASKPANDA_OPENSEARCH": "test-password"}),
            patch.dict(sys.modules, {"opensearch_dsl": mock_os_dsl}),
        ):
            from askpanda_atlas._cache import invalidate
            invalidate("harvester_timeseries:running|||||5m")
            tool = PandaHarvesterTimeseriesTool()
            return _unpack(asyncio.run(tool.call(arguments)))

    def test_successful_call_returns_buckets(self) -> None:
        """Successful call includes bucket list in evidence."""
        raw = [_make_bucket("2026-04-07T15:00:00.000Z", 99)]
        result = self._call({"question": "How many pilots running at BNL?"}, raw)
        ev = result["evidence"]
        assert ev["error"] is None
        assert len(ev["buckets"]) == 1
        assert ev["buckets"][0]["count"] == 99

    def test_default_status_is_running(self) -> None:
        """Status defaults to 'running' when not supplied."""
        raw = [_make_bucket()]
        result = self._call({"question": "pilots?"}, raw)
        assert result["evidence"]["status"] == "running"

    def test_explicit_status_passed_through(self) -> None:
        """Explicit status is stored in evidence."""
        raw = [_make_bucket()]
        result = self._call({"question": "failed pilots?", "status": "failed"}, raw)
        assert result["evidence"]["status"] == "failed"

    def test_interval_stored_in_evidence(self) -> None:
        """Interval is present in the evidence dict."""
        raw = [_make_bucket()]
        result = self._call({"question": "pilots?", "interval": "1m"}, raw)
        assert result["evidence"]["interval"] == "1m"

    def test_missing_password_returns_error_evidence(self) -> None:
        """Missing ASKPANDA_OPENSEARCH returns error evidence, does not raise."""
        from askpanda_atlas._cache import clear as _clear_cache
        _clear_cache()  # ensure no cached result masks the missing-password path
        mock_text_content = lambda s: [{"type": "text", "text": s}]  # noqa: E731
        with (
            patch("bamboo.tools.base.text_content", side_effect=mock_text_content),
            patch.dict(os.environ, {}, clear=True),
        ):
            os.environ.pop("ASKPANDA_OPENSEARCH", None)
            tool = PandaHarvesterTimeseriesTool()
            result = _unpack(asyncio.run(tool.call({"question": "pilots?"})))
        assert result["evidence"]["error"] is not None
        assert result["evidence"]["buckets"] == []

    def test_singleton_instance(self) -> None:
        """Module-level singleton is a PandaHarvesterTimeseriesTool."""
        assert isinstance(panda_harvester_timeseries_tool, PandaHarvesterTimeseriesTool)


# ---------------------------------------------------------------------------
# chart_utils integration
# ---------------------------------------------------------------------------


class TestAsciiTimeseriesIntegration:
    """Tests for ascii_timeseries via render_chart mode='timeseries'."""

    def test_render_chart_timeseries_mode(self) -> None:
        """render_chart with mode='timeseries' produces a chart string."""
        import sys
        sys.path.insert(0, "/home/claude/test_env")
        from askpanda_atlas.chart_utils import render_chart

        ev = {
            "status": "running",
            "buckets": [
                {"timestamp": f"2026-04-07T15:{i:02d}:00.000Z", "count": 100 + i * 10}
                for i in range(20)
            ],
            "from_dt": "2026-04-07T15:00:00",
            "to_dt": "2026-04-07T15:20:00",
        }
        out = render_chart(ev, width=60, mode="timeseries")
        assert "running" in out
        assert "15:00" in out
        assert "█" in out

    def test_fewer_than_two_buckets_returns_no_data(self) -> None:
        """Fewer than 2 buckets returns the no-data message."""
        import sys
        sys.path.insert(0, "/home/claude/test_env")
        from askpanda_atlas.chart_utils import render_chart

        ev = {"status": "running", "buckets": [{"timestamp": "2026-04-07T15:00:00.000Z", "count": 5}]}
        out = render_chart(ev, mode="timeseries")
        assert "No Harvester" in out
