"""Minimal BigPanDA HTTP helpers for standalone (no bamboo core) use.

Used only when ``bamboo.tools._panda_http`` is not importable, i.e. when
the plugin is installed without bamboo core.  Keep this in sync with the
canonical helpers in ``bamboo.tools._panda_http``.
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Any

import requests


def get_base_url() -> str:
    """Return the BigPanDA base URL from the environment."""
    return os.getenv("PANDA_BASE_URL", "https://bigpanda.cern.ch").rstrip("/")


def fetch_jsonish(url: str, timeout: int = 30) -> tuple[int, str, str, dict[str, Any] | None]:
    """Fetch a URL, returning (status, content_type, text, json_or_none)."""
    resp = requests.get(
        url,
        timeout=timeout,
        headers={"Accept": "application/json", "User-Agent": "AskPanDA/1.0"},
        allow_redirects=True,
    )
    status = resp.status_code
    ctype = (resp.headers.get("content-type") or "").lower()
    text = resp.text or ""
    if status < 200 or status >= 300:
        return status, ctype, text, None
    try:
        data = resp.json()
        return status, ctype, text, data if isinstance(data, dict) else {"_data": data}
    except Exception:  # pylint: disable=broad-exception-caught
        return status, ctype, text, None


def job_counts_from_payload(payload: dict[str, Any]) -> dict[str, int]:
    """Count job statuses in a BigPanDA task payload."""
    jobs = next(
        (payload.get(k) for k in ("jobs", "jobList", "joblist")
         if isinstance(payload.get(k), list)),
        None,
    )
    if not jobs:
        return {}
    statuses = [
        j.get("jobStatus") or j.get("status") or j.get("job_status")
        for j in jobs
        if isinstance(j, dict)
    ]
    return dict(Counter(s for s in statuses if isinstance(s, str) and s))


def datasets_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Summarise dataset statuses and file counts in a BigPanDA task payload."""
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        return {}
    sc: Counter[str] = Counter()
    totals: dict[str, int] = {"failed": 0, "finished": 0, "waiting": 0, "missing": 0}
    key_map = {
        "nfilesfailed": "failed",
        "nfilesfinished": "finished",
        "nfileswaiting": "waiting",
        "nfilesmissing": "missing",
    }
    worst: list[dict[str, Any]] = []
    for ds in datasets:
        if not isinstance(ds, dict):
            continue
        st = ds.get("status") or ""
        if st:
            sc[st] += 1
        for k, t in key_map.items():
            v = ds.get(k)
            if isinstance(v, int):
                totals[t] += v
        nff = ds.get("nfilesfailed")
        if isinstance(nff, int) and nff > 0:
            worst.append({
                "datasetname": ds.get("datasetname"),
                "status": ds.get("status"),
                "nfilesfailed": nff,
                "nfiles": ds.get("nfiles"),
            })
    worst = sorted(worst, key=lambda x: x.get("nfilesfailed") or 0, reverse=True)[:5]
    return {
        "dataset_count": len(datasets),
        "status_counts": dict(sc),
        "nfilesfailed_total": totals["failed"],
        "nfilesfinished_total": totals["finished"],
        "nfileswaiting_total": totals["waiting"],
        "nfilesmissing_total": totals["missing"],
        "worst_datasets": worst,
    }
