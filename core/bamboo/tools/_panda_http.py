"""BigPanDA HTTP helpers.

Canonical implementation used by ``askpanda_atlas.task_status_impl``
via ``askpanda_atlas._fallback_http``. The plugin package keeps a copy
of these helpers in ``_fallback_http`` for standalone use (when bamboo
core is not installed) — keep the two in sync when modifying either file.
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Any

import requests


def get_base_url() -> str:
    """Return the BigPanDA base URL from the environment.

    Reads the ``PANDA_BASE_URL`` environment variable, falling back to
    the public BigPanDA instance. Trailing slashes are stripped so
    callers can safely append path segments with ``/``.

    Returns:
        The base URL string, without a trailing slash.
    """
    return os.getenv("PANDA_BASE_URL", "https://bigpanda.cern.ch").rstrip("/")


def fetch_jsonish(
    url: str,
    timeout: int = 30,
) -> tuple[int, str, str, dict[str, Any] | None]:
    """Fetch a URL and return parsed response components.

    Sends a GET request with JSON and AskPanDA user-agent headers.
    Non-2xx responses and non-JSON bodies are returned with
    ``json_or_none`` set to ``None`` so callers can handle them
    uniformly.

    Args:
        url: The URL to fetch.
        timeout: HTTP timeout in seconds.

    Returns:
        A tuple of ``(status_code, content_type, body_text,
        parsed_json_or_none)``. ``parsed_json_or_none`` is ``None``
        when the response is not a 2xx JSON object.
    """
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
    """Count job statuses in a BigPanDA task payload.

    Looks for a job list under the keys ``jobs``, ``jobList``, or
    ``joblist`` and tallies each job's status string. Jobs with a
    missing or non-string status are silently skipped.

    Args:
        payload: Parsed JSON payload returned by the BigPanDA API.

    Returns:
        A mapping of status string to occurrence count, e.g.
        ``{"finished": 42, "failed": 3}``. Empty dict if no job list
        is found.
    """
    jobs: list[Any] | None = next(
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
    """Summarise dataset statuses and file counts in a BigPanDA task payload.

    Aggregates per-dataset file counters and collects the five datasets
    with the most failed files for quick triage.

    Args:
        payload: Parsed JSON payload returned by the BigPanDA API.

    Returns:
        A dict with the following keys:

        - ``dataset_count`` (int): Total number of datasets.
        - ``status_counts`` (dict[str, int]): Tally of dataset statuses.
        - ``nfilesfailed_total`` (int): Sum of failed files across all datasets.
        - ``nfilesfinished_total`` (int): Sum of finished files.
        - ``nfileswaiting_total`` (int): Sum of waiting files.
        - ``nfilesmissing_total`` (int): Sum of missing files.
        - ``worst_datasets`` (list[dict]): Up to five datasets with the
          highest ``nfilesfailed``, sorted descending.

        Returns an empty dict if ``payload`` contains no ``datasets`` list.
    """
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        return {}
    sc: Counter[str] = Counter()
    totals: dict[str, int] = {"failed": 0, "finished": 0, "waiting": 0, "missing": 0}
    key_map: dict[str, str] = {
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
