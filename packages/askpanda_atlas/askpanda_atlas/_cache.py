"""In-process TTL cache for BigPanDA HTTP responses.

Prevents redundant downloads within a session by caching the raw responses
returned by :func:`~askpanda_atlas._fallback_http.fetch_jsonish` and log
text returned by ``requests.get``.

TTL policy
----------
- Task and job **metadata** (``/jobs/``, ``/job?pandaid=``): 60 seconds.
  These may change while the process is running (jobs start, finish, fail),
  but polling more frequently than once per minute is wasteful.
- **Log files** (``/filebrowser/``): infinite TTL (``math.inf``).
  Once a pilot or payload log exists it is immutable.  There is never a
  reason to re-download it during the same process lifetime.

Thread safety
-------------
All cache operations are protected by a :class:`threading.Lock` so the
cache is safe for concurrent use from ``asyncio.to_thread`` workers.

Usage
-----
Replace direct calls to :func:`~askpanda_atlas._fallback_http.fetch_jsonish`
and ``requests.get`` with the cached wrappers::

    from askpanda_atlas._cache import cached_fetch_jsonish, cached_fetch_log

    # Metadata (60-second TTL)
    status, ctype, body, payload = cached_fetch_jsonish(url, timeout)

    # Log text (infinite TTL)
    text = cached_fetch_log(url, timeout)
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METADATA_TTL: float = 60.0   # seconds — task and job metadata
LOG_TTL: float = math.inf     # logs are immutable once written

# ---------------------------------------------------------------------------
# Internal store
# ---------------------------------------------------------------------------

_lock: threading.Lock = threading.Lock()

# key → (expiry_timestamp, value)
# expiry_timestamp == math.inf means the entry never expires.
_store: dict[str, tuple[float, Any]] = {}


# ---------------------------------------------------------------------------
# Core cache primitives
# ---------------------------------------------------------------------------


def _get(key: str) -> Any:
    """Return cached value for *key*, or ``_MISS`` if absent or expired.

    Args:
        key: Cache key (typically a URL string).

    Returns:
        Cached value, or the sentinel :data:`_MISS`.
    """
    with _lock:
        entry = _store.get(key)
    if entry is None:
        return _MISS
    expiry, value = entry
    if expiry != math.inf and time.monotonic() > expiry:
        with _lock:
            _store.pop(key, None)
        return _MISS
    return value


def _set(key: str, value: Any, ttl: float) -> None:
    """Store *value* under *key* with the given *ttl* in seconds.

    Args:
        key: Cache key.
        value: Value to store.
        ttl: Time-to-live in seconds.  ``math.inf`` means never expire.
    """
    expiry = math.inf if ttl == math.inf else time.monotonic() + ttl
    with _lock:
        _store[key] = (expiry, value)


class _MissType:
    """Sentinel singleton used to distinguish a missing cache entry from ``None``."""

    _instance: "_MissType | None" = None

    def __new__(cls) -> "_MissType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<MISS>"


_MISS = _MissType()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cached_fetch_jsonish(
    url: str,
    timeout: int = 30,
    ttl: float = METADATA_TTL,
) -> tuple[int, str, str, dict[str, Any] | None]:
    """Fetch a URL via fetch_jsonish, returning the cached result on repeat calls.

    On a cache hit, returns the previously fetched 4-tuple without making
    an HTTP request.  On a miss, delegates to the real ``fetch_jsonish``,
    stores the result, and returns it.

    Args:
        url: URL to fetch.
        timeout: HTTP timeout in seconds (only used on a cache miss).
        ttl: Time-to-live in seconds for this entry.  Defaults to
            :data:`METADATA_TTL` (60 s).  Pass ``math.inf`` for
            responses that should never expire (e.g. log files served
            through this wrapper).

    Returns:
        4-tuple ``(status_code, content_type, body_text, parsed_json_or_none)``
        as returned by ``fetch_jsonish``.
    """
    cached = _get(url)
    if cached is not _MISS:
        return cached  # type: ignore[return-value]

    from askpanda_atlas._fallback_http import fetch_jsonish  # type: ignore[import]

    result = fetch_jsonish(url, timeout)
    _set(url, result, ttl)
    return result


def cached_fetch_log(
    url: str,
    timeout: int = 60,
) -> str | None:
    """Fetch a log file via requests.get, returning the cached result on repeat calls.

    Log files are immutable once written, so hits are cached with
    :data:`LOG_TTL` (``math.inf``) — they are never re-downloaded within
    the same process lifetime.

    Args:
        url: Full URL of the log file (filebrowser endpoint).
        timeout: HTTP timeout in seconds (only used on a cache miss).

    Returns:
        Log text as a string, or ``None`` if the file is not found or
        the download fails.
    """
    cached = _get(url)
    if cached is not _MISS:
        return cached  # type: ignore[return-value]

    import logging

    import requests  # type: ignore[import]

    _logger = logging.getLogger(__name__)

    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "AskPanDA/1.0"},
            stream=True,
        )
        if resp.status_code == 404:
            _logger.info("Log file not found (404): %s", url)
            result: str | None = None
        else:
            resp.raise_for_status()
            result = resp.text
    except requests.RequestException as exc:
        _logger.warning("Log download failed for %s: %s", url, exc)
        result = None

    # Cache even None so we don't hammer a 404 endpoint.
    _set(url, result, LOG_TTL)
    return result


def invalidate(url: str) -> None:
    """Remove a single URL from the cache.

    Useful in tests or when a caller knows a resource has changed.

    Args:
        url: URL key to evict.
    """
    with _lock:
        _store.pop(url, None)


def clear() -> None:
    """Evict all entries from the cache.

    Primarily intended for tests and the ``/clear`` TUI command.
    """
    with _lock:
        _store.clear()


def stats() -> dict[str, Any]:
    """Return a snapshot of cache statistics for diagnostics.

    Returns:
        Dict with ``"entries"`` (count), ``"urls"`` (sorted list of keys),
        and ``"expired"`` (count of entries past their TTL but not yet
        evicted).
    """
    now = time.monotonic()
    with _lock:
        items = list(_store.items())
    expired = sum(1 for _, (exp, _) in items if exp != math.inf and now > exp)
    return {
        "entries": len(items),
        "expired": expired,
        "urls": sorted(k for k, _ in items),
    }
