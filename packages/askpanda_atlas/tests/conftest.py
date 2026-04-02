"""Pytest configuration for the askpanda_atlas plugin test suite.

These tests are designed to work both when the packages are installed
(editable or wheel) and when running directly from a source checkout.

In a clean checkout, ``bamboo`` (core) lives under ``core/`` and
``askpanda_atlas`` lives under ``packages/askpanda_atlas/``.  Both are
added to ``sys.path`` so pytest can import them without requiring an
editable install.
"""
from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    """Configure sys.path for local-source test runs."""
    # Resolve from packages/askpanda_atlas/tests/ → repo root
    repo_root = Path(__file__).resolve().parents[3]

    core_dir = repo_root / "core"
    atlas_pkg_dir = repo_root / "packages" / "askpanda_atlas"

    for p in (core_dir, atlas_pkg_dir):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
