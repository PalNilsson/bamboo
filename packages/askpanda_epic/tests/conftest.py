"""Pytest configuration for the askpanda_epic plugin test suite.

These tests are designed to work both when the packages are installed
(editable or wheel) and when running directly from a source checkout.

In a clean checkout, ``bamboo`` (core) lives under ``core/`` and
``askpanda_epic`` lives under ``packages/askpanda_epic/``.  Both are
added to ``sys.path`` so pytest can import them without requiring an
editable install.
"""
from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    """Configure sys.path for local-source test runs."""
    # Resolve from packages/askpanda_epic/tests/ → repo root
    repo_root = Path(__file__).resolve().parents[3]

    core_dir = repo_root / "core"
    epic_pkg_dir = repo_root / "packages" / "askpanda_epic"

    for p in (core_dir, epic_pkg_dir):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
