#!/usr/bin/env python3
"""Probe actual atlas_site values in the CRIC queuedata table."""
import os
import sys
try:
    import duckdb
except ImportError:
    print("pip install duckdb")
    sys.exit(1)

path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CRIC_DUCKDB_PATH", "cric.duckdb")
conn = duckdb.connect(path, read_only=True)

print("=== atlas_site values (all) ===")
for row in conn.execute(
    "SELECT atlas_site, COUNT(*) AS n FROM queuedata GROUP BY atlas_site ORDER BY atlas_site"
).fetchall():
    print(f"  {row[0]!r:<35} n={row[1]}")

print()
print("=== BNL-related sites ===")
for row in conn.execute(
    "SELECT DISTINCT atlas_site FROM queuedata WHERE atlas_site ILIKE '%bnl%' ORDER BY atlas_site"
).fetchall():
    print(f"  {row[0]!r}")

print()
print("=== sample queues with atlas_site containing 'BNL' ===")
for row in conn.execute(
    "SELECT queue, atlas_site, status FROM queuedata WHERE atlas_site ILIKE '%bnl%' LIMIT 10"
).fetchall():
    print(f"  queue={row[0]!r:<30} atlas_site={row[1]!r:<20} status={row[2]!r}")

conn.close()
