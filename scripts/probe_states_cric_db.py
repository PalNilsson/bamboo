#!/usr/bin/env python3
"""Probe the actual distinct state values in the CRIC queuedata table."""
import os
import sys
try:
    import duckdb
except ImportError:
    print("pip install duckdb")
    sys.exit(1)

path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CRIC_DUCKDB_PATH", "cric.duckdb")
conn = duckdb.connect(path, read_only=True)

print("=== state values ===")
for row in conn.execute("SELECT state, COUNT(*) AS n FROM queuedata GROUP BY state ORDER BY n DESC").fetchall():
    print(f"  {row[0]!r:20} n={row[1]}")

print()
print("=== status values ===")
for row in conn.execute("SELECT status, COUNT(*) AS n FROM queuedata GROUP BY status ORDER BY n DESC").fetchall():
    print(f"  {row[0]!r:20} n={row[1]}")

print()
print("=== sample of non-ACTIVE state rows ===")
rows = conn.execute(
    "SELECT queue, atlas_site, state, status FROM queuedata "
    "WHERE state NOT IN ('ACTIVE','active') LIMIT 10"
).fetchall()
if rows:
    for r in rows:
        print(f"  {r}")
else:
    print("  (none found with state != ACTIVE)")

print()
print("=== sample of non-ACTIVE status rows ===")
rows = conn.execute(
    "SELECT queue, atlas_site, state, status FROM queuedata "
    "WHERE status NOT IN ('ACTIVE','active') LIMIT 10"
).fetchall()
if rows:
    for r in rows:
        print(f"  {r}")
else:
    print("  (none found with status != ACTIVE)")

conn.close()
