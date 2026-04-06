#!/usr/bin/env python3
"""Quick diagnostic: inspect the CRIC DuckDB file and report its schema.

Usage:
    python probe_cric_db.py /path/to/cric.db
    # or reads CRIC_DUCKDB_PATH env var if no argument given
"""
import os
import sys

try:
    import duckdb
except ImportError:
    print("ERROR: duckdb not installed. Run: pip install duckdb")
    sys.exit(1)

path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CRIC_DUCKDB_PATH", "cric.duckdb")
print(f"DB path: {path}")

if not os.path.exists(path):
    print("ERROR: file not found")
    sys.exit(1)

print(f"File size: {os.path.getsize(path):,} bytes")
print()

try:
    conn = duckdb.connect(path, read_only=True)
except Exception as e:
    print(f"ERROR opening DB (read-only): {e}")
    print("Trying read-write...")
    try:
        conn = duckdb.connect(path, read_only=False)
    except Exception as e2:
        print(f"ERROR opening DB (read-write): {e2}")
        sys.exit(1)

# List tables
try:
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchall()
    print(f"Tables ({len(tables)}):")
    for (t,) in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            cols = [d[0] for d in conn.execute(f"SELECT * FROM {t} LIMIT 0").description or []]
            print(f"  {t}: {count:,} rows, columns: {', '.join(cols)}")
        except Exception as e:
            print(f"  {t}: ERROR reading ({e})")
except Exception as e:
    print(f"ERROR listing tables: {e}")

conn.close()
