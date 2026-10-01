"""Safe execution of model-generated SQL against a throwaway SQLite DB (Phase 4, Task 4).

Layers of protection (this is NOT production hardening, see the PRD non-goals):
  1. A fresh :memory: database per call, built from the caller's schema.
  2. The schema DDL runs with an authorizer that blocks ATTACH/DETACH/PRAGMA,
     so a pasted schema cannot touch the filesystem.
  3. PRAGMA query_only = ON plus an authorizer that only allows reads
     (SELECT / READ / FUNCTION / RECURSIVE) while the generated SQL runs.
  4. Only statements starting with SELECT or WITH are accepted, and sqlite3
     refuses multi-statement strings.
  5. A progress handler aborts queries that run past the time limit.
  6. Returned rows are capped.
"""
import re
import sqlite3
import time
from typing import Any

# Numeric action codes from the SQLite C API (stable across versions).
_SQLITE_OK = 0
_SQLITE_DENY = 1
_SQLITE_PRAGMA = 19
_SQLITE_READ = 20
_SQLITE_SELECT = 21
_SQLITE_ATTACH = 24
_SQLITE_DETACH = 25
_SQLITE_FUNCTION = 31
_SQLITE_RECURSIVE = 33

_READ_ONLY_ACTIONS = {_SQLITE_READ, _SQLITE_SELECT, _SQLITE_FUNCTION, _SQLITE_RECURSIVE}
_SCHEMA_BLOCKED_ACTIONS = {_SQLITE_ATTACH, _SQLITE_DETACH, _SQLITE_PRAGMA}

_SELECT_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


def _schema_authorizer(action, *_):
    return _SQLITE_DENY if action in _SCHEMA_BLOCKED_ACTIONS else _SQLITE_OK


def _read_only_authorizer(action, *_):
    return _SQLITE_OK if action in _READ_ONLY_ACTIONS else _SQLITE_DENY


def _json_safe(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    return value


def _result(columns=None, rows=None, truncated=False, error=None) -> dict:
    return {
        "columns": columns or [],
        "rows": rows or [],
        "truncated": truncated,
        "error": error,
    }


def run_sql(schema_ddl: str, sql: str, max_rows: int = 50, timeout_s: float = 3.0) -> dict:
    """Build a DB from schema_ddl, run one read-only query, return a JSON-safe dict."""
    if not _SELECT_RE.match(sql or ""):
        return _result(error="Only SELECT queries are allowed")

    conn = sqlite3.connect(":memory:")
    try:
        # Phase 1: build the schema (may include INSERT rows for sample data).
        conn.set_authorizer(_schema_authorizer)
        try:
            conn.executescript(schema_ddl)
        except sqlite3.Error as e:
            return _result(error=f"Invalid schema: {e}")

        # Phase 2: lock the connection down before running generated SQL.
        conn.set_authorizer(None)
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(_read_only_authorizer)

        deadline = time.monotonic() + timeout_s
        # Called every 10,000 VM instructions; returning non-zero aborts the query.
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)

        try:
            cur = conn.execute(sql)
            fetched = cur.fetchmany(max_rows + 1)
            columns = [d[0] for d in cur.description] if cur.description else []
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e).lower():
                return _result(error=f"Query exceeded the {timeout_s:g}s time limit")
            return _result(error=str(e))
        except sqlite3.Error as e:
            return _result(error=str(e))

        truncated = len(fetched) > max_rows
        rows = [[_json_safe(v) for v in row] for row in fetched[:max_rows]]
        return _result(columns=columns, rows=rows, truncated=truncated)
    finally:
        conn.close()