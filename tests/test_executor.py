from app.executor import run_sql

SCHEMA = """
CREATE TABLE singer (singer_id number PRIMARY KEY, name text, age number);
INSERT INTO singer VALUES (1, 'Ann', 25), (2, 'Bob', 35), (3, 'Cy', 41);
"""


def test_valid_query():
    r = run_sql(SCHEMA, "SELECT name FROM singer WHERE age > 30 ORDER BY name")
    assert r["error"] is None
    assert r["columns"] == ["name"]
    assert r["rows"] == [["Bob"], ["Cy"]]
    assert r["truncated"] is False


def test_syntax_error():
    r = run_sql(SCHEMA, "SELECT FROM WHERE")
    assert r["error"]
    assert r["rows"] == []


def test_drop_table_rejected():
    r = run_sql(SCHEMA, "DROP TABLE singer")
    assert r["error"]
    # A stacked statement must not sneak through either.
    r2 = run_sql(SCHEMA, "SELECT 1; DROP TABLE singer")
    assert r2["error"]


def test_timeout_on_runaway_cte():
    sql = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
        "SELECT count(*) FROM c"
    )
    r = run_sql(SCHEMA, sql, timeout_s=1)
    assert r["error"] and "time limit" in r["error"]


def test_row_truncation():
    sql = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 100) "
        "SELECT x FROM c"
    )
    r = run_sql(SCHEMA, sql, max_rows=50)
    assert r["error"] is None
    assert len(r["rows"]) == 50
    assert r["truncated"] is True