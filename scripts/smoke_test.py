
"""Milestone 5 smoke test: send Spider dev examples through the running API.

Usage (llama-server on 8080 and uvicorn on 8000 both running, repo root):
    python -m scripts.smoke_test
    python -m scripts.smoke_test --n 25 --spider-db-root path/to/spider_data/database

Reports:
  - agreement with Phase 3's Q4_K_M outputs
  - execution accuracy vs gold through the served path
  - end-to-end p50/p95 latency, and API generation-only latency
"""
import argparse
import json
import math
import os
import sqlite3
import statistics
import time

import httpx
from tqdm import tqdm

PHASE3_RESULTS = "results/phase3_Q4_K_M_eval_results.json"
TEST_JSONL = "data/test.jsonl"


def norm(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").split()).lower()


def run_rows(db_root, db_id, sql):
    """Run sql against the real Spider DB (read-only). Returns a set of rows or None on error."""
    path = os.path.join(db_root, db_id, f"{db_id}.sqlite")
    if not os.path.exists(path):
        return None

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return set(tuple(r) for r in conn.execute(sql).fetchall())
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, math.ceil(q * len(s)) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--spider-db-root", default=None)
    args = ap.parse_args()

    tests = {}
    with open(TEST_JSONL, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tests[row["question"]] = row

    with open(PHASE3_RESULTS, encoding="utf-8") as f:
        ref = json.load(f)["per_example"][:args.n]

    if not ref:
        print("No test examples found.")
        return

    # Fail loudly if the API is down.
    health = httpx.get(f"{args.url}/health", timeout=10)
    assert health.status_code == 200, (
        f"API not healthy: {health.status_code} {health.text}"
    )

    same_sql = same_exec = comparable = 0
    served_correct = phase3_correct = gold_comparable = 0
    e2e_ms, gen_ms = [], []

    # Progress bar
    pbar = tqdm(
        enumerate(ref, 1),
        total=len(ref),
        desc="Smoke testing",
        unit="example",
        dynamic_ncols=True,
    )

    for i, r in pbar:
        t = tests[r["question"]]

        t0 = time.perf_counter()

        resp = httpx.post(
            f"{args.url}/generate",
            json={
                "schema": t["schema_text"],
                "question": t["question"],
                "execute": False,
            },
            timeout=180,
        )

        resp.raise_for_status()

        elapsed = (time.perf_counter() - t0) * 1000
        e2e_ms.append(elapsed)

        body = resp.json()
        gen_ms.append(body["latency_ms"])
        served = body["sql"]

        if norm(served) == norm(r["generated_sql"]):
            same_sql += 1
        elif args.spider_db_root is None:
            tqdm.write(
                f"[{i}] DIFF {t['question']}\n"
                f"   served: {served}\n"
                f"   phase3: {r['generated_sql']}"
            )

        if args.spider_db_root:
            gold = run_rows(
                args.spider_db_root,
                t["db_id"],
                t["gold_sql"],
            )

            srv = run_rows(
                args.spider_db_root,
                t["db_id"],
                served,
            )

            p3 = run_rows(
                args.spider_db_root,
                t["db_id"],
                r["generated_sql"],
            )

            if gold is not None:
                gold_comparable += 1
                served_correct += srv == gold
                phase3_correct += p3 == gold

            if srv is not None and p3 is not None:
                comparable += 1
                same_exec += srv == p3

            if norm(served) != norm(r["generated_sql"]):
                tqdm.write(
                    f"[{i}] DIFF {t['question']}\n"
                    f"   served: {served}\n"
                    f"   phase3: {r['generated_sql']}\n"
                    f"   same rows: "
                    f"{srv == p3 if srv is not None and p3 is not None else 'n/a'}"
                    f"   served correct: "
                    f"{srv == gold if gold is not None else 'n/a'}"
                )

        # Update progress bar with live metrics.
        pbar.set_postfix({
            "SQL match": f"{same_sql}/{i}",
            "E2E": f"{elapsed:.0f}ms",
            "Gen": f"{body['latency_ms']:.0f}ms",
        })

    n = len(ref)

    print(f"\n=== Smoke test: {n} examples ===")
    print(f"Identical SQL vs Phase 3 Q4_K_M: {same_sql}/{n}")

    if args.spider_db_root:
        print(f"Same result rows vs Phase 3:     {same_exec}/{comparable}")
        print(f"Served exec accuracy vs gold:    {served_correct}/{gold_comparable}")
        print(
            f"Phase 3 exec accuracy (same {gold_comparable}): "
            f"{phase3_correct}/{gold_comparable}"
        )
    else:
        print(
            "(pass --spider-db-root to also compare result rows "
            "and accuracy vs gold)"
        )

    print(
        f"End-to-end latency ms: p50={statistics.median(e2e_ms):.0f}  "
        f"p95={pct(e2e_ms, 0.95):.0f}"
    )

    print(
        f"API generation ms:     p50={statistics.median(gen_ms):.0f}  "
        f"p95={pct(gen_ms, 0.95):.0f}"
    )


if __name__ == "__main__":
    main()