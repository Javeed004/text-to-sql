"""Command-line client for the Text-to-SQL API.

Examples:
    python cli.py --example 0 --url http://127.0.0.1:7860
    python cli.py --schema schema.sql --data data.sql --question "How many singers are there?"
"""
import argparse
import sys

import httpx


def print_table(columns, rows):
    cells = [[("NULL" if v is None else str(v)) for v in r] for r in rows]
    widths = [
        max([len(c)] + [len(r[i]) for r in cells]) for i, c in enumerate(columns)
    ]
    line = "-+-".join("-" * w for w in widths)
    print(" | ".join(c.ljust(w) for c, w in zip(columns, widths)))
    print(line)
    for r in cells:
        print(" | ".join(v.ljust(w) for v, w in zip(r, widths)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--schema", help="path to a file with CREATE TABLE statements")
    ap.add_argument("--data", help="optional path to a file with INSERT statements (execution only)")
    ap.add_argument("--question")
    ap.add_argument("--example", type=int, help="use built-in example N from the server")
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()

    try:
        if args.example is not None:
            examples = httpx.get(f"{args.url}/examples", timeout=10).json()
            ex = examples[args.example]
            schema, data, question = ex["schema"], ex["data"], args.question or ex["question"]
        else:
            if not (args.schema and args.question):
                ap.error("provide --schema and --question, or --example N")
            schema = open(args.schema, encoding="utf-8").read()
            data = open(args.data, encoding="utf-8").read() if args.data else ""
            question = args.question

        print(f"Question: {question}\n(this can take a while on a slow CPU...)")
        resp = httpx.post(
            f"{args.url}/generate",
            json={"schema": schema, "data_sql": data, "question": question},
            timeout=args.timeout,
        )
    except httpx.ConnectError:
        sys.exit(f"Could not connect to {args.url}. Is the API running?")
    except httpx.TimeoutException:
        sys.exit("Request timed out.")
    except (IndexError, FileNotFoundError) as e:
        sys.exit(f"Input error: {e}")

    if resp.status_code != 200:
        sys.exit(f"API error {resp.status_code}: {resp.text}")

    body = resp.json()
    print(f"\nSQL ({body['latency_ms'] / 1000:.1f}s):\n  {body['sql']}\n")
    result = body.get("execution_result")
    if not result:
        return
    if result["error"]:
        print(f"Execution error: {result['error']}")
    else:
        print_table(result["columns"], result["rows"])
        print(f"\n{len(result['rows'])} row(s)" + (" (truncated)" if result["truncated"] else ""))


if __name__ == "__main__":
    main()