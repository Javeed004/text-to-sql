
# scripts/compare_outputs.py

import sys
from pathlib import Path
import json
import httpx
from tqdm import tqdm

# Add the project root to Python's module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.prompt import build_messages, render_chatml, clean_sql

# Load test data
with open("data/test.jsonl", "r", encoding="utf-8") as f:
    tests = {
        json.loads(line)["question"]: json.loads(line)
        for line in f
    }

# Load Phase 3 reference outputs
with open("results/phase3_Q4_K_M_eval_results.json", "r", encoding="utf-8") as f:
    ref = json.load(f)["per_example"][:20]

same = 0

# Compare outputs with progress bar
for r in tqdm(ref, desc="Comparing outputs", unit="test", dynamic_ncols=True):
    t = tests[r["question"]]

    prompt = render_chatml(
        build_messages(t["schema_text"], t["question"])
    )

    resp = httpx.post(
        "http://localhost:8080/completion",
        json={
            "prompt": prompt,
            "temperature": 0,
            "n_predict": 256
        },
        timeout=120
    ).json()

    out = clean_sql(resp["content"])

    if out.strip() == r["generated_sql"].strip():
        same += 1
    else:
        tqdm.write(
            f"\nDIFF: {r['question']}\n"
            f"served: {out}\n"
            f"phase3: {r['generated_sql']}\n"
        )

# Final summary
print(f"\n{same}/{len(ref)} identical")
print(f"Match rate: {same / len(ref) * 100:.2f}%")