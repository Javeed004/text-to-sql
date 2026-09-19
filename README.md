# Fine-Tuned Text-to-SQL Engine (LoRA/QLoRA)

Fine-tuning a small open-weight model on Spider to outperform its own zero-shot
baseline on text-to-SQL generation, with a quantified before/after comparison
via an execution-accuracy eval harness.

Full project plan: see `docs/PRD.md` and `docs/phase-0-tasks.md`.

## Status

**Phase 0 (Setup & Baseline) — complete.**

- Base model: `unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit`
- Dataset: Spider (train/val/test splits, dev set held out as test)
- Zero-shot baseline execution accuracy: **61.0%** (122/200, fixed-seed subset)

See `results/phase0_summary.md` for the full write-up, and
`results/baseline_zero_shot_results.json` for per-example results.

## Repo layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── PRD.md
│   └── phase-0-tasks.md
├── notebooks/
│   └── phase0_setup.ipynb        # all pipeline, harness, and eval code lives here
├── data/                          # NOT committed — see .gitignore
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
└── results/
    ├── phase0_summary.md
    └── baseline_zero_shot_results.json
```

All project code — schema serialization, prompt building, the SQLite
execution harness, and the comparison/eval functions — lives directly in
`notebooks/phase0_setup.ipynb` (and its Phase 1+ successors) rather than
being split into a separate package. This keeps each phase's notebook
readable top-to-bottom as a single, self-contained artifact. Key functions
defined in the notebook:

- `schema_to_text(db_id, schema_lookup)` / `build_prompt(schema_text, question, gold_sql=None)`
- `get_db_connection(db_id)` / `run_sql(conn, sql_string)`
- `execution_match(conn, generated_sql, gold_sql)` / `exact_match(generated_sql, gold_sql)`
- `run_eval_harness(test_examples, generate_fn, results_path)`

## Setup

```bash
pip install -r requirements.txt
```

Spider data (question/SQL/schema) loads via Hugging Face `datasets`
(`xlangai/spider`). The per-database `.sqlite` files are pulled separately
via Git LFS from a community mirror (`minktn/spider-data`) since the
official dataset repo only ships parquet:

```bash
git lfs install
git lfs clone https://huggingface.co/datasets/minktn/spider-data
```

## Usage

Open `notebooks/phase0_setup.ipynb` in Colab or Jupyter and run top to
bottom. `run_eval_harness` takes any `generate_fn` with the signature
`(question: str, schema_text: str) -> str`, so a base model, a fine-tuned
checkpoint, or a quantized version can all be dropped in without touching
the harness code — each later phase's notebook re-defines or re-imports
these same cells before swapping in its own `generate_fn`.

## Roadmap

- [x] Phase 0 — Setup & Baseline
- [ ] Phase 1 — First Fine-Tune
- [ ] Phase 2 — Iterate on Data & Hyperparameters
- [ ] Phase 3 — Merge, Quantize, Benchmark
- [ ] Phase 4 — Serve It
- [ ] Phase 5 — Polish & Integrate