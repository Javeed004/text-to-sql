# Fine-Tuned Text-to-SQL Engine (LoRA/QLoRA)

Fine-tuning a small open-weight model on Spider to outperform its own zero-shot
baseline on text-to-SQL generation, with a quantified before/after comparison
via an execution-accuracy eval harness.

Full project plan: see `docs/PRD.md`, `docs/phase-0-tasks.md`, and
`docs/phase-1-tasks.md`.

## Status

**Phase 0 (Setup & Baseline) — complete.**

- Base model: `unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit`
- Dataset: Spider (train/val/test splits, dev set held out as test)
- Zero-shot baseline execution accuracy: **61.0%** (122/200, fixed-seed subset)
- Zero-shot baseline exact match: **10.5%**

See `results/phase0_summary.md` for the full write-up, and
`results/baseline_zero_shot_results.json` for per-example results.

**Phase 1 (First Fine-Tune) — complete.**

- QLoRA fine-tune: `r=16`, `lora_alpha=32`, all 7 attention+MLP projections,
  1 epoch over 1,455 filtered Spider training examples, ~13.3 min on a T4
- Execution accuracy: 61.0% → **75.5%** (+14.5 pts)
- Exact match: 10.5% → **38.0%** (+27.5 pts)
- Failure-pattern review: 1 of Phase 0's 4 named hard patterns
  (hallucinated joins) clearly fixed; self-joins and `!=`/`NOT IN` confusion
  unchanged; set-operations-as-JOIN partially improved with a newly
  introduced join artifact — full breakdown in `results/phase1_summary.md`
- Adapter checkpoint: `qwen2.5-coder-3b-lora-r16-a32-fastpass-v1`
  (saved to Drive; not committed to this repo — see `.gitignore`)

See `results/phase1_summary.md` for the full write-up, and
`results/finetuned_v1_eval_results.json` for per-example results.

## Repo layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── PRD.md
│   ├── phase-0-tasks.md
│   └── phase-1-tasks.md
├── notebooks/
│   ├── phase0_setup.ipynb        # Phase 0: pipeline, harness, zero-shot baseline
│   └── phase1_finetuning.ipynb   # Phase 1: LoRA config, training, eval, comparison
├── data/                          # NOT committed — see .gitignore
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── adapters/                      # NOT committed — see .gitignore
│   └── qwen2.5-coder-3b-lora-r16-a32-fastpass-v1/
└── results/
    ├── phase0_summary.md
    ├── baseline_zero_shot_results.json
    ├── phase1_summary.md
    ├── finetuned_v1_eval_results.json
    └── phase0_vs_phase1_comparison.csv
```

All project code — schema serialization, prompt building, the SQLite
execution harness, LoRA/training config, and the comparison/eval functions —
lives directly in each phase's notebook rather than being split into a
separate package. This keeps each phase readable top-to-bottom as a single,
self-contained artifact. Phase 1's notebook re-defines Phase 0's core
functions at the top (fresh Colab runtimes don't carry state between
sessions) before adding its own. Key functions:

- `schema_to_text(db_id, schema_lookup)` / `build_prompt(schema_text, question, gold_sql=None)`
- `get_db_connection(db_id)` / `run_sql(conn, sql_string)`
- `execution_match(conn, generated_sql, gold_sql)` / `exact_match(generated_sql, gold_sql)`
- `run_eval_harness(test_examples, generate_fn, results_path)`
- `generate_sql(question, schema_text, model, tokenizer)` — the swappable
  inference call; same signature whether `model` is the raw base model, a
  LoRA-adapted checkpoint, or (in later phases) a quantized GGUF version

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

Phase 1 onward also needs a Weights & Biases account for experiment
tracking. Store your API key as a Colab secret named `WANDB_API_KEY`
(never hardcode it) — the notebook reads it via
`google.colab.userdata.get("WANDB_API_KEY")`.

## Usage

Open the relevant notebook in Colab or Jupyter and run top to bottom:

- `notebooks/phase0_setup.ipynb` — environment setup, model selection,
  preprocessing, eval harness, zero-shot baseline
- `notebooks/phase1_finetuning.ipynb` — LoRA config, training, adapter
  reload sanity check, formal eval on the identical baseline test subset,
  before/after comparison table, failure-pattern review

`run_eval_harness` takes any `generate_fn` with the signature
`(question: str, schema_text: str) -> str`, so swapping in a new checkpoint
(fine-tuned, later a hyperparameter-sweep variant, or eventually a quantized
version) never requires touching the harness code itself — only the
`generate_fn` passed into it changes between phases.

## Roadmap

- [x] Phase 0 — Setup & Baseline
- [x] Phase 1 — First Fine-Tune
- [ ] Phase 2 — Iterate on Data & Hyperparameters
- [ ] Phase 3 — Merge, Quantize, Benchmark
- [ ] Phase 4 — Serve It
- [ ] Phase 5 — Polish & Integrate