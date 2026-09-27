# Fine-Tuned Text-to-SQL Engine (LoRA/QLoRA)

Fine-tuning a small open-weight model on Spider to outperform its own zero-shot
baseline on text-to-SQL generation, with a quantified before/after comparison
via an execution-accuracy eval harness.

Full project plan: see `docs/PRD.md`, `docs/phase-0-tasks.md`,
`docs/phase-1-tasks.md`, and `docs/phase-2-tasks.md`.

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

**Phase 2 (Iterate on Data & Hyperparameters) — complete.**

- Scaled to the full Spider training pool (6,440 examples) plus targeted
  oversampling of the two patterns Phase 1 left completely unmoved
  (self-joins, NOT-IN/set-exclusion), for a 9,380-example augmented set
- Swept 3 LoRA hyperparameter variants, one axis changed at a time off
  Phase 1's baseline: scale (more epochs), capacity (`r=32/alpha=64`), and
  stability (lower learning rate)

| Run | Dataset | LoRA | Epochs | LR | Exec. Acc. | Exact Match |
|---|---|---|---|---|---|---|
| Phase 1 | 1,455 | r=16/a=32 | 1 | 2e-4 | 75.5% | 38.0% |
| Variant (a) — scale | 9,380 | r=16/a=32 | 2 | 2e-4 | **72.5%** | **43.5%** |
| Variant (b) — rank | 9,380 | r=32/a=64 | 2 | 2e-4 | 73.0% | 46.0% |
| Variant (c) — low LR | 9,380 | r=16/a=32 | 2 | 5e-5 | 66.5% | 42.5% |

- **Selected checkpoint: variant (a)** — chosen over the marginally
  higher-scoring variant (b) (a 0.5pt aggregate gap, within the test set's
  noise margin) because the failure-pattern review found (a) fixed 3 of
  Phase 1's 4 named hard patterns (self-joins, NOT-IN, hallucinated joins)
  vs. (b)'s 2, with (b) regressing on NOT-IN via a hallucinated schema
  column name that recurred elsewhere in a broader failure sample
- Set-operations remain unresolved in every variant — each now produces a
  real `UNION`/`INTERSECT` shell (an improvement over Phase 1) but breaks
  the second branch's logic differently each time; flagged as an open item
  for Phase 3, not silently dropped
- A new, previously unflagged pattern surfaced during error analysis:
  `model_list`/`car_names` table confusion on car/model questions —
  documented as open, not yet resolved
- Adapter pushed to Hugging Face Hub — see `models/model_card.md` for the
  link, full config, and selection rationale

See `results/phase2_summary.md` for the full write-up,
`results/experiment-log.md` for the complete run-by-run comparison table,
and `configs/phase2_variants.py` for the exact variant configs.

## Repo layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── PRD.md
│   ├── phase-0-tasks.md
│   ├── phase-1-tasks.md
│   └── phase-2-tasks.md
├── configs/
│   └── phase2_variants.py        # Phase 2's 3 hyperparameter variants + final results
├── notebooks/
│   ├── phase0_setup.ipynb        # Phase 0: pipeline, harness, zero-shot baseline
│   ├── phase1_finetuning.ipynb   # Phase 1: LoRA config, training, eval, comparison
│   └── phase2_iteration.ipynb    # Phase 2: full-dataset sweep, curation, checkpoint selection
├── data/                          # NOT committed — see .gitignore and data/README.md
│   └── README.md
├── models/
│   └── model_card.md             # Final Phase 2 adapter — HF Hub link, config, rationale
└── results/
    ├── phase0_summary.md
    ├── baseline_zero_shot_results.json
    ├── phase1_summary.md
    ├── finetuned_v1_eval_results.json
    ├── phase2_summary.md
    ├── experiment-log.md
    ├── phase2_a_epoch1_eval_results.json
    ├── phase2_a_final_eval_results.json
    ├── phase2_a_final_recovered_eval_results.json
    ├── phase2_b_epoch1_eval_results.json
    ├── phase2_b_final_eval_results.json
    ├── phase2_c_epoch1_eval_results.json
    └── phase2_c_final_eval_results.json
```

(LoRA adapter weights and training checkpoints themselves — Phase 1's
adapter and Phase 2's per-variant/per-epoch checkpoints — are **not**
committed; see `.gitignore`. Phase 2's final adapter lives on Hugging Face
Hub, linked from `models/model_card.md`.)

All project code — schema serialization, prompt building, the SQLite
execution harness, LoRA/training config, and the comparison/eval functions —
lives directly in each phase's notebook rather than being split into a
separate package. This keeps each phase readable top-to-bottom as a single,
self-contained artifact. Each phase's notebook re-defines the prior phase's
core functions at the top (fresh Colab runtimes don't carry state between
sessions) before adding its own. Key functions:

- `schema_to_text(db_id, schema_lookup)` / `build_prompt(schema_text, question, gold_sql=None)`
- `get_db_connection(db_id)` / `run_sql(conn, sql_string)`
- `execution_match(conn, generated_sql, gold_sql)` / `exact_match(generated_sql, gold_sql)`
- `run_eval_harness(test_examples, generate_fn, results_path)`
- `generate_sql(question, schema_text, model, tokenizer)` — the swappable
  inference call; same signature whether `model` is the raw base model, a
  LoRA-adapted checkpoint, or (in later phases) a quantized GGUF version
- `check_failure_patterns(model, tokenizer, label)` — added in Phase 2;
  reruns Phase 1's four named hard examples against any checkpoint for a
  fixed/unchanged/broken-differently read, since the aggregate accuracy
  alone repeatedly proved insufficient to judge whether fine-tuning
  actually helped on the patterns that matter
- `log_experiment_row(config, exec_acc, exact_match, notes)` — added in
  Phase 2; appends a run to both `results/experiment-log.md` and a W&B
  Table so every sweep run lands in the comparison table automatically

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

Phase 2 onward also needs a Hugging Face account with write access, for
pushing the final adapter to the Hub (`huggingface_hub`, already in
`requirements.txt`).

## Usage

Open the relevant notebook in Colab or Jupyter and run top to bottom:

- `notebooks/phase0_setup.ipynb` — environment setup, model selection,
  preprocessing, eval harness, zero-shot baseline
- `notebooks/phase1_finetuning.ipynb` — LoRA config, training, adapter
  reload sanity check, formal eval on the identical baseline test subset,
  before/after comparison table, failure-pattern review
- `notebooks/phase2_iteration.ipynb` — full-dataset environment, targeted
  self-join/NOT-IN curation, running experiment comparison table, 3-variant
  hyperparameter sweep (each checkpointed to Drive and resumable across
  Colab disconnects), failure-pattern error analysis, final checkpoint
  selection and Hugging Face Hub push

`run_eval_harness` takes any `generate_fn` with the signature
`(question: str, schema_text: str) -> str`, so swapping in a new checkpoint
(fine-tuned, a hyperparameter-sweep variant, or eventually a quantized
version) never requires touching the harness code itself — only the
`generate_fn` passed into it changes between phases.

## Roadmap

- [x] Phase 0 — Setup & Baseline
- [x] Phase 1 — First Fine-Tune
- [x] Phase 2 — Iterate on Data & Hyperparameters
- [ ] Phase 3 — Merge, Quantize, Benchmark
- [ ] Phase 4 — Serve It
- [ ] Phase 5 — Polish & Integrate