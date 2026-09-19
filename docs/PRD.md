PRD — Track B: Fine-Tuned Text-to-SQL Engine (LoRA/QLoRA)
Owner: Javeed Zulfikar Status: Draft v1 Related: Track A (Coding Agent) — fine-tuned model becomes a pluggable backend in Track A, same pattern as the multi-provider LLM routing in the RAG project.

1. Problem Statement
Calling a hosted LLM API to generate SQL from natural language proves you can prompt a model. It does not prove you can adapt one. Most "AI/ML Engineer" candidates never touch model weights — they only touch prompts. This project closes that gap by fine-tuning a small open-weight model specifically for text-to-SQL, using a task where correctness is objectively measurable (a generated query either executes and returns the right rows, or it doesn't), so the before/after improvement can be proven with numbers rather than asserted.

This also lets you directly reuse the LLM-evaluation methodology from your Kovai.Co internship (faithfulness/accuracy scoring), applied here to a fine-tuning context instead of a vendor-model context.

2. Goals
Fine-tune a small open LLM to outperform its own zero-shot baseline on text-to-SQL generation, with a quantified before/after comparison.
Build a repeatable, scriptable fine-tuning + eval pipeline (not a one-off notebook) that could plausibly be pointed at a different dataset/task later.
Ship a quantized, self-hosted version of the model that a real service (including Track A) can call over an API.
Produce a portfolio artifact with a clear "here's the number that moved, and here's why" story for interviews.
3. Non-Goals
Beating Spider leaderboard SOTA or handling arbitrary enterprise schemas.
Multi-turn conversational SQL (follow-up questions, clarification dialogue).
Full RLHF/DPO pipeline — LoRA/QLoRA supervised fine-tuning only.
Handling every SQL dialect — target SQLite/Postgres-compatible syntax only.
Production security hardening (query sandboxing against a real prod DB is out of scope; execution happens against disposable/sample DBs only).
4. Scope
In scope:

Base model: a small open model that fits free-tier GPU memory (candidates: Qwen2.5-Coder 1.5B/3B, Llama-3.2-3B, or CodeLlama-7B if quantized during training). Final pick happens in Phase 0 based on what fits comfortably in a free Colab/Kaggle T4/P100.
Dataset: Spider (or a curated subset) or WikiSQL — both free, well-known, and defensible in an interview.
Training method: QLoRA via Unsloth.
Eval: execution accuracy (does the generated SQL run and return the correct result set?) + exact-match as a secondary metric, measured before and after fine-tuning on a held-out test split.
Post-training: merge LoRA adapter, quantize to GGUF, serve via llama.cpp or Ollama behind a small FastAPI wrapper.
A minimal UI or CLI: paste a schema + a natural-language question → get generated SQL + execution result.
Out of scope (see Non-Goals).

5. Success Metrics
Metric	Baseline (zero-shot)	Target (post fine-tune)
Execution accuracy on held-out test set	measured in Phase 0	improve by a meaningful margin (aim: +15–30 pts, actual target set after seeing baseline)
Exact-match accuracy	measured in Phase 0	improve, secondary metric
Inference latency (pre-quantization vs GGUF)	measured in Phase 3	quantized version should be notably faster / lower memory, documented even if accuracy dips slightly
The exact target numbers get locked in after Phase 0's baseline run — no point promising a number before you've seen the starting point.

6. Architecture (high level)
[Spider/WikiSQL dataset]
        |
   preprocess (schema + question + gold SQL → instruction format)
        |
   Unsloth + QLoRA fine-tune on Colab/Kaggle GPU
        |
   eval harness (execution accuracy, exact match) ---> baseline vs fine-tuned comparison
        |
   merge adapter → quantize to GGUF
        |
   serve via llama.cpp/Ollama + FastAPI wrapper
        |
   CLI/simple UI  +  pluggable backend for Track A's coding agent
7. Tech Stack (all free)
Compute: Google Colab (free T4) or Kaggle Notebooks (free T4/P100, 30 hrs/week)
Fine-tuning: Unsloth, PEFT (LoRA), bitsandbytes (QLoRA), Hugging Face transformers + datasets
Dataset: Spider or WikiSQL (both freely licensed)
Eval: custom execution-accuracy harness (SQLite in-memory DBs), optionally RAGAS-style LLM-as-judge for SQL "reasonableness"
Serving: llama.cpp or Ollama, FastAPI wrapper
Experiment tracking: MLflow (self-hosted, free) or W&B free tier
Hosting: Hugging Face Hub (free model repo), Render/Railway free tier for the API
8. Phased Plan (work at your own pace — each phase gates the next)
Phase 0 — Setup & Baseline
Tasks:

Set up Colab/Kaggle environment, confirm GPU access and Unsloth install works.
Pick final base model based on what fits comfortably (VRAM headroom for training, not just inference).
Download and preprocess Spider (or WikiSQL): convert schema + question + gold SQL into an instruction-tuning format.
Split into train/val/test (use Spider's existing dev split as test if using Spider).
Build the execution-accuracy eval harness: run generated SQL against an in-memory SQLite DB built from each example's schema, compare result sets to gold.
Run the zero-shot baseline: base model, no fine-tuning, on the test set. Record execution accuracy + exact match.
Milestone/Deliverable: A working eval harness and a documented baseline number. Resume-ready achievement: "Built an automated SQL-execution-based evaluation harness to benchmark LLM text-to-SQL performance."

Phase 1 — First Fine-Tune
Tasks:

Configure LoRA hyperparameters (rank, alpha, target modules) via Unsloth.
Run first QLoRA fine-tuning pass on a subset of training data (fast iteration first, full dataset later).
Log training run (loss curves, config) in MLflow/W&B.
Run the same eval harness on this first fine-tuned checkpoint.
Compare against Phase 0 baseline — does it move at all? Sanity-check outputs manually for a handful of examples.
Milestone/Deliverable: First trained adapter + first before/after comparison (even if not optimal yet). Resume-ready achievement: "Fine-tuned [model] using QLoRA on the Spider dataset, achieving measurable improvement over zero-shot baseline on execution accuracy."

Phase 2 — Iterate on Data & Hyperparameters
Tasks:

Scale up to full training set (or a larger curated subset).
Try 2–3 hyperparameter variants (LoRA rank, learning rate, epochs) — track each in MLflow/W&B.
Add data quality passes if error analysis shows systematic failure patterns (e.g., JOIN-heavy queries failing more).
Re-run eval harness after each significant change; keep a running comparison table.
Pick the best-performing checkpoint as the "final" model for this project.
Milestone/Deliverable: A comparison table across experiments; a chosen best checkpoint with a clear improvement over baseline; push the adapter to Hugging Face Hub. Resume-ready achievement: "Ran systematic hyperparameter experiments tracked in MLflow, improving text-to-SQL execution accuracy from X% to Y%."

Phase 3 — Merge, Quantize, Benchmark
Tasks:

Merge the LoRA adapter into the base model weights.
Convert to GGUF format.
Quantize (try at least two quantization levels, e.g., Q4_K_M and Q8_0) and re-run the eval harness on quantized versions — does accuracy drop, and by how much?
Benchmark inference latency and memory footprint: full-precision vs quantized.
Milestone/Deliverable: A documented accuracy-vs-latency/memory tradeoff table across quantization levels. Resume-ready achievement: "Quantized fine-tuned model to GGUF, benchmarking accuracy/latency tradeoffs across quantization levels for cost-efficient deployment."

Phase 4 — Serve It
Tasks:

Wrap the chosen (quantized) model in Ollama or llama.cpp server.
Build a thin FastAPI layer: accepts {schema, question}, returns {sql, execution_result}.
Deploy on a free-tier host (Render/Railway/HF Spaces) — reuse deployment patterns from the RAG project.
Build a minimal CLI or simple web form for manual testing/demo.
Milestone/Deliverable: A live, callable endpoint + a working demo UI. Resume-ready achievement: "Deployed a self-hosted, quantized fine-tuned LLM as a production-style inference API."

Phase 5 — Polish & Integrate
Tasks:

Write the README: problem statement, architecture diagram, before/after metrics table, how to reproduce.
Record a short demo (Loom/GIF) showing a natural-language question turning into correct SQL + results.
Wire this model in as a selectable backend in Track A's coding agent (same pattern as your RAG project's multi-provider LLM support).
Update resume with final, real numbers (not placeholders).
Milestone/Deliverable: Public GitHub repo, live demo link, resume bullet with real metrics, integration point with Track A.

9. Risks / Things That Might Slow You Down
Free GPU quota limits (Colab/Kaggle): budget for training runs to be interrupted; checkpoint frequently.
Baseline might already be decent: some base models are already okay at SQL out of the box — if the improvement margin is small, lean on the quantization benchmarking and eval harness work as the differentiators instead.
Dataset schema complexity: Spider has genuinely hard multi-table joins — don't be surprised if accuracy plateaus below 100%; that's normal and worth discussing honestly in the README rather than cherry-picking easy examples.
10. Open Questions for Later Phases
Final base model choice — confirm once Phase 0 shows what fits comfortably on the free GPU tier.
Whether to also report an LLM-as-judge "SQL reasonableness" score alongside execution accuracy (nice-to-have, not required for Phase 0–2).