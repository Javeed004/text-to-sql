# Phase 1 — First Fine-Tune: Task Breakdown

**Context carried in from Phase 0:**
- Base model: Qwen2.5-Coder-3B-Instruct (4-bit)
- Dataset: Spider — train pool split 92/8 into train/val (seed=42); dev set = test (unmodified)
- Fixed 200-example test subset (seed=42 sample of Spider dev), greedy decoding — reused here for apples-to-apples comparison
- Baseline: 61.0% execution accuracy (122/200), exact match TBD
- Known failure patterns to watch for: hallucinated joins, missed self-joins, set-ops rewritten as JOIN/OR, `!=` vs `NOT IN` confusion
- Training compute: Google Colab free T4
- Experiment tracking: **Weights & Biases (free tier)** — recommended over self-hosted MLflow here because Colab sessions are ephemeral with no persistent server to host MLflow on; W&B needs only an API key and logs to the cloud automatically, so nothing is lost if the runtime disconnects.

---

## Task 1 — Set up the Colab QLoRA training environment

**Objective:** A Colab notebook with a T4 GPU that has Unsloth, PEFT, bitsandbytes, and W&B installed, and can load Qwen2.5-Coder-3B-Instruct in 4-bit successfully.

**Why this task matters:** Phase 0's VRAM test was for inference-only baseline generation. Training adds optimizer states and gradients on top of the model weights, so the environment needs to be re-verified before you commit a real run to it — an OOM mid-training wastes far more time than one at load time.

**What I will learn:** How Unsloth's `FastLanguageModel.from_pretrained` differs from vanilla `transformers` loading for 4-bit models; how to authenticate W&B from a notebook (`wandb.login()`); reading `nvidia-smi` / `torch.cuda.memory_allocated()` output to sanity-check headroom.

**Prerequisites:** Phase 0's Colab notebook (model loading code, `pip install unsloth` step) as a starting point.

**What to do:**
1. In a fresh Colab notebook, set runtime to T4 GPU.
2. `pip install unsloth bitsandbytes wandb` (pin the same Unsloth version used in Phase 0 if you recorded it).
3. Load the model exactly as in Phase 0 but pass `load_in_4bit=True` in preparation for LoRA attachment (Unsloth handles this internally when you next call `FastLanguageModel.get_peft_model`).
4. Run `wandb.login()` with your API key (stored as a Colab secret, not hardcoded).
5. Print `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` right after model load to record a "before training" VRAM baseline.

**Expected result:** A notebook cell output showing the model loaded in 4-bit, W&B authenticated (`wandb: Logged in as ...`), and a recorded VRAM baseline in MB.

**Completion criteria:**
- Model loads without error and reports 4-bit quantization in the loading logs.
- `wandb.login()` succeeds.
- You can explain, in your own words, why a QLoRA training loop uses more memory than the same model doing plain inference (optimizer states, gradient buffers, activation memory for backprop).

**Connection to next task:** With the environment confirmed, Task 2 attaches the actual LoRA adapters to this loaded model.

---

## Task 2 — Configure LoRA adapter hyperparameters

**Objective:** A `PeftModel` wrapping Qwen2.5-Coder-3B-Instruct with LoRA adapters attached to a deliberately chosen set of target modules, rank, and alpha.

**Why this task matters:** These are the knobs Phase 2 will later sweep systematically — getting a defensible starting point now (not copy-pasted from a random tutorial) matters both for training quality and for being able to explain the choice in an interview.

**What I will learn:** What `r` (rank) and `lora_alpha` actually control in LoRA's low-rank decomposition; why `target_modules` typically includes attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and sometimes MLP projections (`gate_proj`, `up_proj`, `down_proj`); how `lora_dropout` and `use_gradient_checkpointing="unsloth"` trade off memory vs. speed.

**Prerequisites:** Task 1's loaded 4-bit base model.

**What to do:**
1. Call Unsloth's `FastLanguageModel.get_peft_model(model, r=16, target_modules=[...], lora_alpha=16, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", random_state=42)`.
2. Start with `r=16`, `lora_alpha=16` (1:1 ratio is a reasonable, well-documented starting point) and target modules = all seven of `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` — this is what Phase 2's hyperparameter sweep will vary later, so this pass just needs one honest baseline choice, not the optimal one.
3. Print `model.print_trainable_parameters()` and record the trainable % (should be roughly 1-2% of total params for a 3B model at r=16).

**Expected result:** A PEFT-wrapped model object with a logged trainable-parameter count and percentage.

**Completion criteria:**
- `print_trainable_parameters()` output is captured (paste it into your run notes).
- You can explain why only ~1-2% of parameters are trainable and why that's the point of LoRA (vs. full fine-tuning).

**Connection to next task:** Task 3 prepares the actual training data this adapter will see.

---

## Task 3 — Build the fast-iteration training subset

**Objective:** A formatted training subset (not the full Spider train split) ready to feed into the trainer, reusing Phase 0's instruction-format preprocessing.

**Why this task matters:** The PRD explicitly calls for "fast iteration first, full dataset later" — training on the full set now would burn Colab GPU quota before you even know if the pipeline (data format, loss curve shape, eval harness hookup) works end to end.

**What I will learn:** How to slice a Hugging Face `datasets.Dataset` reproducibly with a fixed seed; why a smaller, fast training loop is used to validate a pipeline before scaling up (same principle as a smoke test, one level up).

**Prerequisites:** Phase 0's preprocessing function that converts (schema, question, gold SQL) into the instruction-tuning format, and the 92% train split from Phase 0's 92/8 split.

**What to do:**
1. From the 92% train split, take a fixed-seed random sample of ~1,500-2,000 examples (`dataset.shuffle(seed=42).select(range(1500))`) — large enough to see a real loss curve, small enough to train in well under an hour on a T4.
2. Run Phase 0's preprocessing function over this subset to produce the same instruction format used for baseline generation, but now including the gold SQL as the target completion (not just the prompt).
3. Tokenize with `max_seq_length` set to match the PRD's target range (512-1024 tokens) — pick 1024 if Spider's longer schemas need it, checked by inspecting the token-length distribution of the subset.
4. Spot-check 3 formatted examples by eye to confirm prompt/completion boundaries are correct (this is the #1 place instruction-tuning silently breaks).

**Expected result:** A tokenized `Dataset` object of ~1,500-2,000 examples with correct prompt/completion structure, plus a printed token-length histogram or summary stats.

**Completion criteria:**
- Sample size and seed are recorded in your run notes for reproducibility.
- All 3 spot-checked examples show correctly separated prompt and target.
- You can explain why training/target masking matters (loss should generally only be computed on the completion, not the prompt tokens) — check whether Unsloth's trainer does this automatically or whether you need to set it explicitly.

**Connection to next task:** Task 4 wires this dataset and the Task 2 model into an actual trainer configuration.

---

## Task 4 — Configure the training run and wire up W&B logging

**Objective:** A fully configured `SFTTrainer` (via Unsloth/TRL) pointed at the Task 3 dataset and Task 2 model, logging to W&B, not yet started.

**Why this task matters:** Getting the training arguments right before hitting "run" avoids wasting Colab GPU minutes on a misconfigured job (wrong batch size causing OOM, or logging silently going nowhere).

**What I will learn:** The role of `per_device_train_batch_size` + `gradient_accumulation_steps` in simulating a larger effective batch size on limited VRAM; what `learning_rate`, `warmup_steps`, and `num_train_epochs` do; how `report_to="wandb"` and `run_name` hook a TRL/Transformers `Trainer` up to W&B with no extra logging code.

**Prerequisites:** Task 2's PEFT model, Task 3's tokenized dataset.

**What to do:**
1. Set up `TrainingArguments` (or Unsloth's equivalent) with: `per_device_train_batch_size=2`, `gradient_accumulation_steps=4` (effective batch size 8), `num_train_epochs=1-2` for this fast pass, `learning_rate=2e-4`, `warmup_steps=10`, `logging_steps=1`, `report_to="wandb"`.
2. Set a descriptive `run_name`, e.g. `qwen2.5-coder-3b-lora-r16-fastpass-v1`, so it's identifiable later in the W&B dashboard alongside Phase 2's sweep runs.
3. Instantiate the `SFTTrainer` with the model, tokenizer, dataset, and training args.
4. Do a 5-step dry run (`max_steps=5`) first to confirm no immediate OOM or shape errors before committing to the full run.

**Expected result:** A `Trainer` object that completes a 5-step dry run without error, with a corresponding (short) run visible in your W&B project dashboard.

**Completion criteria:**
- The 5-step dry run finishes and logs at least one loss value to W&B.
- You can explain what "effective batch size" means and why gradient accumulation is used here instead of just increasing `per_device_train_batch_size`.

**Connection to next task:** Task 5 removes the `max_steps` cap and runs the real fast-iteration training pass.

---

## Task 5 — Run the first QLoRA fine-tuning pass

**Objective:** A completed training run over the Task 3 subset, with a full loss curve logged in W&B and a saved adapter checkpoint.

**Why this task matters:** This is the actual fine-tuning event the whole phase is building toward — everything before this was preparation, everything after is evaluation.

**What I will learn:** How to read a loss curve for warning signs (not decreasing at all = learning rate or data problem; decreasing then spiking = instability); how to save a LoRA adapter (`model.save_pretrained(...)`) separately from the merged model (merging comes in Phase 3, not here).

**Prerequisites:** Task 4's dry-run-verified trainer.

**What to do:**
1. Remove the `max_steps=5` cap and call `trainer.train()`.
2. Watch the W&B run live (or check after) for the loss curve shape.
3. On completion, save the adapter weights: `model.save_pretrained("qwen2.5-coder-3b-lora-r16-fastpass-v1")` and push to Hugging Face Hub (or keep in Colab + Google Drive for now if you're saving the Hub push for Phase 2's "best checkpoint").
4. Record final training loss, run duration, and peak VRAM usage in your run notes.

**Expected result:** A saved LoRA adapter directory (or Hub repo) and a completed W&B run showing a declining loss curve.

**Completion criteria:**
- Training completes without crashing.
- Loss curve is visibly declining (not flat, not diverging) by the end of the run.
- Adapter files exist and can be reloaded with `PeftModel.from_pretrained`.

**Connection to next task:** This checkpoint is the artifact Milestone Task 6 packages up as the first working end-to-end proof point.

---

## Milestone Task 6
### First trained adapter — pipeline proven end to end

**Objective:** Prove that the full chain — 4-bit base model → LoRA attach → formatted data → trained → saved adapter → reloadable — actually works, independent of whether the accuracy number is good yet.

**What I should have learned so far:** How LoRA adapters attach to specific modules without touching base weights; how instruction-formatted data flows into an SFT trainer; how effective batch size and learning rate interact with a visible loss curve; how W&B run tracking ties training config to results.

**What I should build without blindly following instructions:** Reload the saved adapter into a fresh model instance (not the same in-memory object you trained) and generate SQL for 2-3 hand-picked questions from the training subset, purely to confirm the adapter changes behavior versus the untuned base model — assemble this reload-and-generate script yourself from what you now know about `PeftModel.from_pretrained` and Phase 0's generation code, rather than copying a finished example.

**Mini challenge:** Pick one of the 2-3 questions and compare the base model's zero-shot answer (from Phase 0) against this adapter's answer side by side — does the fine-tuned output even look different, structurally, from the baseline output?

**Self-assessment:** If someone deleted this adapter and only gave you your W&B run page, could you tell them exactly what data, hyperparameters, and base model produced it — without checking any other notes?

**Milestone that proves your progress:** A reloadable LoRA adapter trained on a real (if small) slice of Spider, with a logged training run, that visibly produces different SQL than the zero-shot base model on at least one example.

**Completion criteria:**
- Adapter reloads successfully in a clean session and produces generations.
- You can point to one hyperparameter choice (rank, target modules, batch size, etc.) you set deliberately rather than copying from a tutorial, and explain why.

**Connection to next task:** Task 7 formally scores this checkpoint with the same eval harness used for the Phase 0 baseline, so "looks different" becomes a real number.

---

## Task 7 — Run the eval harness on the fine-tuned checkpoint

**Objective:** Execution accuracy and exact-match numbers for this checkpoint, measured on the identical 200-example test subset used for the Phase 0 baseline.

**Why this task matters:** Comparing against a different or larger test set than Phase 0 would invalidate the before/after comparison the whole project's success metric depends on.

**What I will learn:** How to swap the model used by an existing eval harness without changing the harness logic itself — reinforces that a good eval harness is decoupled from any one model.

**Prerequisites:** Phase 0's eval harness code and its fixed 200-example test subset (same seed=42 sample); Task 5's saved adapter.

**What to do:**
1. Load the base model + Task 5's LoRA adapter (merged in memory via `PeftModel.from_pretrained`, no need to permanently merge weights yet — that's Phase 3).
2. Run the exact same generation + execution-accuracy scoring loop from Phase 0 over the same 200 examples, same greedy decoding settings.
3. Save the results in the same summary format Phase 0 used (so `loaded["summary"]["execution_accuracy"]` / `["exact_match"]` stay comparable programmatically, not just by eye).

**Expected result:** A results file/dict with execution accuracy and exact-match numbers for the fine-tuned checkpoint, in the same schema as Phase 0's baseline results.

**Completion criteria:**
- Same 200 examples, same decoding settings as Phase 0 (confirm by diffing the example IDs used, not just assuming).
- Execution accuracy and exact match are both recorded, even if one didn't move much.

**Connection to next task:** Task 8 puts this number next to the 61.0% baseline.

---

## Task 8 — Build the baseline vs. fine-tuned comparison table

**Objective:** A single table (this becomes the seed of Phase 2's "running comparison table") showing Phase 0 baseline vs. this first fine-tune, side by side.

**Why this task matters:** A number in isolation ("67% execution accuracy") means nothing without the baseline it's being compared to — this table is the artifact that actually answers the PRD's "does it move at all?" question.

**What I will learn:** How to present a before/after ML result clearly and honestly, including cases where a metric moved less than hoped or even regressed.

**Prerequisites:** Phase 0's 61.0% baseline number, Task 7's fine-tuned results.

**What to do:**
1. Build a small markdown or CSV table: rows = {Baseline (Phase 0), Fine-tuned v1 (Phase 1)}, columns = {Execution accuracy, Exact match, Model, LoRA rank, Training examples seen}.
2. Compute the delta explicitly (percentage points, not just "went up").
3. If exact match wasn't actually recorded in Phase 0 (the doc shows a placeholder), backfill it now for both rows so the table has no missing cells.

**Expected result:** A saved comparison table (markdown or CSV) with no placeholder values.

**Completion criteria:**
- Both rows have real numbers for both metrics.
- The delta is stated explicitly, in either direction (this table doesn't need to show improvement to be "done" — it needs to be honest).

**Connection to next task:** Task 9 explains the *why* behind whatever this table shows, using the failure patterns Phase 0 already identified.

---

## Task 9 — Manual sanity check against known failure patterns

**Objective:** A short written note on whether this first fine-tune reduced, left unchanged, or worsened each of the four failure patterns Phase 0 identified (hallucinated joins, missed self-joins, set-ops-as-JOIN, `!=` vs `NOT IN`).

**Why this task matters:** Execution accuracy is a single aggregate number — it can go up 3 points while one specific failure mode gets worse, and that's exactly the kind of signal Phase 2's data-curation work needs to target.

**What I will learn:** How to do targeted qualitative error analysis instead of just trusting an aggregate metric; how to pull specific example IDs by category for repeatable review (a habit worth carrying into Phase 2).

**Prerequisites:** The specific example IDs from Phase 0's manual review that showed each of the 4 patterns; Task 7's fine-tuned generations for those same example IDs.

**What to do:**
1. Pull up the fine-tuned model's generated SQL for the exact same examples flagged in Phase 0's 4 failure categories.
2. For each of the 4 categories, note: fixed / still broken / broken differently now.
3. Write 2-4 sentences per category summarizing what you see — this is qualitative, not another accuracy number.

**Expected result:** A short markdown note, one paragraph per failure category, appended to your run notes or the comparison doc from Task 8.

**Completion criteria:**
- All 4 categories are addressed (even "no change observed" is a valid, useful finding).
- You can explain, in your own words, why an aggregate accuracy number alone wouldn't have told you this.

**Connection to next task:** Milestone Task 10 packages Tasks 5-9 into the phase's formal deliverable.

---

## Milestone Task 10
### First before/after comparison — Phase 1 complete

**Objective:** Combine the trained adapter, the eval harness re-run, the comparison table, and the qualitative failure-pattern review into the single deliverable the PRD calls for: "First trained adapter + first before/after comparison (even if not optimal yet)."

**What I should have learned so far:** The full LoRA fine-tuning loop end to end (config → data → train → save); how to keep an eval comparison methodologically valid (same test set, same decoding settings); how to read both quantitative deltas and qualitative failure patterns together rather than trusting either alone.

**What I should build without blindly following instructions:** A short (half-page) write-up combining Task 8's table and Task 9's qualitative notes into one coherent "here's what moved and why" narrative — written by you, synthesizing the two artifacts, not just concatenating them.

**Mini challenge:** Without re-running anything, predict which of the 4 failure categories you'd expect to improve *most* from Phase 2's planned data-quality passes, and write one sentence on why — this is the thread Phase 2 picks up.

**Self-assessment:** If you were asked in an interview "did fine-tuning actually help, and how do you know it wasn't just noise from a small test set?" — could you answer with something more rigorous than "the number went up"? (Hint: think about what a 200-example test set's margin of error looks like at ~61-70% accuracy.)

**Milestone that proves your progress:** A documented, reproducible before/after comparison (Phase 0 zero-shot vs. Phase 1 first fine-tune) with both a quantitative table and a qualitative failure-pattern review, plus a saved/reloadable adapter checkpoint — the exact deliverable the PRD specifies for Phase 1.

**Completion criteria:**
- Comparison table (Task 8) and failure-pattern note (Task 9) are both finalized with no placeholder values.
- LoRA adapter checkpoint is saved somewhere durable (Hub or Drive), not only in the Colab runtime's ephemeral disk.
- You can articulate one concrete, evidence-based hypothesis for what Phase 2 should try first, based on this phase's results.

**Connection to next task:** Phase 2 takes this checkpoint's config as a starting point, scales to the full training set, and runs the 2-3 hyperparameter variants this phase's results should now motivate (e.g. if join-hallucination barely moved, that's a signal for the data-quality pass mentioned in Phase 2's tasks, not just a hyperparameter tweak).