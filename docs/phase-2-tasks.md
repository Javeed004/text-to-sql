# Phase 2 Task Breakdown — Iterate on Data & Hyperparameters
Text-to-SQL Finetuning (Track B)

**Scope decisions locked in before this breakdown** (per your answers):
- Full-dataset run happens on Colab/Kaggle T4 directly (not attempted on the local RTX 2050 first).
- Experiment tracker: **W&B free tier**.
- Data-quality pass is **targeted**: curate/augment self-join and NOT-IN examples specifically. Set-ops is tracked in error analysis (Task 9) as a "does scale alone fix it" check, but gets no dedicated curation this phase — per your own Phase-1 hypothesis that it already showed partial learning while self-joins/NOT-IN showed none.

Continuity: no prior task-breakdown output exists for Phase 2, so numbering starts at Task 1.

---

Task 1 — Stand up the full-dataset training environment on Colab/Kaggle with W&B logging

Objective: Have a Colab or Kaggle notebook that loads Qwen2.5-Coder-3B-Instruct in 4-bit via Unsloth, mounts/loads the full Spider training split, and logs to a W&B project — verified with a trivial smoke run.

Why this task matters: Phase 1 ran on a 1455-example filtered subset locally-adjacent to a T4. Moving to the full training set and a persistent cloud notebook changes several things at once (data volume, session lifetime, tracker); if any of those breaks, you want to find out on a 2-minute smoke test, not 40 minutes into a real sweep.

What I will learn: How to authenticate and initialize `wandb.init()` inside a Colab/Kaggle session, how Unsloth's `FastLanguageModel.from_pretrained` reports VRAM usage on a T4 vs your local 2050, and how to structure a notebook so a Colab disconnect doesn't lose your run (checkpointing to HF Hub or Drive).

Prerequisites: Phase 1's fine-tuning notebook/script and the Phase 0 eval harness (both already exist).

What to do:
1. Create a new W&B project (e.g. `text-to-sql-phase2`) and get an API key.
2. In Colab/Kaggle, `pip install unsloth wandb bitsandbytes`, then `wandb.login()`.
3. Port Phase 1's model-loading + LoRA config code into the notebook unchanged.
4. Load the **full** Spider training split (not the 1455-example filtered subset) and print its size.
5. Run a 10-step smoke test with `report_to="wandb"` in your `TrainingArguments`/`SFTConfig` and confirm a run appears in the W&B dashboard with a live loss curve.
6. Add a checkpoint-to-HF-Hub call (or Drive save) after training, so a disconnect mid-run doesn't lose progress.

Expected result: A Colab/Kaggle notebook that completes a 10-step smoke fine-tune on the full dataset, with a corresponding run visible in your W&B project showing loss going down.

Completion criteria:
- W&B dashboard shows a real run with logged loss, learning rate, and step count.
- The notebook loads the full Spider train split and you can state its exact example count (compare it out loud to Phase 1's 1455).
- You can explain, without looking it up, what happens to your run if Colab disconnects at step 500.

Connection to next task: This environment is now the fixed harness every hyperparameter variant in this phase will run inside — Task 4 configures what changes between runs, not the environment itself.

---

Task 2 — Curate targeted training examples for self-joins and NOT-IN exclusion

Objective: Produce an augmented training file that oversamples/adds Spider (and optionally synthetic) examples matching the self-join pattern and the NOT-IN/set-exclusion pattern, tagged so you can measure their representation.

Why this task matters: Phase 1's own analysis found these two patterns had *zero* movement, and hypothesized this was because the 1455-example subset under-represented them — not that the model is structurally incapable of learning them. This task tests that hypothesis directly rather than hoping full-dataset scale fixes it as a side effect.

What I will learn: How to query Spider's training set by SQL structural pattern (e.g., detecting a table joined to itself via alias reuse, or a `NOT IN (SELECT ...)` subquery), and the tradeoff between oversampling existing hard examples vs. writing new synthetic ones.

Prerequisites: Task 1's full training split loaded; the specific failing examples named in phase1_summary.md's failure-pattern section (soccer_2 and friends, the self-join Highschooler example, the NOT-IN example) as your ground-truth pattern definitions.

What to do:
1. Write a small script that scans Spider's full training set's gold SQL for (a) a table name appearing twice with different aliases in the same FROM/JOIN clause (self-join proxy), and (b) `NOT IN` followed by a subquery (vs. a literal list).
2. Count how many natural occurrences of each pattern exist in the full training set — this number is itself a finding worth recording (it tells you whether Phase 1's failure was a subset-sampling artifact or a genuine scarcity in Spider overall).
3. If natural count is low (a real possibility — self-joins are relatively rare in Spider), either (a) oversample the existing matches by duplicating them 3-5x in the training file, or (b) write 10-15 synthetic schema+question+gold-SQL triples following Spider's format for each pattern, validated by actually running the gold SQL against a real in-memory DB to confirm it's correct.
4. Merge these into your full training set as a distinct tagged subset (e.g., add a `pattern_tag` column) so Task 9's error analysis can later check "did examples with this tag actually help."

Expected result: An augmented training file (full Spider set + oversampled/synthetic self-join and NOT-IN examples) with a recorded count of how many examples carry each pattern tag, plus a short note on whether you used oversampling or synthetic generation and why.

Completion criteria:
- You can state the exact number of self-join and NOT-IN examples in vanilla Spider train, before augmentation.
- Every synthetic example (if any) has been verified to execute correctly against a real schema, not just visually checked.
- You can explain why oversampling and synthetic generation have different risks (e.g., oversampling risks overfitting to a handful of surface forms; synthetic data risks distribution mismatch with real Spider style).

Connection to next task: This augmented dataset is one of the two "what changed" variables the hyperparameter sweep will need to isolate — Task 4 decides whether it's tested as part of the sweep or held constant across it.

---

Task 3 — Build a running experiment comparison table (W&B report + local markdown mirror)

Objective: Set up a W&B report (or a structured table logged as a W&B Table artifact) plus a parallel markdown table in your repo that will hold execution accuracy, exact match, and config for every experiment run this phase, starting with Phase 0/Phase 1's numbers as row 0 and row 1.

Why this task matters: The PRD's Phase 2 milestone explicitly requires "a comparison table across experiments" — building the table infrastructure before running experiments (instead of reconstructing it retroactively from W&B logs afterward) means every run automatically lands in the table with no manual backfilling, and you never lose a run's numbers to a later cleared notebook.

What I will learn: How to log a `wandb.Table` alongside training runs, and how to structure a markdown table designed to be diffed/appended to over multiple sessions.

Prerequisites: Phase 0's baseline numbers and Phase 1's fine-tuned numbers (61.0%/10.5% and 75.5%/38.0% from phase1_summary.md) as the first two rows.

What to do:
1. Create `docs/experiment-log.md` with columns: run name, dataset size, LoRA rank/alpha, learning rate, epochs, execution accuracy, exact match, notes.
2. Add Phase 0 (zero-shot baseline) and Phase 1 (first fine-tune) as the first two rows, copied verbatim from your existing results.
3. In the training notebook, add a small helper function `log_experiment_row(config, exec_acc, exact_match, notes)` that appends both to a `wandb.Table` and prints a markdown-formatted row you can paste into `experiment-log.md`.

Expected result: A two-row markdown table already reflecting Phase 0 and Phase 1, plus a reusable logging helper ready to append every subsequent Phase 2 run.

Completion criteria:
- `experiment-log.md` exists with exactly 2 rows and matches phase1_summary.md's numbers exactly.
- Calling the helper function with dummy values produces both a W&B Table row and a correctly formatted markdown row.
- You can explain why this table is being built now rather than after all runs finish.

Connection to next task: Every hyperparameter variant run from here on has a guaranteed place to land — Task 4 defines what those variants are.

---

Task 4 — Define the 2-3 hyperparameter variants for the sweep

Objective: Decide and document the specific 2-3 hyperparameter configurations to run this phase (e.g., varying LoRA rank, learning rate, or epoch count), each as a fully specified config dict, before running any of them.

Why this task matters: The PRD says "try 2-3 hyperparameter variants" without naming which — deciding this upfront (rather than ad hoc between runs) forces you to pick variants that isolate one variable at a time, which is what makes the comparison table in Task 3 actually interpretable instead of confounded.

What I will learn: How to reason about which hyperparameter is likely to matter most given Phase 1's specific result (r=16/alpha=32, 1 epoch) — e.g., whether more epochs on the same rank is a cheaper first experiment than a rank increase, given a full dataset takes meaningfully longer per epoch than 1455 examples did.

Prerequisites: Task 1's environment (to know real per-epoch wall-clock time on the full dataset before committing to a 3-epoch run), Task 2's augmented dataset.

What to do:
1. Run one timed epoch on the full augmented dataset in Task 1's environment and record wall-clock time — this tells you what's actually affordable on free Colab/Kaggle quota (30 hrs/week on Kaggle).
2. Define exactly 3 variants, changing one axis at a time from Phase 1's r=16/alpha=32/1-epoch baseline. A reasonable starting set: (a) same rank/alpha, 2-3 epochs on full data; (b) rank=32/alpha=64, same epoch count as (a); (c) same rank as (a), lower learning rate for more stable convergence.
3. Write each variant as an explicit config dict (rank, alpha, target_modules, learning_rate, epochs, dataset=augmented-full) in a `configs/phase2_variants.py` file.

Expected result: A file with 3 named, fully-specified config dicts and a one-paragraph rationale for why those 3 (not others) were chosen, grounded in the timed-epoch number from step 1.

Completion criteria:
- Each variant changes exactly one hyperparameter axis relative to the Phase 1 baseline (not several at once).
- The timed single-epoch run's wall-clock number is written down and used to justify the epoch counts chosen (not guessed).
- You can explain what confound would exist if two variants changed two axes at once.

Connection to next task: Task 5 runs variant (a) first — the cheapest, most direct extension of Phase 1.

---

Task 5 — Run hyperparameter variant (a): full dataset, same LoRA config, more epochs

Objective: Execute the first Task 4 variant end-to-end on Colab/Kaggle and log the result into the Task 3 comparison table.

Why this task matters: This is the most direct test of "does scale alone help" — same LoRA config as Phase 1, just more data and more epochs — which is exactly the question your Phase-1 hypothesis raised about the set-ops pattern.

What I will learn: How training dynamics differ with ~10x the data and multiple epochs (loss curve shape, whether eval accuracy plateaus or keeps climbing), and how to read a W&B loss curve for early signs of overfitting (e.g., train loss still dropping while an eval subset stagnates, if you log intermediate evals).

Prerequisites: Task 1 (environment), Task 2 (augmented dataset), Task 4 variant (a) config.

What to do:
1. Launch the fine-tune with variant (a)'s config on the full augmented dataset.
2. Save the resulting adapter (locally + push to HF Hub as `phase2-variant-a`).
3. Run Phase 0's eval harness (identical 200-example Spider dev subset) on this checkpoint.
4. Log execution accuracy, exact match, and config into `experiment-log.md` and the W&B Table via Task 3's helper.

Expected result: A trained adapter checkpoint on HF Hub, and a third row in `experiment-log.md` with real execution accuracy / exact match numbers for variant (a).

Completion criteria:
- The eval harness ran on the exact same 200-example test subset as Phase 0 and Phase 1 (no silent test-set drift).
- `experiment-log.md` row 3 is filled in with real numbers, not placeholders.
- You can state whether variant (a) beat, matched, or underperformed Phase 1, and your first-pass guess at why.

Connection to next task: Milestone Task 6 checks this result specifically against the self-join/NOT-IN patterns before you spend compute on variants (b) and (c).

---

Milestone Task 6
First full-dataset run validated against Phase 1's targeted failure patterns

Objective: Confirm the full pipeline — full dataset, W&B logging, augmented data, eval harness, comparison table — works end-to-end, and get a first read on whether Task 2's targeted curation moved the self-join/NOT-IN needle at all.

What I should have learned so far: How to run Unsloth QLoRA fine-tunes on a cloud GPU with persistent experiment tracking, how to structurally detect and augment underrepresented SQL patterns in a training set, and how to keep a comparison table synchronized with actual runs rather than reconstructed after the fact.

What I should build without blindly following instructions: Re-run the specific failure-pattern examples named in phase1_summary.md (the Highschooler self-join example, the NOT-IN example) against variant (a)'s checkpoint yourself — don't just trust the aggregate execution-accuracy number, since Phase 1's own lesson was that the aggregate number hid pattern-level nuance.

Mini challenge: Before looking at variant (a)'s output on the self-join example, write down your own prediction of whether it fixed it, based on how many self-join examples you added in Task 2 relative to Spider's natural count — then check if you were right.

Self-assessment: If someone showed you variant (a)'s execution-accuracy number alone with no other context, could you tell them whether it's safe to conclude the self-join pattern is fixed? Why or why not?

Milestone that proves your progress: A checkpoint that has been fine-tuned on the full augmented dataset, evaluated on the same fixed 200-example test set as every prior checkpoint, with its aggregate numbers in the comparison table *and* a manual re-check of the two targeted failure examples from Phase 1.

Completion criteria:
- `experiment-log.md` has 3 populated rows (Phase 0, Phase 1, variant a) with no placeholder values.
- The self-join and NOT-IN examples from phase1_summary.md have been manually re-run against variant (a) and the outcome (fixed / unchanged / different failure) is written down, not assumed from the aggregate score.
- You can point to one specific reason your Task 2 curation decision (oversampling vs. synthetic) might explain the outcome you observed.

Connection to next task: If self-joins/NOT-IN are still broken here, Task 7-8's variants (b) and (c) become your next lever; if they're fixed, Task 9's error analysis shifts focus to whatever's still broken (likely set-ops or a new pattern).

---

Task 7 — Run hyperparameter variant (b): increased LoRA rank/alpha

Objective: Execute Task 4's variant (b) — rank/alpha increase, same epoch count as variant (a) — and log results.

Why this task matters: Rank controls how much capacity the adapter has to represent new behavior; if variant (a)'s scale-alone approach under-delivered on the structural patterns (self-joins, NOT-IN), more adapter capacity is the next most direct lever to test, isolated from the epoch-count change already tested in (a).

What I will learn: How LoRA rank changes trainable parameter count and per-step compute cost in practice (not just in theory), and how to judge whether a rank increase is worth its cost from your own logged numbers rather than a rule of thumb.

Prerequisites: Task 6's milestone result (to know what specifically variant (b) needs to improve on), Task 4's variant (b) config.

What to do:
1. Launch fine-tuning with variant (b)'s config (rank=32/alpha=64) on the same augmented full dataset.
2. Push adapter to HF Hub as `phase2-variant-b`.
3. Run the identical eval harness and log to `experiment-log.md` / W&B Table.
4. Manually re-check the same self-join/NOT-IN examples as Task 6.

Expected result: A fourth row in the comparison table, plus a manual check of whether higher rank moved the two targeted patterns.

Completion criteria:
- Trainable parameter count for variant (b) is recorded and compared numerically to variant (a)'s.
- `experiment-log.md` row 4 is complete.
- You can state whether the accuracy gain (if any) from higher rank justified its added training time, using your own logged numbers.

Connection to next task: Task 8 tests the remaining variant (c); by this point you'll have two independent axes (epochs, rank) to compare against Phase 1's original config.

---

Task 8 — Run hyperparameter variant (c) and finalize the comparison table

Objective: Execute Task 4's remaining variant (c), log it, and close out the comparison table with all planned runs.

Why this task matters: This completes the "2-3 hyperparameter variants tracked in W&B" deliverable the PRD names explicitly for this phase, and gives you a complete table to select a best checkpoint from in Task 11.

What I will learn: How to read across multiple logged runs in a single W&B project view (parallel coordinates or a run comparison table) instead of only within one run's dashboard.

Prerequisites: Tasks 5 and 7's results (variants a and b) already in the table.

What to do:
1. Launch fine-tuning with variant (c)'s config.
2. Push adapter to HF Hub as `phase2-variant-c`.
3. Run eval harness, log results.
4. Open W&B's run comparison view across all three variants plus Phase 1, and screenshot or export it for your README (Phase 5 will need this).

Expected result: A complete comparison table (5 rows: Phase 0, Phase 1, variants a/b/c) and a W&B multi-run comparison view.

Completion criteria:
- All 5 rows in `experiment-log.md` are complete with real numbers.
- The W&B comparison view has been exported/screenshotted.
- You can identify, from the table alone, which single variant most improved on Phase 1 and by how much.

Connection to next task: Task 9's error analysis runs against whichever variant currently leads the table.

---

Task 9 — Error analysis on the leading checkpoint, including the set-ops "scale alone" check

Objective: Run the same failure-pattern-level review Phase 1 did (per phase1_summary.md's methodology) on the best-performing variant, explicitly checking all four original patterns — self-joins, NOT-IN, set-ops, hallucinated joins — not just the aggregate score.

Why this task matters: This is the direct test of your Q3 decision: self-joins and NOT-IN got dedicated curation (Task 2), while set-ops did not — the hypothesis was that set-ops might resolve from scale alone. This task tells you if that bet paid off, and surfaces any new failure patterns introduced at full-dataset scale (the phase1_summary.md report already flagged "extra unrelated joins" as a possible early overfitting signal worth watching).

What I will learn: How to structure a repeatable failure-pattern review (same test-set slice, same 4 named categories plus an "other/new" catch-all) so it's comparable across phases rather than a one-off manual read.

Prerequisites: Task 8's leading checkpoint; Phase 1's four named failure categories and specific example IDs as the baseline for comparison.

What to do:
1. Re-run the exact same failure-pattern examples from phase1_summary.md (soccer_2, the Highschooler self-join, the set-ops OR-branches example, the NOT-IN example) against the leading Phase 2 checkpoint.
2. For each, classify as: fixed / unchanged / broken differently (matching phase1_summary.md's own classification scheme).
3. Specifically check whether the "extra irrelevant join" artifact from Phase 1's set-ops example reappears or worsens — this is your overfitting-at-scale check.
4. Sample 10-15 additional test-set failures beyond these four named examples to check for any newly emerged pattern not seen in Phase 1.

Expected result: A written failure-pattern breakdown for the leading Phase 2 checkpoint, structured identically to phase1_summary.md's, plus a short verdict on whether set-ops improved without dedicated curation.

Completion criteria:
- All four original patterns have an explicit fixed/unchanged/broken-differently classification, not just a pass/fail.
- The set-ops "scale alone" hypothesis has an explicit verdict (confirmed / not confirmed) with the supporting example.
- At least one newly-sampled failure (beyond the four named ones) has been reviewed for a pattern not present in Phase 1's list.

Connection to next task: If this analysis surfaces a pattern still unaddressed, note it explicitly for Phase 3/README rather than silently dropping it — Task 11's checkpoint selection should weigh this alongside the aggregate score.

---

Task 10 — Add data-quality iteration if Task 9 surfaces a new systematic pattern

Objective: If Task 9 found a new or worsened systematic failure (e.g., the "extra irrelevant join" artifact spreading), make one targeted data or config adjustment and re-run eval only (not a full retrain sweep) to check if it helps — otherwise, skip this task explicitly and record why.

Why this task matters: The PRD frames this as conditional ("add data quality passes if error analysis shows systematic failure patterns") — this task exists so that condition gets a real decision point instead of being silently skipped or triggering unplanned extra work.

What I will learn: How to make a single targeted intervention (e.g., adding negative examples that penalize the specific hallucinated-join pattern, or filtering the augmented set for the artifact's suspected cause) and isolate its effect without re-running the entire sweep.

Prerequisites: Task 9's explicit verdict on whether a new/worsening pattern exists.

What to do (only if Task 9 found something to address):
1. Identify the smallest plausible fix (e.g., a handful of curated negative-ish examples, or removing a suspected noisy subset of your Task 2 augmentation).
2. Apply it to the leading checkpoint's training config and re-run fine-tuning once.
3. Re-run the eval harness and the same 4-pattern check from Task 9.
4. Log this as an explicit extra row in `experiment-log.md`.

If Task 9 found nothing new: Write one sentence in `experiment-log.md`'s notes confirming this task was evaluated and skipped, with the reason.

Expected result: Either an additional logged experiment row addressing a specific newly-found issue, or an explicit documented decision to skip this task.

Completion criteria:
- A decision (act or skip) is explicitly recorded, not implied by silence.
- If acted on, the intervention targeted the specific pattern Task 9 named, not a generic "more epochs" change.

Connection to next task: Task 11 selects the final checkpoint from whatever the table looks like at this point, including this row if it exists.

---

Task 11 — Select the final checkpoint and push it to Hugging Face Hub

Objective: Choose the best-performing checkpoint across all Phase 2 experiments using both the aggregate table and the Task 9 failure-pattern analysis, and push it to HF Hub as the canonical Phase 2 artifact.

Why this task matters: This is the PRD's explicit Phase 2 deliverable ("push the adapter to Hugging Face Hub") and the checkpoint Phase 3 will merge and quantize — picking it deliberately (not just "highest execution accuracy") matters because Phase 1 already showed aggregate accuracy can hide which failure patterns actually improved.

What I will learn: How to weigh a quantitative leaderboard against a qualitative failure-pattern review when they don't perfectly agree (e.g., the highest-accuracy checkpoint might also be the one with the worst hallucinated-join regression).

Prerequisites: Task 8's complete comparison table, Task 9's (and if applicable Task 10's) failure-pattern verdicts.

What to do:
1. Re-read the full comparison table plus your Task 9 written analysis side by side.
2. Write one paragraph justifying the final pick — explicitly stating if you're choosing a checkpoint that isn't the single highest execution-accuracy number, and why.
3. Push the chosen adapter to HF Hub with a clear model card noting the LoRA config, dataset size, and headline metrics.
4. Update `experiment-log.md` to flag the chosen row as "SELECTED — Phase 2 final."

Expected result: One adapter on HF Hub tagged as the Phase 2 final model, with a written justification for the choice that references specific failure-pattern evidence, not just the top-line number.

Completion criteria:
- The HF Hub model card includes config, dataset size, and both execution accuracy and exact match numbers.
- Your written justification references at least one piece of Task 9's qualitative evidence, not only the aggregate score.
- You can defend, out loud, why you didn't just pick whichever row has the single highest execution-accuracy percentage (even if that's the one you picked).

Connection to next task: This checkpoint is exactly what Milestone Task 12 closes out, and what Phase 3 will merge and quantize.

---

Milestone Task 12
Phase 2 complete — comparison table, best checkpoint, and Phase 3 handoff

Objective: Confirm Phase 2's full deliverable is in place: a complete cross-experiment comparison table, a chosen best checkpoint on HF Hub with documented justification, and a clear statement of which failure patterns remain open going into Phase 3.

What I should have learned so far: How to run a disciplined hyperparameter sweep with one-variable-at-a-time variants, how to target a data-quality intervention at a specific diagnosed weakness rather than a generic "more data" pass, and how to select a final model checkpoint using both quantitative and qualitative evidence.

What I should build without blindly following instructions: Write the Phase 2 section of your eventual README now, from your own comparison table and Task 9/11 notes — not by copying phase1_summary.md's structure verbatim, since Phase 2's actual story (what the targeted curation did or didn't fix) may differ from Phase 1's.

Mini challenge: Using only `experiment-log.md` and your Task 9 notes, write the single sentence you'd put in an interview: "Phase 2 improved X from Y% to Z% by doing W, and here's the one thing that still doesn't work." If you can't fill in all four blanks confidently, that's a sign one of Tasks 9-11 needs a second pass before moving to Phase 3.

Self-assessment: If your self-join/NOT-IN curation (Task 2) didn't work, can you explain — from the numbers, not a guess — whether that's because the curation approach was wrong (oversampling vs. synthetic), the volume was still too low, or the pattern needs a fundamentally different fix (e.g., more epochs specifically on those tagged examples)?

Milestone that proves your progress: A best-checkpoint adapter live on HF Hub, a complete 5+ row comparison table, and a written failure-pattern status (fixed / unchanged / new) for all four original Phase 1 patterns, ready to hand to Phase 3's merge-and-quantize step.

Completion criteria:
- HF Hub link to the final adapter is recorded in `experiment-log.md`.
- Every one of Phase 1's four named failure patterns has a final Phase 2 status, not just an aggregate accuracy delta.
- You can point to one specific decision in this phase (e.g., targeted vs. broad curation, which hyperparameter axis you tested first) that you made differently than the "obvious" approach, and explain why.

Connection to next task: Phase 3 (Merge, Quantize, Benchmark) starts from this exact checkpoint — its GGUF conversion and quantization-level comparison will need this adapter's HF Hub link and its full-precision eval numbers as the pre-quantization baseline.