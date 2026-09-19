# Phase 0 — Setup & Baseline: Task Breakdown

**Project:** Track B — Fine-Tuned Text-to-SQL Engine
**Path chosen:** Google Colab (free T4), no local RTX 2050 setup
**Dataset:** Spider (recommendation — see note below)
**Starting point:** Bare setup, nothing exists yet

> **Why Spider over WikiSQL:** Spider's questions span multiple tables with real JOINs, GROUP BY, nested queries, and varying schema complexity, so the before/after fine-tuning story has more room to show a real improvement (and to talk about *where* it still fails, e.g. JOIN-heavy queries — which the PRD explicitly calls out as expected and worth discussing). WikiSQL is single-table and much easier, so a zero-shot model already scores high on it, leaving little room to demonstrate that fine-tuning mattered. Spider also ships a well-known execution-accuracy evaluation methodology, which lines up directly with what Phase 0 asks you to build. If Spider's schema complexity turns out to slow you down too much once you're in it, WikiSQL is a reasonable fallback — but start with Spider.

---

Task 1 — Set up the Colab environment and confirm GPU access

Objective: Get a Colab notebook running with GPU access and all fine-tuning libraries installed and importable.

Why this task matters: Every later task in this phase (model loading, preprocessing, eval, baseline) runs inside this environment. If the install is broken or you don't actually have a GPU attached, you won't discover it until a much more expensive step fails.

What I will learn: How to request a T4 GPU runtime in Colab, how `nvidia-smi` reports GPU allocation in a notebook, and the specific package set (`unsloth`, `bitsandbytes`, `peft`, `transformers`, `datasets`, `accelerate`) needed for QLoRA fine-tuning.

Prerequisites: None — this is the first task.

What to do:
1. Create a new Colab notebook. Go to Runtime → Change runtime type → select "T4 GPU".
2. Run `!nvidia-smi` and confirm it reports a Tesla T4 with ~15GB total memory.
3. Install the core libraries in a cell:
   ```
   !pip install unsloth bitsandbytes peft transformers datasets accelerate trl
   ```
4. Run a version-check cell (`import torch, transformers, peft, bitsandbytes, unsloth; print(torch.__version__, transformers.__version__)`) to confirm everything imported without error.
5. Confirm `torch.cuda.is_available()` returns `True`.

Expected result: A saved Colab notebook (e.g. `phase0_setup.ipynb`) with a working GPU runtime and all five/six libraries importing cleanly with no version conflicts.

Completion criteria:
- `nvidia-smi` output is visible in a notebook cell showing a T4 GPU.
- All imports in the version-check cell run without `ImportError` or `ModuleNotFoundError`.
- You can explain, in your own words, the difference between what `bitsandbytes` and `peft` each do in a QLoRA setup (one handles the 4-bit quantization, the other handles the low-rank adapter layers).

Connection to next task: With a working environment, Task 2 uses it to actually load candidate base models and measure how much GPU memory each one takes, which decides your final model choice.

---

Task 2 — Load candidate base models in 4-bit and pick the final one

Objective: Load 2–3 candidate base models in 4-bit quantization on the T4 and measure GPU memory headroom to select the model Phase 1 onward will fine-tune.

Why this task matters: The PRD is explicit that the final model choice happens after an actual VRAM test, not a guess — training needs headroom beyond just loading weights (optimizer states, gradients, activations), so a model that "fits" for inference can still OOM during fine-tuning.

What I will learn: How to load a model in 4-bit via `unsloth.FastLanguageModel.from_pretrained` (or `bitsandbytes` `BitsAndBytesConfig`), how to read `torch.cuda.memory_allocated()` / `memory_reserved()`, and roughly how much memory overhead LoRA training adds on top of inference-only loading.

Prerequisites: Task 1's working Colab environment.

What to do:
1. Pick 2–3 candidates to test, e.g. `Qwen2.5-Coder-1.5B-Instruct`, `Qwen2.5-Coder-3B-Instruct`, and `Llama-3.2-3B-Instruct` (swap in `unsloth`'s pre-quantized versions if available, e.g. `unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit`, since they load faster).
2. For each candidate, in a fresh runtime (Runtime → Restart), load it in 4-bit:
   ```python
   from unsloth import FastLanguageModel
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name="unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit",
       max_seq_length=1024,
       load_in_4bit=True,
   )
   ```
3. After loading, record `torch.cuda.memory_allocated() / 1e9` (GB).
4. Attach a LoRA adapter with `FastLanguageModel.get_peft_model(...)` (rank 16 as a first guess) and record memory again — this approximates training overhead before you've even run a batch.
5. Tabulate: model name, params, memory after load, memory after LoRA attach, and your headroom estimate (T4 has ~15GB usable).
6. Pick the model with the best balance of capability vs. headroom — bias toward the smaller end if two candidates are close, since Phase 2's full-dataset run needs to fit comfortably too.

Expected result: A small comparison table (in the notebook or a short markdown cell) of 2–3 models with their memory footprints, plus a one-line decision with rationale.

Completion criteria:
- Each candidate was actually loaded and measured — no number in the table is estimated from memory or documentation alone.
- The final pick is written down explicitly with the memory numbers next to it.
- You can explain why LoRA-attached memory is higher than raw 4-bit load memory, even though LoRA adds relatively few trainable parameters.

Connection to next task: The chosen model's tokenizer and prompt format carry into Task 4, where you'll build the instruction-formatted dataset around that specific model's expected chat template.

---

Task 3 — Download and explore the Spider dataset

Objective: Download the Spider dataset into the Colab environment and understand its file structure before writing any preprocessing code.

Why this task matters: Spider isn't a single flat file — it has separate schema (DDL/`tables.json`), question/SQL pairs, and per-database SQLite files. Preprocessing code written without first understanding this structure tends to need a full rewrite once the shape becomes clear.

What I will learn: Spider's directory layout (`train_spider.json`, `dev.json`, `tables.json`, per-database `.sqlite` files), and how question/SQL pairs are linked to their database schema via a `db_id` field.

Prerequisites: Task 1's environment (needs `datasets` or direct download tooling).

What to do:
1. Download Spider, either via Hugging Face (`datasets.load_dataset("spider")` or `"xlangai/spider"`) or the original zip from the Spider project page, into Colab.
2. Inspect one entry from the train split and one from `tables.json` side by side — identify the fields: `db_id`, `question`, `query` (gold SQL), and in `tables.json`: `table_names`, `column_names`, `column_types`, foreign key info.
3. Confirm you can locate the actual SQLite `.sqlite` file for a given `db_id` on disk (needed later for execution).
4. Write a short markdown cell summarizing the schema-to-question-to-SQL relationship in your own words.

Expected result: Spider downloaded into the Colab session (or mounted from Drive for persistence across sessions), with a notebook cell that prints one fully resolved example: question, gold SQL, and the schema it refers to.

Completion criteria:
- You can print, for a single example, the question, gold SQL, and the corresponding table/column names from `tables.json` for its `db_id`.
- You've located the matching `.sqlite` file on disk for that `db_id`.
- You can explain why `db_id` is the join key tying every question to a specific schema and database file.

Connection to next task: Task 4 converts this raw (question, gold SQL, schema) structure into the instruction-tuning prompt format the model will actually be trained and evaluated on.

---

Task 4 — Preprocess Spider into instruction-tuning format

Objective: Write a preprocessing function that converts each (schema, question, gold SQL) triple into a single instruction-formatted prompt/completion pair matching the chosen model's chat template.

Why this task matters: This exact same formatting function will be reused for both the zero-shot baseline (Task 10) and every fine-tuning run in Phase 1–2 — if the prompt format is inconsistent between baseline and fine-tuning, the before/after comparison isn't valid.

What I will learn: How to serialize a database schema into a compact text form (e.g. `CREATE TABLE` statements, or a simpler `table(col1, col2, ...)` listing), how to build a chat-template-formatted prompt using `tokenizer.apply_chat_template`, and why a consistent instruction/response boundary matters for both training and generation.

Prerequisites: Task 3's understanding of Spider's structure and Task 2's chosen tokenizer.

What to do:
1. Write `schema_to_text(db_id, tables_json)` that renders a schema as short `CREATE TABLE` statements listing column names and types (skip full DDL constraints — just names/types/PK/FK is enough for prompting).
2. Write `build_prompt(schema_text, question)` that assembles a system/user message pair, e.g. system = "You are a text-to-SQL model. Given a schema and a question, output only the SQL query.", user = schema + question.
3. Apply `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` to get the final prompt string for inference, and a training-formatted version (prompt + gold SQL + EOS) for fine-tuning use in Phase 1.
4. Run this over 3–5 examples and manually read the full rendered prompt to check it looks correct (schema present, question present, no truncation).
5. Save both the prompt-building functions and a small formatted sample to a `.py` or notebook cell you'll reuse in later phases.

Expected result: A `build_prompt()` / `schema_to_text()` function pair, plus a printed example of the full rendered prompt string for at least one Spider example.

Completion criteria:
- The rendered prompt, when read top to bottom, contains a legible schema and a clear question, formatted according to the chosen model's actual chat template (not a generic hand-rolled template).
- The same function produces both an "inference" version (no gold SQL) and a "training" version (with gold SQL appended) from one shared base.
- You can explain what `add_generation_prompt=True` changes in the output and why it's needed for inference but not for building training examples.

Connection to next task: Task 5 applies this same prompt-building function across the full dataset to produce the train/val/test splits.

---

Task 5 — Build train/val/test splits

Objective: Produce three concrete dataset splits (train/val/test) with the instruction format from Task 4 already applied, ready to be loaded for both baseline evaluation and later fine-tuning.

Why this task matters: Using Spider's existing dev set as your held-out test set (rather than carving your own) keeps your baseline and fine-tuned numbers comparable to published Spider results, which strengthens the "here's the number that moved" story for interviews.

What I will learn: How to carve a validation set out of Spider's train split (since Spider's official split is only train/dev), and how to persist processed splits (e.g. as a Hugging Face `DatasetDict` or JSONL files) so later phases don't repeat this preprocessing.

Prerequisites: Task 4's prompt-building functions; Task 3's raw Spider data.

What to do:
1. Use Spider's `train_spider.json` as your base training pool and `dev.json` as your **test** set (per the PRD's instruction to reuse Spider's existing dev split as test).
2. Carve ~5–10% of the training pool off as a validation set (random split, fixed seed for reproducibility).
3. Apply Task 4's `build_prompt()` to every example in all three splits, producing formatted train/val/test sets.
4. Save each split to disk (JSONL or a saved `DatasetDict`) — ideally to Google Drive if using Colab, so you don't lose this work when the runtime resets.
5. Print the row counts for each split and spot-check 2 examples from test to confirm they still have their gold SQL and schema attached correctly (needed for eval, not just prompting).

Expected result: Three saved, formatted datasets (train/val/test) with row counts printed, persisted somewhere that survives a Colab runtime restart.

Completion criteria:
- Test set size matches Spider's dev set size (no accidental overlap with train).
- A fixed random seed is used for the train/val carve so the split is reproducible.
- You can explain why the test set must remain completely untouched by any preprocessing decision that was tuned by looking at results (e.g. don't drop "hard" examples from test after seeing they lower baseline accuracy).

---

Milestone Task 1
Data pipeline is complete and reproducible

Objective: Prove that the full path from raw Spider files to a formatted, split, persisted dataset works end-to-end without manual patching.

What I should have learned so far: How Spider's schema/question/SQL data is structured, how to render a schema into a model-readable text form, how to build a chat-template-formatted prompt, and how to split data while preserving a clean, official held-out test set.

What I should build without blindly following instructions: A single script or notebook section that runs Tasks 3–5 in sequence from a clean runtime — download → explore → preprocess → split — producing the three saved splits with no manual cell-reordering required.

Mini challenge: Re-run the entire pipeline in a **fresh** Colab runtime (Runtime → Factory reset runtime) and confirm it reproduces identical row counts and an identical first example in each split, without you fixing anything by hand.

Self-assessment: If you deleted every intermediate variable and only kept your `schema_to_text`, `build_prompt`, and split-saving code, could you explain to someone else exactly how a single Spider question turns into a training example — and why the test set can't reuse any decision made while looking at train?

Milestone that proves your progress: A fresh runtime can go from zero to three saved, formatted, correctly-sized dataset splits by running your pipeline code alone.

Completion criteria:
- The pipeline reproduces identical split sizes and first-example content on a second, independent run.
- You can point to one preprocessing decision (e.g. how much schema detail to include, or how to render foreign keys) you made deliberately rather than copying from a Spider baseline repo.

Connection to next task: Task 6 starts the eval harness, which will consume the test split produced here.

---

Task 6 — Build a per-example in-memory SQLite database from schema

Objective: Write a function that takes a Spider `db_id` and produces a ready-to-query in-memory (or temp-file) SQLite connection containing that example's actual data.

Why this task matters: Execution accuracy requires actually *running* both the gold and generated SQL against real data and comparing result sets — this is what makes the eval objective rather than a text-similarity heuristic, and it's the core technical piece the PRD calls out as the differentiator for this project.

What I will learn: How to load Spider's provided per-database `.sqlite` files with Python's `sqlite3` module, and how to safely execute arbitrary (possibly malformed) generated SQL without crashing the eval loop.

Prerequisites: Task 3's located `.sqlite` files on disk.

What to do:
1. Write `get_db_connection(db_id)` that opens the corresponding Spider `.sqlite` file in read-only mode (e.g. `sqlite3.connect(f"file:{path}?mode=ro", uri=True)`) so eval runs can't accidentally mutate the shared database file.
2. Write `run_sql(conn, sql_string)` that executes a query and returns `(success: bool, result_rows: list, error_message: str | None)` — wrapping execution in a try/except so a malformed generated query returns a clean failure instead of crashing the harness.
3. Test it manually: run a known-good gold SQL query from Task 3's example and confirm you get back the expected rows; then run a deliberately broken query (e.g. missing table name) and confirm it returns `success=False` with an error message instead of raising.

Expected result: `get_db_connection()` and `run_sql()` functions that reliably open Spider databases and execute queries against them, failing gracefully on bad SQL.

Completion criteria:
- A known-correct gold SQL query executes and returns rows matching what you'd expect by inspecting the data manually.
- A deliberately malformed query returns `success=False` without raising an unhandled exception.
- You can explain why opening the database in read-only mode matters when you're about to run hundreds of untrusted, model-generated queries against it.

Connection to next task: Task 7 uses these functions to compare a generated query's result set against the gold query's result set — the actual "execution accuracy" comparison.

---

Task 7 — Build the execution-accuracy and exact-match comparison functions

Objective: Write the two comparison functions that turn a pair of SQL strings (generated vs. gold) into a pass/fail signal for each of Phase 0's two metrics.

Why this task matters: These two functions are the entire measurement backbone of the project — every number in the Success Metrics table (Phase 0 baseline, Phase 1–2 fine-tuned results, Phase 3 quantization comparison) comes from this comparison logic, so it needs to be correct and decided on now rather than redefined later.

What I will learn: How to compare SQL result sets in an order-insensitive way (since two syntactically different but logically equivalent queries can return the same rows in different order), and the difference between execution accuracy and exact-match as evaluation signals.

Prerequisites: Task 6's `run_sql()` function.

What to do:
1. Write `execution_match(conn, generated_sql, gold_sql) -> bool`: run both queries via Task 6's `run_sql`, and if both succeed, compare result sets as **sets of tuples** (order-insensitive) rather than raw lists — decide explicitly how you'll handle column ordering differences too (e.g. sort each row's values, or require exact column order — pick one and document it).
2. Write `exact_match(generated_sql, gold_sql) -> bool`: a normalized string comparison (lowercase, strip whitespace/semicolons) — acknowledge in a comment that this is a weak secondary metric since it penalizes any valid rephrasing.
3. Test both functions on 3 handcrafted cases: (a) identical queries, (b) same logic/different column order in a `SELECT`, (c) genuinely different/wrong query — confirm `execution_match` correctly handles case (b) as a match while `exact_match` correctly does not.

Expected result: `execution_match()` and `exact_match()` functions with your handling of result-set ordering explicitly documented in a comment, validated against the 3 handcrafted test cases.

Completion criteria:
- All 3 handcrafted test cases produce the expected true/false output from both functions.
- Your ordering/comparison policy (e.g. "rows compared as sets, columns compared positionally") is written down, not just implicit in the code.
- You can explain, out loud, one specific way `exact_match` could report a false negative for a perfectly correct query.

Connection to next task: Task 8 wraps these per-example comparison functions into a full harness that runs over an entire dataset split and aggregates a percentage score.

---

Task 8 — Assemble the full eval harness

Objective: Combine Tasks 6–7 into a single harness function that takes a list of (question, schema, gold SQL) examples plus a model-generation function, and returns aggregate execution accuracy and exact-match percentages.

Why this task matters: This is the reusable tool the PRD asks for — "not a one-off notebook" — that gets called identically in Phase 0 (baseline), Phase 1–2 (each fine-tuning checkpoint), and Phase 3 (quantized versions), so the numbers stay comparable across the whole project.

What I will learn: How to structure an eval loop that's decoupled from any specific model (it should accept any function with signature `question, schema -> generated_sql_string`), and basic result logging (per-example pass/fail, not just the final aggregate) so failures can be inspected later.

Prerequisites: Tasks 6 and 7's connection and comparison functions; Task 5's test split.

What to do:
1. Write `run_eval_harness(test_examples, generate_fn) -> dict` that, for each example: calls `generate_fn(question, schema)` to get a generated SQL string, runs `execution_match` and `exact_match` against gold, and records the per-example result.
2. Have it return both an aggregate summary (`{"execution_accuracy": 0.xx, "exact_match": 0.xx, "n": ...}`) and the full per-example results list (needed later for error analysis in Phase 2).
3. Save per-example results to a CSV/JSON file (question, generated SQL, gold SQL, execution_match, exact_match) so you can manually inspect failures without re-running generation.
4. Test the harness end-to-end using a **dummy `generate_fn`** that just returns the gold SQL unchanged — this should produce ~100% on both metrics and confirms the harness plumbing itself is correct before you introduce any real model into the loop.

Expected result: A working `run_eval_harness()` function, verified against a dummy "perfect" generator, plus a saved per-example results file format you'll reuse for every future run.

Completion criteria:
- Running the harness with `generate_fn = lambda q, s: gold_sql` produces execution accuracy at or very near 100% (small SQLite quirks aside).
- The per-example results file includes enough detail (question, both SQL strings, both match flags) to debug a specific failure without rerunning anything.
- You can explain why testing the harness with a "perfect" dummy generator first is a more trustworthy sanity check than testing it directly on the real model's first attempt.

Connection to next task: The dummy-generator test proves the harness itself works; Milestone Task 2 confirms this and Task 9 finally swaps in the real base model.

---

Milestone Task 2
Eval harness is proven correct, independent of any real model

Objective: Confirm the entire measurement pipeline (schema → SQLite DB → execution comparison → aggregate metrics) is trustworthy before it's used to judge a real model's output.

What I should have learned so far: How to execute untrusted SQL safely, how to compare result sets in a way that's fair to logically-equivalent-but-differently-ordered queries, and how to structure an eval harness that's reusable across every future phase.

What I should build without blindly following instructions: A short "harness sanity check" report (a markdown cell or small script) that runs the dummy perfect-generator test from Task 8, plus at least one deliberately broken generator (e.g. one that always returns `SELECT 1;`) and confirms that one scores near 0% — proving the harness correctly distinguishes good from bad, not just that it runs.

Mini challenge: Find (or construct) one Spider example where a logically correct but differently-formatted SQL query would be marked wrong by `exact_match` but right by `execution_match`, and include it in your sanity-check report as evidence the two metrics measure different things.

Self-assessment: If your execution-accuracy number came back suspiciously high or low once you plug in the real base model in Task 9, could you tell — from this milestone's sanity checks alone — whether that's a real result or a bug in the harness?

Milestone that proves your progress: Two independent sanity checks (a "perfect" generator scoring near 100%, and a "broken" generator scoring near 0%) both behave as expected, so the harness's numbers can be trusted going forward.

Completion criteria:
- Both the perfect-generator and broken-generator sanity checks produce the expected extreme scores.
- The exact-match-vs-execution-match example is documented with the actual two SQL strings involved.
- You can point to one design decision in your comparison logic (e.g. how you handle row ordering or NULLs) that you chose deliberately rather than copying from a Spider evaluation script.

Connection to next task: With the harness trusted, Task 9 plugs in the actual chosen base model to produce the real zero-shot baseline number.

---

Task 9 — Run the zero-shot baseline with the chosen base model

Objective: Run the actual base model (Task 2's pick) through the eval harness on the full test set (or a representative subset) with no fine-tuning applied, using it as `generate_fn`.

Why this task matters: This is the number every later phase is measured against — the entire "here's the number that moved" story depends on this baseline being measured fairly and under the same conditions the fine-tuned model will later be evaluated under.

What I will learn: How to run batched or looped generation with a Hugging Face/Unsloth model (`model.generate(...)`), how to strip a chat-template response down to just the SQL string, and practical considerations for running inference over a few hundred/thousand examples within Colab's session time limits.

Prerequisites: Task 8's harness, Task 5's test split, Task 2's chosen model already loaded (no LoRA attached — base weights only).

What to do:
1. Write `generate_sql(question, schema)` that builds the inference-mode prompt (Task 4's function, `add_generation_prompt=True`), calls `model.generate()` with reasonable settings (e.g. `max_new_tokens=256`, `do_sample=False` for reproducibility), and extracts just the SQL from the model's raw text output (strip any explanation text, trailing punctuation, or markdown code fences the model might add).
2. Decide test-set size for this first run: if Spider's full dev set (~1000 examples) would take too long in one Colab session, run on a fixed random subset (e.g. 200 examples, fixed seed) and note this explicitly — you can scale to the full set later if time allows.
3. Run `run_eval_harness(test_subset, generate_sql)` and let it complete, watching for a reasonable per-example latency so you can estimate total runtime before committing to the full set.
4. Manually read through 5–10 individual failures in the saved per-example results to sanity-check that failures look like genuine model mistakes (wrong table, wrong join) rather than harness bugs (e.g. every failure being a formatting-extraction issue would indicate Task 9's SQL-extraction step needs fixing, not that the model is bad at SQL).

Expected result: A completed baseline run with an aggregate execution accuracy and exact-match percentage, plus a saved per-example results file, and manual confirmation that a sample of failures are genuine model errors.

Completion criteria:
- The baseline run completes without the harness crashing on any single example (bad generations are caught and scored as failures, not exceptions).
- At least 5 failure cases were manually read, and none of them look like a harness/extraction bug rather than a real model mistake.
- You can explain, from having read real failures, at least one recurring pattern in how the base model gets Spider questions wrong (e.g. missing a JOIN, hallucinating a column name).

Connection to next task: This baseline number is the fixed reference point Phase 1's first fine-tuning checkpoint gets compared against.

---

Milestone Task 3 (Phase 0 completion milestone)
Documented baseline and working eval harness — Phase 0 complete

Objective: Confirm the two deliverables the PRD requires for Phase 0 both exist and are trustworthy: a reusable eval harness and a documented, defensible baseline number.

What I should have learned so far: The full path from raw Spider data to a formatted, split dataset; how to safely execute and compare SQL result sets; how to build a model-agnostic eval harness; and how to run and sanity-check a real zero-shot baseline.

What I should build without blindly following instructions: A short "Phase 0 summary" write-up (a markdown cell, or the start of your eventual README) stating: the chosen base model and why, the dataset and split sizes, the baseline execution accuracy and exact-match numbers, and 2–3 example failures with your own explanation of what went wrong.

Mini challenge: Using the per-example results file, break the baseline execution accuracy down by query difficulty if Spider's difficulty labels are available (easy/medium/hard/extra), and note whether the model's accuracy drops as expected on harder queries — this becomes useful error-analysis material for Phase 2 later.

Self-assessment: If asked in an interview "why is your baseline execution accuracy X% instead of some other number," could you answer with specifics (test set size, model, generation settings, one or two example failures) rather than just restating the percentage?

Milestone that proves your progress: You can run one end-to-end demo showing a Spider question going through your pipeline — schema formatting, prompting, generation, execution comparison — and producing a clear pass/fail result, backed by a documented baseline percentage for the full test set (or your stated subset).

Completion criteria:
- A written Phase 0 summary exists with the specific baseline numbers, model choice, and dataset details — not placeholders.
- The difficulty breakdown (or an explicit note that Spider's difficulty labels weren't used) is included.
- You can name one concrete way the eval harness would need to change, if at all, to evaluate a fine-tuned model in Phase 1 (hint: it shouldn't need to change — that's the point of building it as a standalone harness now).

Connection to next task: Phase 1 begins by attaching a LoRA adapter to this same base model and re-running this exact harness on the same test set to get the first before/after comparison.