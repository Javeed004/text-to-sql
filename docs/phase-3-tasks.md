# Task Breakdown — Phase 3: Merge, Quantize, Benchmark
Source: Track B PRD (Fine-Tuned Text-to-SQL Engine), Phase 3

Scope in from the PRD: merge the LoRA adapter into base weights, convert to GGUF, quantize at two or more levels (e.g. Q4_K_M, Q8_0), re-run the eval harness on the quantized versions, and benchmark latency/memory full-precision vs quantized. Deliverable: a documented accuracy-vs-latency/memory tradeoff table.

Assumption: Phase 2's best checkpoint (base model + trained LoRA adapter, pushed to Hugging Face Hub) already exists and is the input to this phase. If that's not the case yet, say so before starting Task 1.

---

Task 1 — Merge the LoRA Adapter into the Base Model

Objective: Produce a single, standalone set of full model weights that has the LoRA fine-tuning baked in, with no separate adapter needed at inference time.

Why this task matters: GGUF conversion and llama.cpp expect a normal dense model, not a base model plus a separate adapter. Skipping the merge means every later step (conversion, quantization, serving) either breaks or silently serves the un-fine-tuned base model.

What I will learn: How LoRA adapters are mathematically merged into base weights (W' = W + BA), and how to do this with Unsloth's/PEFT's merge utilities instead of treating it as a black box.

Prerequisites: Phase 2's chosen best checkpoint — the base model name/path plus the trained LoRA adapter (local path or Hugging Face Hub repo).

What to do:
- Load the base model and adapter together, e.g. with Unsloth: `model, tokenizer = FastLanguageModel.from_pretrained(base_model_name, ...)` then attach the adapter, or load directly via `PeftModel.from_pretrained(base_model, adapter_path)`.
- Call the merge step: Unsloth's `model.save_pretrained_merged("merged_model", tokenizer, save_method="merged_16bit")` (or PEFT's `model.merge_and_unload()` followed by `model.save_pretrained(...)`).
- Verify the output directory contains a full set of weights (`.safetensors` or `.bin`), `config.json`, and tokenizer files — not just an adapter's `adapter_model.safetensors`/`adapter_config.json`.
- Sanity-check the merge by generating a completion from the merged model directly (no adapter loading) on 2-3 known test prompts, and confirm the SQL output matches what the adapter+base combo produced in Phase 2.

Expected result: A `merged_model/` directory with complete, standalone model weights that produce identical outputs to the adapter-attached version, with no PEFT/adapter loading code required to use it.

Completion criteria:
- The merged model directory loads and runs with plain `transformers` (`AutoModelForCausalLM.from_pretrained`), with no PEFT import needed.
- Output on the 2-3 sanity prompts matches the pre-merge adapter outputs (same or equivalent SQL).
- You can explain, in your own words, why `W' = W + BA` means merging removes the need for a separate adapter forward pass at inference.

Connection to next task: This merged weight directory is the direct input to the GGUF conversion in Task 2.

---

Task 2 — Convert the Merged Model to GGUF Format

Objective: Convert the merged Hugging Face model into a single GGUF file that llama.cpp can load.

Why this task matters: GGUF is the format llama.cpp/Ollama actually run; without this conversion, quantization (Task 3) and serving (Phase 4) have nothing to operate on.

What I will learn: The GGUF conversion pipeline in llama.cpp, what the intermediate "f16"/"f32" GGUF represents (a lossless format conversion, not a quantization step), and how model architecture support in llama.cpp's converter can be a constraint on model choice.

Prerequisites: Task 1's `merged_model/` directory.

What to do:
- Clone or update `llama.cpp` (`git clone https://github.com/ggerganov/llama.cpp` if not already present).
- Install its Python conversion requirements (`pip install -r requirements.txt` inside `llama.cpp`).
- Run the conversion script against the merged model, e.g. `python convert_hf_to_gguf.py /path/to/merged_model --outfile text2sql-f16.gguf --outtype f16`.
- If the script errors on unsupported architecture, note the exact model family and check llama.cpp's supported-architectures list — this may be the point where Phase 0's model choice gets revisited.
- Load the resulting `text2sql-f16.gguf` with `llama.cpp`'s CLI (`./llama-cli -m text2sql-f16.gguf -p "<test prompt>"`) and confirm it generates plausible SQL for a known example.

Expected result: A single `text2sql-f16.gguf` file (full-precision GGUF, no quantization yet) that runs directly under llama.cpp and produces output consistent with the merged Hugging Face model.

Completion criteria:
- `text2sql-f16.gguf` exists and loads without error in `llama-cli` or `llama-server`.
- Output for a known test prompt is consistent with Task 1's merged-model output (same SQL or a reasonable equivalent).
- You can explain why this f16 GGUF step is a format conversion, not a compression step — i.e. why file size here is roughly the same as the original weights.

Connection to next task: This f16 GGUF file is the input that gets quantized down to smaller precision levels in Task 3.

---

Task 3 — Quantize the GGUF Model to Two or More Precision Levels

Objective: Produce at least two quantized GGUF variants of the model (e.g. Q4_K_M and Q8_0) from the f16 GGUF.

Why this task matters: The PRD's core Phase 3 claim — "quantized version should be notably faster / lower memory, documented even if accuracy dips" — can't be measured without multiple concrete quantization levels to compare against each other and against full precision.

What I will learn: What the quantization level naming means (bit-width and the "K_M"/"K_S" mixed-precision grouping scheme), and how to use llama.cpp's `llama-quantize` tool.

Prerequisites: Task 2's `text2sql-f16.gguf` file.

What to do:
- Build llama.cpp's quantize tool if not already built (`cmake -B build && cmake --build build --config Release --target llama-quantize`, or the equivalent make target for your llama.cpp version).
- Run quantization for each target level, e.g.:
  - `./llama-quantize text2sql-f16.gguf text2sql-Q8_0.gguf Q8_0`
  - `./llama-quantize text2sql-f16.gguf text2sql-Q4_K_M.gguf Q4_K_M`
- Record the resulting file size for each variant (`ls -lh`) alongside the original f16 file size.
- Smoke-test each quantized file with `llama-cli` on the same test prompt used in Task 2 and eyeball whether the SQL output still looks reasonable.

Expected result: `text2sql-Q8_0.gguf` and `text2sql-Q4_K_M.gguf` (plus the original `text2sql-f16.gguf` as the full-precision reference), each with a recorded file size, all loadable and generating output.

Completion criteria:
- Both quantized files exist, load, and produce non-garbage SQL on the smoke-test prompt.
- File sizes are recorded for all three variants (f16, Q8_0, Q4_K_M) and show the expected ordering (f16 largest, Q4_K_M smallest).
- You can explain in your own words what "K_M" adds over a naive uniform quantization (i.e. that some tensors/layers are kept at higher precision than others).

Connection to next task: These three model variants (f16 baseline, Q8_0, Q4_K_M) are what gets run through the eval harness in Task 4.

---

Task 4 — Re-run the Eval Harness on Each Quantized Variant

Objective: Get execution accuracy and exact-match numbers for the f16, Q8_0, and Q4_K_M GGUF variants on the same held-out test set used in Phases 0-2.

Why this task matters: This is the accuracy half of the "accuracy-vs-latency/memory tradeoff" the phase deliverable requires — without it you'd only have file sizes, not evidence of what quantization costs in correctness.

What I will learn: How to adapt an eval harness that previously called a Hugging Face `transformers` model so it instead calls a llama.cpp-served model (via `llama-server`'s OpenAI-compatible endpoint or the Python `llama-cpp-python` bindings), while keeping the eval logic (SQLite execution comparison, exact-match) identical.

Prerequisites: Task 3's three GGUF files; the existing eval harness and held-out test split from Phase 0.

What to do:
- Stand up each GGUF variant behind `llama-server` one at a time (`./llama-server -m text2sql-Q4_K_M.gguf --port 8080`), or load it directly with `llama-cpp-python`.
- Point the eval harness's generation call at this server/binding instead of the Phase 0-2 `transformers` pipeline, changing only the inference call — the SQL-execution comparison and exact-match logic stay untouched.
- Run the full held-out test set through each of the three variants in turn, recording execution accuracy and exact-match for each.
- Build a small comparison table: rows = {merged f16, Q8_0, Q4_K_M, and the Phase 2 fine-tuned `transformers` result as a cross-check}, columns = {execution accuracy, exact-match}.

Expected result: A table with execution accuracy and exact-match numbers for all three GGUF precision levels, plus confirmation that f16 GGUF accuracy is close to the original Phase 2 `transformers` result (validating the merge+conversion didn't silently break anything).

Completion criteria:
- All three variants have recorded execution accuracy and exact-match numbers on the same test set.
- f16 GGUF accuracy is within a small, explainable margin of the Phase 2 `transformers` checkpoint's accuracy (any large gap gets investigated, not ignored).
- You can explain why re-using the exact same eval harness logic (not a reimplementation) is what makes this comparison valid.

Connection to next task: This accuracy table is one half of the phase's final deliverable; Milestone Task 5 checks it against expectations before moving to the latency/memory half.

---

Milestone Task 5
Accuracy Impact of Quantization — Checkpoint

Objective: Confirm that quantization's effect on accuracy is understood and correctly measured before spending time on latency/memory benchmarking.

What I should have learned so far: How LoRA merging collapses adapter + base into one set of weights, how GGUF conversion and quantization work mechanically, and how to route an existing eval harness at a llama.cpp-served model without changing its scoring logic.

What I should build without blindly following instructions: A short written comparison (a few paragraphs, in your own words) of the Task 4 table — which quantization level held up best, whether the drop from f16 to Q4_K_M was larger on execution accuracy or exact-match, and a guess at why (e.g. more precision loss hurting exact string formatting more than semantic correctness, or vice versa).

Mini challenge: Look at 3-5 test examples where Q4_K_M got the wrong answer but f16 got it right, and categorize the failure (e.g. wrong column, malformed JOIN, syntax error, numeric rounding). This is manual error analysis, not just a number.

Self-assessment: If someone deleted your accuracy table and only left you the raw per-example outputs from all three variants, could you reconstruct the table and explain which quantization level you'd recommend shipping, and why?

Milestone that proves your progress: You have three working GGUF model variants with verified, comparable accuracy numbers, and a defensible opinion on whether the accuracy cost of quantization is acceptable.

Completion criteria:
- The accuracy comparison table from Task 4 is finalized and saved (not just printed to a terminal you'll lose).
- You can point to at least one specific example (with the model's actual output) that illustrates the accuracy difference between two precision levels, not just the aggregate percentages.

Connection to next task: With accuracy differences understood, Task 6 measures the other side of the tradeoff — speed and memory — for the same three variants.

---

Task 6 — Benchmark Inference Latency and Memory Footprint

Objective: Measure and record inference latency and peak memory usage for the f16, Q8_0, and Q4_K_M GGUF variants, run under the same conditions.

Why this task matters: The PRD explicitly requires documenting that "the quantized version should be notably faster / lower memory... even if accuracy dips slightly" — this is the evidence that justifies quantization as a deployment decision, not just an accuracy comparison.

What I will learn: How to measure real inference latency (time-to-first-token and tokens/sec) and memory footprint for llama.cpp models, and why controlling for hardware/batch-size/sequence-length is necessary for a fair comparison.

Prerequisites: Task 3's three GGUF files, run on the same machine used for Task 4 (ideally the RTX 2050 laptop, or explicitly note if using Colab/CPU instead, since hardware must stay constant across the comparison).

What to do:
- Use llama.cpp's built-in `llama-bench` tool for consistent, repeatable measurement: `./llama-bench -m text2sql-Q4_K_M.gguf` (repeat for each variant).
- Record prompt-processing speed (tokens/sec), generation speed (tokens/sec), and peak VRAM/RAM usage (via `nvidia-smi` during the run, or llama-bench's reported memory if available) for each of the three variants.
- Run each benchmark at least 3 times and average, to smooth out noise — a single run's numbers aren't reliable enough to put in a report.
- Keep prompt length and generation length constant across all three runs so the comparison isn't confounded by different input/output sizes.

Expected result: A table of {model variant, tokens/sec generation speed, peak memory usage} for f16, Q8_0, and Q4_K_M, averaged over multiple runs, all measured on the same hardware.

Completion criteria:
- Latency and memory numbers exist for all three variants, each averaged over 3+ runs.
- Prompt/generation length was held constant across the three benchmark runs (stated explicitly in your notes).
- You can explain why a single-run benchmark number would be misleading and why averaging matters here.

Connection to next task: This latency/memory table combines with Task 4's accuracy table to form the final Phase 3 deliverable, assembled in Milestone Task 7.

---

Milestone Task 7
Documented Accuracy-vs-Latency/Memory Tradeoff Table

Objective: Combine the accuracy results (Task 4) and the latency/memory results (Task 6) into the single documented tradeoff table the PRD names as this phase's milestone/deliverable.

What I should have learned so far: The full merge → convert → quantize → re-evaluate → benchmark pipeline, and how to reason about accuracy/speed/memory as a three-way tradeoff rather than optimizing one in isolation.

What I should build without blindly following instructions: One combined table (not two separate ones) with rows for f16 (baseline), Q8_0, and Q4_K_M, and columns for execution accuracy, exact-match, generation speed, and peak memory — plus a short written recommendation on which variant you'd actually ship for Phase 4's serving step, and why.

Mini challenge: Write the 3-4 sentence version of this tradeoff you'd say out loud in an interview if asked "why did you pick this quantization level?" — it should reference actual numbers from your table, not vague claims like "it was faster."

Self-assessment: Could you defend your chosen quantization level to someone who argues you should have picked a different one, using only the numbers in your table?

Milestone that proves your progress: A single, self-contained artifact (table + short writeup) that fully answers the PRD's Phase 3 deliverable, and a specific model file chosen to carry forward.

Completion criteria:
- The combined table includes all three variants across all four metrics (execution accuracy, exact-match, speed, memory).
- A written recommendation names one specific GGUF file as the one to serve next, backed by a reason drawn from the table.
- You can point to one design decision in this phase (e.g. which quantization levels you chose to test, or how you controlled the latency benchmark) that you made deliberately rather than copying from a tutorial.

Connection to next task: The chosen GGUF file and this tradeoff table are the direct inputs to Phase 4 (serving the model behind a FastAPI wrapper) — the served model is exactly the one selected here.