# Phase 3 Summary — Merge, Quantize, Benchmark

**Input checkpoint:** Phase 2 variant (a) — `JaveedHabeeb/text-to-sql-qwen2.5-coder-3b-phase2` (LoRA r=16/alpha=32, 9,380 augmented examples, 2 epochs, LR 2e-4)
**Base model:** `unsloth/Qwen2.5-Coder-3B-Instruct` (fp16 weights, merged from the 4-bit-trained adapter)
**Test set:** same fixed 200-example Spider dev subset (seed=42), greedy decoding, used in every prior phase

## Result in one paragraph

Quantizing the merged model to Q8_0 and Q4_K_M produced **no measurable accuracy loss** on the 200-example test set. Q4_K_M is **68.8% smaller** than f16 (1.93 GB vs 6.18 GB) and generated tokens about **2x faster** in `llama-bench`. Q4_K_M is the recommended file to ship to Phase 4.

## Pipeline

1. **Merge.** Loaded the fp16 base, attached the Phase 2 adapter with `PeftModel.from_pretrained`, folded it in with `merge_and_unload()`, and saved dense safetensors. The merge was verified from the saved index: no tensor names containing `lora` or `base_layer`, and no `adapter_config.json` in the output directory.
2. **Convert.** `llama.cpp/convert_hf_to_gguf.py --outtype f16` produced `text2sql-f16.gguf`.
3. **Quantize.** `llama-quantize` produced `text2sql-Q8_0.gguf` and `text2sql-Q4_K_M.gguf`.
4. **Evaluate.** Each GGUF was served with `llama-server` (`-c 2048`, greedy, `n_predict=256`) and scored by the unchanged Phase 0 harness. The prompt is built client-side with the HF tokenizer's chat template, so it is identical to the prompt used in Phases 0-2.
5. **Benchmark.** `llama-bench` (`-p 512 -n 256`, 3 repeats) for generation tokens/sec.

## Accuracy vs. size vs. speed

| Variant | File size | Exec. acc. | Exact match | Gen tok/s | Speed vs f16 | Storage saved vs f16 |
|---|---|---|---|---|---|---|
| Phase 2 checkpoint (transformers, 4-bit base + adapter) | n/a | 72.5% | 43.5% | n/a | n/a | n/a |
| f16 GGUF | 6.18 GB | 74.5% | 45.5% | 32.50 | baseline | baseline |
| Q8_0 | 3.29 GB | 73.5% | 45.5% | 53.39 | +64.3% | 46.8% |
| Q4_K_M | 1.93 GB | 75.5% | 45.0% | 64.28 | +97.8% | 68.8% |

Peak memory was not captured (`peak_memory_gb` is empty in the saved table). **TODO:** fill it in, or state that it was not measured.
**TODO:** state whether the `llama-bench` numbers are CPU or GPU. The server log showed `n_threads = 1` and ~29 tok/s on f16, which suggests a CPU-only build, so confirm this before quoting the speed figures.

## How to read the accuracy numbers

- **A 200-example test set has a 95% margin of error of about ±6 points** at ~75% accuracy, and one example is worth 0.5 points. The 2-point spread across f16, Q8_0 and Q4_K_M is well inside that noise.
- **Q4_K_M scoring above f16 is not an improvement.** The honest reading is that accuracy is unchanged within noise across all three variants.
- **f16 is 2 points above the Phase 2 number.** A plausible reason is that Phase 2 evaluated the adapter on top of a 4-bit base, while Phase 3 merges into an fp16 base. This was not isolated by a separate experiment.
- **Exact match** is stable at 45-45.5%, which points the same way as execution accuracy.

**TODO (pairwise check):** run the f16-vs-Q4_K_M per-example comparison and record both counts here (f16 right / Q4_K_M wrong, and the reverse). Small, similar counts confirm noise; a large one-sided count would indicate a real regression.
**TODO (failure patterns):** record fixed / unchanged / broken-differently for the four named patterns on each variant, from `quant_eval_summary.json`. Phase 2's status is the reference: self-joins, NOT-IN and hallucinated joins fixed; set-operations still open.

## Recommendation

**Ship `text2sql-Q4_K_M.gguf` to Phase 4.** Against f16 it shows no accuracy difference beyond noise, 68.8% less storage, and about 2x generation speed. Q8_0 remains the fallback if the failure-pattern recheck shows Q4_K_M regressing on a named pattern.

Interview answer: "I benchmarked f16, Q8_0 and Q4_K_M on the same 200 held-out Spider questions. Accuracy was flat within the test set's ±6 point margin, while Q4_K_M cut the file from 6.2 GB to 1.9 GB and roughly doubled generation speed, so I shipped Q4_K_M. The limitation is that 200 examples can't detect a change of a couple of points, so I also checked the named failure patterns rather than relying on the aggregate."

## Problems hit (worth knowing for reproduction)

- `pip install -r llama.cpp/requirements.txt` downgrades `transformers`, `numpy`, `protobuf` and `huggingface-hub`, and Colab asks for a runtime restart. Merge and save the model first, then install the requirements and restart.
- `cmake --build ... -j` with no job limit exhausts Colab's RAM on a CUDA build. Build one target at a time with `-j1` or `-j2`.
- **A failed run that looked like a result.** The first eval scored 5% because `llama-server` had not started, and the harness scores exceptions as empty SQL. It finished in 3 seconds, which was the giveaway. The fix was a health-checked server start, a port other than 8080 (Colab uses it), `raise_for_status()`, and a smoke-test assert before each eval.
- `PHASE2_EXEC_ACC` was set to `72.5` while harness outputs are fractions, which produced a 7175.5% "gap". The constants should be `0.725` and `0.435`.
- In the saved tradeoff table, f16's storage saving shows `-0.03` where it should be 0. It is a rounding artifact from comparing a rounded size against the raw one.

## Limitations

- 200-example subset: differences under roughly 6 points cannot be resolved.
- Speed and memory are from one hardware configuration and are not portable to other machines.
- Set-operation failures from Phase 2 are still open and were not addressed by quantization or merging.
- The f16 GGUF (6.18 GB) is not stored locally. It can be regenerated from the merged model with the conversion step.

## Handoff to Phase 4

- Model file: `text2sql-Q4_K_M.gguf` (1.93 GB), served with `llama-server`.
- Serving flags that worked: `-c 2048`, and a port other than 8080.
- The FastAPI wrapper should reuse `build_prompt`, the chat-template rendering and `extract_sql` unchanged, since the accuracy numbers above depend on that exact prompt path.