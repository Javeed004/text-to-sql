# Text-to-SQL Fine-Tuning — Experiment Log

| run_name | dataset_size | lora_rank | lora_alpha | learning_rate | epochs | execution_accuracy | exact_match | notes |
|---|---|---|---|---|---|---|---|---|
| Phase 0 (zero-shot baseline) | 0 | N/A | N/A | N/A | 0 | 61.0% | 10.5% | Qwen2.5-Coder-3B 4-bit, no adapter |
| Phase 1 (first fine-tune) | 1455 | 16 | 32 | 2e-4 | 1 | 69.5% | 38.0% | 1-epoch fast pass; hallucinated-joins fixed, self-join/NOT-IN unchanged, set-ops partial |
| phase2-variant-a-scale-epoch1 | 9380 | 16 | 32 | 0.0002 | 1 | 73.5% | 43.5% | epochs 1->2, everything else = Phase 1 (dropped from 3->2 after epoch-1 timing probe showed loss already at 0.042 — risk of memorizing the oversampled self-join/NOT-IN rows, not generalizing) — mid-training checkpoint after epoch 1 of 2 |
| phase2-variant-a-scale-epoch1 | 9380 | 16 | 32 | 0.0002 | 1 | 71.5% | 41.0% | epochs 1->2, everything else = Phase 1 (dropped from 3->2 after epoch-1 timing probe showed loss already at 0.042 — risk of memorizing the oversampled self-join/NOT-IN rows, not generalizing) — mid-training checkpoint after epoch 1 of 2 |
| phase2-variant-a-scale | 9380 | 16 | 32 | 0.0002 | 2 | 72.5% | 44.0% | epochs 1->2, everything else = Phase 1 (dropped from 3->2 after epoch-1 timing probe showed loss already at 0.042 — risk of memorizing the oversampled self-join/NOT-IN rows, not generalizing) — epoch1->final delta +1.0 pts exec acc |
| phase2-variant-b-rank32-epoch1 | 9380 | 32 | 64 | 0.0002 | 1 | 76.0% | 42.5% | rank/alpha 16/32 -> 32/64, same epochs as (a) — mid-training checkpoint after epoch 1 of 2 |
| phase2-variant-b-rank32 | 9380 | 32 | 64 | 0.0002 | 2 | 73.0% | 46.0% | rank/alpha 16/32 -> 32/64, same epochs as (a) — epoch1->final delta -3.0 pts exec acc |
| phase2-variant-c-lowlr-epoch1 | 9380 | 16 | 32 | 5e-05 | 1 | 69.0% | 41.5% | learning_rate 2e-4 -> 5e-5, same rank/epochs as (a) — mid-training checkpoint after epoch 1 of 2 |
| phase2-variant-c-lowlr | 9380 | 16 | 32 | 5e-05 | 2 | 66.5% | 42.5% | learning_rate 2e-4 -> 5e-5, same rank/epochs as (a) — epoch1->final delta -2.5 pts exec acc |

> SELECTED — Phase 2 final: variant a (72.5% / 43.5%). HF Hub: JaveedHabeeb/text-to-sql-qwen2.5-coder-3b-phase2
