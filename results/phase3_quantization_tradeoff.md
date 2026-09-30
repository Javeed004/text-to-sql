# Phase 3 — Quantization Tradeoff

| variant   |   file_size_gb |   execution_accuracy |   exact_match |   gen_tokens_per_sec | peak_memory_gb   |   speed_gain_vs_f16_pct |   storage_savings_vs_f16_pct |
|:----------|---------------:|---------------------:|--------------:|---------------------:|:-----------------|------------------------:|-----------------------------:|
| f16       |           6.18 |                0.745 |         0.455 |                32.5  |                  |                    0    |                        -0.03 |
| Q8_0      |           3.29 |                0.735 |         0.455 |                53.39 |                  |                   64.28 |                        46.75 |
| Q4_K_M    |           1.93 |                0.755 |         0.45  |                64.28 |                  |                   97.78 |                        68.76 |

**Note:** Peak GPU memory was not captured during the benchmark. GGUF file size is reported separately and should not be interpreted as peak runtime memory.
