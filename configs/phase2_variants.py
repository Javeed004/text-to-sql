"""Phase 2 hyperparameter sweep — variant configs.

Each variant changes exactly one axis off Phase 1's baseline
(r=16, alpha=32, 1 epoch, lr=2e-4, 1455-example filtered subset), per the
one-variable-at-a-time constraint in phase-2-tasks.md Task 4.

dataset_size reflects the full Spider train split plus Task 2's targeted
self-join / NOT-IN oversampling (9,380 rows total: 6,440 base + 2,124
self-join oversamples [531 natural x4] + 816 NOT-IN oversamples
[204 natural x4], 0 synthetic — natural counts were well above the <30
threshold that would have triggered synthetic generation).

Epoch count was dropped from the original plan of 3 -> 2 after a timed
1-epoch probe on the full augmented set (train_runtime=5866.97s, loss
already down to ~0.042) suggested diminishing returns / overfitting risk
from a third pass over data where ~39% of rows are repeats of only 735
unique underlying examples.
"""

PHASE2_VARIANTS = {
    "a": {  # scale alone: same LoRA config as Phase 1, more epochs, full+augmented data
        "run_name": "phase2-variant-a-scale",
        "lora_rank": 16,
        "lora_alpha": 32,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "learning_rate": 2e-4,
        "epochs": 2,
        "dataset_size": 9380,
        "notes_axis": "epochs 1->2, everything else = Phase 1 "
                       "(dropped from 3->2 after epoch-1 timing probe showed "
                       "loss already at 0.042 — risk of memorizing the "
                       "oversampled self-join/NOT-IN rows, not generalizing)",
    },
    "b": {  # capacity: rank/alpha up, same epoch count as (a)
        "run_name": "phase2-variant-b-rank32",
        "lora_rank": 32,
        "lora_alpha": 64,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "learning_rate": 2e-4,
        "epochs": 2,
        "dataset_size": 9380,
        "notes_axis": "rank/alpha 16/32 -> 32/64, same epochs as (a)",
    },
    "c": {  # stability: lower LR, same rank as (a)
        "run_name": "phase2-variant-c-lowlr",
        "lora_rank": 16,
        "lora_alpha": 32,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "learning_rate": 5e-5,
        "epochs": 2,
        "dataset_size": 9380,
        "notes_axis": "learning_rate 2e-4 -> 5e-5, same rank/epochs as (a)",
    },
}

# Final results (200-example Spider dev subset, seed=42, identical across
# Phase 0/1/2 — see results/experiment-log.md for full history including
# the epoch-1 checkpoints and the corrupted-then-recovered variant (a) run):
#
#   Phase 0 (zero-shot):  exec_acc=61.0%  exact_match=10.5%
#   Phase 1 (first tune): exec_acc=75.5%  exact_match=38.0%
#   Variant (a):           exec_acc=72.5%  exact_match=43.5%  [3/4 failure patterns fixed]
#   Variant (b):           exec_acc=73.0%  exact_match=46.0%  [2/4 failure patterns fixed]
#   Variant (c):           exec_acc=66.5%  exact_match=42.5%  [not deep-reviewed]
#
# SELECTED: variant (a) — see results/phase2_summary.md for the full
# selection rationale (chosen over (b) despite (b)'s marginally higher
# aggregate score, which was within noise and masked a regression on the
# NOT-IN failure pattern).