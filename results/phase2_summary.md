## Findings — Phase 2 — Iterate on Data & Hyperparameters

**What moved and why:**

Phase 2 scaled from Phase 1's 1,455-example filtered subset to the full
Spider training pool (6,440 examples) plus targeted oversampling of the two
patterns Phase 1 found completely unmoved — self-joins and NOT-IN/set-exclusion
— for a total augmented training set of 9,380 rows (531 natural self-join
examples and 204 natural NOT-IN examples, each oversampled 4x on top of their
one natural occurrence, for an effective 5x representation; no synthetic
examples were needed since both natural counts were well above the <30
threshold that would have triggered synthetic generation).

Three LoRA hyperparameter variants were swept, each changing exactly one axis
off Phase 1's baseline (r=16, alpha=32, 1 epoch, lr=2e-4):

| Run | Dataset | LoRA | Epochs | LR | Exec. Acc. | Exact Match |
|---|---|---|---|---|---|---|
| Phase 0 (zero-shot) | — | — | 0 | — | 61.0% | 10.5% |
| Phase 1 (first tune) | 1,455 | r=16/a=32 | 1 | 2e-4 | 75.5% | 38.0% |
| Variant (a) — scale | 9,380 | r=16/a=32 | 2 | 2e-4 | **72.5%** | **43.5%** |
| Variant (b) — rank | 9,380 | r=32/a=64 | 2 | 2e-4 | 73.0% | 46.0% |
| Variant (c) — low LR | 9,380 | r=16/a=32 | 2 | 5e-5 | 66.5% | 42.5% |

(Epoch count was planned at 3 but cut to 2 after a timed 1-epoch probe on
the full augmented set showed loss already down to ~0.042 — a third pass
over data where ~39% of rows are repeats of only 735 unique underlying
examples looked more likely to encourage memorization than generalization.)

**The aggregate numbers alone are misleading here, exactly as Phase 1's own
report warned.** No variant beat Phase 1's 75.5% execution accuracy — scaling
data 6.4x and adding a full-dataset targeted-curation pass did not reproduce
anything like Phase 1's original jump off zero-shot. Exact match improved
across the board (43.5–46.0% vs. Phase 1's 38.0%), suggesting continued
exposure to real Spider examples keeps refining stylistic conformance even
where raw correctness plateaus or dips slightly.

**Variant (b) edged out variant (a) by only 0.5 points on execution accuracy
(73.0% vs. 72.5%, n=200) — well inside the ~±6.4pt margin Phase 1 calculated
for this test-set size, i.e. statistically indistinguishable.** The
failure-pattern review broke the tie. Re-running Phase 1's four named hard
examples against each checkpoint:

| Pattern | Phase 1 | Variant (a) | Variant (b) |
|---|---|---|---|
| Hallucinated/unnecessary joins | FIXED | FIXED | FIXED |
| Self-join failures | unchanged | **FIXED** | **FIXED** |
| Set operations (JOIN/WHERE → UNION) | partial | broken differently | broken differently |
| != vs NOT IN confusion | unchanged | **FIXED** | **regressed** |

Variant (a) fixed 3 of 4 named patterns; variant (b) fixed only 2, regressing
on the NOT-IN pattern it should have inherited the same curation benefit
for. Its failure there was a hallucinated column name
(`T3.pet_type` instead of the schema's actual `T3.pettype`) — plausible-looking
but wrong, not a logic error. A 15-example sample of variant (b)'s broader
test-set failures confirmed this wasn't a one-off: the same
confident-but-schema-wrong pattern recurred twice more (another `pet_type`
instance, and a fabricated `car_makers.country` column where the real schema
requires a join through `countries`), for 3 of 15 sampled failures showing
this specific failure mode.

**That same 15-sample review surfaced a new, previously unflagged systematic
pattern**: 4 of 15 failures confuse `model_list` with `car_names` as the
correct join table for car/model questions — routing `model_list` directly
to `cars_data` when the schema requires `car_names` as the bridge table. This
was not one of Phase 1's four named patterns and was more prevalent in this
sample than the NOT-IN regression. It is flagged here as an open item for
Phase 3, not resolved by any variant this phase, and was only spot-checked
against variant (b) — unverified whether variant (a) shares it.

**Set operations remain unresolved in every variant.** All three produced a
real `UNION`/`INTERSECT` shell for the first time (Phase 1 never did), but
each broke the second branch's logic differently: Phase 1 introduced an
extra irrelevant join; variant (a) referenced an undefined table alias;
variant (b) used an aggregate function (`count(*)`) directly in a `WHERE`
clause with no `GROUP BY`, which SQLite rejects outright. **Verdict: the
"does scale alone fix set-ops" hypothesis from Phase 1 is not confirmed.**
Scale produces the right shell structure but not the right branch-level
logic — this looks like it needs the same kind of targeted curation applied
to self-joins/NOT-IN in Task 2, not more data volume.

**Checkpoint selection (Task 11):** Variant (a) was selected as the Phase 2
final checkpoint over variant (b), despite (b)'s marginally higher aggregate
execution accuracy, on the strength of the failure-pattern evidence above —
consistent with Phase 1's own lesson that an aggregate number alone can hide
which specific hard patterns actually improved or regressed.

**Operational lessons (worth carrying into Phase 3):**
- A double `PeftModel.from_pretrained()` wrap (once via `attach_lora()`,
  again via a resume-from-checkpoint reload) silently produces a
  near-inert adapter rather than erroring — caught only because generated
  SQL for the four named examples exactly matched Phase 0's zero-shot
  outputs character-for-character. Any future resume/reload path should
  assert the model isn't already a `PeftModel` before wrapping again.
- `save_adapter_to_drive()`'s fixed save path meant a corrupted run
  silently overwrote a good checkpoint with no warning. Recovery was only
  possible because the HF `Trainer`'s own `save_steps` checkpointing wrote
  to a separate path independently of that call.
- Colab session disconnects during the sweep (once mid-training, once via
  `KeyboardInterrupt`) made per-stage checkpointing and a completion
  marker file (to make reruns skip already-finished stages rather than
  silently repeat them) necessary rather than optional.

**Net for Phase 2:** self-join and NOT-IN curation worked — both patterns
that showed zero movement in Phase 1 are now fixed in the selected
checkpoint. Set-ops remains open and looks like a curation problem, not a
scale problem. A new pattern (`model_list`/`car_names` confusion) surfaced
that wasn't visible in Phase 1's smaller sample. Aggregate execution
accuracy did not beat Phase 1's number, but exact match improved and the
pattern-level story — the thing this project is actually trying to
demonstrate — moved from 1/4 fixed to 3/4 fixed.