# Text-to-SQL LoRA adapter — Phase 2 final (a)

Base model: unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit
LoRA config: r=16, alpha=32
Training set: 9380 examples (full Spider train + targeted
self-join/NOT-IN augmentation), 2 epochs, lr=0.0002

## Metrics (200-example Spider dev subset, seed=42, identical across Phase 0/1/2)
- Execution accuracy: 72.5%
- Exact match: 43.5%

## Selection rationale
Variant (b) (r=32/alpha=64) edges out variant (a) on aggregate execution accuracy by only 0.5 points (73.0% vs 72.5%, n=200) — well inside the ~±6.4pt margin Phase 1 calculated for this test set size, so the two are statistically indistinguishable on the top-line number alone. The failure-pattern review breaks the tie: variant (a) fixed 3 of Phase 1's 4 named hard patterns (hallucinated joins, self-joins, NOT-IN/set-exclusion), while variant (b) fixed only 2 — it regressed on the NOT-IN pattern, producing a plausible-looking but hallucinated column name (T3.pet_type instead of the schema's actual T3.pettype) that would fail outright rather than just return the wrong rows. [Sampled review of {N} additional (b) failures found this same confident-but-wrong-schema-reference pattern in {M} cases beyond the named NOT-IN example / did not recur beyond the single named example — fill in from the sampled_failures_b review above.] Given the PRD's explicit goal of measurable improvement on hard structural patterns, not just the aggregate score, variant (a) is the more defensible pick even though it is not the single highest execution-accuracy row in the table. Set-operations remains unresolved in all three variants (a/b/c each produce a real UNION shell but break the second branch differently), so this is flagged as Phase 2's one open failure pattern going into Phase 3, not silently dropped.

## Known open issues (not fixed by this checkpoint)
- Set-operations (UNION/INTERSECT) queries: model produces the correct set-op
  shell but reliably breaks the second branch's logic.
- (Confirmed via variant (b)'s broader sample, likely shared risk in (a) too,
  unverified) model_list vs car_names table confusion on car/model questions.
