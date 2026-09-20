## Findings — Phase-1 — Text-to-SQL Finetuning v1
 
**What moved and why:**

Fine-tuning Qwen2.5-Coder-3B-Instruct (4-bit) with a single QLoRA pass
(r=16, alpha=32, all 7 attention+MLP projections, 1 epoch, 1455 filtered
Spider training examples, ~13.3 min on a T4) moved execution accuracy from
61.0% to 75.5% (+14.5 pts) and exact match from 10.5% to 38.0% (+27.5 pts),
measured on the identical 200-example Spider dev subset used for Phase 0's
baseline.
 
The exact-match delta being nearly double the execution-accuracy delta is
the most interesting single finding here: Phase 0's baseline was often
semantically correct but styled differently from Spider's SQL conventions
(extra DESC, different casing, unnecessary GROUP BY/joins). Fine-tuning on
real Spider examples appears to have taught the model Spider's specific
style on top of raw correctness — visible directly in the soccer_2
milestone example (zero-shot added a wrong GROUP BY; fine-tuned didn't) and
in failure-pattern example 1 (fine-tuned matched gold's execution exactly,
missing only on a quote-style difference).
 
The failure-pattern review (Task 9) tells a more nuanced story than the
aggregate numbers: of Phase 0's four named hard patterns, only "hallucinated
joins" was clearly fixed. Self-joins and NOT-IN-based exclusion were
unchanged, and the set-operations pattern showed a mixed result — partial
structural improvement alongside a newly introduced irrelevant join. This
is a legitimate current limitation, not a hidden failure: a 1-epoch pass on
a ~1500-example subset was always going to fix the failure modes with
enough representation in that subset before the rarer, harder structural
ones.
 
**Hypothesis for what Phase 2 should try first** (mini challenge): the
self-join and NOT-IN patterns are the best candidates for Phase 2's planned
error-analysis-driven data-quality pass specifically because they showed
*zero* movement here — that's a stronger signal of "needs more/better
examples of this exact pattern" than the set-ops category, which already
showed partial learning and might resolve simply by scaling to the full
training set rather than needing curated examples.
 
**On "is this just noise from a small test set?"**: at n=200 and ~62-76%
accuracy, the standard error on a binomial proportion is roughly
sqrt(p(1-p)/n) ≈ sqrt(0.7*0.3/200) ≈ 3.2 percentage points, so a rough 95%
CI is about ±6.4 pts. A 14.5-pt move is more than 2x that margin — not
noise. The 27.5-pt exact-match move is even further outside any plausible
margin-of-error explanation.

## Actual failure-pattern results
 
**1. Hallucinated/unnecessary joins — FIXED.** Fine-tuned query dropped the
unnecessary countrylanguage join entirely, matching gold's structure.
execution_match=True confirms real correctness; exact_match=False is a pure
quote-style artifact (' vs ") on an otherwise identical query, not a real
error — reinforces Phase 0's own note that exact_match is a weak metric
that penalizes valid textual variation.
 
**2. Self-join failures — STILL BROKEN, unchanged.** Fine-tuned still only
joins Highschooler once (would return Kyle, not his friends) — identical
structural mistake to zero-shot. Self-joins weren't learned at this
training scale; plausible cause is too few self-join examples in the
1455-example subset to generalize the pattern.
 
**3. Set-ops as JOIN/WHERE — STILL BROKEN, but differently.** No UNION
produced (core misunderstanding persists), but each OR branch is now a
cleaner, correctly-scoped subquery — a real decomposition improvement. Cost:
a new, irrelevant "JOIN continents" appeared in the outer FROM clause that
wasn't in zero-shot's output at all. Partial improvement + a newly
introduced hallucination in the same query. Worth watching in Phase 2 for
whether "extra unrelated join" becomes a broader pattern at full-dataset
scale (possible early overfitting signal from a small subset).
 
**4. != vs NOT IN confusion — STILL BROKEN, unchanged.** Same row-level
`!=` filter instead of set-exclusion via NOT IN subquery; only cosmetic
formatting changed (lowercase aliases, quote style).
 
**Net: 1/4 targeted hard patterns genuinely fixed, 2 unchanged, 1 broken
differently with a new artifact.** An aggregate 75.5%/38.0% alone wouldn't
have surfaced any of this — the model learned the "easy" fix (drop an
unneeded join on a simple single-table aggregate) but the harder structural
patterns (self-joins, UNION, NOT IN subqueries) need more signal than a
1-epoch pass on 1455 examples provided.