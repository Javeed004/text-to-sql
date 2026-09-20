## Findings — Phase-0 — Text-to-SQL Baseline

**Base model:** Qwen2.5-Coder-3B-Instruct (4-bit) — chosen over Qwen2.5-Coder-1.5B and Llama-3.2-3B for stronger code/structured-output priors, with ~12.8GB headroom on the T4 (see Task 2 table).

**Dataset:** Spider — train pool split 92/8 into train/val (seed=42), dev set used as test (unmodified, per PRD).

**Test subset:** 200 examples (fixed seed=42 sample of Spider dev), greedy decoding.

**Baseline results:**
- Execution accuracy: **61.0%** (122/200)
- Exact match: *[insert from `loaded["summary"]["exact_match"]`]*

**Recurring failure patterns (from manual review of 10 failures):**
1. **Unnecessary/hallucinated joins** — most common pattern. Model joins to unrelated tables (e.g. `country` → `countrylanguage` for a query needing only `country`'s own columns) instead of checking if needed columns already sit on one table.
2. **Self-join failures** — e.g. "Kyle's friends" needed `Highschooler` joined to itself twice (once per role); model only joined once, returning Kyle instead of his friends. Known hard case for text-to-SQL.
3. **Set operations replaced with JOIN/WHERE logic** — `UNION`/`INTERSECT` queries rewritten as single joins with `OR`, breaking aggregate semantics.
4. **`!=` vs `NOT IN` (subquery) confusion** — filters individual rows instead of excluding entire entities.

**Difficulty breakdown:** *not performed — Spider's difficulty labels weren't used for this pass* (noted per completion criteria rather than skipped silently).

**Takeaway:** 61% execution accuracy leaves clear room for fine-tuning to move the needle, and the join-hallucination pattern is a good target for Phase 2 error-analysis-driven data curation.