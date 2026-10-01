# Serving the fine-tuned Text-to-SQL model

Qwen2.5-Coder-3B-Instruct, fine-tuned with QLoRA on Spider, merged and quantized to
`Q4_K_M` (1.93 GB), served with llama.cpp behind a FastAPI layer.

## Architecture

```
Browser / CLI / Track A agent
        |  POST /generate {schema, data_sql?, question}
        v
 [ RequestMiddleware ]  per-IP rate limit (10/min), in-flight cap (4), X-Request-Id, access log
        |
 [ FastAPI  :7860 ]  validate -> render training prompt (ChatML) -> call llama-server
        |                                                   ^
        |  POST /completion (temperature 0, stop <|im_end|>) |
        v                                                   |
 [ llama-server  127.0.0.1:8080 ]  Q4_K_M GGUF, -c 2048, CPU  
        |
        v  raw text
 [ clean_sql ] -> [ executor ] throwaway :memory: SQLite, read-only, 3 s timeout, 50-row cap
        |
        v
 {sql, latency_ms, execution_result}
```

Both processes run in one container. `start.sh` launches llama-server on loopback only, waits for
`/health`, then starts the API on `0.0.0.0:${PORT:-7860}`.

## Faithfulness of the serving path

The API renders the Qwen ChatML prompt by hand (no `transformers` at runtime). A dev script compared it
with `tokenizer.apply_chat_template` on 20 test examples: 0 mismatches.

| Check (25 Spider dev examples, Phase 3 Q4_K_M as reference) | Native server | Container |
|---|---|---|
| Identical SQL text | 23/25 | 24/25 |
| Same result rows (where both queries ran) | 22/23 | 23/23 |
| Execution accuracy vs gold, served path | 17/25 | 18/25 |
| Execution accuracy vs gold, Phase 3, same 25 | 18/25 | 18/25 |

With 25 examples one example is 4 points. The two runs differ from Phase 3 on different questions
(native: value literals such as `'right'` vs `'L'`; container: a join and ORDER BY choice on one car question),
which points to borderline-token flips between llama.cpp builds, not a systematic serving error.

## Measured numbers

Hardware: Intel Core i5-11400H (6 cores / 12 threads), 23.7 GB RAM, Windows.

| Metric | Native (llama-server.exe, no limits) | Container (`--cpus 2 --memory 4g`) | Public tunnel |
|---|---|---|---|
| End-to-end p50 | 3126 ms | 4950 ms | MEASURE |
| End-to-end p95 | 9548 ms | 12240 ms | MEASURE |
| Generation-only p50 | 2942 ms | 4786 ms | n/a |
| Generation-only p95 | 9362 ms | 12068 ms | n/a |
| llama-server memory | 2.09 GB (working set) | 1.47 GiB idle after load (`docker stats`); MEASURE under load | n/a |
| Cold start (container start to ready) | n/a | 54 s (model load about 52 s from a bind mount) | n/a |
| Image size | n/a | 1.3 GB on disk (333 MB compressed) | n/a |

Notes on method: 25 examples, sequential, greedy decoding, `127.0.0.1` rather than `localhost`.
Native numbers come from a CPU with far more threads than a free host, so treat them as optimistic.

## Limits and what I found

- **`localhost` on Windows cost about 2.2 s per request** (end-to-end minus generation) until the client
  used `127.0.0.1`; afterwards the gap was about 0.18 s. IPv6-first resolution is the likely cause.
- **Latency scales with schema length.** Prompt processing on CPU dominates long schemas, which is why
  p95 is about 3x p50.
- **Free hosting:** Hugging Face Docker Spaces need a paid plan, so the public demo runs from the local
  container behind a Cloudflare quick tunnel. The URL changes on every restart and works only while
  the machine is on.
- **Concurrency:** requests queue inside llama-server, so a burst makes every request slower. The API
  caps in-flight generations at 4 (extra requests get a 503 with `Retry-After`) and each IP at 10
  requests/minute (429). State is in-memory and per process.
- **Execution accuracy can pass wrong queries.** One served query filtered on a city that does not appear in the
  question and still matched gold, likely because both returned an empty/zero result.
- **Executor guardrails are not production hardening:** it blocks writes, `ATTACH` and `PRAGMA`, times out
  at 3 s and caps rows, but it does not bound memory use and was not security-audited.
- **Schema format:** the model was trained on `CREATE TABLE name (col type PRIMARY KEY, ...);` plus `-- FK:`
  lines. Other DDL styles are off-distribution. Sample data goes in a separate `data_sql` field so the prompt
  stays in the training format.

## What I would change for real traffic

Shared rate-limit state (Redis), a request queue with backpressure instead of a flat in-flight cap, a
GPU or larger CPU host, prompt caching for repeated schemas, and schema-aware prompt truncation.