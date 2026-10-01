# Phase 4 Summary — Serve It

**Input model:** `text2sql-Q4_K_M.gguf` (1.93 GB), the Phase 3 recommendation (Phase 2 variant (a), merged and quantized)
**Reference outputs:** `results/quantization_results/phase3_Q4_K_M_eval_results.json`
**Test data:** Spider dev examples from `data/test.jsonl`, the same split used in every prior phase
**Hardware:** Windows laptop, Intel Core i5-11400H (6 cores / 12 threads), 23.7 GB RAM, CPU only

## Result in one paragraph

The Q4_K_M model is now a callable service: `llama-server` behind a FastAPI layer that takes `{schema, question}` and returns `{sql, latency_ms, execution_result}`, with a guarded SQL executor, a per-IP rate limit, a demo UI, a CLI and a Docker image. The served path reproduces Phase 3: the hand-rendered prompt is byte-identical to the tokenizer's chat template, and on 25 dev examples the served SQL matches Phase 3's Q4_K_M output on 23/25 (native) and 24/25 (container), with the same result rows on every query where both ran. Latency on CPU is about 3.1 s p50 / 9.5 s p95 native and about 5.0 s p50 / 12.2 s p95 in a 2-CPU container. The public demo runs from the local container behind a Cloudflare quick tunnel, because no free host was found for a 1.9 GB CPU model.

## Pipeline

1. **Serve.** `llama-server -m text2sql-Q4_K_M.gguf -c 2048 --port 8080`, CPU only.
2. **Prompt.** `app/prompt.py` renders the training-time ChatML by hand (`build_messages`, `render_chatml`) and reuses the Phase 0 `extract_sql` logic as `clean_sql`. No `transformers` is needed at runtime.
3. **API.** `app/main.py` validates the request, calls `/completion` (`temperature=0`, stop `<|im_end|>`) and maps failures to 502 / 504.
4. **Execute.** `app/executor.py` builds a `:memory:` SQLite database from the schema, locks it down (`PRAGMA query_only`, a read-only authorizer, `SELECT`/`WITH` only), and applies a 3 s timeout and a 50-row cap. Sample rows travel in a separate `data_sql` field so the model prompt stays schema-only.
5. **Protect.** `app/middleware.py` applies 10 requests/minute per client IP and a cap of 4 in-flight generations to `POST /generate`, and adds `X-Request-Id` plus one access-log line per request.
6. **Present.** A static page at `/` with built-in examples, and `cli.py`.
7. **Package.** The Dockerfile builds on the llama.cpp server image; `start.sh` fetches the model if missing, starts `llama-server` on loopback, waits for `/health`, then starts `uvicorn`.

## Faithfulness and speed

| Check | Native server | Container (`--cpus 2 --memory 4g`) |
|---|---|---|
| Prompt vs `apply_chat_template` (20 examples) | 0 mismatches | not rerun |
| Identical SQL vs Phase 3 Q4_K_M (25 examples) | 23/25 | 24/25 |
| Same result rows (where both queries ran) | 22/23 | 23/23 |
| Execution accuracy vs gold, served | 17/25 | 18/25 |
| Execution accuracy vs gold, Phase 3 (same 25) | 18/25 | 18/25 |
| End-to-end latency p50 / p95 | 3,126 / 9,548 ms | 4,950 / 12,240 ms |
| Generation-only latency p50 / p95 | 2,942 / 9,362 ms | 4,786 / 12,068 ms |

| Resource | Value |
|---|---|
| Container cold start | 54 s (about 52 s loading the model from a bind mount) |
| Image size | 1.3 GB on disk, 333 MB compressed |
| `llama-server` working set (native) | 2.09 GB at `-c 2048` |
| Container memory (`docker stats`, idle after load) | 1.47 GiB |

**TODO:** re-measure container memory while requests are running. The idle figure understates it, since file-backed model pages may not be counted.
**TODO:** public tunnel latency for `POST /generate` (cold and warm). The only tunnel timing taken so far was a request for `/`, which shows tunnel overhead and not generation time.

## How to read the numbers

- **25 examples is a coarse instrument.** One example is 4 points and the 95% margin is roughly ±17, so these runs measure agreement with Phase 3, not accuracy. Equal accuracy (18/25 vs 18/25) does not mean the same examples were right.
- **The differing examples change between runs.** The native run differed on value literals (`'right'` vs `'L'`, a city value on the `APG` question); the container run differed on a join choice on one car question. Different llama.cpp builds flipping borderline tokens fits this. It is not proven, since the two builds were not isolated.
- **Container vs native latency is not a pure CPU effect.** The container used 2 threads and a 2-CPU limit while the native server used default threads, so about 1.6x at p50 is the combined effect.
- **p95 over 25 requests is close to the maximum.** Treat it as indicative.

## Problems hit (worth knowing for reproduction)

- **`localhost` cost about 2.2 s per request on Windows.** End-to-end minus generation latency was about 2.2 s until the client used `127.0.0.1`, after which it was about 0.18 s. IPv6-first resolution is the likely cause; it was not isolated further.
- **Hugging Face Docker Spaces need a paid plan.** The task plan assumed a free Docker Space. The demo therefore runs from the local container behind a Cloudflare quick tunnel.
- **The rate limit breaks the smoke test.** 25 requests in about 90 s from one IP exceed 10/minute. Start the container with `-e RATE_LIMIT_PER_MIN=1000` for measurements, and without it before sharing a public URL.
- **First test prompt was off-distribution.** An early `curl` used a different system message and a placeholder schema. It only proved the pipe worked; the faithful prompt came from `app/prompt.py`.
- **Windows line endings.** `start.sh` can pick up CRLF and fail in the container, so the Dockerfile strips `\r` with `sed`.
- **The `APG` question passes a wrong query.** Phase 3 scored `City = "APG"` as correct even though `APG` is an airport code, most likely because gold and generated both returned zero. Execution accuracy can pass wrong queries when the result is empty.

## Limitations

- CPU latency is high for long schemas, and a free 2-vCPU host will be slower than this i5. Expect a cold start of about a minute.
- The public demo URL is temporary and works only while the machine is on.
- Rate limit and in-flight state are in-memory and per process.
- The executor is not production hardening: it does not bound memory use and has not been audited.
- Schemas should follow the training format; other DDL styles are off-distribution.
- Not run: a parallel-load test (for example 8 simultaneous requests) and a live check that the rate limit returns 429 against the running service. The middleware is covered by unit tests only. `MAX_INFLIGHT=4` matches the server's 4 slots, but 2 may suit a 2-CPU host and is untested.

## Handoff to Phase 5

- The service contract is `POST /generate` with `{schema, question, data_sql?, execute?}`. Track A can call it as a selectable backend.
- README, demo recording and the final resume numbers are next. Use the measured figures above, and fill in the two TODO items first.
- Interview answer: "I served the quantized model with llama.cpp behind FastAPI, rendering the exact training prompt by hand and verifying it byte-for-byte against the tokenizer. On 25 held-out questions the served SQL matched my offline evaluation on 23 to 24, with identical results where both ran. On CPU it takes about 3 to 5 seconds at the median, with a roughly 12 second tail. The limits I measured and wrote down are the CPU latency, the one-minute cold start, and that 25 examples can only show agreement, not accuracy."