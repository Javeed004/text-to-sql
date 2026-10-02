# Phase 4 — Serve It: Task Breakdown

**Project:** Track B — Fine-Tuned Text-to-SQL Engine
**Phase goal:** Wrap the chosen quantized GGUF model in a server, put a thin FastAPI layer on top (`{schema, question}` → `{sql, execution_result}`), deploy it on a free tier, and build a minimal CLI/web form for demos.
**Phase deliverable:** A live, callable endpoint + a working demo UI.
**Resume-ready achievement:** "Deployed a self-hosted, quantized fine-tuned LLM as a production-style inference API."

**Assumed inputs from earlier phases:** the best quantized GGUF from Phase 3 (e.g. Q4_K_M), the exact prompt template used in Phase 1/2 training, and the execution-accuracy eval harness from Phase 0.

**Decision made for this breakdown:** serve with **llama.cpp's `llama-server`** (OpenAI-compatible API, easy to containerize). Ollama works too for local testing, but one path keeps the deployment simple. Target host: **Hugging Face Spaces (Docker, free CPU)**. See the warning in Task 6 about Render/Railway memory limits.

---

## Task 1 — Serve the GGUF locally with llama-server

**Objective:** Run the quantized model through `llama-server` and get a valid completion back from `curl`.

**Why this task matters:** Everything else in this phase sits on top of a working inference server. If the model doesn't load or respond correctly here, you'd be debugging FastAPI, Docker and hosting problems at the same time later.

**What I will learn:** What `llama-server` is and how it exposes `/v1/chat/completions` and `/health`; the flags `-m` (model path), `-c` (context size), `-ngl` (GPU layers offloaded), `--port`, `-t` (CPU threads); the difference between CPU-only and GPU-offloaded inference.

**Prerequisites:** The Phase 3 quantized `.gguf` file on disk (Q4_K_M, plus Q8_0 if you want to compare).

**What to do:**
1. Install llama.cpp (prebuilt release binary, or build from source with CUDA if you want GPU offload on the RTX 2050).
2. Start the server: `llama-server -m ./models/text2sql-q4_k_m.gguf -c 2048 --port 8080 -ngl 0` (CPU first).
3. Check `curl http://localhost:8080/health`.
4. Send a hand-written request to `/v1/chat/completions` with a short schema and question; set `temperature` to `0`.
5. Repeat with `-ngl 99` (GPU) and note tokens/second for both modes in a small `notes/serving.md`.
6. Note the RAM used by the process (Task Manager / `htop`). You'll need this number for hosting in Task 6.

**Expected result:** A running server on port 8080 that returns a JSON completion containing SQL, plus a notes file with CPU vs GPU speed and memory numbers.

**Completion criteria:**
- `/health` returns OK and a completion request returns SQL text.
- You have recorded RAM usage and tokens/sec for CPU-only mode.
- You can explain what `-c` (context size) limits and why it must be at least as large as your longest prompt plus output.

**Connection to next task:** The raw output may look wrong or rambling if the prompt doesn't match training. Task 2 fixes that.

---

## Task 2 — Reproduce the training prompt format exactly

**Objective:** Build a single `build_prompt(schema, question)` function that produces the identical format the model saw in fine-tuning, and confirm the server output matches what the eval harness got.

**Why this task matters:** A fine-tuned model is very sensitive to its prompt format. A mismatch in the template (missing system text, different schema layout, wrong chat markers) quietly costs accuracy, and you'd blame quantization or serving for it.

**What I will learn:** How chat templates work (`apply_chat_template` vs llama-server's built-in template handling); how to stop generation cleanly (stop tokens / EOS); how to post-process a completion into a single clean SQL statement.

**Prerequisites:** Task 1 server running; your Phase 1 preprocessing code (the script that turned schema + question + gold SQL into instruction format).

**What to do:**
1. Copy the prompt-building logic out of the preprocessing script into a shared module, e.g. `app/prompt.py`, so training-side and serving-side code import the same function where possible.
2. Decide how the template is applied: either send a fully formatted prompt to `/completion`, or send `messages` to `/v1/chat/completions` and confirm the GGUF's embedded chat template matches training.
3. Add `clean_sql(text)`: strip markdown fences, trailing explanations, and anything after the first `;`.
4. Take 20 examples from the Spider dev split, run them through the server, and compare outputs to what your eval harness produced for the same quantized model in Phase 3.
5. Investigate any difference in outputs (template, temperature, stop tokens).

**Expected result:** `app/prompt.py` with `build_prompt` and `clean_sql`, and a small script showing the server's outputs agree with the Phase 3 eval outputs on the 20 examples.

**Completion criteria:**
- At least 18 of 20 served outputs match the Phase 3 outputs (or differences are explained).
- Temperature is `0` and stop tokens are configured, so the same input gives the same output twice.
- You can explain what would go wrong if the serving prompt differed from the training prompt.

**Connection to next task:** With a trustworthy prompt and clean SQL, you can wrap it in an API with a proper contract.

---

## Task 3 — Build the FastAPI layer (`/generate`)

**Objective:** An API endpoint that accepts `{schema, question}` and returns `{sql}` by calling llama-server.

**Why this task matters:** This is the contract Track A's coding agent and your UI will call. Getting request/response models and error handling right now keeps later integration painless.

**What I will learn:** FastAPI app structure; Pydantic request/response models and validation (max lengths); `httpx.AsyncClient` for calling the LLM server with timeouts; `lifespan` events; returning proper HTTP errors (422, 502, 504).

**Prerequisites:** Task 2's `app/prompt.py`; llama-server from Task 1 running.

**What to do:**
1. Create `app/main.py` with `POST /generate`, request model `GenerateRequest(schema: str, question: str)` (cap schema at roughly 8k chars, question at 500).
2. Read the LLM server URL from an environment variable (`LLM_BASE_URL`, default `http://localhost:8080`) so the same code works locally and in Docker.
3. Call the server with `httpx`, a timeout (e.g. 60 s), `temperature=0`, and `max_tokens` around 256.
4. Return `{sql, latency_ms}`. Map server-down to 502 and timeout to 504.
5. Add `GET /health` that checks the LLM server too.
6. Test with FastAPI's auto-generated `/docs` page and with `curl`.

**Expected result:** A FastAPI app where `/docs` shows both endpoints and `POST /generate` returns correct SQL for a sample schema.

**Completion criteria:**
- Valid request returns 200 with `sql` and `latency_ms`.
- Oversized schema returns 422; stopping llama-server makes `/generate` return 502 (not a 500 stack trace).
- You can explain why the LLM URL lives in an environment variable.

**Connection to next task:** Now that SQL is generated, the API needs to execute it and return rows.

---

## Task 4 — Add safe SQL execution (`execution_result`)

**Objective:** Extend the API so it builds a throwaway SQLite database from the provided schema, runs the generated SQL safely, and returns rows or a readable error.

**Why this task matters:** The PRD's endpoint returns `{sql, execution_result}`, and the demo story ("question → SQL → actual results") depends on it. Even though production hardening is a non-goal, you are still running model-generated text against a database, so basic guardrails are required.

**What I will learn:** Building a SQLite DB in memory from DDL (`executescript`); making a connection read-only after setup (`PRAGMA query_only = ON`); enforcing a query timeout with `sqlite3`'s `set_progress_handler`; allow-listing only `SELECT` statements; capping returned rows.

**Prerequisites:** Task 3 API; your Phase 0 eval harness (reuse its schema-to-SQLite code rather than rewriting it).

**What to do:**
1. Move the reusable DB-building code from the eval harness into `app/executor.py`.
2. Implement `run_sql(schema_ddl, sql, max_rows=50, timeout_s=3)`:
   - create a fresh `:memory:` DB and run the schema DDL (reject DDL containing anything suspicious if it fails to parse),
   - set `PRAGMA query_only = ON`,
   - reject anything whose first keyword is not `SELECT` or `WITH`,
   - install a progress handler that aborts after the time limit,
   - return `{columns, rows, truncated, error}`.
3. Wire it into `/generate` and add an `execute: bool = true` request flag.
4. Write 5 pytest cases: valid query, syntax error, `DROP TABLE` attempt, infinite-style query (recursive CTE) hitting the timeout, and >50 rows truncation.
5. Note in your README what this does *not* protect against, per the PRD's non-goals.

**Expected result:** `/generate` returns `{sql, execution_result: {columns, rows, truncated, error}, latency_ms}`; the test file passes.

**Completion criteria:**
- All 5 pytest cases pass.
- A destructive statement produced by the model (simulate by sending one directly) is rejected, not executed.
- You can explain why the DB is empty of data (schema only) and what that means for what the demo can show. *(Hint: decide whether to accept optional `INSERT` sample rows in the schema field so results are meaningful.)*

**Connection to next task:** The pieces (model server, prompt, API, executor) now exist separately; the milestone proves they work as one system.

---

## Milestone Task 5
**Local End-to-End Text-to-SQL Service**

**Objective:** Combine llama-server, the shared prompt module, FastAPI, and the safe executor into one locally running service and verify it on real examples.

**What I should have learned so far:** How a GGUF model is served over HTTP; why the serving prompt must match training; how to design a small API contract with validation and error mapping; how to run untrusted, model-generated SQL with basic guardrails.

**What I should build without blindly following instructions:** A script `scripts/smoke_test.py` that sends 25 Spider dev examples (with their schemas and sample data where available) to `/generate` and reports execution accuracy through the *served* path, written by you from the pieces above.

**Mini challenge:** Compare your served-path execution accuracy with the Phase 3 number for the same quantization. If they differ by more than about 2 points, find and explain the cause. Also log p50 and p95 latency across the 25 calls.

**Self-assessment:** If you deleted `app/` and only had the smoke-test output, could you explain to a colleague each hop a request takes (API → prompt → llama-server → cleaner → executor) and what can fail at each hop?

**Milestone that proves your progress:** `curl` a schema and question to your local API and get back correct SQL plus real result rows, with measured accuracy and latency through the full serving stack.

**Completion criteria:**
- Smoke test runs end-to-end and prints accuracy, p50 and p95 latency.
- Served-path accuracy is within about 2 points of Phase 3's number, or the gap is explained.
- You can point to one design decision you made differently from the obvious approach (for example where you placed the SQL cleaning or how you handled timeouts) and justify it.

**Connection to next task:** The service only works on your machine; next you package it so it runs anywhere.

---

## Task 6 — Containerize the model server and API together

**Objective:** One Docker image that starts llama-server and FastAPI and answers `/generate` when run locally.

**Why this task matters:** Free hosts deploy containers, not your laptop environment. If it runs in Docker locally, deployment mostly becomes a push.

**What I will learn:** Writing a `Dockerfile` for a Python + llama.cpp service; a startup script that launches two processes and waits for the model server to be ready; `.dockerignore`; image size trade-offs; fetching the model at build/start time from Hugging Face Hub instead of baking it into git.

**Prerequisites:** Milestone Task 5 (working local service); model pushed to Hugging Face Hub in Phase 2/3 (upload the GGUF to a model repo if not already there).

**What to do:**
1. **Check hosting memory limits first.** Use your RAM number from Task 1. As of this writing, Render's free web service has a small RAM limit (512 MB) that a Q4 1.5B model won't fit, and Railway no longer has a permanently free tier. Confirm current limits on each provider's pricing page. HF Spaces free CPU hardware has much more RAM and is the better fit; 0.5B models are the only realistic fit for very small hosts.
2. Write a `Dockerfile` based on a slim Python image that installs a prebuilt `llama-server` (llama.cpp release binary or the official `ghcr.io/ggml-org/llama.cpp:server` image as a base) plus `fastapi`, `uvicorn`, `httpx`.
3. Write `start.sh`: download the GGUF with `huggingface_hub` if absent, start `llama-server` in the background on `127.0.0.1:8080`, poll `/health` until ready, then start `uvicorn app.main:app --host 0.0.0.0 --port 7860` (the port HF Spaces expects).
4. Build and run: `docker build -t text2sql .` then `docker run -p 7860:7860 text2sql`.
5. Hit `/generate` from your host and confirm it works; record the image size and cold-start time.

**Expected result:** A Docker image that serves `/generate` on port 7860 with no dependency on your local Python environment, plus recorded image size and cold-start time.

**Completion criteria:**
- `docker run` followed by a `curl` to `/generate` returns correct SQL and rows.
- The model file is not committed to git (pulled from the Hub at start).
- You can explain why the API talks to llama-server on `localhost` inside the container and what would break if the two started in the wrong order.

**Connection to next task:** The container is ready to be pushed to a free host.

---

## Task 7 — Deploy to a free host and verify the live endpoint

**Objective:** The API is reachable at a public URL and returns correct SQL and results.

**Why this task matters:** "Live, callable endpoint" is the phase's deliverable, and a public URL is what makes the project demoable to recruiters without setup.

**What I will learn:** Creating a Hugging Face Docker Space; Space metadata (`sdk: docker`, `app_port: 7860`) in the README front matter; secrets and environment variables on a host; reading build and runtime logs; dealing with cold starts and sleeping free apps.

**Prerequisites:** Task 6 image working locally; a Hugging Face account; model GGUF on the Hub.

**What to do:**
1. Create a new Docker Space and push your repo (Dockerfile, `start.sh`, `app/`, `requirements.txt`, README with the required front matter).
2. Set `MODEL_REPO` / `MODEL_FILE` as variables if your `start.sh` reads them.
3. Watch the build logs; fix failures (missing packages, wrong port, out-of-memory kill).
4. Once running, call `https://<your-space>.hf.space/generate` with `curl` and through `/docs`.
5. Measure cold-start time and warm latency; note both in your README. Free Spaces sleep after inactivity, so document that the first request is slow.
6. If the model doesn't fit or is too slow on the free CPU tier, fall back to the smaller model/quantization (for example Q4_K_M of the smallest model that still performed well) and record why.

**Expected result:** A public URL where `POST /generate` returns `{sql, execution_result}`.

**Completion criteria:**
- A request from a device that is not your laptop (your phone, for example) returns correct SQL.
- Warm latency and cold-start time are recorded.
- You can explain what you would change to handle real traffic (concurrency limits, rate limiting, paid hardware), even though that's out of scope here.

**Connection to next task:** The endpoint works but only for people who can write `curl`; the next task gives it a human-friendly front end.

---

## Task 8 — Build the CLI and a minimal demo UI

**Objective:** A CLI command and a one-page web form that take a schema and question and display the generated SQL and result table, both talking to the live API.

**Why this task matters:** The demo is how people actually experience the project. A clean "paste schema, ask question, see SQL and rows" flow is what you'll screen-record in Phase 5.

**What I will learn:** Building a CLI with `argparse` or `typer`; calling an HTTP API with `httpx`/`requests`; a minimal UI with Gradio (simplest on HF Spaces) or a single static HTML page with `fetch`; rendering result rows as a table; handling loading and error states in a UI.

**Prerequisites:** Task 7 live endpoint (or the local API from Milestone 5).

**What to do:**
1. CLI: `cli.py --schema schema.sql --question "How many singers are older than 30?" --url <API_URL>`; print the SQL, then a formatted table of rows, or the error.
2. Web UI: either a Gradio app (two text boxes, a submit button, outputs for SQL and a dataframe) or one static `index.html` served by FastAPI at `/`.
3. Add 2–3 built-in example schemas (for instance a singers/concerts database with sample `INSERT` rows) behind an "Examples" dropdown so a visitor can try it in one click.
4. Show a "model is waking up" message if the request takes more than a few seconds (free-tier cold start).
5. Redeploy and test the UI on the public URL.

**Expected result:** `cli.py` plus a web page at your public URL that completes the question → SQL → rows flow using a built-in example.

**Completion criteria:**
- CLI and web UI both return SQL and rows for an example question against the live endpoint.
- An invalid or failing query shows a readable error instead of a blank screen.
- You can explain why you chose Gradio or plain HTML and what the trade-off was.

**Connection to next task:** With the endpoint and UI live, the final milestone verifies the whole phase deliverable.

---

## Milestone Task 9
**Live Endpoint + Working Demo**

**Objective:** Prove the full Phase 4 deliverable: a publicly reachable, quantized, fine-tuned text-to-SQL service with a demo UI, and documented serving numbers.

**What I should have learned so far:** Serving a GGUF with llama.cpp, designing an API contract around a model, executing generated SQL safely, containerizing a two-process service, deploying to a free host and dealing with its limits, and presenting it through a UI.

**What I should build without blindly following instructions:** A `docs/serving.md` page (for the README later) containing your own architecture sketch of the deployed system, a table of cold-start time, warm p50/p95 latency and RAM use, and a short "limits and what I'd do next" section written from what you actually observed.

**Mini challenge:** Add a simple in-memory rate limit (for example 10 requests per minute per IP) to `/generate` and demonstrate it returning HTTP 429. Then add an `X-Request-Id` header and log each request's latency, so you could debug a slow call.

**Self-assessment:** A recruiter opens your URL and asks, "Why is the first request slow, what happens if 50 people use it at once, and how do I know the answers are right?" Can you answer each one with numbers and evidence from your own system?

**Milestone that proves your progress:** Anyone with the link can open the demo, pick an example schema, ask a question, and see correct SQL with result rows, served from your own quantized fine-tuned model.

**Completion criteria:**
- Public URL works from a fresh device and the UI completes an example end-to-end.
- `docs/serving.md` contains real measured numbers (no placeholders).
- You can point to one limitation you found (memory, cold start, concurrency) and explain the decision you made because of it.

