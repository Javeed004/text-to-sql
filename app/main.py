"""FastAPI wrapper around llama-server (Phase 4, Task 3).

Run locally:
    uvicorn app.main:app --reload --port 8000

Config (environment variables):
    LLM_BASE_URL   where llama-server listens (default http://localhost:8080)
    LLM_TIMEOUT_S  read timeout for a generation call (default 120)
"""
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from app.examples import EXAMPLES
from app.executor import run_sql
from app.middleware import RequestMiddleware
from app.prompt import build_messages, clean_sql, render_chatml

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080").rstrip("/")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "120"))

# llama-server runs with -c 2048 (prompt + output). Spider prompts are mostly
# under 512 tokens, and n_predict is 256, so ~6000 chars of schema (~1.5k tokens)
# is a conservative cap. Re-tune this after your ~3,000-token overflow test.
MAX_SCHEMA_CHARS = 6000
MAX_QUESTION_CHARS = 500
N_PREDICT = 256


class GenerateRequest(BaseModel):
    # "schema" shadows a BaseModel attribute in pydantic v2, so the Python
    # attribute is db_schema while the JSON key stays "schema".
    model_config = ConfigDict(populate_by_name=True)

    db_schema: str = Field(alias="schema", min_length=1, max_length=MAX_SCHEMA_CHARS)
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    # INSERT statements used only by the executor. Never put into the model prompt,
    # because the model was trained on schema-only prompts.
    data_sql: str = Field(default="", max_length=20000)
    execute: bool = True


class GenerateResponse(BaseModel):
    sql: str
    latency_ms: int  # generation only, execution time is not included
    execution_result: Optional[dict[str, Any]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One shared client for the whole process, not one per request.
    app.state.client = httpx.AsyncClient(
        base_url=LLM_BASE_URL,
        timeout=httpx.Timeout(LLM_TIMEOUT_S, connect=5.0),
    )
    yield
    await app.state.client.aclose()


app = FastAPI(title="Text-to-SQL API", lifespan=lifespan)
app.add_middleware(RequestMiddleware)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    prompt = render_chatml(build_messages(req.db_schema, req.question))
    payload = {
        "prompt": prompt,
        "temperature": 0,
        "n_predict": N_PREDICT,
        "stop": ["<|im_end|>"],
    }

    start = time.perf_counter()
    try:
        resp = await app.state.client.post("/completion", json=payload)
    except httpx.ConnectError:
        raise HTTPException(502, "LLM server is unreachable")
    except httpx.TimeoutException:
        raise HTTPException(504, "LLM server timed out")

    if resp.status_code != 200:
        raise HTTPException(502, f"LLM server returned {resp.status_code}")

    latency_ms = int((time.perf_counter() - start) * 1000)

    sql = clean_sql(resp.json().get("content", ""))
    if not sql:
        raise HTTPException(502, "Model returned no SQL")

    execution_result = None
    if req.execute:
        # sqlite3 is blocking, so keep it off the event loop.
        execution_result = await run_in_threadpool(
            run_sql, req.db_schema + "\n" + req.data_sql, sql
        )

    return GenerateResponse(sql=sql, latency_ms=latency_ms, execution_result=execution_result)


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/examples")
async def examples():
    return EXAMPLES


@app.get("/health")
async def health():
    try:
        r = await app.state.client.get("/health", timeout=5.0)
        llm_ok = r.status_code == 200
    except httpx.HTTPError:
        llm_ok = False
    if not llm_ok:
        raise HTTPException(503, "LLM server not ready")
    return {"status": "ok", "llm": "ok"}