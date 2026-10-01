#!/usr/bin/env bash
# Starts llama-server in the background, waits until it is healthy, then runs the API.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/text2sql-Q4_K_M.gguf}"

# 1. Get the model: use the mounted/baked file, otherwise pull it from the Hub.
if [ ! -f "$MODEL_PATH" ]; then
  : "${MODEL_REPO:?Model not found at $MODEL_PATH and MODEL_REPO is not set}"
  MODEL_FILE="${MODEL_FILE:-$(basename "$MODEL_PATH")}"
  echo "Downloading $MODEL_FILE from $MODEL_REPO ..."
  python - <<PY
from huggingface_hub import hf_hub_download
import os
hf_hub_download(
    repo_id=os.environ["MODEL_REPO"],
    filename="$MODEL_FILE",
    local_dir=os.path.dirname("$MODEL_PATH") or ".",
)
PY
fi

# 2. Start the model server on loopback only (the API is the public entry point).
llama-server -m "$MODEL_PATH" -c 2048 \
  --host 127.0.0.1 --port 8080 -t "${LLAMA_THREADS:-2}" &
LLAMA_PID=$!

# 3. Wait for it to load. The API must not start before this.
for i in $(seq 1 300); do
  if curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "llama-server ready after ${i}s"
    break
  fi
  if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
    echo "llama-server exited during startup" >&2
    exit 1
  fi
  sleep 1
done

# 4. Run the API. PORT is set by some hosts (Cloud Run); 7860 is the HF default.
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-7860}"