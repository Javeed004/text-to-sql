
FROM ghcr.io/ggml-org/llama.cpp:server

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv curl \
    && rm -rf /var/lib/apt/lists/*

# Verify llama-server binary
RUN /app/llama-server --version

# Create Python virtual environment
RUN python3 -m venv /opt/venv

# Configure executable and shared library paths
ENV PATH="/opt/venv/bin:/app:${PATH}" \
    LD_LIBRARY_PATH="/app"
    
WORKDIR /srv

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app ./app
COPY start.sh .

# Fix Windows CRLF line endings
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

ENV LLM_BASE_URL=http://127.0.0.1:8080 \
    MODEL_PATH=/models/text2sql-Q4_K_M.gguf \
    LLAMA_THREADS=2

EXPOSE 7860

ENTRYPOINT ["./start.sh"]