# rag-eval REST API image.
# Ships the pipeline + eval harness behind FastAPI. Local BGE embeddings run on
# CPU (no GPU needed). Provide GROQ_API_KEY (and any opt-in provider keys) at
# runtime via env or --env-file.
FROM python:3.12-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    TRANSFORMERS_VERBOSITY=error

# Install with the CPU build of torch to keep the image lean.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[api]"

COPY configs ./configs
COPY data ./data

EXPOSE 8000

# Serve the API. Build the index once via `POST /ingest` (or bake it into a
# derived image). Health check at GET /health.
CMD ["uvicorn", "rag_eval.api:app", "--host", "0.0.0.0", "--port", "8000"]
