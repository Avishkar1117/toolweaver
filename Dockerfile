# syntax=docker/dockerfile:1.7
# App image for the agentic-tool-use API (mirrors rag-service's multi-stage uv
# build, Dockerfile at repo root so HF Spaces' Docker SDK picks it up). The
# code_exec sandbox is NOT run from inside this container -- it needs the host
# Docker daemon, which a managed deploy (HF Spaces) doesn't provide, so code_exec
# degrades gracefully there (CLAUDE.md §4 tool-failure-as-observation).

# --- Builder: install dependencies with uv into an isolated .venv ---
FROM python:3.10-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

COPY --from=ghcr.io/astral-sh/uv:0.5.0 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies only (cached until the lockfile changes). We don't install the
# project itself; the app runs from source on PYTHONPATH (set in the runtime stage).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# --- Runtime: slim image with the venv + source ---
FROM python:3.10-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ /app/src/
COPY api/ /app/api/
# Bake the minimal doc_eval_corpus store (built by scripts/bundle_corpus.py).
# Managed hosting has no runtime volume mount, so the corpus ships in the image;
# doc_lookup reads it from DOC_CHROMA_PATH below.
COPY chroma_store/ /app/chroma_store/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:/app" \
    PYTHONUNBUFFERED=1 \
    DOC_CHROMA_PATH="/app/chroma_store" \
    SANDBOX_ENABLED="false"

RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
