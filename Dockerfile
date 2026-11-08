# syntax=docker/dockerfile:1
# Multi-stage build (uv's Docker guide): dependencies from uv.lock in a builder image,
# then only the virtualenv + app + index are copied into a slim, non-root runtime image.

# ---------- builder ----------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# 1) dependencies only -> this layer is cached until pyproject.toml/uv.lock change
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# 2) the project itself
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# 3) the corpus + the index, baked into the image (fine for a portfolio demo)
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY scripts/get_corpus.sh scripts/
RUN bash scripts/get_corpus.sh \
    && /app/.venv/bin/findex index data/pydocs --out /app/index.pkl --positions --executor serial

# ---------- runtime ----------
FROM python:3.13-slim-bookworm
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/data /app/data
COPY --from=builder --chown=app:app /app/index.pkl /app/index.pkl
USER app
ENV PATH="/app/.venv/bin:$PATH" \
    INDEX_PATH=/app/index.pkl \
    LOG_FORMAT=json \
    PORT=8000 \
    PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=2)"
# sh -c so that $PORT (set by Render/Fly/Railway) is expanded; exec so uvicorn gets SIGTERM
CMD ["sh", "-c", "exec uvicorn findex.web:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
