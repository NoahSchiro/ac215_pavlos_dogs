# Use the official Debian-hosted Python image (same base as the class tutorial)
FROM python:3.12-slim-bookworm

# Install uv
RUN pip install uv

WORKDIR /app

# Install dependencies first so code changes don't invalidate this layer.
# --frozen: use the exact versions in uv.lock. --no-dev: skip pytest/ruff.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy the source code
COPY app ./app

# Putting the venv on PATH is the same as `source .venv/bin/activate`.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8080
EXPOSE 8080

# A web service starts itself (no interactive shell). Cloud Run sets $PORT.
# One worker: game rooms live in this process's memory.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
