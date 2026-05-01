# ==============================================================================
# liveball Dashboard + Data Feed Image
# ==============================================================================
# Build context: repo root (for workspace pyproject.toml + uv.lock access)
# Usage: docker build -f liveball/Dockerfile .
# ==============================================================================

# Stage 1: Build frontend
FROM node:22-alpine AS frontend
WORKDIR /app/frontend
COPY liveball/frontend/package.json liveball/frontend/package-lock.json ./
RUN npm ci
COPY liveball/frontend/ ./
RUN npm run build

# Stage 2: Python runtime
FROM python:3.12-slim AS runtime
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /usr/local/bin/uv

# Layer 1: deps only (cached unless any pyproject.toml or uv.lock change)
COPY pyproject.toml uv.lock ./
COPY packages/simulator/pyproject.toml packages/simulator/
COPY liveball/pyproject.toml liveball/
RUN uv sync --frozen --no-install-workspace --no-dev --package liveball

# Layer 2: source + workspace install
COPY packages/simulator/sim/ packages/simulator/sim/
COPY packages/simulator/xg/ packages/simulator/xg/
COPY liveball/backend/ liveball/backend/
COPY liveball/data_feed/ liveball/data_feed/
RUN uv sync --frozen --no-dev --package liveball

# Static frontend assets
COPY --from=frontend /app/frontend/dist ./liveball/static/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
