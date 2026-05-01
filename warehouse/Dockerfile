# ==============================================================================
# stateball Dagster User Code Image
# ==============================================================================

FROM ghcr.io/astral-sh/uv:0.9.26 AS uv

FROM python:3.12-slim

WORKDIR /app

COPY --from=uv /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Layer 1: deps only (cached unless any pyproject.toml or uv.lock change)
COPY pyproject.toml uv.lock ./
COPY packages/simulator/pyproject.toml packages/simulator/
COPY warehouse/pyproject.toml warehouse/
RUN uv sync --frozen --no-install-workspace --no-dev --package warehouse

# Layer 2: source + workspace install
COPY packages/simulator/sim/ packages/simulator/sim/
COPY packages/simulator/xg/ packages/simulator/xg/
COPY warehouse/orchestration/ warehouse/orchestration/
RUN uv sync --frozen --no-dev --package warehouse

# Activate venv for all subsequent RUN/CMD
ENV PATH="/app/.venv/bin:$PATH"

# Generate dbt manifest at build time
RUN cd warehouse/orchestration && dbt parse --profiles-dir . --project-dir .

EXPOSE 3030

CMD ["dagster", "api", "grpc", "-h", "0.0.0.0", "-p", "3030", "-m", "orchestration.definitions"]
