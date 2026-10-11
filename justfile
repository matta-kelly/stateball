set dotenv-load

image := "docker.io/library/postgres:17.10"
volume := "stateball-pgdata"

default:
    @just --list

# Local Postgres + dagster dev
up:
    @echo "TODO"

down:
    @echo "TODO"

psql:
    podman exec -it postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"

# The whole check set, cheapest first; the pipeline runs this and nothing else
check:
    #!/usr/bin/env bash
    set -euo pipefail
    export POSTGRES_DB=stateball_check
    uv run ruff format --check .
    uv run ruff check .
    uv run dagster definitions validate -m orchestration.definitions
    uv run python tests/fixtures/land.py
    just build

build *ARGS:
    cd orchestration && uv run dbt build --profiles-dir . {{ ARGS }}

debug:
    cd orchestration && uv run dbt debug --profiles-dir .
