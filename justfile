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

build *ARGS:
    cd orchestration && uv run dbt build --profiles-dir . {{ ARGS }}

debug:
    cd orchestration && uv run dbt debug --profiles-dir .
