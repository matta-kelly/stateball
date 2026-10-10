# stateball

MLB prediction-market recording and a small warehouse. This is the rebuild; version one is kept at the [`v1` tag](https://github.com/matta-kelly/stateball/tree/v1).

Today it does one thing: every hour it pulls Kalshi's MLB game markets into Postgres and decodes each ticker into a game date, two teams and a side.

- ingestion: a dlt source for the Kalshi API, landing in the `kalshi` schema
- transform: one dbt model, `proc_kalshi__markets`, in the `processed` schema, and one seed, `kalshi_team_map`, in the `reference` schema
- orchestration: a Dagster code location that runs the ingestion on an hourly cron

Built with Python 3.12, uv, Dagster, dlt and dbt, on Postgres.

## Run it

Needs [uv](https://docs.astral.sh/uv/), [just](https://just.systems), a Postgres server you can reach, and a Kalshi API key.

The code reads its configuration from the environment. `just` loads a `.env` file from the repo root if there is one.

| Variable             | Used for                                  | Default     |
| -------------------- | ----------------------------------------- | ----------- |
| `POSTGRES_HOST`      | the database host                         | `127.0.0.1` |
| `POSTGRES_PORT`      | the database port                         | `5432`      |
| `POSTGRES_DB`        | the database name                         | `stateball` |
| `POSTGRES_USER`      | the database user                         | `stateball` |
| `POSTGRES_PASSWORD`  | the database password                     | none        |
| `KALSHI_KEY_ID`      | the id of the Kalshi API key              | none        |
| `KALSHI_PRIVATE_KEY` | the key's RSA private key, as PEM text    | none        |

The three without a default are required.

```sh
uv sync
uv run dagster dev
```

`dagster dev` loads the code location named in `pyproject.toml`. The hourly ingestion is armed as soon as it loads.

The `justfile` is not finished:

- `just up` and `just down` are stubs. They print `TODO` and start nothing, so starting Postgres is up to you.
- `just psql` expects a running container named `postgres`, which no recipe creates.
- `just build` and `just debug` work. They run `dbt build` and `dbt debug`.

Run the ingestion once before `just build`. The dbt model reads the table the ingestion lands.

## Check it

```sh
just build
```

`dbt build` loads the seed, builds the model and runs the dbt data tests declared beside it. It needs the database.

That is the only check today. `ruff` and `pytest` are installed as dev dependencies, but there is no `ruff` configuration and there are no `pytest` tests. No check runs on a pull request.

## Ship it

Nothing is deployed yet. There is no container image and no pipeline.

## Layout

| Path                                 | Holds                                             |
| ------------------------------------ | ------------------------------------------------- |
| `orchestration/definitions.py`       | the Dagster code location                         |
| `orchestration/ingestion.py`         | the Dagster assets and their cron                 |
| `orchestration/shared.py`            | the dbt project wiring and per-run config         |
| `orchestration/dlt/`                 | the dlt pipeline and the Kalshi source and client |
| `orchestration/dbt/`                 | dbt sources, models and seeds                     |
| `orchestration/macros/`              | dbt macros                                        |
| `orchestration/dbt_project.yml`      | the dbt project                                   |
| `orchestration/profiles.yml`         | the dbt connection, read from the environment     |
