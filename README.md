# stateball

Records MLB prediction markets from Kalshi into a Postgres warehouse.

This is a rebuild. Version one, a win-probability model built on a Markov simulator and XGBoost, is kept at the [`v1` tag](https://github.com/matta-kelly/stateball/tree/v1).

## Where it stands

It does one thing so far:

- every hour, it pulls Kalshi's MLB game markets into Postgres
- it decodes each market ticker into a game date, two teams and a side

Three pieces do that:

- **ingestion**: a dlt source for the Kalshi API
- **transform**: one dbt model and one seed that maps Kalshi's team codes
- **orchestration**: a Dagster code location on an hourly schedule

Built with Python 3.12, uv, Dagster, dlt and dbt.

## Run it

Needs [uv](https://docs.astral.sh/uv/), a Postgres server and a Kalshi API key.

Configuration comes from the environment:

- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
- `KALSHI_KEY_ID`, `KALSHI_PRIVATE_KEY`

```sh
uv sync
uv run dagster dev
```

## Check it

```sh
just check
```

Runs the whole check set, in this order, and stops at the first failure:

- format: `ruff format --check`
- lint: `ruff check`
- load: the Dagster code location loads
- build: `dbt build`, which runs the seed, the unit tests, the model and its data tests

Needs [uv](https://docs.astral.sh/uv/), [just](https://just.systems/) and a Postgres server with a database named `stateball_check`. It reads and writes only that database, and it needs no Kalshi API key: the build runs on two invented markets in `tests/fixtures/`.

The same command runs on every pull request into main.

## Ship it

Not deployed yet.
