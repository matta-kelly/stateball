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
just build
```

Runs `dbt build`: the seed, the model and its data tests. It needs the database. No check runs on a pull request yet.

## Ship it

Not deployed yet.
