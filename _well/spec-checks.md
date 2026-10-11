---
type: spec
description: "Run stateball's checks on GitHub for every pull request, and make main refuse a change that fails them. Reference for what the check set is, how it gets a Postgres without Kalshi, and how the change is checked."
status: design
---
## Need

- nothing checks a change before it merges
	- the repo has no workflow
	- the ruleset on main requires a pull request and a squash merge, and no check
- the one check, `just build`, cannot run on a clean machine
	- it needs a Postgres server
	- it needs the table `kalshi.markets`, which only a pull from Kalshi makes
- a check must never need a Kalshi credential
	- the repo is public; a pull request from a stranger gets no secrets
- the repo has no formatter config, no linter config, no type checker and no tests of its own

## Done looks like

### Acceptance criteria

- a pull request into main starts one job, named `checks`, that runs `just check`
- `just check` does these, in this order, and stops at the first failure
	- format: `ruff format --check`
	- lint: `ruff check`, with the default rules and `ERA001` (commented-out code)
	- load: the Dagster code location loads
	- build: `dbt build` on Postgres, which runs the seed, the three unit tests, the model and the data tests
- on this spec's own pull request, the run passes, for the last commit
- the run needs no credential: it passes while the repo holds no Actions secret
- `just check` needs no `KALSHI_KEY_ID` and no `KALSHI_PRIVATE_KEY`, and makes no request to Kalshi
- `just check` reads and writes one database, `stateball_check`, whatever `POSTGRES_DB` holds
- the job follows the rules in [[ci-cd#Pipeline file]]
	- runs on `ubuntu-24.04`
	- time limit of ten minutes
	- rights: `contents: read`, and no more
	- each action is pinned to a full commit hash
	- a newer run on the same pull request cancels the older one
- the ruleset on main requires `checks`: a pull request cannot merge while the run is failing or missing
- `just build` does what it did before
- the README says how to run the check set, and that it runs on every pull request

Checked in code review, by breaking it once on purpose, on a throwaway pull request:

- a file that `ruff format` would change: the run fails
- an unused import: the run fails
- `side` swapped in `proc_kalshi__markets.sql`: the run fails at a unit test
- `POSTGRES_DB` set to a different name in the workflow: the run still passes

### Approach

- the check set, and why each part
	- format and lint: `ruff`, already in the `dev` group and never configured
		- `[tool.ruff.lint]` in `pyproject.toml`: `extend-select = ["ERA001"]`; nothing else
		- [[shit_sdlc#Loop]] names "no commented-out code" as a lint; [ERA001](https://docs.astral.sh/ruff/rules/commented-out-code/) is that rule
		- the existing Python files are reformatted by `ruff format`, in a commit that changes nothing else
	- load: `uv run dagster definitions validate -m orchestration.definitions`
		- exits 1 when the code location has an error ([Dagster CLI](https://dagster.io/docs/api/clis/cli))
		- this is the only step that imports the orchestration code
	- build: the existing `just build`
		- [dbt unit tests](https://docs.getdbt.com/docs/build/unit-tests) run inside `dbt build`, before the model is made
		- "the direct parents of the model that you're unit testing need to exist in the warehouse": so `kalshi.markets` must exist first
- the landing table, without Kalshi
	- `markets()` in `orchestration/dlt/sources/kalshi/streams.py` takes one more argument, `client`, default `None`
		- `None`: it builds a `KalshiClient` from the environment, as it does today
		- the two `os.environ` reads move inside that branch
	- `tests/fixtures/kalshi_markets.json`: the two markets of one invented game, in the shape of Kalshi's `GET /markets` response
	- `tests/fixtures/land.py`: runs `markets(client=...)` through `source_pipeline("kalshi")`, with a client that returns the file
		- the real resource, the real declared columns and the real pipeline make the table; no second copy of the schema
		- dlt "creates tables only when a resource yields data" ([dlt resource](https://dlthub.com/docs/general-usage/resource)), so the file must hold rows
- `justfile`: one new recipe, `check`
	- sets `POSTGRES_DB=stateball_check` for all its steps
	- `uv run ruff format --check .`
	- `uv run ruff check .`
	- `uv run dagster definitions validate -m orchestration.definitions`
	- `uv run python tests/fixtures/land.py`
	- `just build`
- `.github/workflows/checks.yml`
	- trigger: pull requests into main; no path filter, or a required check never reports
	- `permissions: contents: read`
	- `concurrency`: one group per pull request, `cancel-in-progress: true`
	- one job, `checks`, on `ubuntu-24.04`, `timeout-minutes: 10`
	- Postgres as a service container ([GitHub: PostgreSQL service containers](https://docs.github.com/en/actions/tutorials/use-containerized-services/create-postgresql-service-containers))
		- image `postgres:17.10`, the version the `justfile` names
		- `ports: 5432:5432`; the job runs on the runner, so the host is `localhost`
		- `options`: `--health-cmd pg_isready --health-interval 10s --health-timeout 5s --health-retries 5`
		- service env: `POSTGRES_USER: stateball`, `POSTGRES_DB: stateball_check`, `POSTGRES_PASSWORD`: a literal in the file
			- not a secret: the database lives for one run and holds invented rows
	- job env: `POSTGRES_HOST: localhost`, `POSTGRES_PORT: 5432`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
	- steps, in order
		- `actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1` (v7.0.1), with `persist-credentials: false`
		- `astral-sh/setup-uv@1c37ad07a6a961277cf70c0d37d6f313000f5884` (v10.3.0), with `version: "0.9.26"`, `python-version: "3.12"`, `enable-cache: false` ([uv: GitHub Actions](https://docs.astral.sh/uv/guides/integration/github/))
			- 0.9.26 is the uv that wrote `uv.lock`
		- `extractions/setup-just@53165ef7e734c5c07cb06b3c8e7b647c5aa16db3` (v4), with `just-version: "1.56.0"`
		- `uv sync --locked`: fails when `uv.lock` does not match `pyproject.toml`
		- `just check`
- `README.md`, section "Check it": `just check`, what it runs, what it needs (uv, just, a Postgres with a database `stateball_check`)
- the required check: the owner's step, not the builder's
	- it changes a ruleset, which an agent does not do
	- it comes after the first green run; before that it blocks every merge, this one included ([[github-guide#Rulesets and settings]])
	- the command, which adds one rule and keeps the others:
		- `gh api repos/matta-kelly/stateball/rulesets/24824223 --jq '{rules: (.rules + [{"type":"required_status_checks","parameters":{"strict_required_status_checks_policy":false,"do_not_enforce_on_create":false,"required_status_checks":[{"context":"checks"}]}}])}' | gh api -X PUT repos/matta-kelly/stateball/rulesets/24824223 --input -`
- assumptions, not run on a machine by Design
	- `dagster definitions validate` needs no `DAGSTER_HOME` and no database
	- the three unit tests pass against the model as it is on main
	- if one is false, the spec comes back to Design; the builder does not edit a test

### How to go back

- the owner removes the `required_status_checks` rule from the ruleset `main` first
	- while it is required, a pull request that removes the workflow cannot merge
- then one pull request reverts the squash commit
- no data to restore: the change touches no running system

### Out of scope

- types: the repo has no type checker; a later spec picks one
- Python tests: `pytest` stays in the `dev` group and out of `just check` until a test exists (it exits 5 with none)
- format and lint of SQL and YAML
- a local Postgres: the `up` and `down` recipes stay as they are
- a dev container
- the pull from Kalshi itself, and the hourly schedule
- an image, a deploy, a ship workflow: the project is not deployed
- a cache between runs
- a separate GitHub identity for agents

## Tests

- `orchestration/dbt/processed/proc_kalshi__markets_unit.yml`: three dbt unit tests, run by `dbt build`
	- `decodes_date_teams_and_side_when_the_ticker_is_regular_season`
	- `reads_the_game_number_when_the_ticker_is_a_doubleheader`
	- `strips_the_start_time_when_the_ticker_is_spring_training`
	- today nothing runs them: `kalshi.markets` exists on no clean machine, and no pipeline exists
- `tests/e2e/pipeline.sh`, run from the repo root, on this spec's branch, after a push
	- one block per criterion that GitHub can answer
		- the job and its command
		- the green run for the last commit
		- the three unit tests and a data test, passed in that run's log
		- no Actions secret
		- the pipeline file rules
		- the ruleset
		- `just build` unchanged
	- fails today at the first block: there is no `.github/workflows/checks.yml`
	- the ruleset block stays red until the owner runs the command above
- no test for three criteria; they are in the code review list or read from the diff
	- no request to Kalshi
	- the database name
	- the README
