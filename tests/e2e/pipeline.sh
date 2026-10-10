#!/usr/bin/env bash
# Checks the spec in _well/spec-checks.md. Run from the repo root, on the spec's branch,
# after the branch is pushed. Reads GitHub through `gh`; installs and builds nothing.
set -euo pipefail

repo=matta-kelly/stateball
workflow=.github/workflows/checks.yml
branch=$(git branch --show-current)
head=$(git rev-parse HEAD)

fail() { echo "FAIL: $1" >&2; exit 1; }
pass() { echo "ok: $1"; }

# YAML reads a bare `on` key as the boolean true, so the trigger is looked up under both.
python3 - "$workflow" <<'PY' || fail "no workflow runs \`just check\` in one job named checks on pull requests into main"
import sys, yaml
wf = yaml.safe_load(open(sys.argv[1]))
on = wf.get("on", wf.get(True))
assert "main" in on["pull_request"]["branches"]
assert list(wf["jobs"]) == ["checks"], list(wf["jobs"])
runs = [step.get("run", "").strip() for step in wf["jobs"]["checks"]["steps"]]
assert "just check" in runs, runs
PY
pass "a pull request into main starts one job, named checks, that runs just check"

# A run that never finishes must fail here, not hang.
timeout -k 5 900 gh pr checks "$branch" --repo "$repo" --watch --fail-fast >/dev/null \
  || fail "the run on this pull request did not pass within 900s"
run=$(timeout 60 gh run list --repo "$repo" --workflow checks.yml --event pull_request \
  --commit "$head" --status success --limit 1 --json databaseId --jq '.[0].databaseId // empty')
[ -n "$run" ] || fail "no passing run of checks for commit $head"
pass "the run passes on this pull request, for the commit checked out here"

log=$(timeout 120 gh run view "$run" --repo "$repo" --log)
for test in \
  decodes_date_teams_and_side_when_the_ticker_is_regular_season \
  reads_the_game_number_when_the_ticker_is_a_doubleheader \
  strips_the_start_time_when_the_ticker_is_spring_training
do
  grep -E "PASS .*$test" <<<"$log" >/dev/null || fail "the run did not pass the dbt unit test $test"
done
grep -E "PASS .*proc_kalshi__markets" <<<"$log" | grep -v "::" >/dev/null \
  || fail "the run did not build the model and pass a data test on it"
pass "the run builds the seed and the model on Postgres and passes the three unit tests"

secrets=$(timeout 60 gh api "repos/$repo/actions/secrets" --jq .total_count)
[ "$secrets" = 0 ] || fail "the repo holds $secrets Actions secrets; the run must pass with none"
pass "the run passes with no credential: the repo holds no Actions secret"

labels=$(timeout 60 gh api "repos/$repo/actions/runs/$run/jobs" \
  --jq '.jobs[] | select(.name == "checks") | .labels | join(",")')
[ "$labels" = ubuntu-24.04 ] || fail "the checks job ran on '$labels', not ubuntu-24.04"
python3 - "$workflow" <<'PY' || fail "the workflow breaks a pipeline file rule"
import re, sys, yaml
wf = yaml.safe_load(open(sys.argv[1]))
job = wf["jobs"]["checks"]
assert 0 < job.get("timeout-minutes", 0) <= 10, "time limit missing or over ten minutes"
assert wf.get("permissions") == {"contents": "read"}, "rights are wider than contents: read"
assert "permissions" not in job, "the job widens its own rights"
for step in job["steps"]:
    if "uses" in step:
        assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"]), step["uses"] + " is not pinned to a commit"
assert wf["concurrency"]["cancel-in-progress"] is True, "a newer run does not cancel an older one"
PY
pass "the job runs on ubuntu-24.04, under ten minutes, read-only, with pinned actions"

timeout 60 gh api "repos/$repo/rules/branches/main" \
  --jq '.[] | select(.type == "required_status_checks") | .parameters.required_status_checks[].context' \
  | grep -qx checks || fail "the ruleset on main does not require checks"
pass "main refuses a pull request whose run is failing or missing"

# --dry-run prints the recipe and runs nothing.
build=$(timeout 20 just --dry-run build 2>&1 | xargs)
[ "$build" = "cd orchestration && uv run dbt build --profiles-dir ." ] \
  || fail "\`just build\` no longer runs only dbt build: $build"
pass "just build does what it did before"
