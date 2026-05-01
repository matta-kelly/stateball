"""Build empirical win expectancy table from proc_mlb__events + proc_mlb__games.

Computes P(home_win) by (inning, half, outs, base_state, run_diff).
Hierarchical fallback for sparse state combinations. Output is a JSON
artifact consumed by the game simulator for Rao-Blackwellization and
blowout pruning.

Built from our own historical data — reflects matchup-specific
XGBoost-era game states, not league-average FanGraphs/BR tables.
Improves as backfill continues.

Usage:
    # Local dev (DuckDB file)
    python -m sim.build_win_expectancy_table --db-path stateball.duckdb

    # With custom output path
    python -m sim.build_win_expectancy_table --db-path stateball.duckdb --output sim/data/win_expectancy.json

    # Cluster (DuckLake) — called by Dagster asset, not directly
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Clamp run_diff beyond these bounds — essentially decided games
RUN_DIFF_MIN = -15
RUN_DIFF_MAX = 15

_HALF_MAP = {"Top": 0, "Bot": 1}
_RD_OFFSET = -RUN_DIFF_MIN  # add to run_diff to get array index


# ---------------------------------------------------------------------------
# Lookup functions (consumed by control variates + Rao-Blackwell)
# ---------------------------------------------------------------------------


def lookup(table: dict, inning: int, half: str, outs: int, bases: int, run_diff: int) -> float:
    """Look up P(home_win) with hierarchical fallback.

    Args:
        table: WE artifact dict with "levels" key.
        inning: 1+ (extras clamped to 9).
        half: "Top" or "Bot".
        outs: 0, 1, or 2.
        bases: bitmask 0-7.
        run_diff: home_score - away_score (clamped to [-15, 15]).
    """
    levels = table["levels"]
    run_diff = max(RUN_DIFF_MIN, min(RUN_DIFF_MAX, run_diff))
    inning = min(inning, 9)

    key = f"{inning}|{half}|{outs}|{bases}|{run_diff}"
    entry = levels.get("full", {}).get(key)
    if entry and entry["n"] >= 20:
        return entry["p_home_win"]

    key = f"{inning}|{half}|{outs}|{run_diff}"
    entry = levels.get("no_bases", {}).get(key)
    if entry and entry["n"] >= 20:
        return entry["p_home_win"]

    key = f"{inning}|{half}|{run_diff}"
    entry = levels.get("coarse", {}).get(key)
    if entry:
        return entry["p_home_win"]

    return levels["baseline"]["p_home_win"]


def build_lookup_array(table: dict) -> np.ndarray:
    """Pre-compute WE into a dense NumPy array for vectorized batch lookup.

    Returns array shaped (10, 2, 3, 8, 31) indexed by
    [inning][half_idx][outs][bases][run_diff + 15]
    where half_idx: 0=Top, 1=Bot.

    Populated via hierarchical fallback (full → no_bases → coarse → baseline)
    matching the same logic as lookup().
    """
    rd_range = RUN_DIFF_MAX - RUN_DIFF_MIN + 1  # 31
    arr = np.full((10, 2, 3, 8, rd_range), np.nan, dtype=np.float64)

    baseline = table["levels"]["baseline"]["p_home_win"]

    # Fill from coarsest to finest — finer levels overwrite coarser
    coarse = table["levels"].get("coarse", {})
    no_bases = table["levels"].get("no_bases", {})
    full = table["levels"].get("full", {})

    for inn in range(1, 10):
        for half_idx, half_str in enumerate(("Top", "Bot")):
            for rd in range(RUN_DIFF_MIN, RUN_DIFF_MAX + 1):
                rd_idx = rd + _RD_OFFSET

                # Coarse fallback
                ckey = f"{inn}|{half_str}|{rd}"
                c_entry = coarse.get(ckey)
                p = c_entry["p_home_win"] if c_entry else baseline

                for outs in range(3):
                    # No-bases fallback
                    nbkey = f"{inn}|{half_str}|{outs}|{rd}"
                    nb_entry = no_bases.get(nbkey)
                    p_nb = nb_entry["p_home_win"] if (nb_entry and nb_entry["n"] >= 20) else p

                    for bases in range(8):
                        # Full lookup
                        fkey = f"{inn}|{half_str}|{outs}|{bases}|{rd}"
                        f_entry = full.get(fkey)
                        if f_entry and f_entry["n"] >= 20:
                            arr[inn, half_idx, outs, bases, rd_idx] = f_entry["p_home_win"]
                        else:
                            arr[inn, half_idx, outs, bases, rd_idx] = p_nb

    # Fill inning 0 (unused) with baseline
    arr[0, :, :, :, :] = baseline

    logger.info(
        "Built WE lookup array: shape=%s, non-NaN=%d/%d",
        arr.shape, int(np.isfinite(arr).sum()), arr.size,
    )
    return arr


def build_sensitivity(table: dict) -> dict:
    """Compute dP(home_win)/d(run_diff) at half-inning boundaries.

    For each (inning, half, run_diff), takes the finite difference of WE
    at outs=0, bases=0 (the state at a half-inning boundary). This is the
    WE table's expected sensitivity to a one-run change — used by SMC
    tempering to compare sim spread against table spread.

    Returns dict with "coarse" level (keyed like "1|Top|-15") and metadata.
    """
    halves = ("Top", "Bot")
    coarse: dict[str, dict] = {}

    for inn in range(1, 10):
        for half in halves:
            for rd in range(RUN_DIFF_MIN, RUN_DIFF_MAX):
                we_lo = lookup(table, inn, half, outs=0, bases=0, run_diff=rd)
                we_hi = lookup(table, inn, half, outs=0, bases=0, run_diff=rd + 1)
                key = f"{inn}|{half}|{rd}"
                coarse[key] = {
                    "dwe_drd": round(we_hi - we_lo, 6),
                    "we_at_rd": round(we_lo, 6),
                    "we_at_rd_plus_1": round(we_hi, 6),
                }

    logger.info("Built WE sensitivity: %d keys", len(coarse))
    return {
        "coarse": coarse,
        "metadata": {
            "description": "dP(home_win)/d(run_diff) at half-inning boundaries (outs=0, bases=0)",
            "rd_range": [RUN_DIFF_MIN, RUN_DIFF_MAX - 1],
        },
    }


def build_sensitivity_array(table: dict) -> np.ndarray:
    """Dense sensitivity array for vectorized access.

    Returns array shaped (10, 2, 30) indexed by
    [inning][half_idx][rd + 15] for rd in [-15, 14].
    Values are dWE/dRD at outs=0, bases=0 (half-inning boundaries).
    """
    we_arr = build_lookup_array(table)
    boundary_we = we_arr[:, :, 0, 0, :]  # (10, 2, 31)
    sens = np.diff(boundary_we, axis=2)   # (10, 2, 30)
    logger.info("Built WE sensitivity array: shape=%s", sens.shape)
    return sens


def _detect_tables(conn) -> tuple[str, str]:
    """Auto-detect DuckLake vs local table paths. Returns (events_table, games_table)."""
    try:
        conn.execute("SELECT 1 FROM lakehouse.main.proc_mlb__events LIMIT 0")
        return "lakehouse.main.proc_mlb__events", "lakehouse.main.proc_mlb__games"
    except Exception:
        return "main.proc_mlb__events", "main.proc_mlb__games"


def _pa_states_cte(events_table: str, games_table: str) -> str:
    """Return SQL CTE that produces one row per PA with game state and outcome.

    Joins terminal PAs from events to game final scores.
    Computes run_diff from home perspective (pre-pitch), base state as
    bitmask, and home_win from final score.
    """
    return f"""
    WITH pa_states AS (
        SELECT
            e.game_pk,
            e.inning,
            e.inning_topbot,
            e.outs_when_up,
            (CASE WHEN e.on_1b IS NOT NULL THEN 1 ELSE 0 END) +
            (CASE WHEN e.on_2b IS NOT NULL THEN 2 ELSE 0 END) +
            (CASE WHEN e.on_3b IS NOT NULL THEN 4 ELSE 0 END) AS bases,
            GREATEST({RUN_DIFF_MIN}, LEAST({RUN_DIFF_MAX},
                CAST(e.home_score - e.away_score AS INT))) AS run_diff,
            CASE WHEN g.home_score > g.away_score THEN 1.0 ELSE 0.0 END AS home_win
        FROM {events_table} e
        JOIN {games_table} g ON e.game_pk = g.game_pk
        WHERE e.pa_result IS NOT NULL
          AND e.pa_result != 'truncated_pa'
          AND e.inning BETWEEN 1 AND 9
          AND g.abstract_game_state = 'Final'
          AND g.status NOT IN ('Postponed', 'Cancelled')
    )
    """


def _level_queries() -> list[tuple[str, str]]:
    """Return (level_name, SELECT) pairs for each fallback level."""
    return [
        (
            "full",
            "SELECT inning, inning_topbot, outs_when_up, bases, run_diff, "
            "COUNT(*) AS n, AVG(home_win) AS p_home_win "
            "FROM pa_states GROUP BY 1, 2, 3, 4, 5",
        ),
        (
            "no_bases",
            "SELECT inning, inning_topbot, outs_when_up, run_diff, "
            "COUNT(*) AS n, AVG(home_win) AS p_home_win "
            "FROM pa_states GROUP BY 1, 2, 3, 4",
        ),
        (
            "coarse",
            "SELECT inning, inning_topbot, run_diff, "
            "COUNT(*) AS n, AVG(home_win) AS p_home_win "
            "FROM pa_states GROUP BY 1, 2, 3",
        ),
        (
            "baseline",
            "SELECT COUNT(*) AS n, AVG(home_win) AS p_home_win "
            "FROM pa_states",
        ),
    ]


def _key_columns(level: str) -> list[str]:
    """Return the column names that form the key for a given level."""
    return {
        "full": ["inning", "inning_topbot", "outs_when_up", "bases", "run_diff"],
        "no_bases": ["inning", "inning_topbot", "outs_when_up", "run_diff"],
        "coarse": ["inning", "inning_topbot", "run_diff"],
        "baseline": [],
    }[level]


def build(conn) -> dict:
    """Query PA states and game outcomes, return hierarchical WE table.

    Args:
        conn: DuckDB connection (local file or DuckLake).

    Returns:
        Dict with "metadata" and "levels" keys.
    """
    events_table, games_table = _detect_tables(conn)
    logger.info("Querying %s + %s for win expectancy", events_table, games_table)

    cte = _pa_states_cte(events_table, games_table)

    levels: dict[str, dict] = {}
    n_pas = 0
    baseline_entry: dict = {}

    for level_name, select in _level_queries():
        query = cte + select
        df = conn.execute(query).fetchdf()

        if level_name == "baseline":
            row = df.iloc[0]
            baseline_entry = {
                "p_home_win": round(float(row["p_home_win"]), 6),
                "n": int(row["n"]),
            }
            n_pas = baseline_entry["n"]
            levels["baseline"] = baseline_entry
            logger.info(
                "Baseline: p_home_win=%.4f, n=%d",
                baseline_entry["p_home_win"],
                baseline_entry["n"],
            )
            continue

        key_cols = _key_columns(level_name)
        level_dict: dict[str, dict] = {}
        for _, row in df.iterrows():
            key = "|".join(str(row[c]) for c in key_cols)
            level_dict[key] = {
                "p_home_win": round(float(row["p_home_win"]), 6),
                "n": int(row["n"]),
            }
        levels[level_name] = level_dict
        logger.info("Level %s: %d keys", level_name, len(level_dict))

    # Count distinct games
    game_count_query = cte + "SELECT COUNT(DISTINCT game_pk) AS n_games FROM pa_states"
    n_games = int(conn.execute(game_count_query).fetchdf().iloc[0]["n_games"])

    result = {
        "metadata": {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "source_events": events_table,
            "source_games": games_table,
            "n_pas": n_pas,
            "n_games": n_games,
            "n_full_keys": len(levels.get("full", {})),
            "n_no_bases_keys": len(levels.get("no_bases", {})),
            "n_coarse_keys": len(levels.get("coarse", {})),
            "run_diff_range": [RUN_DIFF_MIN, RUN_DIFF_MAX],
        },
        "levels": levels,
    }

    # Sensitivity: dWE/dRD at half-inning boundaries
    sensitivity = build_sensitivity(result)
    result["sensitivity"] = sensitivity
    result["metadata"]["n_sensitivity_keys"] = len(sensitivity["coarse"])

    logger.info(
        "Built win expectancy table: %d PAs, %d games, "
        "%d full / %d no_bases / %d coarse / %d sensitivity keys",
        n_pas,
        n_games,
        len(levels.get("full", {})),
        len(levels.get("no_bases", {})),
        len(levels.get("coarse", {})),
        len(sensitivity["coarse"]),
    )
    return result


def validate(table: dict) -> None:
    """Verify all p_home_win values are in [0, 1] and n > 0."""
    errors = []
    for level_name, level_data in table["levels"].items():
        if level_name == "baseline":
            # baseline is a flat dict, not nested
            if not (0.0 <= level_data["p_home_win"] <= 1.0):
                errors.append(
                    f"baseline: p_home_win={level_data['p_home_win']} out of [0,1]"
                )
            if level_data["n"] <= 0:
                errors.append(f"baseline: n={level_data['n']} <= 0")
            continue

        for key, entry in level_data.items():
            if not (0.0 <= entry["p_home_win"] <= 1.0):
                errors.append(
                    f"{level_name}/{key}: p_home_win={entry['p_home_win']} out of [0,1]"
                )
            if entry["n"] <= 0:
                errors.append(f"{level_name}/{key}: n={entry['n']} <= 0")

    # Baseline must exist with meaningful sample
    baseline = table["levels"].get("baseline")
    if baseline is None:
        errors.append("baseline level missing")
    elif baseline["n"] < 1000:
        errors.append(f"baseline n={baseline['n']} < 1000 — insufficient data")

    if errors:
        raise ValueError(
            f"Validation failed for {len(errors)} entries:\n"
            + "\n".join(errors[:10])
        )

    total_keys = sum(
        len(v) if isinstance(v, dict) and "p_home_win" not in v else 1
        for v in table["levels"].values()
    )
    logger.info("Validation passed: %d keys across all levels", total_keys)


def save(table: dict, path: str) -> str:
    """Write win expectancy table as JSON. Supports local paths and s3:// URIs."""
    if path.startswith("s3://"):
        import s3fs

        fs = s3fs.S3FileSystem(
            key=os.environ.get("S3_ACCESS_KEY_ID", ""),
            secret=os.environ.get("S3_SECRET_ACCESS_KEY", ""),
            endpoint_url=os.environ.get("S3_ENDPOINT", ""),
        )
        with fs.open(path, "w") as f:
            json.dump(table, f, indent=2)
    else:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(table, f, indent=2)

    logger.info("Saved win expectancy table to %s", path)
    return path


if __name__ == "__main__":
    import argparse

    import duckdb

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Build win expectancy lookup table"
    )
    parser.add_argument(
        "--db-path",
        default="stateball.duckdb",
        help="Path to local DuckDB file (default: stateball.duckdb)",
    )
    parser.add_argument(
        "--output",
        default="sim/data/win_expectancy.json",
        help="Output path for JSON artifact (default: sim/data/win_expectancy.json)",
    )
    args = parser.parse_args()

    conn = duckdb.connect(str(args.db_path), read_only=True)
    try:
        tbl = build(conn)
    finally:
        conn.close()

    validate(tbl)
    save(tbl, args.output)
