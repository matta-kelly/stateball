"""Build empirical baserunning transition table from proc_mlb__events.

Computes runner advancement frequencies by (outcome, pre_base_state, outs).
Output is a JSON artifact consumed by the game simulator at startup.

Usage:
    # Local dev (DuckDB file)
    python -m sim.build_baserunning_table --db-path stateball.duckdb

    # With custom output path
    python -m sim.build_baserunning_table --db-path stateball.duckdb --output sim/data/baserunning.json

    # Cluster (DuckLake) — called by Dagster asset, not directly
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_query(table: str) -> str:
    """Return SQL that produces grouped transition counts.

    Uses LEAD on first pitches to derive post-PA state from the next PA's
    pre-pitch state. This avoids mid-PA stolen base contamination.
    """
    return f"""
    WITH first_pitches AS (
        -- First pitch of each PA = pre-PA state (before any mid-PA steals)
        SELECT
            game_pk, at_bat_number, inning, inning_topbot,
            outs_when_up, on_1b, on_2b, on_3b, bat_score
        FROM {table}
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY game_pk, at_bat_number ORDER BY pitch_number
        ) = 1
    ),

    terminal_pitches AS (
        -- Terminal pitch of each PA = outcome + post-pitch score
        SELECT
            game_pk, at_bat_number, pa_result, post_bat_score
        FROM {table}
        WHERE pa_result IS NOT NULL
          AND pa_result != 'truncated_pa'
    ),

    pa_view AS (
        SELECT
            f.game_pk,
            f.at_bat_number,
            t.pa_result,
            f.outs_when_up,
            f.inning,
            f.inning_topbot,
            -- Pre-PA base state (0-7 bitmask)
            (CASE WHEN f.on_1b IS NOT NULL THEN 1 ELSE 0 END)
            | (CASE WHEN f.on_2b IS NOT NULL THEN 2 ELSE 0 END)
            | (CASE WHEN f.on_3b IS NOT NULL THEN 4 ELSE 0 END) AS pre_bases,
            -- Runs scored on this play
            CAST(t.post_bat_score - f.bat_score AS INT) AS runs_scored,
            -- Next PA's first-pitch state (= post-PA state)
            LEAD(f.outs_when_up) OVER w AS next_outs,
            LEAD(f.on_1b) OVER w AS next_on_1b,
            LEAD(f.on_2b) OVER w AS next_on_2b,
            LEAD(f.on_3b) OVER w AS next_on_3b,
            LEAD(f.inning) OVER w AS next_inning,
            LEAD(f.inning_topbot) OVER w AS next_topbot
        FROM first_pitches f
        INNER JOIN terminal_pitches t USING (game_pk, at_bat_number)
        WINDOW w AS (PARTITION BY f.game_pk ORDER BY f.at_bat_number)
    ),

    with_post_state AS (
        SELECT
            pa_result,
            pre_bases,
            outs_when_up AS outs,
            runs_scored,
            -- Post-PA base state
            CASE
                WHEN next_inning IS NULL THEN NULL
                WHEN next_inning = inning AND next_topbot = inning_topbot THEN
                    (CASE WHEN next_on_1b IS NOT NULL THEN 1 ELSE 0 END)
                    | (CASE WHEN next_on_2b IS NOT NULL THEN 2 ELSE 0 END)
                    | (CASE WHEN next_on_3b IS NOT NULL THEN 4 ELSE 0 END)
                ELSE 0
            END AS post_bases,
            -- Outs added
            CASE
                WHEN next_inning IS NULL THEN NULL
                WHEN next_inning = inning AND next_topbot = inning_topbot THEN
                    next_outs - outs_when_up
                ELSE 3 - outs_when_up
            END AS outs_added
        FROM pa_view
    )

    SELECT
        pa_result,
        pre_bases,
        outs,
        post_bases,
        runs_scored,
        outs_added,
        COUNT(*) AS n
    FROM with_post_state
    WHERE post_bases IS NOT NULL
    GROUP BY 1, 2, 3, 4, 5, 6
    ORDER BY 1, 2, 3, n DESC
    """


def build(conn) -> dict:
    """Query transition frequencies and return the baserunning table.

    Args:
        conn: DuckDB connection (local file or DuckLake).
              For DuckLake, table is lakehouse.main.proc_mlb__events.
              For local DuckDB, table is main.proc_mlb__events.

    Returns:
        Dict with "metadata" and "transitions" keys.
    """
    # Detect table path: DuckLake uses lakehouse schema
    try:
        conn.execute("SELECT 1 FROM lakehouse.main.proc_mlb__events LIMIT 0")
        table = "lakehouse.main.proc_mlb__events"
    except Exception:
        table = "main.proc_mlb__events"

    logger.info("Querying %s for baserunning transitions", table)
    query = _build_query(table)
    df = conn.execute(query).fetchdf()
    logger.info("Query returned %d grouped rows", len(df))

    # Count total PAs and excluded PAs
    total_pas = int(df["n"].sum())

    # Build transition dict
    transitions: dict[str, list[dict]] = {}
    for (pa_result, pre_bases, outs), group in df.groupby(
        ["pa_result", "pre_bases", "outs"]
    ):
        group_total = group["n"].sum()
        entries = []
        for _, row in group.iterrows():
            entries.append({
                "post_bases": int(row["post_bases"]),
                "runs_scored": int(row["runs_scored"]),
                "outs_added": int(row["outs_added"]),
                "p": round(int(row["n"]) / int(group_total), 6),
            })
        key = f"{pa_result}|{int(pre_bases)}|{int(outs)}"
        transitions[key] = entries

    result = {
        "metadata": {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "source": table,
            "n_pas": total_pas,
            "n_keys": len(transitions),
            "n_transition_entries": sum(len(v) for v in transitions.values()),
        },
        "transitions": transitions,
    }

    logger.info(
        "Built table: %d PAs, %d keys, %d transition entries",
        total_pas,
        result["metadata"]["n_keys"],
        result["metadata"]["n_transition_entries"],
    )
    return result


def save(table: dict, path: str) -> str:
    """Write baserunning table as JSON. Supports local paths and s3:// URIs.

    Returns the path written.
    """
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

    logger.info("Saved baserunning table to %s", path)
    return path


def validate(table: dict) -> None:
    """Verify all transition probability groups sum to 1.0."""
    errors = []
    for key, transitions in table["transitions"].items():
        total = sum(t["p"] for t in transitions)
        if abs(total - 1.0) > 1e-4:
            errors.append(f"{key}: probabilities sum to {total:.6f}")
    if errors:
        raise ValueError(
            f"Probability validation failed for {len(errors)} keys:\n"
            + "\n".join(errors[:10])
        )
    logger.info("Validation passed: all %d keys sum to 1.0", len(table["transitions"]))


if __name__ == "__main__":
    import argparse

    import duckdb

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build baserunning transition table")
    parser.add_argument(
        "--db-path",
        default="stateball.duckdb",
        help="Path to local DuckDB file (default: stateball.duckdb)",
    )
    parser.add_argument(
        "--output",
        default="sim/data/baserunning.json",
        help="Output path for JSON artifact (default: sim/data/baserunning.json)",
    )
    args = parser.parse_args()

    conn = duckdb.connect(str(args.db_path), read_only=True)
    try:
        tbl = build(conn)
    finally:
        conn.close()

    validate(tbl)
    save(tbl, args.output)
