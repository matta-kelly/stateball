"""Artifact registry, sim configuration, and eval tracking.

Three DuckLake tables manage the lifecycle of simulation artifacts:
  - artifact_registry: catalog of every built artifact with metrics
  - sim_config: named combinations of artifacts (one active at a time)
  - sim_eval: evaluation results tied to sim configs

Tables are created lazily via ensure_registry_tables(). All functions
accept a DuckDB connection (local or DuckLake).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema prefix detection
# ---------------------------------------------------------------------------

def _schema_prefix(conn) -> str:
    """Return 'lakehouse.main' or 'main' depending on connection type."""
    try:
        conn.execute("SELECT 1 FROM lakehouse.main.proc_mlb__events LIMIT 0")
        return "lakehouse.main"
    except Exception:
        return "main"


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

_TABLES_ENSURED = False


def ensure_registry_tables(conn) -> None:
    """Create registry tables if they don't exist. Idempotent."""
    global _TABLES_ENSURED
    if _TABLES_ENSURED:
        return

    prefix = _schema_prefix(conn)

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}.artifact_registry (
            artifact_id   VARCHAR,
            artifact_type VARCHAR NOT NULL,
            run_id        VARCHAR NOT NULL,
            s3_path       VARCHAR NOT NULL,
            created_at    TIMESTAMP NOT NULL,
            metrics       VARCHAR
        )
    """)

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}.sim_config (
            config_id          VARCHAR,
            name               VARCHAR NOT NULL,
            status             VARCHAR NOT NULL DEFAULT 'candidate',
            created_at         TIMESTAMP NOT NULL,
            promoted_at        TIMESTAMP,
            xgboost_id         VARCHAR NOT NULL,
            baserunning_id     VARCHAR NOT NULL,
            pitcher_exit_id    VARCHAR NOT NULL,
            win_expectancy_id  VARCHAR NOT NULL
        )
    """)

    # Migrate: add per-artifact slot columns to artifact_registry
    for col in ("is_prod", "is_test"):
        try:
            conn.execute(
                f"ALTER TABLE {prefix}.artifact_registry ADD COLUMN {col} BOOLEAN"
            )
        except Exception:
            pass  # Column already exists

    # Migrate: rename legacy artifact types → n_lookup
    for old_type in ('smc_config', 'convergence_config'):
        conn.execute(f"""
            UPDATE {prefix}.artifact_registry
            SET artifact_type = 'n_lookup',
                artifact_id = REPLACE(artifact_id, '{old_type}/', 'n_lookup/')
            WHERE artifact_type = '{old_type}'
        """)

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}.sim_eval (
            eval_id    VARCHAR,
            config_id  VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            n_games    INTEGER,
            n_sims     INTEGER,
            accuracy   DOUBLE,
            mean_p_home DOUBLE,
            mean_mc_time DOUBLE,
            results    VARCHAR
        )
    """)

    # New sim_eval columns (idempotent via try/except)
    for col_def in (
        "n_per_inning INTEGER",
        "estimator VARCHAR",
        "seed INTEGER",
        "total_time DOUBLE",
        "setup_time DOUBLE",
        "score_error_home DOUBLE",
        "score_error_away DOUBLE",
        "score_mae DOUBLE",
        "prune_rate DOUBLE",
        "artifact_path VARCHAR",
    ):
        try:
            conn.execute(
                f"ALTER TABLE {prefix}.sim_eval ADD COLUMN {col_def}"
            )
        except Exception:
            pass

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}.sim_eval_games (
            eval_id           VARCHAR NOT NULL,
            game_pk           INTEGER NOT NULL,
            game_date         VARCHAR,
            entry_inning      INTEGER,
            entry_half        INTEGER,
            entry_outs        INTEGER,
            entry_bases       INTEGER,
            entry_run_diff    INTEGER,
            entry_phase       VARCHAR,
            entry_rd_bucket   VARCHAR,
            entry_we_baseline DOUBLE,
            p_home_win        DOUBLE,
            p_home_win_se     DOUBLE,
            mean_home_score   DOUBLE,
            mean_away_score   DOUBLE,
            actual_home_score INTEGER,
            actual_away_score INTEGER,
            actual_home_win   BOOLEAN,
            correct           BOOLEAN,
            mc_time_s         DOUBLE
        )
    """)

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}.sim_eval_levels (
            eval_id       VARCHAR NOT NULL,
            level         VARCHAR NOT NULL,
            n             INTEGER,
            entry_mae     DOUBLE,
            pred_mae      DOUBLE,
            improvement   DOUBLE,
            pred_we_std   DOUBLE,
            delta_std     DOUBLE,
            brier         DOUBLE,
            brier_skill   DOUBLE,
            reliability   DOUBLE,
            resolution    DOUBLE,
            uncertainty   DOUBLE
        )
    """)

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}.sim_eval_convergence (
            eval_id         VARCHAR NOT NULL,
            inning          INTEGER NOT NULL,
            rd_bucket       VARCHAR NOT NULL,
            level           VARCHAR NOT NULL,
            n_games         INTEGER,
            stabilization_n INTEGER,
            std_at_25       DOUBLE,
            std_at_50       DOUBLE,
            std_at_100      DOUBLE,
            std_at_200      DOUBLE,
            std_at_500      DOUBLE
        )
    """)

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}.sim_eval_horizons (
            eval_id            VARCHAR NOT NULL,
            game_pk            INTEGER NOT NULL,
            horizon            VARCHAR NOT NULL,
            n_reached          INTEGER,
            pred_we            DOUBLE,
            pred_we_std        DOUBLE,
            actual_we          DOUBLE,
            pred_run_diff_mean DOUBLE,
            pred_run_diff_std  DOUBLE,
            actual_home_score  INTEGER,
            actual_away_score  INTEGER
        )
    """)

    _TABLES_ENSURED = True
    logger.info("Registry tables ensured in %s", prefix)


# ---------------------------------------------------------------------------
# Artifact registry
# ---------------------------------------------------------------------------


def register_artifact(
    conn,
    artifact_type: str,
    run_id: str,
    s3_path: str,
    metrics: dict,
) -> str:
    """Insert a new artifact into the registry. Returns artifact_id."""
    ensure_registry_tables(conn)
    prefix = _schema_prefix(conn)

    artifact_id = f"{artifact_type}/{run_id}"
    now = datetime.now(timezone.utc).isoformat()
    metrics_json = json.dumps(metrics, default=str)

    conn.execute(
        f"""
        INSERT INTO {prefix}.artifact_registry
            (artifact_id, artifact_type, run_id, s3_path, created_at, metrics)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [artifact_id, artifact_type, run_id, s3_path, now, metrics_json],
    )

    logger.info("Registered artifact: %s → %s", artifact_id, s3_path)
    return artifact_id


def list_artifacts(conn, artifact_type: str | None = None) -> list[dict]:
    """List all artifacts, optionally filtered by type."""
    ensure_registry_tables(conn)
    prefix = _schema_prefix(conn)

    if artifact_type:
        df = conn.execute(
            f"SELECT * FROM {prefix}.artifact_registry WHERE artifact_type = ? ORDER BY created_at DESC",
            [artifact_type],
        ).fetchdf()
    else:
        df = conn.execute(
            f"SELECT * FROM {prefix}.artifact_registry ORDER BY artifact_type, created_at DESC"
        ).fetchdf()

    rows = []
    for _, row in df.iterrows():
        d = dict(row)
        if d.get("metrics"):
            try:
                d["metrics"] = json.loads(d["metrics"])
            except (json.JSONDecodeError, TypeError):
                pass
        rows.append(d)
    return rows


# ---------------------------------------------------------------------------
# Per-artifact slot assignment
# ---------------------------------------------------------------------------


def _slot_col(slot: str) -> str:
    """Map slot name to column name."""
    return {"prod": "is_prod", "test": "is_test"}[slot]


def set_artifact_slot(conn, artifact_id: str, slot: str, active: bool) -> None:
    """Set or clear an artifact's assignment to a slot.

    An artifact can be in both prod and test simultaneously. Setting
    active=True clears any other artifact of the same type from that
    slot first (at most one per type per slot).
    """
    if slot not in ("prod", "test"):
        raise ValueError(f"slot must be 'prod' or 'test', got {slot!r}")

    ensure_registry_tables(conn)
    prefix = _schema_prefix(conn)
    col = _slot_col(slot)

    if not active:
        conn.execute(
            f"UPDATE {prefix}.artifact_registry SET {col} = FALSE WHERE artifact_id = ?",
            [artifact_id],
        )
        logger.info("Unassigned artifact %s from %s", artifact_id, slot)
        return

    # Look up artifact type
    row = conn.execute(
        f"SELECT artifact_type FROM {prefix}.artifact_registry WHERE artifact_id = ?",
        [artifact_id],
    ).fetchone()
    if row is None:
        raise ValueError(f"Artifact {artifact_id!r} not found")
    artifact_type = row[0]

    # Clear previous holder of this (type, slot)
    conn.execute(
        f"UPDATE {prefix}.artifact_registry SET {col} = FALSE "
        f"WHERE artifact_type = ? AND {col} = TRUE",
        [artifact_type],
    )

    # Assign
    conn.execute(
        f"UPDATE {prefix}.artifact_registry SET {col} = TRUE WHERE artifact_id = ?",
        [artifact_id],
    )
    logger.info("Assigned artifact %s to %s", artifact_id, slot)


CALIBRATION_ARTIFACT_TYPES = ("n_lookup", "stopping_thresholds", "gamma_schedule", "horizon_cutoff")


def promote_calibration(conn, eval_id: str, slot: str = "prod") -> list[str]:
    """Promote all calibration artifacts from a given eval run to a slot.

    Each calibration eval produces up to 4 artifacts (n_lookup,
    stopping_thresholds, gamma_schedule, horizon_cutoff). This promotes
    whichever exist for the given eval_id.
    """
    promoted = []
    for atype in CALIBRATION_ARTIFACT_TYPES:
        artifact_id = f"{atype}/{eval_id}"
        try:
            set_artifact_slot(conn, artifact_id, slot, active=True)
            promoted.append(artifact_id)
        except ValueError:
            pass  # artifact not built for this eval
    logger.info("Promoted calibration %s to %s: %d artifacts", eval_id, slot, len(promoted))
    return promoted


def get_slot_artifacts(conn, slot: str = "prod") -> dict[str, dict]:
    """Return artifacts assigned to a slot, keyed by artifact_type.

    Returns {artifact_type: {artifact_id, s3_path, run_id, created_at, metrics}}
    for each type that has a slot assignment. Empty dict if no assignments.
    """
    if slot not in ("prod", "test"):
        raise ValueError(f"slot must be 'prod' or 'test', got {slot!r}")

    ensure_registry_tables(conn)
    prefix = _schema_prefix(conn)
    col = _slot_col(slot)

    rows = conn.execute(
        f"""
        SELECT artifact_id, artifact_type, run_id, s3_path, created_at, metrics
        FROM {prefix}.artifact_registry
        WHERE {col} = TRUE
        """,
    ).fetchall()

    result: dict[str, dict] = {}
    for row in rows:
        artifact_id, artifact_type, run_id, s3_path, created_at, metrics = row
        parsed_metrics = {}
        if metrics:
            try:
                parsed_metrics = json.loads(metrics)
            except (json.JSONDecodeError, TypeError):
                pass
        result[artifact_type] = {
            "artifact_id": artifact_id,
            "s3_path": s3_path,
            "run_id": run_id,
            "created_at": str(created_at),
            "metrics": parsed_metrics,
        }

    return result


def get_manifest_path(conn, slot: str = "test") -> str | None:
    """Return the S3 path of the feature manifest in the given slot, or None."""
    artifacts = get_slot_artifacts(conn, slot=slot)
    entry = artifacts.get("feature_manifest")
    return entry["s3_path"] if entry else None


# ---------------------------------------------------------------------------
# Sim configuration (legacy — kept for backward compat)
# ---------------------------------------------------------------------------


def create_sim_config(
    conn,
    name: str,
    xgboost_id: str,
    baserunning_id: str,
    pitcher_exit_id: str,
    win_expectancy_id: str,
    auto_promote: bool = False,
) -> str:
    """Create a new sim config. Returns config_id."""
    ensure_registry_tables(conn)
    prefix = _schema_prefix(conn)

    config_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    now = datetime.now(timezone.utc).isoformat()

    conn.execute(
        f"""
        INSERT INTO {prefix}.sim_config
            (config_id, name, status, created_at, xgboost_id,
             baserunning_id, pitcher_exit_id, win_expectancy_id)
        VALUES (?, ?, 'candidate', ?, ?, ?, ?, ?)
        """,
        [config_id, name, now, xgboost_id, baserunning_id,
         pitcher_exit_id, win_expectancy_id],
    )

    logger.info("Created sim config: %s (%s)", config_id, name)

    if auto_promote:
        promote_sim_config(conn, config_id)

    return config_id


def promote_sim_config(conn, config_id: str, *, target: str = "prod") -> None:
    """Promote config to target slot (prod or test), demote previous holder."""
    if target not in ("prod", "test"):
        raise ValueError(f"target must be 'prod' or 'test', got {target!r}")

    ensure_registry_tables(conn)
    prefix = _schema_prefix(conn)

    now = datetime.now(timezone.utc).isoformat()

    # Demote current holder of the target slot
    conn.execute(
        f"UPDATE {prefix}.sim_config SET status = 'archived' WHERE status = ?",
        [target],
    )

    # Promote target
    conn.execute(
        f"UPDATE {prefix}.sim_config SET status = ?, promoted_at = ? WHERE config_id = ?",
        [target, now, config_id],
    )

    logger.info("Promoted sim config %s to %s", config_id, target)


def get_active_config(conn, *, slot: str = "prod") -> dict | None:
    """Return the config in the given slot with resolved S3 paths, or None.

    Args:
        conn: DuckDB connection.
        slot: "prod" or "test" — which active config to return.
    """
    if slot not in ("prod", "test"):
        raise ValueError(f"slot must be 'prod' or 'test', got {slot!r}")

    ensure_registry_tables(conn)
    prefix = _schema_prefix(conn)

    row = conn.execute(
        f"""
        SELECT
            c.config_id, c.name, c.status, c.created_at, c.promoted_at,
            c.xgboost_id, c.baserunning_id, c.pitcher_exit_id, c.win_expectancy_id,
            xg.s3_path  AS xgboost_path,
            br.s3_path  AS baserunning_path,
            pe.s3_path  AS pitcher_exit_path,
            we.s3_path  AS win_expectancy_path
        FROM {prefix}.sim_config c
        LEFT JOIN {prefix}.artifact_registry xg ON c.xgboost_id = xg.artifact_id
        LEFT JOIN {prefix}.artifact_registry br ON c.baserunning_id = br.artifact_id
        LEFT JOIN {prefix}.artifact_registry pe ON c.pitcher_exit_id = pe.artifact_id
        LEFT JOIN {prefix}.artifact_registry we ON c.win_expectancy_id = we.artifact_id
        WHERE c.status = ?
        LIMIT 1
        """,
        [slot],
    ).fetchone()

    if row is None:
        return None

    cols = [
        "config_id", "name", "status", "created_at", "promoted_at",
        "xgboost_id", "baserunning_id", "pitcher_exit_id", "win_expectancy_id",
        "xgboost_path", "baserunning_path", "pitcher_exit_path", "win_expectancy_path",
    ]
    return dict(zip(cols, row))


def list_configs(conn) -> list[dict]:
    """List all sim configs."""
    ensure_registry_tables(conn)
    prefix = _schema_prefix(conn)

    df = conn.execute(
        f"SELECT * FROM {prefix}.sim_config ORDER BY created_at DESC"
    ).fetchdf()
    return [dict(row) for _, row in df.iterrows()]


# ---------------------------------------------------------------------------
# Eval results
# ---------------------------------------------------------------------------


def _bulk_insert(conn, table: str, columns: list[str], rows: list[tuple]) -> None:
    """Insert rows via a temp table to avoid DuckLake per-row transaction overhead.

    DuckLake's executemany creates a separate transaction per row, causing
    serialization failures. Instead: load into a local temp table, then
    INSERT INTO ... SELECT FROM in one shot.
    """
    if not rows:
        return
    col_str = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(columns))
    conn.execute("CREATE OR REPLACE TEMP TABLE _bulk_staging AS "
                 f"SELECT * FROM {table} WHERE 1=0")
    conn.executemany(
        f"INSERT INTO _bulk_staging ({col_str}) VALUES ({placeholders})", rows
    )
    conn.execute(f"INSERT INTO {table} SELECT {col_str} FROM _bulk_staging")
    conn.execute("DROP TABLE IF EXISTS _bulk_staging")


def _decompose_eval(conn, prefix: str, eval_id: str, summary: dict) -> None:
    """Insert per-game, per-level, and convergence rows for an eval."""
    # sim_eval_games
    games = summary.get("games", [])
    if games:
        rows = []
        for g in games:
            entry = g.get("entry", {})
            pred = g.get("prediction", {})
            actual = g.get("actual", {})
            rows.append((
                eval_id,
                g.get("game_pk"),
                g.get("game_date"),
                entry.get("inning"),
                entry.get("half"),
                entry.get("outs"),
                entry.get("bases"),
                entry.get("run_diff"),
                entry.get("phase"),
                entry.get("rd_bucket"),
                entry.get("we_baseline"),
                pred.get("p_home_win"),
                pred.get("p_home_win_se"),
                pred.get("mean_home_score"),
                pred.get("mean_away_score"),
                actual.get("home_score"),
                actual.get("away_score"),
                actual.get("home_win"),
                g.get("correct"),
                g.get("timing_s", {}).get("mc"),
            ))
        _bulk_insert(
            conn, f"{prefix}.sim_eval_games",
            ["eval_id", "game_pk", "game_date", "entry_inning", "entry_half",
             "entry_outs", "entry_bases", "entry_run_diff", "entry_phase",
             "entry_rd_bucket", "entry_we_baseline", "p_home_win", "p_home_win_se",
             "mean_home_score", "mean_away_score", "actual_home_score",
             "actual_away_score", "actual_home_win", "correct", "mc_time_s"],
            rows,
        )

    # sim_eval_levels
    levels = summary.get("level_diagnostics", {})
    if levels:
        rows = []
        for level, v in levels.items():
            rows.append((
                eval_id, level,
                v.get("n"),
                v.get("entry_mae"),
                v.get("pred_mae"),
                v.get("improvement"),
                v.get("pred_we_std"),
                v.get("delta_std"),
                v.get("brier"),
                v.get("brier_skill"),
                v.get("reliability"),
                v.get("resolution"),
                v.get("uncertainty"),
            ))
        _bulk_insert(
            conn, f"{prefix}.sim_eval_levels",
            ["eval_id", "level", "n", "entry_mae", "pred_mae", "improvement",
             "pred_we_std", "delta_std", "brier", "brier_skill",
             "reliability", "resolution", "uncertainty"],
            rows,
        )

    # sim_eval_convergence
    conv = summary.get("convergence", {})
    if conv:
        rows = []
        for state_key, level_data in conv.items():
            parts = state_key.split("|", 1)
            if len(parts) != 2:
                continue
            inning, rd_bucket = parts
            for level, curve in level_data.items():
                # Curve keys may be int (live) or str (JSON deserialized)
                def _cv(n):
                    return curve.get(n) or curve.get(str(n))

                rows.append((
                    eval_id,
                    int(inning),
                    rd_bucket,
                    level,
                    curve.get("n_games"),
                    curve.get("stabilization_n"),
                    _cv(25),
                    _cv(50),
                    _cv(100),
                    _cv(200),
                    _cv(500),
                ))
        _bulk_insert(
            conn, f"{prefix}.sim_eval_convergence",
            ["eval_id", "inning", "rd_bucket", "level", "n_games",
             "stabilization_n", "std_at_25", "std_at_50", "std_at_100",
             "std_at_200", "std_at_500"],
            rows,
        )

    # sim_eval_horizons
    if games:
        rows = []
        for g in games:
            game_pk = g.get("game_pk")
            horizons = g.get("horizons", {})
            for hz_key, hz in horizons.items():
                rd = hz.get("pred_run_diff", {})
                rows.append((
                    eval_id,
                    game_pk,
                    hz_key,
                    hz.get("n_reached"),
                    hz.get("pred_we"),
                    hz.get("pred_we_std"),
                    hz.get("actual_we"),
                    rd.get("mean"),
                    rd.get("std"),
                    hz.get("actual_home_score"),
                    hz.get("actual_away_score"),
                ))
        if rows:
            _bulk_insert(
                conn, f"{prefix}.sim_eval_horizons",
                ["eval_id", "game_pk", "horizon", "n_reached",
                 "pred_we", "pred_we_std", "actual_we",
                 "pred_run_diff_mean", "pred_run_diff_std",
                 "actual_home_score", "actual_away_score"],
                rows,
            )


def record_eval(
    conn,
    config_id: str,
    eval_id: str,
    summary: dict,
) -> None:
    """Insert eval results tied to a sim config."""
    ensure_registry_tables(conn)
    prefix = _schema_prefix(conn)

    now = datetime.now(timezone.utc).isoformat()

    # Extract top-level scalar metrics
    n_games = summary.get("n_games", 0)
    params = summary.get("params", {})
    n_sims = params.get("n_sims") or summary.get("n_sims", 0)
    accuracy = summary.get("accuracy", 0.0)
    mean_p_home = summary.get("mean_p_home", 0.0)
    mean_mc_time = summary.get("mean_mc_time", 0.0)

    # New scalar columns
    n_per_inning = summary.get("n_per_inning") or params.get("n_per_inning")
    estimator = params.get("estimator")
    seed = params.get("seed")
    total_time = summary.get("total_time")
    setup_time = summary.get("setup_time")
    scores = summary.get("scores") or {}
    score_error_home = scores.get("mean_error_home")
    score_error_away = scores.get("mean_error_away")
    score_mae = scores.get("mean_abs_error")
    pruning = summary.get("pruning") or {}
    prune_rate = pruning.get("mean_prune_rate")
    artifact_path = summary.get("artifact_path")

    # Full results as JSON (kept for backward compat)
    results_json = json.dumps(summary, default=str)

    conn.execute(
        f"""
        INSERT INTO {prefix}.sim_eval
            (eval_id, config_id, created_at, n_games, n_sims,
             accuracy, mean_p_home, mean_mc_time, results,
             n_per_inning, estimator, seed, total_time, setup_time,
             score_error_home, score_error_away, score_mae,
             prune_rate, artifact_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [eval_id, config_id, now, n_games, n_sims,
         accuracy, mean_p_home, mean_mc_time, results_json,
         n_per_inning, estimator, seed, total_time, setup_time,
         score_error_home, score_error_away, score_mae,
         prune_rate, artifact_path],
    )

    # Decompose into child tables
    _decompose_eval(conn, prefix, eval_id, summary)

    logger.info(
        "Recorded eval: %s (config=%s, accuracy=%.4f, n_games=%d)",
        eval_id, config_id, accuracy, n_games,
    )


def list_evals(conn, config_id: str | None = None) -> list[dict]:
    """List eval results, optionally filtered by config."""
    ensure_registry_tables(conn)
    prefix = _schema_prefix(conn)

    if config_id:
        df = conn.execute(
            f"SELECT * FROM {prefix}.sim_eval WHERE config_id = ? ORDER BY created_at DESC",
            [config_id],
        ).fetchdf()
    else:
        df = conn.execute(
            f"SELECT * FROM {prefix}.sim_eval ORDER BY created_at DESC"
        ).fetchdf()

    rows = []
    for _, row in df.iterrows():
        d = dict(row)
        if d.get("results"):
            try:
                d["results"] = json.loads(d["results"])
            except (json.JSONDecodeError, TypeError):
                pass
        rows.append(d)
    return rows


def backfill_eval_tables(conn) -> dict:
    """One-time backfill: decompose existing JSON blobs into eval child tables.

    Reads sim_eval.results JSON, updates scalar columns on sim_eval, and inserts
    into sim_eval_games, sim_eval_levels, sim_eval_convergence, sim_eval_horizons.

    Returns {"backfilled": N, "skipped": N, "errors": N, "horizons_backfilled": N}.
    """
    ensure_registry_tables(conn)
    prefix = _schema_prefix(conn)

    rows = conn.execute(f"""
        SELECT eval_id, results FROM {prefix}.sim_eval
        WHERE results IS NOT NULL
    """).fetchall()

    # Find eval_ids already in child tables to skip
    existing_game_evals = {
        r[0] for r in conn.execute(
            f"SELECT DISTINCT eval_id FROM {prefix}.sim_eval_games"
        ).fetchall()
    }
    existing_horizon_evals = {
        r[0] for r in conn.execute(
            f"SELECT DISTINCT eval_id FROM {prefix}.sim_eval_horizons"
        ).fetchall()
    }

    stats = {"backfilled": 0, "skipped": 0, "errors": 0, "horizons_backfilled": 0}

    for eval_id, results_json in rows:
        if eval_id in existing_game_evals:
            stats["skipped"] += 1
            continue

        try:
            summary = json.loads(results_json)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse results JSON for eval %s", eval_id)
            stats["errors"] += 1
            continue

        try:
            # Update scalar columns on sim_eval
            params = summary.get("params", {})
            scores = summary.get("scores") or {}
            pruning = summary.get("pruning") or {}
            conn.execute(
                f"""UPDATE {prefix}.sim_eval SET
                    n_per_inning = ?,
                    seed = ?,
                    total_time = ?,
                    setup_time = ?,
                    score_error_home = ?,
                    score_error_away = ?,
                    score_mae = ?,
                    prune_rate = ?,
                    artifact_path = ?
                WHERE eval_id = ?""",
                [
                    summary.get("n_per_inning") or params.get("n_per_inning"),
                    params.get("seed"),
                    summary.get("total_time"),
                    summary.get("setup_time"),
                    scores.get("mean_error_home"),
                    scores.get("mean_error_away"),
                    scores.get("mean_abs_error"),
                    pruning.get("mean_prune_rate"),
                    summary.get("artifact_path"),
                    eval_id,
                ],
            )

            # Decompose into child tables
            _decompose_eval(conn, prefix, eval_id, summary)
            stats["backfilled"] += 1
            logger.info("Backfilled eval %s", eval_id)

        except Exception as e:
            logger.warning("Failed to backfill eval %s: %s", eval_id, e)
            stats["errors"] += 1

    # Backfill horizons for evals that already had games but not horizons
    for eval_id, results_json in rows:
        if eval_id not in existing_game_evals or eval_id in existing_horizon_evals:
            continue
        try:
            summary = json.loads(results_json)
            games = summary.get("games", [])
            hz_rows = []
            for g in games:
                game_pk = g.get("game_pk")
                for hz_key, hz in g.get("horizons", {}).items():
                    rd = hz.get("pred_run_diff", {})
                    hz_rows.append((
                        eval_id, game_pk, hz_key,
                        hz.get("n_reached"), hz.get("pred_we"),
                        hz.get("pred_we_std"), hz.get("actual_we"),
                        rd.get("mean"), rd.get("std"),
                        hz.get("actual_home_score"), hz.get("actual_away_score"),
                    ))
            if hz_rows:
                _bulk_insert(
                    conn, f"{prefix}.sim_eval_horizons",
                    ["eval_id", "game_pk", "horizon", "n_reached",
                     "pred_we", "pred_we_std", "actual_we",
                     "pred_run_diff_mean", "pred_run_diff_std",
                     "actual_home_score", "actual_away_score"],
                    hz_rows,
                )
                stats["horizons_backfilled"] += 1
                logger.info("Backfilled horizons for eval %s (%d rows)", eval_id, len(hz_rows))
        except Exception as e:
            logger.warning("Failed to backfill horizons for eval %s: %s", eval_id, e)

    logger.info(
        "Backfill complete: %d backfilled, %d skipped, %d errors, %d horizons_backfilled",
        stats["backfilled"], stats["skipped"], stats["errors"], stats["horizons_backfilled"],
    )
    return stats


# ---------------------------------------------------------------------------
# Bootstrap — register existing artifacts from S3
# ---------------------------------------------------------------------------


def bootstrap_existing_artifacts(conn) -> dict:
    """Scan S3 for existing artifacts and register them.

    Also creates and promotes an initial sim config from the current
    DEFAULT_* paths in sample_sim.py. Idempotent — skips already-registered
    artifact_ids.

    Returns summary of what was registered.
    """
    import os

    import s3fs

    from sim.infra.artifact_loaders import _read_json

    ensure_registry_tables(conn)
    prefix = _schema_prefix(conn)

    fs = s3fs.S3FileSystem(
        key=os.environ.get("S3_ACCESS_KEY_ID", ""),
        secret=os.environ.get("S3_SECRET_ACCESS_KEY", ""),
        endpoint_url=os.environ.get("S3_ENDPOINT", ""),
    )

    _bucket = os.environ.get("S3_BUCKET", "dazoo")
    base = f"{_bucket}/stateball/artifacts"

    # Get existing artifact_ids to skip
    existing = set()
    try:
        rows = conn.execute(
            f"SELECT artifact_id FROM {prefix}.artifact_registry"
        ).fetchall()
        existing = {r[0] for r in rows}
    except Exception:
        pass

    registered = []

    # Scan sim artifacts (baserunning, win_expectancy)
    artifact_files = {
        "baserunning": "baserunning.json",
        "win_expectancy": "win_expectancy.json",
    }

    try:
        sim_runs = sorted(fs.ls(f"{base}/sim/"))
    except FileNotFoundError:
        sim_runs = []

    for run_dir in sim_runs:
        run_id = run_dir.split("/")[-1]
        try:
            files = fs.ls(run_dir)
        except Exception:
            continue

        fnames = {f.split("/")[-1]: f for f in files}
        for atype, fname in artifact_files.items():
            if fname in fnames:
                aid = f"{atype}/{run_id}"
                if aid in existing:
                    continue
                s3_path = f"s3://{fnames[fname]}"
                try:
                    data = _read_json(s3_path)
                    metrics = data.get("metadata", {})
                    register_artifact(conn, atype, run_id, s3_path, metrics)
                    registered.append(aid)
                except Exception as e:
                    logger.warning("Failed to register %s: %s", aid, e)

    # Scan xgboost artifacts
    try:
        xgb_runs = sorted(fs.ls(f"{base}/xgboost/"))
    except FileNotFoundError:
        xgb_runs = []

    for run_dir in xgb_runs:
        run_id = run_dir.split("/")[-1]
        aid = f"xgboost/{run_id}"
        if aid in existing:
            continue

        results_path = f"s3://{run_dir}/results.json"
        try:
            if not fs.exists(f"{run_dir}/results.json"):
                continue
            metrics = _read_json(results_path)
            # Classify by run_id suffix: _sim or _live
            if run_id.endswith("_sim"):
                register_artifact(conn, "xgboost_sim", run_id, f"s3://{run_dir}", metrics)
            elif run_id.endswith("_live"):
                register_artifact(conn, "xgboost_live", run_id, f"s3://{run_dir}", metrics)
            else:
                # Legacy runs before sim/live split — skip
                logger.info("Skipping legacy xgboost run (no _sim/_live suffix): %s", run_id)
                continue
            registered.append(aid)
        except Exception as e:
            logger.warning("Failed to register %s: %s", aid, e)

    # Auto-promote latest artifacts to prod if no prod artifacts exist
    if registered:
        required_types = ("xgboost_sim", "xgboost_live", "baserunning", "win_expectancy")
        slot_artifacts = get_slot_artifacts(conn, slot="prod")
        unslotted = [t for t in required_types if t not in slot_artifacts]

        if unslotted:
            for atype in unslotted:
                rows = conn.execute(
                    f"""
                    SELECT artifact_id FROM {prefix}.artifact_registry
                    WHERE artifact_type = ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    [atype],
                ).fetchall()
                if rows:
                    set_artifact_slot(conn, rows[0][0], "prod", True)
                    logger.info("Auto-promoted %s to prod: %s", atype, rows[0][0])
                else:
                    logger.warning("No %s artifact found to promote", atype)

    logger.info("Bootstrap complete: %d artifacts registered", len(registered))
    return {"registered": registered, "total": len(registered)}
