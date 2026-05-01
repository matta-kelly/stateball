from fastapi import APIRouter, HTTPException

from backend.db import get_conn

router = APIRouter()


def _brier_decomposition(
    predictions: list[float], outcomes: list[bool], n_bins: int = 20,
) -> dict | None:
    """Murphy (1973) Brier decomposition for binary predictions."""
    n = len(predictions)
    if n < 20:
        return None

    o_bar = sum(outcomes) / n
    uncertainty = o_bar * (1 - o_bar)
    brier = sum((p - float(o)) ** 2 for p, o in zip(predictions, outcomes)) / n

    bins: dict[int, list[tuple[float, float]]] = {}
    for p, o in zip(predictions, outcomes):
        b = min(int(p * n_bins), n_bins - 1)
        bins.setdefault(b, []).append((p, float(o)))

    reliability = 0.0
    resolution = 0.0
    bin_details: list[dict] = []

    for b in range(n_bins):
        entries = bins.get(b, [])
        n_k = len(entries)
        if n_k == 0:
            continue
        p_bar_k = sum(p for p, _ in entries) / n_k
        o_bar_k = sum(o for _, o in entries) / n_k
        rel_k = n_k * (p_bar_k - o_bar_k) ** 2
        res_k = n_k * (o_bar_k - o_bar) ** 2
        reliability += rel_k
        resolution += res_k
        bin_details.append({
            "bin_lo": round(b / n_bins, 3),
            "bin_hi": round((b + 1) / n_bins, 3),
            "n": n_k,
            "mean_pred": round(p_bar_k, 4),
            "actual_rate": round(o_bar_k, 4),
            "gap": round(p_bar_k - o_bar_k, 4),
            "reliability_contrib": round(rel_k / n, 6),
        })

    reliability /= n
    resolution /= n
    brier_skill = 1.0 - (brier / uncertainty) if uncertainty > 0 else 0.0

    return {
        "brier": round(brier, 6),
        "reliability": round(reliability, 6),
        "resolution": round(resolution, 6),
        "uncertainty": round(uncertainty, 6),
        "brier_skill": round(brier_skill, 4),
        "bins": bin_details,
    }


@router.get("/evals")
def list_evals():
    conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT eval_id, config_id, created_at, n_games, n_sims,
                   accuracy, mean_p_home, mean_mc_time,
                   estimator, score_mae, prune_rate, n_per_inning
            FROM lh.main.sim_eval
            WHERE NOT (
                (estimator = 'naive_mc' OR estimator IS NULL)
                AND (n_per_inning >= 100 OR n_per_inning IS NULL)
            )
            ORDER BY created_at DESC
        """).fetchall()

        evals = []
        for row in rows:
            (eval_id, config_id, created_at, n_games, n_sims,
             accuracy, mean_p_home, mean_mc_time,
             estimator, score_mae, prune_rate, n_per_inning) = row
            evals.append({
                "eval_id": eval_id,
                "config_id": config_id,
                "created_at": str(created_at),
                "n_games": n_games,
                "n_sims": n_sims,
                "accuracy": accuracy,
                "mean_p_home": mean_p_home,
                "mean_mc_time": mean_mc_time,
                "estimator": estimator,
                "score_mae": score_mae,
                "prune_rate": prune_rate,
                "n_per_inning": n_per_inning,
            })
        return evals
    finally:
        conn.close()


@router.get("/evals/{eval_id}")
def get_eval(eval_id: str):
    conn = get_conn()
    try:
        row = conn.execute("""
            SELECT eval_id, config_id, created_at, n_games, n_sims,
                   accuracy, mean_p_home, mean_mc_time,
                   estimator, score_mae, prune_rate, n_per_inning,
                   total_time, setup_time, score_error_home, score_error_away,
                   seed, artifact_path
            FROM lh.main.sim_eval
            WHERE eval_id = ?
        """, [eval_id]).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"Eval {eval_id} not found")

        (eid, config_id, created_at, n_games, n_sims,
         accuracy, mean_p_home, mean_mc_time,
         estimator, score_mae, prune_rate, n_per_inning,
         total_time, setup_time, score_error_home, score_error_away,
         seed, artifact_path) = row

        # Per-game details from child table
        game_rows = conn.execute("""
            SELECT game_pk, game_date, entry_inning, entry_half, entry_outs,
                   entry_bases, entry_run_diff, entry_phase, entry_rd_bucket,
                   entry_we_baseline, p_home_win, p_home_win_se,
                   mean_home_score, mean_away_score,
                   actual_home_score, actual_away_score, actual_home_win,
                   correct, mc_time_s
            FROM lh.main.sim_eval_games
            WHERE eval_id = ?
            ORDER BY game_pk
        """, [eval_id]).fetchall()

        games = []
        for gr in game_rows:
            games.append({
                "game_pk": gr[0], "game_date": gr[1],
                "entry_inning": gr[2], "entry_half": gr[3],
                "entry_outs": gr[4], "entry_bases": gr[5],
                "entry_run_diff": gr[6], "entry_phase": gr[7],
                "entry_rd_bucket": gr[8], "entry_we_baseline": gr[9],
                "p_home_win": gr[10], "p_home_win_se": gr[11],
                "mean_home_score": gr[12], "mean_away_score": gr[13],
                "actual_home_score": gr[14], "actual_away_score": gr[15],
                "actual_home_win": gr[16], "correct": gr[17],
                "mc_time_s": gr[18],
            })

        # Level diagnostics from child table
        level_rows = conn.execute("""
            SELECT level, n, entry_mae, pred_mae, improvement,
                   pred_we_std, delta_std, brier, brier_skill,
                   reliability, resolution, uncertainty
            FROM lh.main.sim_eval_levels
            WHERE eval_id = ?
            ORDER BY level
        """, [eval_id]).fetchall()

        levels = {}
        for lr in level_rows:
            levels[lr[0]] = {
                "n": lr[1], "entry_mae": lr[2], "pred_mae": lr[3],
                "improvement": lr[4], "pred_we_std": lr[5],
                "delta_std": lr[6], "brier": lr[7], "brier_skill": lr[8],
                "reliability": lr[9], "resolution": lr[10],
                "uncertainty": lr[11],
            }

        # Accuracy by inning
        inning_rows = conn.execute("""
            SELECT entry_inning, COUNT(*) as n,
                   SUM(CASE WHEN correct THEN 1 ELSE 0 END) as correct
            FROM lh.main.sim_eval_games WHERE eval_id = ?
            GROUP BY entry_inning ORDER BY entry_inning
        """, [eval_id]).fetchall()
        accuracy_by_inning = {}
        for inn, n, correct_count in inning_rows:
            accuracy_by_inning[str(inn)] = {
                "n": n, "correct": correct_count,
                "accuracy": round(correct_count / n, 4) if n > 0 else None,
            }

        # Accuracy by phase
        phase_rows = conn.execute("""
            SELECT entry_phase, COUNT(*) as n,
                   SUM(CASE WHEN correct THEN 1 ELSE 0 END) as correct
            FROM lh.main.sim_eval_games WHERE eval_id = ?
            GROUP BY entry_phase ORDER BY entry_phase
        """, [eval_id]).fetchall()
        accuracy_by_phase = {}
        for phase, n, correct_count in phase_rows:
            if phase is not None:
                accuracy_by_phase[phase] = {
                    "n": n, "correct": correct_count,
                    "accuracy": round(correct_count / n, 4) if n > 0 else None,
                }

        # Win probability Brier decomposition
        wp_rows = conn.execute("""
            SELECT p_home_win, actual_home_win
            FROM lh.main.sim_eval_games WHERE eval_id = ?
        """, [eval_id]).fetchall()
        preds = [r[0] for r in wp_rows if r[0] is not None and r[1] is not None]
        outcomes = [bool(r[1]) for r in wp_rows if r[0] is not None and r[1] is not None]
        win_probability = _brier_decomposition(preds, outcomes)

        return {
            "eval_id": eid,
            "config_id": config_id,
            "created_at": str(created_at),
            "n_games": n_games,
            "n_sims": n_sims,
            "accuracy": accuracy,
            "mean_p_home": mean_p_home,
            "mean_mc_time": mean_mc_time,
            "estimator": estimator,
            "score_mae": score_mae,
            "prune_rate": prune_rate,
            "n_per_inning": n_per_inning,
            "total_time": total_time,
            "setup_time": setup_time,
            "score_error_home": score_error_home,
            "score_error_away": score_error_away,
            "seed": seed,
            "artifact_path": artifact_path,
            "games": games,
            "level_diagnostics": levels,
            "accuracy_by_inning": accuracy_by_inning,
            "accuracy_by_phase": accuracy_by_phase,
            "win_probability": win_probability,
            "scores": {
                "mean_error_home": score_error_home,
                "mean_error_away": score_error_away,
                "mean_abs_error": score_mae,
            } if score_mae is not None else None,
        }
    finally:
        conn.close()


def _artifact_label(s3_path: str | None) -> str | None:
    """Extract a short label from an S3 path like 's3://bucket/stateball/artifacts/xgboost/20260310_abc'."""
    if not s3_path:
        return None
    parts = s3_path.rstrip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


# ---------------------------------------------------------------------------
# Eval diagnostics
# ---------------------------------------------------------------------------


@router.get("/evals/{eval_id}/diagnostics")
def get_eval_diagnostics(eval_id: str, inning: str = ""):
    """Compute horizon-level diagnostics from raw tables.

    Price accuracy, price movement, calibration — all computed at query time.
    Optional inning filter (1-9) to slice by entry inning.
    """
    conn = get_conn()
    try:
        inning_clause = ""
        params = [eval_id]
        if inning and inning != "all":
            inning_clause = "AND g.entry_inning = ?"
            params.append(int(inning))

        # Aggregate metrics per horizon
        agg_rows = conn.execute(f"""
            SELECT
                h.horizon,
                COUNT(*) as n,
                AVG(ABS(h.pred_we - h.actual_we)) as pred_mae,
                AVG(ABS(g.entry_we_baseline - h.actual_we)) as entry_mae,
                AVG(h.pred_we - g.entry_we_baseline) as mean_predicted_move,
                AVG(h.actual_we - g.entry_we_baseline) as mean_actual_move,
                AVG(ABS((h.pred_we - g.entry_we_baseline) - (h.actual_we - g.entry_we_baseline))) as move_mae,
                AVG((h.pred_we - g.entry_we_baseline) - (h.actual_we - g.entry_we_baseline)) as move_bias
            FROM lh.main.sim_eval_horizons h
            JOIN lh.main.sim_eval_games g
                ON h.eval_id = g.eval_id AND h.game_pk = g.game_pk
            WHERE h.eval_id = ?
                AND h.pred_we IS NOT NULL
                AND h.actual_we IS NOT NULL
                AND g.entry_we_baseline IS NOT NULL
                AND h.horizon LIKE '+%hi'
                {inning_clause}
            GROUP BY h.horizon
            ORDER BY CAST(SUBSTR(h.horizon, 2, LENGTH(h.horizon) - 3) AS INTEGER)
        """, params).fetchall()

        # Per-horizon Brier: need raw pred_we + actual_home_win
        brier_rows = conn.execute(f"""
            SELECT h.horizon, h.pred_we, g.actual_home_win
            FROM lh.main.sim_eval_horizons h
            JOIN lh.main.sim_eval_games g
                ON h.eval_id = g.eval_id AND h.game_pk = g.game_pk
            WHERE h.eval_id = ?
                AND h.pred_we IS NOT NULL
                AND h.horizon LIKE '+%hi'
                {inning_clause}
            ORDER BY h.horizon
        """, params).fetchall()

        # Group Brier data by horizon
        brier_by_hz: dict[str, tuple[list[float], list[bool]]] = {}
        for hz, pred_we, home_win in brier_rows:
            if hz not in brier_by_hz:
                brier_by_hz[hz] = ([], [])
            brier_by_hz[hz][0].append(float(pred_we))
            brier_by_hz[hz][1].append(bool(home_win))

        horizons = []
        for row in agg_rows:
            hz, n, pred_mae, entry_mae, pred_move, actual_move, move_mae, move_bias = row
            improvement = (
                round((entry_mae - pred_mae) / entry_mae, 4)
                if entry_mae and entry_mae > 0 else 0.0
            )

            bss = None
            reliability = None
            if hz in brier_by_hz:
                preds, outcomes = brier_by_hz[hz]
                decomp = _brier_decomposition(preds, outcomes)
                if decomp:
                    bss = decomp.get("brier_skill")
                    reliability = decomp.get("reliability")

            horizons.append({
                "horizon": hz,
                "n": int(n),
                "pred_mae": round(float(pred_mae), 4) if pred_mae else None,
                "entry_mae": round(float(entry_mae), 4) if entry_mae else None,
                "improvement": improvement,
                "move_mae": round(float(move_mae), 4) if move_mae else None,
                "move_bias": round(float(move_bias), 4) if move_bias else None,
                "bss": bss,
                "reliability": reliability,
            })

        return {
            "eval_id": eval_id,
            "inning_filter": inning if inning and inning != "all" else None,
            "horizons": horizons,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Calibration endpoints
# ---------------------------------------------------------------------------

_CALIBRATION_TYPES = ("n_lookup", "stopping_thresholds", "gamma_schedule", "horizon_weights")


@router.get("/calibrations")
def list_calibrations():
    """List calibration eval runs with their derived artifact status."""
    conn = get_conn()
    try:
        # Get all naive_mc evals with high n_per_inning (calibration runs)
        evals = conn.execute("""
            SELECT eval_id, created_at, n_games, n_sims, accuracy,
                   mean_mc_time, n_per_inning, total_time
            FROM lh.main.sim_eval
            WHERE (estimator = 'naive_mc' OR estimator IS NULL)
              AND n_per_inning >= 100
            ORDER BY created_at DESC
        """).fetchall()

        # Get all calibration artifacts
        artifacts = conn.execute("""
            SELECT artifact_id, artifact_type, run_id, is_prod, is_test
            FROM lh.main.artifact_registry
            WHERE artifact_type IN (?, ?, ?, ?)
        """, list(_CALIBRATION_TYPES)).fetchall()

        # Index artifacts by run_id
        by_run: dict[str, dict] = {}
        for aid, atype, run_id, is_prod, is_test in artifacts:
            by_run.setdefault(run_id, {})[atype] = {
                "artifact_id": aid,
                "is_prod": bool(is_prod),
                "is_test": bool(is_test),
            }

        result = []
        for (eval_id, created_at, n_games, n_sims, accuracy,
             mean_mc_time, n_per_inning, total_time) in evals:
            arts = by_run.get(eval_id, {})
            result.append({
                "eval_id": eval_id,
                "created_at": str(created_at),
                "n_games": n_games,
                "n_sims": n_sims,
                "accuracy": accuracy,
                "mean_mc_time": mean_mc_time,
                "n_per_inning": n_per_inning,
                "total_time": total_time,
                "artifacts": {
                    atype: arts.get(atype) for atype in _CALIBRATION_TYPES
                },
                "is_promoted": any(
                    arts.get(atype, {}).get("is_prod", False)
                    for atype in _CALIBRATION_TYPES
                ),
            })
        return result
    finally:
        conn.close()


@router.get("/artifacts/{artifact_type}")
def list_artifacts_by_type(artifact_type: str):
    """List all artifacts of a given type with their slot status."""
    conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT artifact_id, run_id, s3_path, is_prod, is_test, created_at, metrics
            FROM lh.main.artifact_registry
            WHERE artifact_type = ?
            ORDER BY created_at DESC
        """, [artifact_type]).fetchall()

        import json
        result = []
        for aid, run_id, s3_path, is_prod, is_test, created_at, metrics_json in rows:
            metrics = {}
            if metrics_json:
                try:
                    metrics = json.loads(metrics_json)
                except Exception:
                    pass
            result.append({
                "artifact_id": aid,
                "run_id": run_id,
                "s3_path": s3_path,
                "is_prod": bool(is_prod) if is_prod is not None else False,
                "is_test": bool(is_test) if is_test is not None else False,
                "created_at": str(created_at),
                "metrics": metrics,
            })
        return result
    finally:
        conn.close()


@router.post("/artifacts/{artifact_id:path}/promote")
def promote_artifact(artifact_id: str, slot: str = "prod"):
    """Promote a single artifact to a slot."""
    if slot not in ("prod", "test"):
        raise HTTPException(400, f"Invalid slot: {slot}")
    conn = get_conn()
    try:
        from sim.infra.artifact_catalog import set_artifact_slot
        set_artifact_slot(conn, artifact_id, slot, active=True)
        return {"promoted": artifact_id, "slot": slot}
    except ValueError as e:
        raise HTTPException(404, str(e))
    finally:
        conn.close()
