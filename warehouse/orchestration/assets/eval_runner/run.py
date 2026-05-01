"""Sample simulation — stratified by inning for balanced evaluation.

Picks random feature vectors from the warehouse, stratified by inning
(n_per_inning games for each of innings 1-9), builds GameInputs, runs
Monte Carlo, and compares predicted winners to actual outcomes.

Usage (CLI):
    python -m orchestration.assets.eval_runner.cli --n-per-inning 100 --n-sims 1000

Also callable as a Dagster asset via orchestration/assets.py.
Requires DuckLake connection (env vars) and S3 access for model artifacts.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Callable

from .helpers import bases_str, entry_phase, half_str, reconstruct_scores

logger = logging.getLogger(__name__)

_S3_BUCKET = os.environ.get("S3_BUCKET", "dazoo")
_S3_ARTIFACTS = f"s3://{_S3_BUCKET}/stateball/artifacts"


def _get_actual_horizons(conn, game_pk: int, entry_ab: int) -> dict:
    """Query actual game events after entry and extract states at each horizon.

    Returns {horizon_name: {home_score, away_score, inning, half, outs, bases}}.
    Keys: "+1pa" for first PA, "+1hi", "+2hi", ... for half-inning boundaries.
    """
    rows = conn.execute(
        """
        SELECT at_bat_number, inning, half, outs, bases, home_score, away_score
        FROM _local_game_events
        WHERE game_pk = ? AND at_bat_number > ?
        ORDER BY at_bat_number
        """,
        [game_pk, entry_ab],
    ).fetchall()

    if not rows:
        return {}

    # Deduplicate to one row per PA (first pitch of each AB)
    pas = []
    seen_abs = set()
    for r in rows:
        ab = r[0]
        if ab not in seen_abs:
            seen_abs.add(ab)
            pas.append({
                "at_bat_number": r[0], "inning": r[1], "half": r[2],
                "outs": r[3], "bases": r[4], "home_score": r[5], "away_score": r[6],
            })

    result = {}

    # +1 PA
    if pas:
        result["+1pa"] = pas[0]

    # Half-inning boundaries
    hi_idx = 1
    for i in range(1, len(pas)):
        if (pas[i]["inning"] != pas[i-1]["inning"] or
                pas[i]["half"] != pas[i-1]["half"]):
            result[f"+{hi_idx}hi"] = pas[i]
            hi_idx += 1

    return result


def _build_horizons(result, actual_horizons: dict, we_table, we_entry: float = 0.5) -> dict | None:
    """Build horizon comparison dict from sim result and actual game events.

    At each checkpoint, computes:
    - sim_p_home: fraction of sims active at that point that went on to win
    - pred_run_diff: distribution of run differentials across sims
    - correction_var: variance of (WE_h - WE_{h-1}) across sims (for MLMC weights)
    - actual state and WE at actual state
    """
    import numpy as _np
    from sim.engine.lookups.win_expectancy import lookup as we_lookup

    hd = result.horizon_data
    if hd is None:
        return None

    # Per-sim final outcomes (did home win?)
    sim_home_wins = _np.array(
        [r.home_score > r.away_score for r in result.results], dtype=bool,
    )

    horizons = {}

    def _checkpoint_dict(
        n_reached, sim_p_home, rd, actual, pred_inning, pred_half, pred_outs, pred_bases,
        correction_var_val=None,
    ):
        actual_rd = actual["home_score"] - actual["away_score"]
        actual_half_str = "Bot" if actual["half"] else "Top"
        actual_we = we_lookup(
            we_table, actual["inning"], actual_half_str,
            actual["outs"], actual["bases"], actual_rd,
        )
        # WE distribution across sims — one lookup per sim at its actual run diff
        pred_half_str = "Bot" if int(pred_half) else "Top"
        we_vals = [
            we_lookup(we_table, int(pred_inning), pred_half_str,
                      int(pred_outs), int(pred_bases), int(r))
            for r in rd
        ]
        we_arr = _np.array(we_vals)

        # Convergence: std at increasing subsample sizes.
        # First-N is valid — sims are exchangeable (independent, no ordering).
        convergence = {}
        for sub_n in (25, 50, 100, 200, 500):
            if len(we_arr) >= sub_n:
                convergence[sub_n] = round(float(we_arr[:sub_n].std()), 4)

        return {
            "n_reached": n_reached,
            "sim_p_home": round(sim_p_home, 4),
            "pred_we": round(float(we_arr.mean()), 4),
            "pred_we_std": round(float(we_arr.std()), 4),
            "convergence": convergence,
            "pred_run_diff": {
                "mean": round(float(rd.mean()), 2),
                "std": round(float(rd.std()), 2),
            },
            "actual_home_score": actual["home_score"],
            "actual_away_score": actual["away_score"],
            "actual_we": round(actual_we, 4),
            "correction_var": round(correction_var_val, 6) if correction_var_val is not None else None,
        }

    # +1 PA
    if "+1pa" in actual_horizons and hd.pa1_active.any():
        actual = actual_horizons["+1pa"]
        active = hd.pa1_active
        n_reached = int(active.sum())
        sim_p_home = float(sim_home_wins[active].mean())
        rd = (hd.pa1_home_score[active] - hd.pa1_away_score[active]).astype(float)
        # Use modal state for WE lookup (most common inning/half/outs/bases)
        horizons["+1pa"] = _checkpoint_dict(
            n_reached, sim_p_home, rd, actual,
            _np.median(hd.pa1_inning[active]),
            _np.median(hd.pa1_half[active]),
            _np.median(hd.pa1_outs[active]),
            _np.median(hd.pa1_bases[active]),
        )

    # Half-inning boundaries — with correction_var for MLMC weights
    hi_idx = 1
    while f"+{hi_idx}hi" in actual_horizons:
        actual = actual_horizons[f"+{hi_idx}hi"]
        boundary = hi_idx - 1

        reached = hd.hi_count > boundary
        if not reached.any():
            break

        n_reached = int(reached.sum())
        sim_p_home = float(sim_home_wins[reached].mean())
        rd = (
            hd.hi_home_score[reached, boundary]
            - hd.hi_away_score[reached, boundary]
        ).astype(float)

        # Correction variance: var(WE_h - WE_{h-1}) across sims
        corr_var = None
        current_we = hd.hi_we[reached, boundary]
        valid_we = current_we >= 0
        if valid_we.sum() > 1:
            if boundary == 0:
                # First boundary: correction = WE_1 - WE_entry
                corrections = current_we[valid_we] - we_entry
            else:
                prev_we = hd.hi_we[reached, boundary - 1]
                both_valid = valid_we & (prev_we >= 0)
                if both_valid.sum() > 1:
                    corrections = current_we[both_valid] - prev_we[both_valid]
                else:
                    corrections = None
            if corrections is not None and len(corrections) > 1:
                corr_var = float(_np.var(corrections, ddof=1))

        horizons[f"+{hi_idx}hi"] = _checkpoint_dict(
            n_reached, sim_p_home, rd, actual,
            _np.median(hd.hi_inning[reached, boundary]),
            _np.median(hd.hi_half[reached, boundary]),
            0, 0,
            corr_var,
        )
        hi_idx += 1

    return horizons if horizons else None


def run(
    n_per_inning: int = 100,
    n_sims: int = 1000,
    seed: int = 42,
    log_fn: Callable[[str], None] | None = None,
    profile: bool = False,
    slot: str = "test",
    estimator: str = "naive_mc",
    adaptive_n: bool = False,
    enable_pruning: bool = False,
    estimator_config=None,
) -> dict:
    """Run sample simulations stratified by inning × run_diff_bucket.

    Samples across 45 strata (9 innings × 5 run diff buckets: blowout_away,
    lead_away, tied, lead_home, blowout_home) for even coverage of the
    state space. n_per_inning games per inning, split evenly across buckets.

    Args:
        n_per_inning: Games to sample per inning (1-9).
        n_sims: Monte Carlo iterations per game.
        seed: RNG seed for reproducibility.
        log_fn: Logging callback. Defaults to logger.info.
        profile: Enable per-PA timing instrumentation in the engine.
        slot: Artifact slot to resolve ("prod" or "test").

    Returns:
        (result_dict, simulator) — result has run_id, n_games, accuracy, timing.
    """
    if log_fn is None:
        log_fn = logger.info

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    t_total = time.perf_counter()

    # --- [setup 1/6] Connect to DuckLake ---
    log_fn("[setup 1/6] Connecting to DuckLake...")
    t0 = time.perf_counter()
    from orchestration.lib import get_ducklake_connection
    conn = get_ducklake_connection()
    log_fn(f"[setup 1/6] Connected ({time.perf_counter() - t0:.1f}s)")

    # --- [setup 2/6] Load simulator artifacts from S3 ---
    log_fn("[setup 2/6] Resolving config and loading artifacts...")
    t0 = time.perf_counter()
    from sim.infra.bootstrap import bootstrap
    if estimator_config is None:
        from sim.engine.estimators.config import NaiveMcConfig, SmcConfig, TruncatedMcConfig
        if estimator == "naive_mc":
            estimator_config = NaiveMcConfig(
                n_sims=n_sims, adaptive_n=adaptive_n,
                enable_pruning=enable_pruning,
            )
        elif estimator == "smc":
            estimator_config = SmcConfig()
        elif estimator == "truncated_mc":
            estimator_config = TruncatedMcConfig(
                n_sims=n_sims, adaptive_n=adaptive_n,
                enable_pruning=enable_pruning,
            )
    sim, config_id, artifact_paths = bootstrap(
        conn, slot=slot, seed=seed, estimator=estimator,
        estimator_config=estimator_config, log_fn=log_fn,
    )
    log_fn(f"[setup 2/6] Simulator ready ({time.perf_counter() - t0:.1f}s)")

    # --- [setup 3/6] Cache batter profiles ---
    log_fn("[setup 3/6] Caching batter profiles from DuckLake to local memory...")
    t0 = time.perf_counter()
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE _local_batter_profiles AS
        SELECT * FROM lakehouse.main.int_mlb__batter_profile
    """)
    log_fn(f"[setup 3/6] Batter profiles cached ({time.perf_counter() - t0:.1f}s)")

    # --- [setup 4/6] Cache pitcher profiles ---
    log_fn("[setup 4/6] Caching pitcher profiles from DuckLake to local memory...")
    t0 = time.perf_counter()
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE _local_pitcher_profiles AS
        SELECT * FROM lakehouse.main.int_mlb__pitcher_profile
    """)
    log_fn(f"[setup 4/6] Pitcher profiles cached ({time.perf_counter() - t0:.1f}s)")

    # --- [setup 5/6] Stratified sampling: inning × run_diff_bucket ---
    from sim.engine.estimators.buckets import SQL_CASE, N_BUCKETS

    n_per_bucket = max(1, n_per_inning // N_BUCKETS)
    log_fn(f"[setup 5/6] Sampling {n_per_bucket}/bucket × {N_BUCKETS} buckets × 9 innings "
           f"(target {n_per_bucket * N_BUCKETS * 9} games)...")
    t0 = time.perf_counter()
    samples_df = conn.execute(
        f"""
        WITH bucketed AS (
            SELECT v.*, g.home_score AS actual_home_score, g.away_score AS actual_away_score,
                {SQL_CASE} AS rd_bucket
            FROM lakehouse.main.feat_mlb__vectors v
            JOIN lakehouse.main.proc_mlb__games g
              ON v.game_pk = g.game_pk
            WHERE g.abstract_game_state = 'Final'
              AND g.status NOT IN ('Postponed', 'Cancelled')
              AND v.inning BETWEEN 1 AND 9
              AND EXISTS (SELECT 1 FROM lakehouse.main.proc_mlb__rosters r WHERE r.game_pk = g.game_pk)
        )
        SELECT * FROM bucketed
        QUALIFY ROW_NUMBER() OVER (PARTITION BY inning, rd_bucket ORDER BY RANDOM()) <= ?
        ORDER BY inning, game_pk
        """,
        [n_per_bucket],
    ).fetchdf()
    n_total = len(samples_df)
    per_inn_counts = samples_df.groupby("inning").size().to_dict()
    # Log distribution by inning and bucket
    if "rd_bucket" in samples_df.columns:
        bucket_counts = samples_df.groupby(["inning", "rd_bucket"]).size()
        log_fn(f"[setup 5/6] Sampled {n_total} vectors across {len(bucket_counts)} strata: "
               + ", ".join(f"inn{k}={v}" for k, v in sorted(per_inn_counts.items()))
               + f" ({time.perf_counter() - t0:.1f}s)")
    else:
        log_fn(f"[setup 5/6] Sampled {n_total} vectors: "
               + ", ".join(f"inn{k}={v}" for k, v in sorted(per_inn_counts.items()))
               + f" ({time.perf_counter() - t0:.1f}s)")

    # --- [setup 6/6] Cache game events for horizon evaluation ---
    t0 = time.perf_counter()
    game_pks = samples_df["game_pk"].unique().tolist()
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _local_game_events AS
        SELECT game_pk, at_bat_number, inning,
               CASE WHEN inning_topbot = 'Bot' THEN 1 ELSE 0 END AS half,
               outs_when_up AS outs,
               (CASE WHEN on_1b IS NOT NULL THEN 1 ELSE 0 END)
               + (CASE WHEN on_2b IS NOT NULL THEN 2 ELSE 0 END)
               + (CASE WHEN on_3b IS NOT NULL THEN 4 ELSE 0 END) AS bases,
               CAST(home_score AS INT) AS home_score,
               CAST(away_score AS INT) AS away_score
        FROM lakehouse.main.proc_mlb__events
        WHERE game_pk IN ({','.join(str(pk) for pk in game_pks)})
          AND pa_result IS NOT NULL
          AND pa_result != 'truncated_pa'
    """)
    log_fn(f"[setup 6/6] Game events cached ({time.perf_counter() - t0:.1f}s)")

    setup_elapsed = time.perf_counter() - t_total
    log_fn(f"Setup complete in {setup_elapsed:.1f}s — starting {n_total} simulations")

    # --- Run simulations ---
    from sim.game_inputs.game import init_game

    correct = 0
    total_mc_elapsed = 0.0
    game_details: list[dict] = []

    log_fn("")
    log_fn("=" * 120)
    log_fn(f"{'#':>3}  {'game_pk':>8}  {'date':>10}  {'entry state':>30}  "
           f"{'P(home)':>16}  {'pred':>4}  {'actual':>8}  {'sim score':>9}  "
           f"{'ok':>2}  {'time':>6}")
    log_fn("=" * 120)

    for i, (_, row) in enumerate(samples_df.iterrows(), 1):
        game_pk = int(row["game_pk"])
        game_date = str(row["game_date"])
        vector = row.to_dict()

        # Build game input
        t0 = time.perf_counter()
        game_input = init_game(vector, conn)
        state = game_input.game_state
        init_elapsed = time.perf_counter() - t0

        # Entry state
        bases = bases_str(state.bases)
        run_diff = int(row["run_diff"])
        home_lead = reconstruct_scores(
            state.half, run_diff,
            int(row["actual_home_score"]), int(row["actual_away_score"]),
        )
        score_str = f"H{home_lead:+d}" if home_lead != 0 else "tied"
        phase = entry_phase(state.inning)
        state_desc = f"{half_str(state.half)} {state.inning}, {state.outs}o {bases} ({score_str})"

        # Log entry state
        log_fn(f"[{i}/{n_total}] {game_pk}  {game_date}  {state_desc}  "
               f"AB#{int(row['at_bat_number'])}  {phase}  "
               f"lineup {len(game_input.home_lineup)}v{len(game_input.away_lineup)}  "
               f"pen {len(game_input.home_bullpen)}H/{len(game_input.away_bullpen)}A")

        # Run MC
        t0 = time.perf_counter()
        result = sim.simulate(game_input, profile=profile)
        mc_elapsed = time.perf_counter() - t0
        total_mc_elapsed += mc_elapsed

        # Compare
        actual_home_win = int(row["actual_home_score"]) > int(row["actual_away_score"])
        predicted_home_win = result.p_home_win > 0.5
        is_correct = predicted_home_win == actual_home_win
        if is_correct:
            correct += 1
        log_fn(
            f"{i:>3}  {game_pk:>8}  P(home)={result.p_home_win:.3f}  "
            f"{'Y' if is_correct else 'N'}  {mc_elapsed:.1f}s"
        )

        # --- Per-sim analysis ---
        import numpy as _np
        _results = result.results
        _n = result.n_sims

        # WE-only fallback (no sims ran — uncalibrated state bucket)
        if _n == 0 or not _results:
            _we_baseline = None
            if sim.we_table is not None:
                from sim.engine.lookups.win_expectancy import lookup as _we_lookup
                _we_baseline = round(_we_lookup(
                    sim.we_table, state.inning, half_str(state.half),
                    state.outs, state.bases, home_lead,
                ), 4)

            detail = {
                "game_pk": game_pk,
                "game_date": game_date,
                "entry": {
                    "inning": state.inning, "half": state.half,
                    "outs": state.outs, "bases": state.bases,
                    "run_diff": run_diff, "home_run_diff": home_lead,
                    "at_bat_number": int(row["at_bat_number"]),
                    "phase": phase,
                    "rd_bucket": str(row["rd_bucket"]) if "rd_bucket" in row.index else None,
                    "we_baseline": _we_baseline,
                },
                "prediction": {
                    "p_home_win": round(result.p_home_win, 4),
                    "p_home_win_se": 0.0,
                    "ci_95": [round(result.p_home_win, 4), round(result.p_home_win, 4)],
                    "we_only": True,
                },
                "actual": {
                    "home_score": int(row["actual_home_score"]),
                    "away_score": int(row["actual_away_score"]),
                    "home_win": actual_home_win,
                },
                "correct": is_correct,
                "timing_s": {"mc": round(mc_elapsed, 2), "init_game": round(init_elapsed, 2)},
            }
            game_details.append(detail)
            continue

        _home_scores = _np.array([r.home_score for r in _results])
        _away_scores = _np.array([r.away_score for r in _results])
        _total_scores = _home_scores + _away_scores
        _home_wins = _home_scores > _away_scores
        _home_win_count = int(_home_wins.sum())

        _extras = sum(1 for r in _results if r.innings > 9)
        _pitcher_changes_h = [r.home_pitcher_changes for r in _results]
        _pitcher_changes_a = [r.away_pitcher_changes for r in _results]
        _pcts = [10, 25, 50, 75, 90]

        def _score_pcts(arr):
            ps = _np.percentile(arr, _pcts)
            return {f"p{p}": int(v) for p, v in zip(_pcts, ps)} | {"mean": round(float(arr.mean()), 2)}

        # WE baseline for this entry state
        _we_baseline = None
        if sim.we_table is not None:
            from sim.engine.lookups.win_expectancy import lookup as _we_lookup
            _we_baseline = round(_we_lookup(
                sim.we_table, state.inning, half_str(state.half),
                state.outs, state.bases, home_lead,
            ), 4)

        # Build structured game detail
        detail = {
            "game_pk": game_pk,
            "game_date": game_date,
            "entry": {
                "inning": state.inning,
                "half": state.half,
                "outs": state.outs,
                "bases": state.bases,
                "run_diff": run_diff,
                "home_run_diff": home_lead,
                "at_bat_number": int(row["at_bat_number"]),
                "phase": phase,
                "rd_bucket": str(row["rd_bucket"]) if "rd_bucket" in row.index else None,
                "we_baseline": _we_baseline,
            },
            "prediction": {
                "p_home_win": round(result.p_home_win, 4),
                "p_home_win_se": round(result.p_home_win_se, 4),
                "ci_95": [
                    round(max(0, result.p_home_win - 1.96 * result.p_home_win_se), 4),
                    round(min(1, result.p_home_win + 1.96 * result.p_home_win_se), 4),
                ],
                "home_win_frac": round(_home_win_count / _n, 4),
                "mean_home_score": round(result.mean_home_score, 2),
                "mean_away_score": round(result.mean_away_score, 2),
                "std_home_score": round(float(_home_scores.std()), 2),
                "std_away_score": round(float(_away_scores.std()), 2),
                "mean_innings": round(result.mean_innings, 2),
                "predicted_home_win": predicted_home_win,
                "extras_frac": round(_extras / _n, 4),
                "mean_pitcher_changes_home": round(sum(_pitcher_changes_h) / _n, 2),
                "mean_pitcher_changes_away": round(sum(_pitcher_changes_a) / _n, 2),
            },
            "actual": {
                "home_score": int(row["actual_home_score"]),
                "away_score": int(row["actual_away_score"]),
                "home_win": actual_home_win,
            },
            "correct": is_correct,
            "timing_s": {
                "mc": round(mc_elapsed, 2),
                "init_game": round(init_elapsed, 2),
            },
            "sim_scores": {
                "home": _score_pcts(_home_scores),
                "away": _score_pcts(_away_scores),
                "total": _score_pcts(_total_scores),
            },
        }

        # Pruning stats
        pruned_results = [r for r in _results if r.pruned]
        n_pruned = len(pruned_results)
        detail["pruning"] = {
            "n_pruned": n_pruned,
            "prune_rate": round(n_pruned / _n, 4),
            "mean_pa_at_prune": round(
                sum(r.pruned_at_pa for r in pruned_results) / n_pruned, 1
            ) if n_pruned else None,
        }

        # Per-game outcome counts
        if result.outcome_counts:
            detail["outcome_counts"] = dict(result.outcome_counts)

        # Include profiling per-game if available
        if result.mean_timings is not None:
            detail["profiling"] = asdict(result.mean_timings)

        # Horizon evaluation
        actual_horizons = _get_actual_horizons(
            conn, game_pk, int(row["at_bat_number"]),
        )
        horizons = _build_horizons(result, actual_horizons, sim.we_table, we_entry=_we_baseline or 0.5)
        if horizons:
            detail["horizons"] = horizons

        game_details.append(detail)

    # --- Record to DuckLake ---
    total_elapsed = time.perf_counter() - t_total
    n = len(game_details)
    accuracy = correct / n if n > 0 else 0.0
    mean_mc_time = total_mc_elapsed / n if n > 0 else 0.0

    log_fn(
        f"Eval {run_id}: {correct}/{n} correct ({accuracy:.0%}), "
        f"{mean_mc_time:.2f}s/game, {total_elapsed:.1f}s total"
    )

    if config_id is not None:
        try:
            from sim.infra.artifact_catalog import record_eval
            from orchestration.lib import get_ducklake_connection

            conn_reg = get_ducklake_connection()
            try:
                record_eval(
                    conn_reg, config_id, run_id, game_details,
                    n_sims=n_sims, n_per_inning=n_per_inning,
                    estimator=estimator, seed=seed,
                    total_time=total_elapsed, setup_time=setup_elapsed,
                )
                log_fn(f"[eval] Recorded eval {run_id} for config {config_id}")
            finally:
                conn_reg.close()
        except Exception as e:
            log_fn(f"WARNING: Failed to record eval: {e}")

    result = {
        "run_id": run_id,
        "n_games": n,
        "correct": correct,
        "accuracy": accuracy,
        "mean_mc_time": mean_mc_time,
        "total_time": total_elapsed,
    }
    return result, sim
