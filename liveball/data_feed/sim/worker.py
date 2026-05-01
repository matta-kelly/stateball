"""Sim worker — runs in child process via ProcessPoolExecutor.

Each worker loads the Simulator once at process init. The parent
submits pickled GameInputs; only 5 numbers come back.
"""

import pickle

_simulator = None


def init_worker(from_s3_kwargs: dict, seed: int, estimator: str = "smc"):
    """Called once per child process — loads Simulator from S3."""
    global _simulator
    from sim.simulator import Simulator

    _simulator = Simulator.from_s3(**from_s3_kwargs, seed=seed, estimator=estimator)


def run_sim(game_input_bytes: bytes) -> dict:
    """Run Monte Carlo sim on a pickled GameInput."""
    import numpy as np

    game_input = pickle.loads(game_input_bytes)
    result = _simulator.simulate(game_input)

    # Baseline WE from lookup table at current game state
    we_baseline = None
    if _simulator.we_array is not None:
        gs = game_input.game_state
        inn = min(gs.inning, 9)
        outs = min(gs.outs, 2)  # clamp — outs=3 means half-inning over
        rd = int(np.clip(gs.home_score - gs.away_score, -15, 15))
        we_baseline = float(_simulator.we_array[inn, gs.half, outs, gs.bases, rd + 15])

    return {
        "p_home_win": result.p_home_win,
        "p_home_win_se": result.p_home_win_se,
        "mean_home_score": result.mean_home_score,
        "mean_away_score": result.mean_away_score,
        "n_sims": result.n_sims,
        "we_baseline": we_baseline,
    }
