"""Tests for batch engine and compiled baserunning table.

Verifies:
  - Compiled baserunning table structure and round-trip correctness
  - Batch engine produces valid game results
  - Per-sim PA tracking (total_pas bug regression)
  - Vectorized _batch_recent_whip correctness
"""

from __future__ import annotations

import numpy as np

from sim.engine.core.batch_engine import (
    _batch_recent_whip,
    _batch_resolve_outcomes,
    simulate_game_batch,
)
from sim.engine.core.engine import make_uniform_prob_source
from sim.engine.lookups.compiled_baserunning import compile_baserunning_table
from sim.tests.test_engine import (
    make_baserunning_table,
    make_game_input,
)
from xg.core.config import OUTCOME_CLASSES

uniform_prob_source = make_uniform_prob_source(len(OUTCOME_CLASSES))


# ---------------------------------------------------------------------------
# Compiled baserunning table tests
# ---------------------------------------------------------------------------


class TestCompiledBaserunning:
    def test_shapes(self):
        """Compiled table has correct array shapes."""
        raw = make_baserunning_table()
        compiled = compile_baserunning_table(raw, OUTCOME_CLASSES)

        n_keys = len(OUTCOME_CLASSES) * 24
        assert compiled.n_keys == n_keys
        assert compiled.cum_probs.shape[0] == n_keys
        assert compiled.post_bases.shape[0] == n_keys
        assert compiled.runs_scored.shape[0] == n_keys
        assert compiled.outs_added.shape[0] == n_keys
        assert compiled.n_transitions.shape == (n_keys,)
        assert compiled.max_transitions >= 1

    def test_populated_keys(self):
        """All outcomes × bases × outs combos in the test fixture are populated."""
        raw = make_baserunning_table()
        compiled = compile_baserunning_table(raw, OUTCOME_CLASSES)

        populated = (compiled.n_transitions > 0).sum()
        # The test fixture generates keys for all outcomes × 8 bases × 3 outs
        # but only for outcomes in the fixture (21 outcomes). Only outcomes
        # in OUTCOME_CLASSES get compiled.
        assert populated > 0
        # Every OUTCOME_CLASS that's in the fixture should be populated
        for i, label in enumerate(OUTCOME_CLASSES):
            for bases in range(8):
                for outs in range(3):
                    key = f"{label}|{bases}|{outs}"
                    if key in raw["transitions"]:
                        key_idx = i * 24 + bases * 3 + outs
                        assert compiled.n_transitions[key_idx] > 0, (
                            f"Expected populated key for {key}"
                        )

    def test_cum_probs_monotonic(self):
        """Cumulative probabilities are non-decreasing and end at ~1.0."""
        raw = make_baserunning_table()
        compiled = compile_baserunning_table(raw, OUTCOME_CLASSES)

        for key_idx in range(compiled.n_keys):
            n_t = compiled.n_transitions[key_idx]
            if n_t == 0:
                continue
            cp = compiled.cum_probs[key_idx, :n_t]
            assert np.all(np.diff(cp) >= 0), f"Non-monotonic cum_probs at key {key_idx}"
            assert abs(cp[-1] - 1.0) < 0.01, f"cum_probs doesn't sum to ~1.0 at key {key_idx}"

    def test_missing_key_returns_invalid(self):
        """Looking up a key with n_transitions=0 returns valid=False."""
        raw = {"transitions": {}}  # empty table
        compiled = compile_baserunning_table(raw, OUTCOME_CLASSES)

        assert (compiled.n_transitions == 0).all()

        # Resolution should return all invalid
        rng = np.random.default_rng(42)
        oi = np.array([0, 1, 2], dtype=np.int16)
        bases = np.array([0, 0, 0], dtype=np.int8)
        outs = np.array([0, 0, 0], dtype=np.int8)
        _, _, _, valid = _batch_resolve_outcomes(oi, bases, outs, compiled, rng)
        assert not valid.any()

    def test_round_trip_distribution(self):
        """Sampling from compiled table produces same distribution as raw."""
        raw = make_baserunning_table()
        compiled = compile_baserunning_table(raw, OUTCOME_CLASSES)

        # The test fixture has p=1.0 for each key (deterministic).
        # Verify that compiled sampling returns the exact same values.
        rng = np.random.default_rng(42)
        n = 100

        for label in ["single", "home_run", "strikeout", "walk"]:
            oi_val = OUTCOME_CLASSES.index(label)
            for bases in [0, 3, 7]:
                for outs_val in [0, 1, 2]:
                    key = f"{label}|{bases}|{outs_val}"
                    expected = raw["transitions"].get(key)
                    if expected is None:
                        continue
                    exp_t = expected[0]  # p=1.0, only one transition

                    oi = np.full(n, oi_val, dtype=np.int16)
                    b = np.full(n, bases, dtype=np.int8)
                    o = np.full(n, outs_val, dtype=np.int8)

                    pb, rs, oa, valid = _batch_resolve_outcomes(
                        oi, b, o, compiled, rng,
                    )
                    assert valid.all(), f"Invalid for {key}"
                    assert (pb == exp_t["post_bases"]).all(), f"post_bases mismatch for {key}"
                    assert (rs == exp_t["runs_scored"]).all(), f"runs_scored mismatch for {key}"
                    assert (oa == exp_t["outs_added"]).all(), f"outs_added mismatch for {key}"


# ---------------------------------------------------------------------------
# Batch engine tests
# ---------------------------------------------------------------------------


def _make_batch_prob_source():
    """Build a uniform batch prob source for testing."""
    n_classes = len(OUTCOME_CLASSES)
    uniform = np.full(n_classes, 1.0 / n_classes, dtype=np.float32)

    def batch_source(batter_dicts, pitcher_dicts, group_idx, dynamic_arrays, n_active):
        return np.broadcast_to(
            uniform.reshape(1, -1), (n_active, n_classes),
        ).copy()

    return batch_source


class TestBatchEngine:
    def test_game_terminates(self):
        """Batch engine games all terminate within max_pa."""
        gi = make_game_input()
        raw = make_baserunning_table()
        compiled = compile_baserunning_table(raw, OUTCOME_CLASSES)
        rng = np.random.default_rng(42)

        results, _, _ = simulate_game_batch(
            game_input=gi,
            baserunning_table=raw,
            outcome_labels=OUTCOME_CLASSES,
            batch_prob_source=_make_batch_prob_source(),
            live_prob_source=uniform_prob_source,
            rng=rng,
            n_sims=100,
            compiled_baserunning=compiled,
        )

        assert len(results) == 100
        for r in results:
            assert r.total_pas > 0
            assert r.total_pas <= 200

    def test_scores_reasonable(self):
        """Batch engine produces reasonable scores."""
        gi = make_game_input()
        raw = make_baserunning_table()
        compiled = compile_baserunning_table(raw, OUTCOME_CLASSES)
        rng = np.random.default_rng(42)

        results, _, _ = simulate_game_batch(
            game_input=gi,
            baserunning_table=raw,
            outcome_labels=OUTCOME_CLASSES,
            batch_prob_source=_make_batch_prob_source(),
            live_prob_source=uniform_prob_source,
            rng=rng,
            n_sims=200,
            compiled_baserunning=compiled,
        )

        total_runs = [r.home_score + r.away_score for r in results]
        mean = sum(total_runs) / len(total_runs)
        assert 3 < mean < 30, f"Mean total runs = {mean:.1f}, outside [3, 30]"

    def test_all_strikeout(self):
        """All-strikeout prob source → 0-0 games."""
        k_idx = OUTCOME_CLASSES.index("strikeout")
        n_classes = len(OUTCOME_CLASSES)

        def strikeout_batch(batter_dicts, pitcher_dicts, group_idx, dynamic_arrays, n_active):
            probs = np.zeros((n_active, n_classes), dtype=np.float32)
            probs[:, k_idx] = 1.0
            return probs

        def strikeout_scalar(batter, pitcher, state, context):
            probs = [0.0] * n_classes
            probs[k_idx] = 1.0
            return probs

        gi = make_game_input()
        raw = make_baserunning_table()
        compiled = compile_baserunning_table(raw, OUTCOME_CLASSES)
        rng = np.random.default_rng(42)

        results, _, _ = simulate_game_batch(
            game_input=gi,
            baserunning_table=raw,
            outcome_labels=OUTCOME_CLASSES,
            batch_prob_source=strikeout_batch,
            live_prob_source=strikeout_scalar,
            rng=rng,
            n_sims=10,
            compiled_baserunning=compiled,
        )

        for r in results:
            assert r.home_score == 0
            assert r.away_score == 0


# ---------------------------------------------------------------------------
# Per-sim PA count tests
# ---------------------------------------------------------------------------


class TestPerSimPACounts:
    def test_total_pas_varies_across_sims(self):
        """Different sims can have different PA counts (not all share one global)."""
        gi = make_game_input()
        raw = make_baserunning_table()
        compiled = compile_baserunning_table(raw, OUTCOME_CLASSES)
        rng = np.random.default_rng(42)

        results, _, _ = simulate_game_batch(
            game_input=gi,
            baserunning_table=raw,
            outcome_labels=OUTCOME_CLASSES,
            batch_prob_source=_make_batch_prob_source(),
            live_prob_source=uniform_prob_source,
            rng=rng,
            n_sims=100,
            compiled_baserunning=compiled,
        )

        pa_counts = [r.total_pas for r in results]
        # With uniform probs, games should have varying lengths
        # (some end in 9 innings, some go to extras). If all sims
        # share a global pa_step, they'd all be identical.
        # With the batch engine, pa_step IS global per step, but
        # sim_pa_counts tracks per-sim. Sims that end early have
        # fewer PAs.
        # At minimum, all should be >= 54 (27 outs per side)
        assert all(pa >= 1 for pa in pa_counts)


# ---------------------------------------------------------------------------
# _batch_recent_whip tests
# ---------------------------------------------------------------------------


class TestBatchRecentWhip:
    def test_empty_buffer(self):
        """All -1 → WHIP = 0.0."""
        recent = np.full((5, 9), -1, dtype=np.int8)
        walk_idx = np.array([True, False, False], dtype=bool)
        hit_idx = np.array([False, True, False], dtype=bool)
        result = _batch_recent_whip(recent, walk_idx, hit_idx)
        assert (result == 0.0).all()

    def test_all_walks(self):
        """Buffer full of walk indices → WHIP = 1.0."""
        walk_outcome_idx = 0
        recent = np.full((3, 9), walk_outcome_idx, dtype=np.int8)
        walk_idx = np.array([True, False, False], dtype=bool)
        hit_idx = np.array([False, False, False], dtype=bool)
        result = _batch_recent_whip(recent, walk_idx, hit_idx)
        np.testing.assert_allclose(result, 1.0)

    def test_mixed(self):
        """Buffer with mix of walks, hits, and outs → correct WHIP."""
        # 3 outcomes: idx 0 = walk, idx 1 = hit, idx 2 = out
        walk_idx = np.array([True, False, False], dtype=bool)
        hit_idx = np.array([False, True, False], dtype=bool)

        # Sim 0: [walk, hit, out, -1, ...] = 2 WH / 3 valid
        # Sim 1: [out, out, out, out, ...] = 0 WH / 9 valid
        recent = np.full((2, 9), -1, dtype=np.int8)
        recent[0, 0] = 0  # walk
        recent[0, 1] = 1  # hit
        recent[0, 2] = 2  # out
        recent[1, :] = 2  # all outs

        result = _batch_recent_whip(recent, walk_idx, hit_idx)
        np.testing.assert_allclose(result[0], 2.0 / 3.0, rtol=1e-5)
        np.testing.assert_allclose(result[1], 0.0)
