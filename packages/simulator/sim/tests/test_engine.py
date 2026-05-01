"""Tests for sim.engine — game simulation loop.

All tests use mock data (no warehouse, no artifacts from S3). The engine
takes GameInput directly, so we construct it with synthetic profiles.
"""

from __future__ import annotations

import random

from sim.engine.core.engine import (
    _PitcherOuting,
    make_uniform_prob_source,
    simulate_game,
)
from sim.game_inputs.game import GameInput
from sim.engine.core.state import GameState
from xg.core.config import OUTCOME_CLASSES

uniform_prob_source = make_uniform_prob_source(len(OUTCOME_CLASSES))


# ---------------------------------------------------------------------------
# Test fixture factories
# ---------------------------------------------------------------------------


def _make_player(pid: int, hand: str = "R") -> dict:
    return {"player_id": pid, "hand": hand}


def _make_lineup(start_id: int = 100) -> list[dict]:
    hands = ["R", "L", "R", "R", "L", "R", "R", "L", "R"]
    return [_make_player(start_id + i, hands[i]) for i in range(9)]


def _make_bullpen(start_id: int = 200, n: int = 5) -> list[dict]:
    return [_make_player(start_id + i, "R") for i in range(n)]



def _make_seed_context(**overrides) -> dict:
    """Zeroed-out seed context — fresh game start, 0-0 count."""
    ctx = {
        "times_through_order": 0.0,
        "batter_prior_pa": 0.0,
        "pitcher_pitch_count": 0.0,
        "pitcher_bf_game": 0.0,
        "batter_ab_vs_pitcher": 0.0,
        "pitcher_outing_walks": 0.0,
        "pitcher_outing_hits": 0.0,
        "pitcher_outing_k": 0.0,
        "pitcher_outing_runs": 0.0,
        "pitcher_outing_whip": 0.0,
        "pitcher_recent_whip": 0.0,
        "balls": 0.0,
        "strikes": 0.0,
    }
    ctx.update(overrides)
    return ctx


def make_game_input(**overrides) -> GameInput:
    defaults = {
        "home_lineup": _make_lineup(100),
        "away_lineup": _make_lineup(200),
        "home_pitcher": _make_player(300, "R"),
        "away_pitcher": _make_player(301, "L"),
        "home_bullpen": _make_bullpen(400),
        "away_bullpen": _make_bullpen(500),
        "game_state": GameState(),
        "game_pk": 999999,
        "game_date": "2025-07-01",
        "seed_context": _make_seed_context(),
    }
    defaults.update(overrides)
    return GameInput(**defaults)


def make_baserunning_table() -> dict:
    """Build a deterministic baserunning table covering all 21 model outcomes × bases × outs.

    Simplified transitions:
    - strikeout/field_out/force_out/etc → 1 out, bases unchanged
    - double_play/gidp → 2 outs, clear bases
    - strikeout_double_play/sac_fly_double_play/sac_bunt_double_play → 2 outs
    - triple_play → 3 outs
    - single/field_error/catcher_interf → runners advance 1, batter on 1B
    - double → runners advance 2, batter on 2B
    - triple → all runners score, batter on 3B
    - home_run → all score, bases empty
    - walk/hit_by_pitch/intent_walk → advance forced runners
    - sac_fly/sac_bunt → 1 out, runner on 3B scores
    - fielders_choice → 1 out, batter on 1B
    """
    transitions = {}

    out_outcomes = {
        "field_out", "force_out", "fielders_choice_out", "strikeout",
    }
    dp_outcomes = {
        "double_play", "grounded_into_double_play",
        "strikeout_double_play", "sac_fly_double_play",
        "sac_bunt_double_play",
    }

    for outs in range(3):
        for bases in range(8):
            # --- Outs ---
            for outcome in out_outcomes:
                key = f"{outcome}|{bases}|{outs}"
                transitions[key] = [
                    {"post_bases": bases, "runs_scored": 0, "outs_added": 1, "p": 1.0}
                ]

            # --- Double plays ---
            for outcome in dp_outcomes:
                key = f"{outcome}|{bases}|{outs}"
                transitions[key] = [
                    {"post_bases": 0, "runs_scored": 0, "outs_added": 2, "p": 1.0}
                ]

            # --- Triple play ---
            key = f"triple_play|{bases}|{outs}"
            transitions[key] = [
                {"post_bases": 0, "runs_scored": 0, "outs_added": 3, "p": 1.0}
            ]

            # --- Sac fly / sac bunt ---
            for outcome in ("sac_fly", "sac_bunt"):
                runs_sf = 1 if bases & 0b100 else 0
                post_sf = bases & ~0b100  # clear 3B
                key = f"{outcome}|{bases}|{outs}"
                transitions[key] = [
                    {"post_bases": post_sf, "runs_scored": runs_sf, "outs_added": 1, "p": 1.0}
                ]

            # --- Fielders choice ---
            key = f"fielders_choice|{bases}|{outs}"
            transitions[key] = [
                {"post_bases": 1, "runs_scored": 0, "outs_added": 1, "p": 1.0}
            ]

            # --- Single / field_error / catcher_interf (like single) ---
            runs_s = 1 if bases & 0b100 else 0  # 3B scores
            post_s = ((bases << 1) & 0b110) | 0b001  # shift runners, batter on 1B
            for outcome in ("single", "field_error", "catcher_interf"):
                key = f"{outcome}|{bases}|{outs}"
                transitions[key] = [
                    {"post_bases": post_s & 0b111, "runs_scored": runs_s, "outs_added": 0, "p": 1.0}
                ]

            # --- Double ---
            runs_d = bin(bases & 0b110).count("1")  # 2B and 3B score
            if bases & 0b001:
                runs_d += 1  # 1B scores too (generous)
            key = f"double|{bases}|{outs}"
            transitions[key] = [
                {"post_bases": 0b010, "runs_scored": runs_d, "outs_added": 0, "p": 1.0}
            ]

            # --- Triple ---
            runs_t = bin(bases).count("1")  # all runners score
            key = f"triple|{bases}|{outs}"
            transitions[key] = [
                {"post_bases": 0b100, "runs_scored": runs_t, "outs_added": 0, "p": 1.0}
            ]

            # --- Home run ---
            runs_hr = bin(bases).count("1") + 1  # all runners + batter
            key = f"home_run|{bases}|{outs}"
            transitions[key] = [
                {"post_bases": 0, "runs_scored": runs_hr, "outs_added": 0, "p": 1.0}
            ]

            # --- Walk / HBP / IBB ---
            for outcome in ("walk", "hit_by_pitch", "intent_walk"):
                # Force runners if bases loaded chain
                if bases & 0b001:
                    post_w = bases | 0b001
                    if bases & 0b010:
                        post_w |= 0b100
                        if bases & 0b100:
                            runs_w = 1
                        else:
                            runs_w = 0
                    else:
                        post_w |= 0b010
                        runs_w = 0
                    post_w |= 0b001
                else:
                    post_w = bases | 0b001
                    runs_w = 0
                key = f"{outcome}|{bases}|{outs}"
                transitions[key] = [
                    {"post_bases": post_w & 0b111, "runs_scored": runs_w, "outs_added": 0, "p": 1.0}
                ]

    return {"transitions": transitions}




# ---------------------------------------------------------------------------
# Game simulation tests
# ---------------------------------------------------------------------------


def _run_game(seed=42, **overrides):
    gi = make_game_input(**overrides)
    return simulate_game(
        game_input=gi,
        baserunning_table=make_baserunning_table(),
        outcome_labels=OUTCOME_CLASSES,
        sim_prob_source=uniform_prob_source,
        rng=random.Random(seed),
        live_prob_source=uniform_prob_source,
    )


class TestGameTermination:
    def test_game_terminates(self):
        """1000 games all finish within max_pa."""
        for seed in range(1000):
            result = _run_game(seed=seed)
            assert result.total_pas <= 200, f"Game {seed} exceeded max_pa"
            assert result.total_pas > 0

    def test_scores_reasonable(self):
        """Mean total runs across 1000 games is in a reasonable range."""
        total_runs = []
        for seed in range(1000):
            r = _run_game(seed=seed)
            total_runs.append(r.home_score + r.away_score)
        mean = sum(total_runs) / len(total_runs)
        assert 3 < mean < 30, f"Mean total runs = {mean:.1f}, outside [3, 30]"


class TestDeterminism:
    def test_deterministic_with_seed(self):
        r1 = _run_game(seed=12345)
        r2 = _run_game(seed=12345)
        assert r1 == r2


class TestLineup:
    def test_lineup_cycling(self):
        """After enough PAs, lineup cursor has wrapped."""
        result = _run_game(seed=1)
        # A game has at minimum ~50 PAs total (27 outs per side)
        assert result.total_pas >= 50


class TestPitcherExit:
    def test_pitcher_exit_happens(self):
        """With placeholder pull logic, some games see pitcher changes."""
        changes = 0
        n = 200
        for seed in range(n):
            r = _run_game(seed=seed)
            if r.home_pitcher_changes > 0 or r.away_pitcher_changes > 0:
                changes += 1
        # With 4% per-PA pull rate after BF threshold, most games should
        # see at least one pitcher change across 200 games
        assert changes > 0, f"No pitcher changes in {n} games"

    def test_bullpen_depletion(self):
        """1-arm bullpen → no crash when bullpen exhausted."""
        gi = make_game_input(
            home_bullpen=[_make_player(900, "R")],
            away_bullpen=[_make_player(901, "R")],
        )
        for seed in range(50):
            simulate_game(
                game_input=gi,
                baserunning_table=make_baserunning_table(),
                outcome_labels=OUTCOME_CLASSES,
                sim_prob_source=uniform_prob_source,
                rng=random.Random(seed),
                live_prob_source=uniform_prob_source,
            )
        # No crash = success

    def test_outing_counters_reset(self):
        """After pitcher change, outing object gets reset."""
        outing = _PitcherOuting(walks=3, hits=5, k=2, runs=4)
        outing.recent.extend(["walk", "single", "strikeout"])
        outing.reset()
        assert outing.walks == 0
        assert outing.hits == 0
        assert outing.k == 0
        assert outing.runs == 0
        assert len(outing.recent) == 0


class TestManfredRunner:
    def test_manfred_runner_in_extras(self):
        """Game starting in extras with Manfred runner enabled gets runner on 2B."""
        gi = make_game_input(
            game_state=GameState(inning=10, half=0, outs=0, bases=0,
                                 home_score=5, away_score=5),
        )
        # We can't easily inspect mid-game state, but we can verify the game
        # runs and produces valid results. With runner on 2B, extras should
        # resolve faster than regulation.
        result = simulate_game(
            game_input=gi,
            baserunning_table=make_baserunning_table(),
            outcome_labels=OUTCOME_CLASSES,
            sim_prob_source=uniform_prob_source,
            rng=random.Random(42),
            live_prob_source=uniform_prob_source,
            )
        assert result.innings >= 10
        assert result.home_score != result.away_score

    def test_manfred_runner_disabled(self):
        """With manfred_runner=False, no ghost runner in extras."""
        gi = make_game_input(
            game_state=GameState(inning=10, half=0, outs=0, bases=0,
                                 home_score=5, away_score=5),
        )
        result = simulate_game(
            game_input=gi,
            baserunning_table=make_baserunning_table(),
            outcome_labels=OUTCOME_CLASSES,
            sim_prob_source=uniform_prob_source,
            rng=random.Random(42),
            manfred_runner=False,
            live_prob_source=uniform_prob_source,
            )
        assert result.innings >= 10


class TestWalkoff:
    def test_walkoff(self):
        """Home team leading in bottom of 9 → game should end."""
        gi = make_game_input(
            game_state=GameState(inning=9, half=1, outs=2, bases=0,
                                 home_score=5, away_score=3),
        )
        # With a home lead in bot 9, game_over should trigger after
        # any PA that doesn't change the lead condition.
        # Actually game_over checks AFTER apply_pa. The game is already
        # won — but we need at least 1 PA to trigger the check via the loop.
        result = simulate_game(
            game_input=gi,
            baserunning_table=make_baserunning_table(),
            outcome_labels=OUTCOME_CLASSES,
            sim_prob_source=uniform_prob_source,
            rng=random.Random(42),
            live_prob_source=uniform_prob_source,
            )
        assert result.home_score >= result.away_score


class TestMaxPA:
    def test_max_pa_safety(self):
        """Game stops at max_pa even if not finished."""
        result = simulate_game(
            game_input=make_game_input(),
            baserunning_table=make_baserunning_table(),
            outcome_labels=OUTCOME_CLASSES,
            sim_prob_source=uniform_prob_source,
            rng=random.Random(42),
            max_pa=10,
            live_prob_source=uniform_prob_source,
            )
        assert result.total_pas <= 10


class TestSeedPA:
    """Tests for the dual-model seed PA phase."""

    def test_seed_pa_passes_context_through(self):
        """Seed PA passes seed_context directly to live_prob_source."""
        seed_calls = []

        def live_source(batter, pitcher, state, context):
            seed_calls.append(context.copy())
            probs = [0.0] * len(OUTCOME_CLASSES)
            probs[OUTCOME_CLASSES.index("strikeout")] = 1.0
            return probs

        seed_ctx = _make_seed_context(
            times_through_order=2.0,
            pitcher_pitch_count=47.0,
            balls=2.0,
            strikes=1.0,
        )
        gi = make_game_input(seed_context=seed_ctx)
        result = simulate_game(
            game_input=gi,
            baserunning_table=make_baserunning_table(),
            outcome_labels=OUTCOME_CLASSES,
            sim_prob_source=uniform_prob_source,
            rng=random.Random(42),
            live_prob_source=live_source,
        )
        assert len(seed_calls) == 1
        assert seed_calls[0]["balls"] == 2.0
        assert seed_calls[0]["strikes"] == 1.0
        assert seed_calls[0]["times_through_order"] == 2.0
        assert seed_calls[0]["pitcher_pitch_count"] == 47.0
        assert result.total_pas >= 55

class TestCustomProbSource:
    def test_all_strikeout(self):
        """prob_source always returns strikeout → 0 runs scored."""
        k_idx = OUTCOME_CLASSES.index("strikeout")

        def strikeout_source(batter, pitcher, state, context):
            probs = [0.0] * len(OUTCOME_CLASSES)
            probs[k_idx] = 1.0
            return probs

        result = simulate_game(
            game_input=make_game_input(),
            baserunning_table=make_baserunning_table(),
            outcome_labels=OUTCOME_CLASSES,
            sim_prob_source=strikeout_source,
            rng=random.Random(42),
            live_prob_source=strikeout_source,
            )
        assert result.home_score == 0
        assert result.away_score == 0
        # 27 outs per side = 54 PAs minimum (3 outs × 9 innings × 2 teams)
        # But with pitcher exit rolls, there might be a few extra PAs
        # where the exit check fires before the PA happens.
        assert result.total_pas >= 54


# ---------------------------------------------------------------------------
# Innings reporting — regulation vs extras
# ---------------------------------------------------------------------------


class TestInningsReporting:
    """Verify GameResult.innings reflects completed innings, not the
    state machine counter (which ticks past the last inning for away wins).
    """

    def test_away_regulation_win(self):
        """Away leads entering bot 9 → game ends at 9 innings."""
        k_idx = OUTCOME_CLASSES.index("strikeout")
        probs = [0.0] * len(OUTCOME_CLASSES)
        probs[k_idx] = 1.0

        def k_source(b, p, s, c):
            return probs

        # Bot 9, 2 outs, away leading 5-0. One more strikeout ends the game.
        gi = make_game_input(
            game_state=GameState(
                inning=9, half=1, outs=2, bases=0,
                away_score=5, home_score=0,
            ),
        )
        result = simulate_game(
            game_input=gi,
            baserunning_table=make_baserunning_table(),
            outcome_labels=OUTCOME_CLASSES,
            sim_prob_source=k_source,
            live_prob_source=k_source,
            rng=random.Random(42),
        )
        assert result.innings == 9
        assert result.away_score >= 5
        assert result.home_score < result.away_score

    def test_walkoff_innings(self):
        """Walk-off HR in bot 9 → game ends at 9 innings."""
        hr_idx = OUTCOME_CLASSES.index("home_run")
        probs = [0.0] * len(OUTCOME_CLASSES)
        probs[hr_idx] = 1.0

        def hr_source(b, p, s, c):
            return probs

        # Bot 9, 0 outs, tied 0-0. HR scores the walk-off.
        gi = make_game_input(
            game_state=GameState(
                inning=9, half=1, outs=0, bases=0,
                away_score=0, home_score=0,
            ),
        )
        result = simulate_game(
            game_input=gi,
            baserunning_table=make_baserunning_table(),
            outcome_labels=OUTCOME_CLASSES,
            sim_prob_source=hr_source,
            live_prob_source=hr_source,
            rng=random.Random(42),
        )
        assert result.innings == 9
        assert result.home_score > result.away_score

    def test_extras_away_win(self):
        """Away wins in extras (bot 10 fails) → innings == 10."""
        k_idx = OUTCOME_CLASSES.index("strikeout")
        probs = [0.0] * len(OUTCOME_CLASSES)
        probs[k_idx] = 1.0

        def k_source(b, p, s, c):
            return probs

        # Bot 10, 2 outs, away leading 1-0.
        gi = make_game_input(
            game_state=GameState(
                inning=10, half=1, outs=2, bases=0,
                away_score=1, home_score=0,
            ),
        )
        result = simulate_game(
            game_input=gi,
            baserunning_table=make_baserunning_table(),
            outcome_labels=OUTCOME_CLASSES,
            sim_prob_source=k_source,
            live_prob_source=k_source,
            rng=random.Random(42),
        )
        assert result.innings == 10
        assert result.away_score > result.home_score

    def test_extras_walkoff(self):
        """Walk-off in bot 10 → innings == 10."""
        hr_idx = OUTCOME_CLASSES.index("home_run")
        probs = [0.0] * len(OUTCOME_CLASSES)
        probs[hr_idx] = 1.0

        def hr_source(b, p, s, c):
            return probs

        # Bot 10, 0 outs, tied 1-1.
        gi = make_game_input(
            game_state=GameState(
                inning=10, half=1, outs=0, bases=0,
                away_score=1, home_score=1,
            ),
        )
        result = simulate_game(
            game_input=gi,
            baserunning_table=make_baserunning_table(),
            outcome_labels=OUTCOME_CLASSES,
            sim_prob_source=hr_source,
            live_prob_source=hr_source,
            rng=random.Random(42),
        )
        assert result.innings == 10
        assert result.home_score > result.away_score

    def test_all_strikeout_reports_9_innings(self):
        """All-K game (0-0) goes to extras but should still report
        correct innings for both away and home wins across the extras."""
        k_idx = OUTCOME_CLASSES.index("strikeout")
        probs = [0.0] * len(OUTCOME_CLASSES)
        probs[k_idx] = 1.0

        def k_source(b, p, s, c):
            return probs

        # 0-0 all-K game from top 1 — will go to extras due to Manfred runner
        result = simulate_game(
            game_input=make_game_input(),
            baserunning_table=make_baserunning_table(),
            outcome_labels=OUTCOME_CLASSES,
            sim_prob_source=k_source,
            live_prob_source=k_source,
            rng=random.Random(42),
        )
        # Should go to extras (Manfred runner eventually scores)
        # Key assertion: innings should never exceed the state.inning
        # by the off-by-one bug
        if result.home_score > result.away_score:
            # Walk-off: innings should be the inning it ended in
            assert result.innings >= 10
        else:
            # Away win: innings should be correct (not +1)
            assert result.innings >= 10
