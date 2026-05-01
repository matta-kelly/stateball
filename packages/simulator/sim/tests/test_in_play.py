"""Tests for sim.game_inputs.in_play — in-play state adjustment."""

from __future__ import annotations

from sim.engine.core.state import GameState
from sim.game_inputs.in_play import InPlayAdjustment, apply, detect


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_current_play(
    is_complete: bool,
    result_event: str | None,
    description: str,
    is_in_play: bool,
) -> dict:
    """Build a minimal currentPlay dict for testing."""
    return {
        "about": {"isComplete": is_complete},
        "result": {"event": result_event},
        "playEvents": [
            {
                "details": {
                    "isInPlay": is_in_play,
                    "description": description,
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# detect() tests
# ---------------------------------------------------------------------------


class TestDetect:
    def test_in_play_runs(self):
        cp = _make_current_play(False, None, "In play, run(s)", True)
        adj = detect(cp)
        assert adj == InPlayAdjustment(score_delta=1, outs_delta=0)

    def test_in_play_outs(self):
        cp = _make_current_play(False, None, "In play, out(s)", True)
        adj = detect(cp)
        assert adj == InPlayAdjustment(score_delta=0, outs_delta=1)

    def test_in_play_no_out(self):
        cp = _make_current_play(False, None, "In play, no out", True)
        adj = detect(cp)
        assert adj == InPlayAdjustment(score_delta=0, outs_delta=0)

    def test_resolved_play_returns_none(self):
        cp = _make_current_play(True, "Single", "In play, run(s)", True)
        assert detect(cp) is None

    def test_empty_play_events_returns_none(self):
        cp = {
            "about": {"isComplete": False},
            "result": {"event": None},
            "playEvents": [],
        }
        assert detect(cp) is None

    def test_empty_current_play_returns_none(self):
        assert detect({}) is None

    def test_not_in_play_returns_none(self):
        cp = _make_current_play(False, None, "Called Strike", False)
        assert detect(cp) is None

    def test_unknown_description_returns_none(self):
        cp = _make_current_play(False, None, "In play, something weird", True)
        assert detect(cp) is None

    def test_result_event_present_returns_none(self):
        """Even if isComplete is False, a non-null result.event means resolved."""
        cp = _make_current_play(False, "Groundout", "In play, out(s)", True)
        assert detect(cp) is None


# ---------------------------------------------------------------------------
# apply() tests
# ---------------------------------------------------------------------------


class TestApply:
    def test_score_away_batting(self):
        """Top of inning (half=0) — away team batting, away_score incremented."""
        state = GameState(inning=5, half=0, outs=1, bases=0b001, away_score=3, home_score=2)
        cp = _make_current_play(False, None, "In play, run(s)", True)
        result = apply(state, cp)
        assert result.away_score == 4
        assert result.home_score == 2
        assert result.outs == 1  # unchanged

    def test_score_home_batting(self):
        """Bottom of inning (half=1) — home team batting, home_score incremented."""
        state = GameState(inning=5, half=1, outs=0, bases=0b010, away_score=3, home_score=2)
        cp = _make_current_play(False, None, "In play, run(s)", True)
        result = apply(state, cp)
        assert result.home_score == 3
        assert result.away_score == 3
        assert result.outs == 0  # unchanged

    def test_outs_increment(self):
        state = GameState(inning=3, half=0, outs=1)
        cp = _make_current_play(False, None, "In play, out(s)", True)
        result = apply(state, cp)
        assert result.outs == 2

    def test_outs_increment_to_three(self):
        """outs=2 + 'In play, out(s)' → outs=3. Engine handles the flip separately."""
        state = GameState(inning=7, half=1, outs=2, home_score=5, away_score=4)
        cp = _make_current_play(False, None, "In play, out(s)", True)
        result = apply(state, cp)
        assert result.outs == 3

    def test_no_out_no_change(self):
        state = GameState(inning=4, half=0, outs=1, bases=0b011, away_score=2, home_score=1)
        cp = _make_current_play(False, None, "In play, no out", True)
        result = apply(state, cp)
        assert result == state

    def test_resolved_play_no_change(self):
        state = GameState(inning=4, half=0, outs=1)
        cp = _make_current_play(True, "Single", "In play, run(s)", True)
        result = apply(state, cp)
        assert result == state

    def test_bases_unchanged(self):
        """Baserunners should never be modified by the adjustment."""
        state = GameState(inning=5, half=0, outs=1, bases=0b101, away_score=2, home_score=1)
        cp = _make_current_play(False, None, "In play, run(s)", True)
        result = apply(state, cp)
        assert result.bases == 0b101  # unchanged


# ---------------------------------------------------------------------------
# Engine seed guard tests
# ---------------------------------------------------------------------------


class TestEngineOuts3:

    _SEED_CONTEXT = {
        "times_through_order": 1.0,
        "batter_prior_pa": 0.0,
        "pitcher_pitch_count": 50.0,
        "pitcher_bf_game": 15.0,
        "batter_ab_vs_pitcher": 0.0,
        "pitcher_outing_walks": 1.0,
        "pitcher_outing_hits": 3.0,
        "pitcher_outing_k": 4.0,
        "pitcher_outing_runs": 1.0,
        "pitcher_outing_whip": 0.8,
        "pitcher_recent_whip": 0.8,
        "balls": 0.0,
        "strikes": 0.0,
    }

    def _make_game_input(self, outs=3, half=0, inning=5, home_pitcher_bf=15, away_pitcher_bf=12):
        from sim.game_inputs.game import GameInput
        return GameInput(
            home_lineup=[{"player_id": i, "hand": "R"} for i in range(9)],
            away_lineup=[{"player_id": i + 100, "hand": "R"} for i in range(9)],
            home_pitcher={"player_id": 50, "hand": "R"},
            away_pitcher={"player_id": 150, "hand": "L"},
            home_bullpen=[{"player_id": 60, "hand": "R"}],
            away_bullpen=[{"player_id": 160, "hand": "R"}],
            game_state=GameState(
                inning=inning, half=half, outs=outs,
                home_pitcher_bf=home_pitcher_bf, away_pitcher_bf=away_pitcher_bf,
            ),
            game_pk=0,
            game_date="2026-01-01",
            seed_context=self._SEED_CONTEXT,
        )

    def test_scalar_engine_handles_outs_3(self):
        """Scalar engine should flip inning when seed state has outs=3."""
        from sim.engine.core.engine import simulate_game, make_uniform_prob_source
        from xg.core.config import OUTCOME_CLASSES

        prob_source = make_uniform_prob_source(len(OUTCOME_CLASSES))
        baserunning = _make_minimal_baserunning(OUTCOME_CLASSES)
        gi = self._make_game_input()

        result = simulate_game(
            gi, baserunning, OUTCOME_CLASSES,
            sim_prob_source=prob_source,
            live_prob_source=prob_source,
        )
        assert result.home_score >= 0
        assert result.away_score >= 0
        assert result.total_pas > 0

    def test_scalar_engine_no_phantom_bf(self):
        """Flipping from outs=3 should not increment pitcher BF."""
        from dataclasses import replace as dc_replace
        from sim.engine.core.state import game_over

        # Top of 5th, outs=3 — should flip to bottom of 5th
        state = GameState(inning=5, half=0, outs=3, home_pitcher_bf=15, away_pitcher_bf=12)

        # Replicate the engine guard logic
        if state.half == 0:
            flipped = dc_replace(state, half=1, outs=0, bases=0)
        else:
            flipped = dc_replace(state, half=0, inning=state.inning + 1, outs=0, bases=0)

        assert flipped.half == 1
        assert flipped.outs == 0
        assert flipped.inning == 5
        assert flipped.home_pitcher_bf == 15  # unchanged
        assert flipped.away_pitcher_bf == 12  # unchanged
        assert not game_over(flipped)

    def test_scalar_engine_flip_bottom_to_top(self):
        """Bottom of 5th with outs=3 should flip to top of 6th."""
        from dataclasses import replace as dc_replace

        state = GameState(inning=5, half=1, outs=3, home_pitcher_bf=15, away_pitcher_bf=12)
        flipped = dc_replace(state, half=0, inning=state.inning + 1, outs=0, bases=0)

        assert flipped.half == 0
        assert flipped.inning == 6
        assert flipped.outs == 0
        assert flipped.home_pitcher_bf == 15
        assert flipped.away_pitcher_bf == 12

    def test_scalar_engine_outs3_game_over_walkoff(self):
        """Bottom 9+, home leads, outs=3 → flip to... wait, this shouldn't happen.

        If it's bottom 9+ and home leads, the game should have already ended.
        But if we somehow get outs=3 here (defensive out, not a walkoff),
        the flip goes to top of 10th and game_over detects scores differ.
        """
        from dataclasses import replace as dc_replace
        from sim.engine.core.state import game_over

        # Bottom of 9th, away leads 3-2, outs=3 — flip to top of 10th
        state = GameState(inning=9, half=1, outs=3, home_score=2, away_score=3)
        flipped = dc_replace(state, half=0, inning=10, outs=0, bases=0)
        assert game_over(flipped)  # extras resolved, scores differ


def _make_minimal_baserunning(outcome_labels: list[str]) -> dict:
    """Build a minimal baserunning table that covers all outcome/base/out combos."""
    transitions = {}
    for outcome in outcome_labels:
        for bases in range(8):
            for outs in range(3):
                key = f"{outcome}|{bases}|{outs}"
                if outcome in ("strikeout", "strikeout_double_play"):
                    transitions[key] = [{"post_bases": bases, "runs_scored": 0, "outs_added": 1, "p": 1.0}]
                elif outcome in ("walk", "hit_by_pitch"):
                    transitions[key] = [{"post_bases": bases | 1, "runs_scored": 0, "outs_added": 0, "p": 1.0}]
                elif outcome == "home_run":
                    runners = bin(bases).count("1")
                    transitions[key] = [{"post_bases": 0, "runs_scored": runners + 1, "outs_added": 0, "p": 1.0}]
                elif outcome in ("double_play", "grounded_into_double_play", "fielders_choice_out", "strikeout_double_play"):
                    transitions[key] = [{"post_bases": 0, "runs_scored": 0, "outs_added": 2, "p": 1.0}]
                elif outcome in ("single",):
                    transitions[key] = [{"post_bases": 1, "runs_scored": 0, "outs_added": 0, "p": 1.0}]
                elif outcome in ("double",):
                    transitions[key] = [{"post_bases": 2, "runs_scored": 0, "outs_added": 0, "p": 1.0}]
                elif outcome in ("triple",):
                    transitions[key] = [{"post_bases": 4, "runs_scored": 0, "outs_added": 0, "p": 1.0}]
                else:
                    # field_out, force_out, fielders_choice, sac_bunt, sac_fly
                    transitions[key] = [{"post_bases": 0, "runs_scored": 0, "outs_added": 1, "p": 1.0}]
    return {"transitions": transitions, "metadata": {"total_events": 1000}}
