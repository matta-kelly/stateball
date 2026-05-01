"""SimContext — per-game state for live simulation.

Bridges the MLB live feed (gumbo) + cached warehouse profiles into
GameInputs that the Simulator can consume. Hydrated once per game
(profiles from warehouse), then to_game_input() is called each time
we want to re-sim from the current live state.

The engine simulates forward from the seed state on its own — it
tracks outing counters, lineup advancement, pitcher exits internally.
SimContext's job is narrow: assemble a correct starting point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sim.engine.core.state import GameState
from sim.game_inputs.in_play import apply as apply_in_play
from sim.game_inputs.profiles import (
    build_player,
    detect_schema,
    fetch_batter_profiles,
    fetch_handedness,
    fetch_pitcher_profiles,
)
from sim.game_inputs.game import GameInput

logger = logging.getLogger(__name__)


@dataclass
class SimContext:
    """Per-game context for live simulation.

    Holds cached warehouse data (profiles, handedness) that doesn't
    change mid-game. Game state and lineups are read fresh from gumbo
    each time to_game_input() is called.
    """

    game_pk: int
    game_date: str
    player_profiles: dict[int, dict]  # {pid: profile_dict} — batters + pitchers
    handedness: dict[int, dict]       # {pid: {bats, throws}}

    @classmethod
    def hydrate(
        cls,
        game_pk: int,
        game_date: str,
        gumbo: dict,
        conn,
    ) -> SimContext:
        """One-time setup: extract player IDs from gumbo, fetch profiles.

        Args:
            game_pk: MLB game primary key.
            game_date: Game date string (YYYY-MM-DD).
            gumbo: Full gumbo response dict from MLB API.
            conn: DuckDB connection (local or DuckLake).

        Returns:
            A hydrated SimContext ready for to_game_input() calls.
        """
        schema = detect_schema(conn)

        # Collect all player IDs from both teams' boxscore
        boxscore = gumbo["liveData"]["boxscore"]
        all_ids: set[int] = set()
        batter_ids: set[int] = set()
        pitcher_ids: set[int] = set()

        for side in ("home", "away"):
            team = boxscore["teams"][side]
            lineup = team.get("battingOrder", [])
            bullpen = team.get("bullpen", [])
            pitchers = team.get("pitchers", [])

            batter_ids.update(lineup)
            pitcher_ids.update(bullpen)
            pitcher_ids.update(pitchers)
            all_ids.update(lineup)
            all_ids.update(bullpen)
            all_ids.update(pitchers)

        handedness = fetch_handedness(conn, schema, list(all_ids))
        batter_profiles = fetch_batter_profiles(
            conn, schema, list(batter_ids), game_date
        )
        pitcher_profiles = fetch_pitcher_profiles(
            conn, schema, list(pitcher_ids), game_date
        )

        # Merge into one dict
        profiles = {**batter_profiles, **pitcher_profiles}

        logger.info(
            "[%d] SimContext hydrated: %d batters, %d pitchers, %d total profiles",
            game_pk, len(batter_profiles), len(pitcher_profiles), len(profiles),
        )

        return cls(
            game_pk=game_pk,
            game_date=game_date,
            player_profiles=profiles,
            handedness=handedness,
        )

    def to_game_input(self, gumbo: dict) -> GameInput:
        """Build a GameInput from cached profiles + current gumbo state.

        Args:
            gumbo: Full gumbo response dict from MLB API.

        Returns:
            GameInput ready for Simulator.simulate().
        """
        boxscore = gumbo["liveData"]["boxscore"]
        linescore = gumbo["liveData"]["linescore"]
        current_play = gumbo["liveData"]["plays"].get("currentPlay", {})
        matchup = current_play.get("matchup", {})

        # --- Lineups ---
        home_lineup_ids = boxscore["teams"]["home"].get("battingOrder", [])
        away_lineup_ids = boxscore["teams"]["away"].get("battingOrder", [])

        # --- Current pitchers (from matchup + linescore.defense) ---
        # Determine who's batting: Top = away bats, Bot = home bats
        inning_half = linescore.get("inningHalf", "Top")
        if inning_half == "Top":
            # Away batting → home is on defense → home pitcher from defense
            home_pitcher_id = matchup.get("pitcher", {}).get("id")
            away_pitcher_id = _defensive_pitcher_for(
                boxscore, "away", linescore
            )
        else:
            # Home batting → away is on defense → away pitcher from defense
            away_pitcher_id = matchup.get("pitcher", {}).get("id")
            home_pitcher_id = _defensive_pitcher_for(
                boxscore, "home", linescore
            )

        # --- Bullpen ---
        home_bullpen_ids = boxscore["teams"]["home"].get("bullpen", [])
        away_bullpen_ids = boxscore["teams"]["away"].get("bullpen", [])

        # --- Build player dicts ---
        home_lineup = [
            build_player(pid, self.player_profiles, self.handedness, "bats")
            for pid in home_lineup_ids
        ]
        away_lineup = [
            build_player(pid, self.player_profiles, self.handedness, "bats")
            for pid in away_lineup_ids
        ]
        home_pitcher = build_player(
            home_pitcher_id, self.player_profiles, self.handedness, "throws"
        )
        away_pitcher = build_player(
            away_pitcher_id, self.player_profiles, self.handedness, "throws"
        )
        home_bullpen = [
            build_player(pid, self.player_profiles, self.handedness, "throws")
            for pid in home_bullpen_ids
        ]
        away_bullpen = [
            build_player(pid, self.player_profiles, self.handedness, "throws")
            for pid in away_bullpen_ids
        ]

        # --- Game state ---
        game_state = _build_live_game_state(linescore)
        game_state = apply_in_play(game_state, current_play)

        # --- Seed context ---
        # The defensive pitcher (facing the current batter) provides outing stats
        defensive_pitcher_id = matchup.get("pitcher", {}).get("id")
        seed_context = _build_live_seed_context(
            gumbo, defensive_pitcher_id, linescore
        )

        return GameInput(
            home_lineup=home_lineup,
            away_lineup=away_lineup,
            home_pitcher=home_pitcher,
            away_pitcher=away_pitcher,
            home_bullpen=home_bullpen,
            away_bullpen=away_bullpen,
            game_state=game_state,
            game_pk=self.game_pk,
            game_date=self.game_date,
            seed_context=seed_context,
        )


# ---------------------------------------------------------------------------
# Gumbo parsing helpers
# ---------------------------------------------------------------------------


def _defensive_pitcher_for(
    boxscore: dict, side: str, linescore: dict
) -> int | None:
    """Get the last pitcher used by a team (when they're NOT currently on defense).

    When team is batting, we can't get their pitcher from the matchup.
    Fall back to the last entry in their pitchers array.
    """
    pitchers = boxscore["teams"][side].get("pitchers", [])
    if pitchers:
        return pitchers[-1]
    return None


def _build_live_game_state(linescore: dict) -> GameState:
    """Build GameState from gumbo linescore.

    Maps gumbo field names to GameState fields:
      inningHalf "Top"/"Bottom" → half 0/1
      offense.{first,second,third} → bases bitmask
      teams.{home,away}.runs → scores
    """
    inning = linescore.get("currentInning", 1)
    inning_half = linescore.get("inningHalf", "Top")
    is_bottom = 1 if inning_half == "Bottom" else 0
    outs = linescore.get("outs", 0) or 0

    # Runners: offense section has runner objects when occupied
    offense = linescore.get("offense", {})
    bases = 0
    if offense.get("first"):
        bases |= 0b001
    if offense.get("second"):
        bases |= 0b010
    if offense.get("third"):
        bases |= 0b100

    home_score = linescore.get("teams", {}).get("home", {}).get("runs", 0) or 0
    away_score = linescore.get("teams", {}).get("away", {}).get("runs", 0) or 0

    return GameState(
        inning=inning,
        half=is_bottom,
        outs=outs,
        bases=bases,
        home_score=home_score,
        away_score=away_score,
    )


def _build_live_seed_context(
    gumbo: dict, pitcher_id: int | None, linescore: dict
) -> dict:
    """Build seed_context from gumbo pitcher stats + linescore count.

    Provides all 13 keys expected by the engine's seed PA.
    Fields not available from gumbo get reasonable defaults.
    """
    # Get pitcher's game stats from boxscore
    bf = 0
    pitch_count = 0
    walks = 0
    hits = 0
    k = 0
    runs = 0

    if pitcher_id is not None:
        pitching_stats = _get_pitcher_stats(gumbo, pitcher_id)
        if pitching_stats:
            bf = pitching_stats.get("battersFaced", 0) or 0
            pitch_count = pitching_stats.get("numberOfPitches", 0) or 0
            walks = pitching_stats.get("baseOnBalls", 0) or 0
            hits = pitching_stats.get("hits", 0) or 0
            k = pitching_stats.get("strikeOuts", 0) or 0
            runs = pitching_stats.get("runs", 0) or 0

    outing_whip = (walks + hits) / bf if bf > 0 else 0.0

    return {
        "times_through_order": float(min(bf // 9 + 1, 4)),
        "batter_prior_pa": 0.0,
        "pitcher_pitch_count": float(pitch_count),
        "pitcher_bf_game": float(bf),
        "batter_ab_vs_pitcher": 0.0,
        "pitcher_outing_walks": float(walks),
        "pitcher_outing_hits": float(hits),
        "pitcher_outing_k": float(k),
        "pitcher_outing_runs": float(runs),
        "pitcher_outing_whip": outing_whip,
        "pitcher_recent_whip": outing_whip,  # best available approximation
        "balls": float(linescore.get("balls", 0) or 0),
        "strikes": float(linescore.get("strikes", 0) or 0),
    }


def _get_pitcher_stats(gumbo: dict, pitcher_id: int) -> dict:
    """Look up a pitcher's game stats from the boxscore players section.

    Searches both teams since we may not know which side the pitcher is on.
    """
    boxscore = gumbo["liveData"]["boxscore"]
    player_key = f"ID{pitcher_id}"

    for side in ("home", "away"):
        players = boxscore["teams"][side].get("players", {})
        player = players.get(player_key)
        if player:
            return player.get("stats", {}).get("pitching", {})

    return {}
