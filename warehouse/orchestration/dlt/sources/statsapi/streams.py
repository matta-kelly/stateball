"""MLB Stats API dlt resources."""
from datetime import date

import dlt

from .client import MLBClient


def _season_ranges(start_date: str, end_date: str):
    """Break a date range into per-season chunks (Mar 1 - Nov 30)."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    year = start.year
    while year <= end.year:
        season_start = max(start, date(year, 3, 1))
        season_end = min(end, date(year, 11, 30))
        if season_start <= season_end:
            yield season_start.isoformat(), season_end.isoformat()
        year += 1


@dlt.resource(write_disposition="append")
def games(start_date: str, end_date: str, log=None):
    """Fetch MLB game schedules for date range.

    Chunks requests by season to avoid MLB API truncation.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        log: Optional logger

    Yields:
        Raw game dicts from MLB API (dlt handles flattening via max_table_nesting=0)
    """
    client = MLBClient()

    if log:
        log.info(f"[games] Fetching {start_date} to {end_date}")

    total = 0
    for chunk_start, chunk_end in _season_ranges(start_date, end_date):
        if log:
            log.info(f"[games] Season chunk: {chunk_start} to {chunk_end}")

        response = client.get(
            "/api/v1/schedule",
            params={"sportId": 1, "startDate": chunk_start, "endDate": chunk_end},
            log=log,
        )

        for date_entry in response.get("dates", []):
            batch = date_entry.get("games", [])
            if batch:
                total += len(batch)
                yield batch

    if log:
        log.info(f"[games] Done. Total: {total}")


@dlt.resource(write_disposition="append")
def players(player_ids: list[int], log=None):
    """Fetch player details from MLB Stats API.

    Hits the bulk endpoint in batches of 50 IDs.
    Flattens nested fields (position, handedness, team) before yielding.

    Args:
        player_ids: List of MLB player IDs to fetch
        log: Optional logger
    """
    client = MLBClient()
    batch_size = 50

    if log:
        log.info(f"[players] Fetching {len(player_ids)} players")

    total = 0
    for i in range(0, len(player_ids), batch_size):
        batch_ids = player_ids[i : i + batch_size]
        ids_param = ",".join(str(pid) for pid in batch_ids)

        response = client.get(
            "/api/v1/people",
            params={"personIds": ids_param, "hydrate": "currentTeam"},
            log=log,
        )

        people = response.get("people", [])
        total += len(people)

        if log:
            log.info(f"[players] Batch {i // batch_size + 1}: {len(people)} players")

        records = []
        for person in people:
            current_team = person.get("currentTeam") or {}
            primary_position = person.get("primaryPosition") or {}
            bat_side = person.get("batSide") or {}
            pitch_hand = person.get("pitchHand") or {}

            records.append({
                "player_id": person["id"],
                "full_name": person.get("fullName"),
                "first_name": person.get("firstName"),
                "last_name": person.get("lastName"),
                "position": primary_position.get("abbreviation"),
                "bats": bat_side.get("code"),
                "throws": pitch_hand.get("code"),
                "team_id": current_team.get("id"),
                "team_name": current_team.get("name"),
                "active": person.get("active"),
                "birth_date": person.get("birthDate"),
                "mlb_debut_date": person.get("mlbDebutDate"),
            })

        if records:
            yield records

    if log:
        log.info(f"[players] Done. Total: {total}")


@dlt.resource(write_disposition="append")
def boxscores(game_pks: list[int], log=None):
    """Fetch boxscore data from MLB Stats API.

    One API call per game_pk. Yields one record per player per game
    (roster grain). Per-game error isolation — failed games are skipped.

    Args:
        game_pks: List of MLB game PKs to fetch
        log: Optional logger
    """
    client = MLBClient()

    if log:
        log.info(f"[boxscores] Fetching {len(game_pks)} games")

    total = 0
    errors = 0
    for i, game_pk in enumerate(game_pks):
        try:
            response = client.get(f"/api/v1/game/{game_pk}/boxscore", log=log)

            records = []
            for side in ("away", "home"):
                team = response.get("teams", {}).get(side, {})
                team_id = team.get("team", {}).get("id")
                batting_order = team.get("battingOrder", [])
                pitchers = team.get("pitchers", [])
                sp_id = pitchers[0] if pitchers else None

                for player_key, player_data in team.get("players", {}).items():
                    pid = player_data["person"]["id"]
                    pos = player_data.get("position", {}).get("abbreviation")
                    bat_pos = None
                    if pid in batting_order:
                        bat_pos = batting_order.index(pid) + 1

                    records.append({
                        "game_pk": game_pk,
                        "player_id": pid,
                        "team_id": team_id,
                        "side": side,
                        "position": pos,
                        "batting_order": bat_pos,
                        "is_starting_pitcher": pid == sp_id,
                    })

            if records:
                total += len(records)
                yield records

            if log and (i + 1) % 100 == 0:
                log.info(f"[boxscores] Progress: {i + 1}/{len(game_pks)} games")

        except Exception as e:
            errors += 1
            if log:
                log.warning(f"[boxscores] Game {game_pk} failed, skipping: {e}")

    if log:
        log.info(f"[boxscores] Done. {total} records, {errors} errors")
