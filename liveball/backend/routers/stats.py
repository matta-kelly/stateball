"""Player stats endpoints for data verification against public sources."""

from datetime import datetime, timezone

from fastapi import APIRouter, Query

from backend.db import get_conn

router = APIRouter(prefix="/stats", tags=["stats"])

DEFAULT_LIMIT = 50

BATTER_SQL = """
SELECT
    p.full_name, p.team_name, p.position, p.bats,
    b.batter_id, b.game_date,
    b.s_pa                                              AS season_pa,
    b.season_ba, b.season_obp, b.season_slg,
    b.season_ops, b.season_woba,
    b.season_k_pct, b.season_bb_pct,
    (b.s_ev_sum_vs_l + b.s_ev_sum_vs_r)
        / NULLIF(b.s_bip_vs_l + b.s_bip_vs_r, 0)      AS season_avg_ev,
    (b.s_barrel_vs_l + b.s_barrel_vs_r)::double
        / NULLIF(b.s_bip_vs_l + b.s_bip_vs_r, 0)      AS season_barrel_pct,
    b.c_pa                                              AS career_pa,
    b.career_ba, b.career_obp, b.career_slg,
    b.career_ops, b.career_woba,
    b.career_k_pct, b.career_bb_pct,
    (b.c_ev_sum_vs_l + b.c_ev_sum_vs_r)
        / NULLIF(b.c_bip_vs_l + b.c_bip_vs_r, 0)      AS career_avg_ev,
    (b.c_barrel_vs_l + b.c_barrel_vs_r)::double
        / NULLIF(b.c_bip_vs_l + b.c_bip_vs_r, 0)      AS career_barrel_pct
FROM lh.main.int_mlb__batters b
LEFT JOIN lh.main.ref_mlb__players p ON b.batter_id = p.player_id
WHERE b.season = ?
  AND b.game_date = (SELECT MAX(game_date) FROM lh.main.int_mlb__batters WHERE season = ?)
  AND b.s_pa >= 1
ORDER BY b.s_pa DESC
LIMIT ?
OFFSET ?
"""

PITCHER_SQL = """
SELECT
    p.full_name, p.team_name, p.position, p.throws,
    pi.pitcher_id, pi.game_date,
    pi.s_bf                     AS season_bf,
    pi.season_ip, pi.season_whip,
    pi.season_k_pct, pi.season_bb_pct,
    pi.season_hr9, pi.season_woba,
    pi.c_bf                     AS career_bf,
    pi.career_ip, pi.career_whip,
    pi.career_k_pct, pi.career_bb_pct,
    pi.career_hr9, pi.career_woba
FROM lh.main.int_mlb__pitchers pi
LEFT JOIN lh.main.ref_mlb__players p ON pi.pitcher_id = p.player_id
WHERE pi.season = ?
  AND pi.game_date = (SELECT MAX(game_date) FROM lh.main.int_mlb__pitchers WHERE season = ?)
  AND pi.s_bf >= 1
ORDER BY pi.s_bf DESC
LIMIT ?
OFFSET ?
"""


def _query(sql: str, params: list) -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(sql, params).fetchall()
        cols = [d[0] for d in conn.description]
        return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()


@router.get("/batters")
def list_batters(
    season: int = Query(default_factory=lambda: datetime.now(timezone.utc).year),
    offset: int = Query(default=0),
    limit: int = Query(default=DEFAULT_LIMIT),
):
    return _query(BATTER_SQL, [season, season, limit, offset])


@router.get("/pitchers")
def list_pitchers(
    season: int = Query(default_factory=lambda: datetime.now(timezone.utc).year),
    offset: int = Query(default=0),
    limit: int = Query(default=DEFAULT_LIMIT),
):
    return _query(PITCHER_SQL, [season, season, limit, offset])


@router.get("/seasons")
def list_seasons():
    return [
        row[0]
        for row in get_conn()
        .execute(
            "SELECT DISTINCT season FROM lh.main.int_mlb__batters ORDER BY season DESC"
        )
        .fetchall()
    ]
