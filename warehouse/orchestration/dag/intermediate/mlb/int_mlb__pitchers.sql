-- int_mlb__pitchers.sql
-- Incremental pitcher rates: cumulative season + career counting stats
-- and derived rate stats, split by batter handedness.
-- Self-join pattern: prev cumulatives + bounded window within chunk.
--
-- Grain: (pitcher_id, game_date)

{{ config(
    materialized='incremental',
    unique_key=['pitcher_id', 'game_date'],
    incremental_strategy='delete+insert',
    on_schema_change='fail',
) }}

{%- set chunk_start = var('chunk_start', none) -%}
{%- set chunk_end = var('chunk_end', none) -%}

{%- set count_cols = [
    'bf', 'outs', 'hits', 'singles', 'doubles', 'triples',
    'hr', 'bb', 'ibb', 'k', 'hbp', 'sf', 'sh',
    'woba_value', 'woba_denom'
] -%}
{%- set hands = ['l', 'r'] -%}

with new_counts as (
    select * from {{ ref('int_mlb__pitcher_counts') }}
    {% if is_incremental() %}
        {% if chunk_start is not none %}
    where game_date between '{{ chunk_start }}' and '{{ chunk_end }}'
        {% else %}
    where game_date > (select max(game_date) from {{ this }})
        {% endif %}
    {% elif chunk_start is not none %}
    where game_date between '{{ chunk_start }}' and '{{ chunk_end }}'
    {% endif %}
),

{% if is_incremental() %}
prev_season as (
    select t.*
    from {{ this }} t
    inner join (
        select pitcher_id, season, max(game_date) as max_gd
        from {{ this }}
        where game_date < (select min(game_date) from new_counts)
        group by pitcher_id, season
    ) p on t.pitcher_id = p.pitcher_id and t.season = p.season and t.game_date = p.max_gd
),

prev_career as (
    select t.*
    from {{ this }} t
    inner join (
        select pitcher_id, max(game_date) as max_gd
        from {{ this }}
        where game_date < (select min(game_date) from new_counts)
        group by pitcher_id
    ) p on t.pitcher_id = p.pitcher_id and t.game_date = p.max_gd
),
{% endif %}

cumulative as (
    select
        n.pitcher_id,
        n.game_date,
        n.season,

        -- === SEASON CUMULATIVE COUNTS ===
        {% for col in count_cols %}
        {% for h in hands %}
        {% if is_incremental() %}coalesce(ps.s_{{ col }}_vs_{{ h }}, 0) + {% endif %}sum(n.{{ col }}_vs_{{ h }}) over w_season as s_{{ col }}_vs_{{ h }},
        {% endfor %}
        {% endfor %}

        -- === CAREER CUMULATIVE COUNTS ===
        {% for col in count_cols %}
        {% for h in hands %}
        {% if is_incremental() %}coalesce(pc.c_{{ col }}_vs_{{ h }}, 0) + {% endif %}sum(n.{{ col }}_vs_{{ h }}) over w_career as c_{{ col }}_vs_{{ h }},
        {% endfor %}
        {% endfor %}

        -- === APPEARANCE + PITCH TRACKING (not hand-split) ===
        {% if is_incremental() %}coalesce(ps.s_pitches, 0) + {% endif %}sum(n.pitches) over w_season as s_pitches,
        {% if is_incremental() %}coalesce(pc.c_pitches, 0) + {% endif %}sum(n.pitches) over w_career as c_pitches,
        {% if is_incremental() %}coalesce(ps.s_games, 0) + {% endif %}sum(n.games) over w_season as s_games,
        {% if is_incremental() %}coalesce(pc.c_games, 0) + {% endif %}sum(n.games) over w_career as c_games

    from new_counts n
    {% if is_incremental() %}
    left join prev_season ps on n.pitcher_id = ps.pitcher_id and n.season = ps.season
    left join prev_career pc on n.pitcher_id = pc.pitcher_id
    {% endif %}
    window
        w_season as (partition by n.pitcher_id, n.season order by n.game_date rows unbounded preceding),
        w_career as (partition by n.pitcher_id order by n.game_date rows unbounded preceding)
)

select
    pitcher_id,
    game_date,
    season,

    -- === SEASON CUMULATIVE COUNTS ===
    {% for col in count_cols %}
    {% for h in hands %}
    s_{{ col }}_vs_{{ h }},
    {% endfor %}
    {% endfor %}
    s_bf_vs_l + s_bf_vs_r as s_bf,
    s_outs_vs_l + s_outs_vs_r as s_outs,
    s_hits_vs_l + s_hits_vs_r as s_hits,
    s_hr_vs_l + s_hr_vs_r as s_hr,
    s_bb_vs_l + s_bb_vs_r as s_bb,
    s_k_vs_l + s_k_vs_r as s_k,
    s_hbp_vs_l + s_hbp_vs_r as s_hbp,
    s_sf_vs_l + s_sf_vs_r as s_sf,
    s_sh_vs_l + s_sh_vs_r as s_sh,

    -- === CAREER CUMULATIVE COUNTS ===
    {% for col in count_cols %}
    {% for h in hands %}
    c_{{ col }}_vs_{{ h }},
    {% endfor %}
    {% endfor %}
    c_bf_vs_l + c_bf_vs_r as c_bf,
    c_outs_vs_l + c_outs_vs_r as c_outs,
    c_hits_vs_l + c_hits_vs_r as c_hits,
    c_hr_vs_l + c_hr_vs_r as c_hr,
    c_bb_vs_l + c_bb_vs_r as c_bb,
    c_k_vs_l + c_k_vs_r as c_k,
    c_hbp_vs_l + c_hbp_vs_r as c_hbp,
    c_sf_vs_l + c_sf_vs_r as c_sf,
    c_sh_vs_l + c_sh_vs_r as c_sh,

    -- === APPEARANCE + PITCH TRACKING ===
    s_pitches, c_pitches, s_games, c_games,

    -- =========================================================================
    -- SEASON RATE STATS
    -- =========================================================================

    -- IP (innings pitched = outs / 3)
    (s_outs_vs_l)::double / 3.0 as season_ip_vs_l,
    (s_outs_vs_r)::double / 3.0 as season_ip_vs_r,
    (s_outs_vs_l + s_outs_vs_r)::double / 3.0 as season_ip,

    -- WHIP = (BB + H) / IP
    (s_bb_vs_l + s_hits_vs_l)::double / nullif(s_outs_vs_l::double / 3.0, 0) as season_whip_vs_l,
    (s_bb_vs_r + s_hits_vs_r)::double / nullif(s_outs_vs_r::double / 3.0, 0) as season_whip_vs_r,
    (s_bb_vs_l + s_bb_vs_r + s_hits_vs_l + s_hits_vs_r)::double
        / nullif((s_outs_vs_l + s_outs_vs_r)::double / 3.0, 0) as season_whip,

    -- K/9
    s_k_vs_l::double * 9 / nullif(s_outs_vs_l::double / 3.0, 0) as season_k9_vs_l,
    s_k_vs_r::double * 9 / nullif(s_outs_vs_r::double / 3.0, 0) as season_k9_vs_r,
    (s_k_vs_l + s_k_vs_r)::double * 9 / nullif((s_outs_vs_l + s_outs_vs_r)::double / 3.0, 0) as season_k9,

    -- BB/9
    s_bb_vs_l::double * 9 / nullif(s_outs_vs_l::double / 3.0, 0) as season_bb9_vs_l,
    s_bb_vs_r::double * 9 / nullif(s_outs_vs_r::double / 3.0, 0) as season_bb9_vs_r,
    (s_bb_vs_l + s_bb_vs_r)::double * 9 / nullif((s_outs_vs_l + s_outs_vs_r)::double / 3.0, 0) as season_bb9,

    -- HR/9
    s_hr_vs_l::double * 9 / nullif(s_outs_vs_l::double / 3.0, 0) as season_hr9_vs_l,
    s_hr_vs_r::double * 9 / nullif(s_outs_vs_r::double / 3.0, 0) as season_hr9_vs_r,
    (s_hr_vs_l + s_hr_vs_r)::double * 9 / nullif((s_outs_vs_l + s_outs_vs_r)::double / 3.0, 0) as season_hr9,

    -- H/9
    s_hits_vs_l::double * 9 / nullif(s_outs_vs_l::double / 3.0, 0) as season_h9_vs_l,
    s_hits_vs_r::double * 9 / nullif(s_outs_vs_r::double / 3.0, 0) as season_h9_vs_r,
    (s_hits_vs_l + s_hits_vs_r)::double * 9 / nullif((s_outs_vs_l + s_outs_vs_r)::double / 3.0, 0) as season_h9,

    -- K%
    s_k_vs_l::double / nullif(s_bf_vs_l, 0) as season_k_pct_vs_l,
    s_k_vs_r::double / nullif(s_bf_vs_r, 0) as season_k_pct_vs_r,
    (s_k_vs_l + s_k_vs_r)::double / nullif(s_bf_vs_l + s_bf_vs_r, 0) as season_k_pct,

    -- BB%
    s_bb_vs_l::double / nullif(s_bf_vs_l, 0) as season_bb_pct_vs_l,
    s_bb_vs_r::double / nullif(s_bf_vs_r, 0) as season_bb_pct_vs_r,
    (s_bb_vs_l + s_bb_vs_r)::double / nullif(s_bf_vs_l + s_bf_vs_r, 0) as season_bb_pct,

    -- FIP = ((13*HR + 3*(BB+HBP) - 2*K) / IP) + 3.20
    (13.0 * s_hr_vs_l + 3.0 * (s_bb_vs_l + s_hbp_vs_l) - 2.0 * s_k_vs_l)
        / nullif(s_outs_vs_l::double / 3.0, 0) + 3.20 as season_fip_vs_l,
    (13.0 * s_hr_vs_r + 3.0 * (s_bb_vs_r + s_hbp_vs_r) - 2.0 * s_k_vs_r)
        / nullif(s_outs_vs_r::double / 3.0, 0) + 3.20 as season_fip_vs_r,
    (13.0 * (s_hr_vs_l + s_hr_vs_r) + 3.0 * (s_bb_vs_l + s_bb_vs_r + s_hbp_vs_l + s_hbp_vs_r)
        - 2.0 * (s_k_vs_l + s_k_vs_r))
        / nullif((s_outs_vs_l + s_outs_vs_r)::double / 3.0, 0) + 3.20 as season_fip,

    -- BABIP against = (H - HR) / (BF - K - HR - BB - HBP + SF)
    (s_hits_vs_l - s_hr_vs_l)::double
        / nullif(s_bf_vs_l - s_k_vs_l - s_hr_vs_l - s_bb_vs_l - s_hbp_vs_l + s_sf_vs_l, 0) as season_babip_vs_l,
    (s_hits_vs_r - s_hr_vs_r)::double
        / nullif(s_bf_vs_r - s_k_vs_r - s_hr_vs_r - s_bb_vs_r - s_hbp_vs_r + s_sf_vs_r, 0) as season_babip_vs_r,
    (s_hits_vs_l + s_hits_vs_r - s_hr_vs_l - s_hr_vs_r)::double
        / nullif(s_bf_vs_l + s_bf_vs_r - s_k_vs_l - s_k_vs_r - s_hr_vs_l - s_hr_vs_r
            - s_bb_vs_l - s_bb_vs_r - s_hbp_vs_l - s_hbp_vs_r + s_sf_vs_l + s_sf_vs_r, 0) as season_babip,

    -- wOBA against (Savant precomputed)
    s_woba_value_vs_l / nullif(s_woba_denom_vs_l, 0) as season_woba_vs_l,
    s_woba_value_vs_r / nullif(s_woba_denom_vs_r, 0) as season_woba_vs_r,
    (s_woba_value_vs_l + s_woba_value_vs_r) / nullif(s_woba_denom_vs_l + s_woba_denom_vs_r, 0) as season_woba,

    -- =========================================================================
    -- CAREER RATE STATS
    -- =========================================================================

    -- IP
    (c_outs_vs_l)::double / 3.0 as career_ip_vs_l,
    (c_outs_vs_r)::double / 3.0 as career_ip_vs_r,
    (c_outs_vs_l + c_outs_vs_r)::double / 3.0 as career_ip,

    -- WHIP
    (c_bb_vs_l + c_hits_vs_l)::double / nullif(c_outs_vs_l::double / 3.0, 0) as career_whip_vs_l,
    (c_bb_vs_r + c_hits_vs_r)::double / nullif(c_outs_vs_r::double / 3.0, 0) as career_whip_vs_r,
    (c_bb_vs_l + c_bb_vs_r + c_hits_vs_l + c_hits_vs_r)::double
        / nullif((c_outs_vs_l + c_outs_vs_r)::double / 3.0, 0) as career_whip,

    -- K/9
    c_k_vs_l::double * 9 / nullif(c_outs_vs_l::double / 3.0, 0) as career_k9_vs_l,
    c_k_vs_r::double * 9 / nullif(c_outs_vs_r::double / 3.0, 0) as career_k9_vs_r,
    (c_k_vs_l + c_k_vs_r)::double * 9 / nullif((c_outs_vs_l + c_outs_vs_r)::double / 3.0, 0) as career_k9,

    -- BB/9
    c_bb_vs_l::double * 9 / nullif(c_outs_vs_l::double / 3.0, 0) as career_bb9_vs_l,
    c_bb_vs_r::double * 9 / nullif(c_outs_vs_r::double / 3.0, 0) as career_bb9_vs_r,
    (c_bb_vs_l + c_bb_vs_r)::double * 9 / nullif((c_outs_vs_l + c_outs_vs_r)::double / 3.0, 0) as career_bb9,

    -- HR/9
    c_hr_vs_l::double * 9 / nullif(c_outs_vs_l::double / 3.0, 0) as career_hr9_vs_l,
    c_hr_vs_r::double * 9 / nullif(c_outs_vs_r::double / 3.0, 0) as career_hr9_vs_r,
    (c_hr_vs_l + c_hr_vs_r)::double * 9 / nullif((c_outs_vs_l + c_outs_vs_r)::double / 3.0, 0) as career_hr9,

    -- H/9
    c_hits_vs_l::double * 9 / nullif(c_outs_vs_l::double / 3.0, 0) as career_h9_vs_l,
    c_hits_vs_r::double * 9 / nullif(c_outs_vs_r::double / 3.0, 0) as career_h9_vs_r,
    (c_hits_vs_l + c_hits_vs_r)::double * 9 / nullif((c_outs_vs_l + c_outs_vs_r)::double / 3.0, 0) as career_h9,

    -- K%
    c_k_vs_l::double / nullif(c_bf_vs_l, 0) as career_k_pct_vs_l,
    c_k_vs_r::double / nullif(c_bf_vs_r, 0) as career_k_pct_vs_r,
    (c_k_vs_l + c_k_vs_r)::double / nullif(c_bf_vs_l + c_bf_vs_r, 0) as career_k_pct,

    -- BB%
    c_bb_vs_l::double / nullif(c_bf_vs_l, 0) as career_bb_pct_vs_l,
    c_bb_vs_r::double / nullif(c_bf_vs_r, 0) as career_bb_pct_vs_r,
    (c_bb_vs_l + c_bb_vs_r)::double / nullif(c_bf_vs_l + c_bf_vs_r, 0) as career_bb_pct,

    -- FIP
    (13.0 * c_hr_vs_l + 3.0 * (c_bb_vs_l + c_hbp_vs_l) - 2.0 * c_k_vs_l)
        / nullif(c_outs_vs_l::double / 3.0, 0) + 3.20 as career_fip_vs_l,
    (13.0 * c_hr_vs_r + 3.0 * (c_bb_vs_r + c_hbp_vs_r) - 2.0 * c_k_vs_r)
        / nullif(c_outs_vs_r::double / 3.0, 0) + 3.20 as career_fip_vs_r,
    (13.0 * (c_hr_vs_l + c_hr_vs_r) + 3.0 * (c_bb_vs_l + c_bb_vs_r + c_hbp_vs_l + c_hbp_vs_r)
        - 2.0 * (c_k_vs_l + c_k_vs_r))
        / nullif((c_outs_vs_l + c_outs_vs_r)::double / 3.0, 0) + 3.20 as career_fip,

    -- BABIP against
    (c_hits_vs_l - c_hr_vs_l)::double
        / nullif(c_bf_vs_l - c_k_vs_l - c_hr_vs_l - c_bb_vs_l - c_hbp_vs_l + c_sf_vs_l, 0) as career_babip_vs_l,
    (c_hits_vs_r - c_hr_vs_r)::double
        / nullif(c_bf_vs_r - c_k_vs_r - c_hr_vs_r - c_bb_vs_r - c_hbp_vs_r + c_sf_vs_r, 0) as career_babip_vs_r,
    (c_hits_vs_l + c_hits_vs_r - c_hr_vs_l - c_hr_vs_r)::double
        / nullif(c_bf_vs_l + c_bf_vs_r - c_k_vs_l - c_k_vs_r - c_hr_vs_l - c_hr_vs_r
            - c_bb_vs_l - c_bb_vs_r - c_hbp_vs_l - c_hbp_vs_r + c_sf_vs_l + c_sf_vs_r, 0) as career_babip,

    -- wOBA against
    c_woba_value_vs_l / nullif(c_woba_denom_vs_l, 0) as career_woba_vs_l,
    c_woba_value_vs_r / nullif(c_woba_denom_vs_r, 0) as career_woba_vs_r,
    (c_woba_value_vs_l + c_woba_value_vs_r) / nullif(c_woba_denom_vs_l + c_woba_denom_vs_r, 0) as career_woba,

    -- =========================================================================
    -- APPEARANCE AVERAGES + PITCHING EFFICIENCY
    -- =========================================================================

    s_bf::double / nullif(s_games, 0) as season_avg_bf_per_app,
    c_bf::double / nullif(c_games, 0) as career_avg_bf_per_app,
    s_pitches::double / nullif(s_games, 0) as season_avg_pitches_per_app,
    c_pitches::double / nullif(c_games, 0) as career_avg_pitches_per_app,
    s_pitches::double / nullif(s_outs, 0) as season_pitches_per_out,
    c_pitches::double / nullif(c_outs, 0) as career_pitches_per_out,

    -- Days since pitcher's previous game (rest days)
    coalesce(
        (game_date::date - lag(game_date::date) over (partition by pitcher_id order by game_date))::int,
        -1
    ) as pitcher_rest_days

from cumulative
