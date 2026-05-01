-- int_mlb__batters.sql
-- Incremental batter rates: cumulative season + career counting stats
-- and derived rate stats, split by pitcher handedness.
-- Self-join pattern: prev cumulatives + bounded window within chunk.
--
-- Grain: (batter_id, game_date)

{{ config(
    materialized='incremental',
    unique_key=['batter_id', 'game_date'],
    incremental_strategy='delete+insert',
    on_schema_change='fail',
) }}

{%- set chunk_start = var('chunk_start', none) -%}
{%- set chunk_end = var('chunk_end', none) -%}

{%- set count_cols = [
    'pa', 'abs', 'hits', 'singles', 'doubles', 'triples',
    'hr', 'bb', 'ibb', 'k', 'hbp', 'sf',
    'woba_value', 'woba_denom',
    'ev_sum', 'bip', 'hard_hit', 'barrel'
] -%}
{%- set hands = ['l', 'r'] -%}

with new_counts as (
    select * from {{ ref('int_mlb__batter_counts') }}
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
        select batter_id, season, max(game_date) as max_gd
        from {{ this }}
        where game_date < (select min(game_date) from new_counts)
        group by batter_id, season
    ) p on t.batter_id = p.batter_id and t.season = p.season and t.game_date = p.max_gd
),

prev_career as (
    select t.*
    from {{ this }} t
    inner join (
        select batter_id, max(game_date) as max_gd
        from {{ this }}
        where game_date < (select min(game_date) from new_counts)
        group by batter_id
    ) p on t.batter_id = p.batter_id and t.game_date = p.max_gd
),
{% endif %}

cumulative as (
    select
        n.batter_id,
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
        {%- set is_last_col = loop.last -%}
        {% for h in hands %}
        {% if is_incremental() %}coalesce(pc.c_{{ col }}_vs_{{ h }}, 0) + {% endif %}sum(n.{{ col }}_vs_{{ h }}) over w_career as c_{{ col }}_vs_{{ h }}{% if not (loop.last and is_last_col) %},{% endif %}
        {% endfor %}
        {% endfor %}

    from new_counts n
    {% if is_incremental() %}
    left join prev_season ps on n.batter_id = ps.batter_id and n.season = ps.season
    left join prev_career pc on n.batter_id = pc.batter_id
    {% endif %}
    window
        w_season as (partition by n.batter_id, n.season order by n.game_date rows unbounded preceding),
        w_career as (partition by n.batter_id order by n.game_date rows unbounded preceding)
)

select
    batter_id,
    game_date,
    season,

    -- === SEASON CUMULATIVE COUNTS ===
    {% for col in count_cols %}
    {% for h in hands %}
    s_{{ col }}_vs_{{ h }},
    {% endfor %}
    {% endfor %}
    s_pa_vs_l + s_pa_vs_r as s_pa,
    s_abs_vs_l + s_abs_vs_r as s_abs,
    s_hits_vs_l + s_hits_vs_r as s_hits,
    s_hr_vs_l + s_hr_vs_r as s_hr,
    s_bb_vs_l + s_bb_vs_r as s_bb,
    s_k_vs_l + s_k_vs_r as s_k,
    s_hbp_vs_l + s_hbp_vs_r as s_hbp,
    s_sf_vs_l + s_sf_vs_r as s_sf,

    -- === CAREER CUMULATIVE COUNTS ===
    {% for col in count_cols %}
    {% for h in hands %}
    c_{{ col }}_vs_{{ h }},
    {% endfor %}
    {% endfor %}
    c_pa_vs_l + c_pa_vs_r as c_pa,
    c_abs_vs_l + c_abs_vs_r as c_abs,
    c_hits_vs_l + c_hits_vs_r as c_hits,
    c_hr_vs_l + c_hr_vs_r as c_hr,
    c_bb_vs_l + c_bb_vs_r as c_bb,
    c_k_vs_l + c_k_vs_r as c_k,
    c_hbp_vs_l + c_hbp_vs_r as c_hbp,
    c_sf_vs_l + c_sf_vs_r as c_sf,

    -- =========================================================================
    -- SEASON RATE STATS
    -- =========================================================================

    -- BA
    s_hits_vs_l::double / nullif(s_abs_vs_l, 0) as season_ba_vs_l,
    s_hits_vs_r::double / nullif(s_abs_vs_r, 0) as season_ba_vs_r,
    (s_hits_vs_l + s_hits_vs_r)::double / nullif(s_abs_vs_l + s_abs_vs_r, 0) as season_ba,

    -- OBP
    (s_hits_vs_l + s_bb_vs_l + s_hbp_vs_l)::double
        / nullif(s_abs_vs_l + s_bb_vs_l + s_hbp_vs_l + s_sf_vs_l, 0) as season_obp_vs_l,
    (s_hits_vs_r + s_bb_vs_r + s_hbp_vs_r)::double
        / nullif(s_abs_vs_r + s_bb_vs_r + s_hbp_vs_r + s_sf_vs_r, 0) as season_obp_vs_r,
    (s_hits_vs_l + s_hits_vs_r + s_bb_vs_l + s_bb_vs_r + s_hbp_vs_l + s_hbp_vs_r)::double
        / nullif(s_abs_vs_l + s_abs_vs_r + s_bb_vs_l + s_bb_vs_r + s_hbp_vs_l + s_hbp_vs_r + s_sf_vs_l + s_sf_vs_r, 0) as season_obp,

    -- SLG
    (s_singles_vs_l + 2 * s_doubles_vs_l + 3 * s_triples_vs_l + 4 * s_hr_vs_l)::double
        / nullif(s_abs_vs_l, 0) as season_slg_vs_l,
    (s_singles_vs_r + 2 * s_doubles_vs_r + 3 * s_triples_vs_r + 4 * s_hr_vs_r)::double
        / nullif(s_abs_vs_r, 0) as season_slg_vs_r,
    (s_singles_vs_l + s_singles_vs_r + 2 * (s_doubles_vs_l + s_doubles_vs_r)
        + 3 * (s_triples_vs_l + s_triples_vs_r) + 4 * (s_hr_vs_l + s_hr_vs_r))::double
        / nullif(s_abs_vs_l + s_abs_vs_r, 0) as season_slg,

    -- OPS (derived inline)
    (s_hits_vs_l + s_bb_vs_l + s_hbp_vs_l)::double / nullif(s_abs_vs_l + s_bb_vs_l + s_hbp_vs_l + s_sf_vs_l, 0)
        + (s_singles_vs_l + 2 * s_doubles_vs_l + 3 * s_triples_vs_l + 4 * s_hr_vs_l)::double / nullif(s_abs_vs_l, 0)
        as season_ops_vs_l,
    (s_hits_vs_r + s_bb_vs_r + s_hbp_vs_r)::double / nullif(s_abs_vs_r + s_bb_vs_r + s_hbp_vs_r + s_sf_vs_r, 0)
        + (s_singles_vs_r + 2 * s_doubles_vs_r + 3 * s_triples_vs_r + 4 * s_hr_vs_r)::double / nullif(s_abs_vs_r, 0)
        as season_ops_vs_r,
    (s_hits_vs_l + s_hits_vs_r + s_bb_vs_l + s_bb_vs_r + s_hbp_vs_l + s_hbp_vs_r)::double
        / nullif(s_abs_vs_l + s_abs_vs_r + s_bb_vs_l + s_bb_vs_r + s_hbp_vs_l + s_hbp_vs_r + s_sf_vs_l + s_sf_vs_r, 0)
        + (s_singles_vs_l + s_singles_vs_r + 2 * (s_doubles_vs_l + s_doubles_vs_r)
            + 3 * (s_triples_vs_l + s_triples_vs_r) + 4 * (s_hr_vs_l + s_hr_vs_r))::double
            / nullif(s_abs_vs_l + s_abs_vs_r, 0)
        as season_ops,

    -- wOBA (Savant precomputed)
    s_woba_value_vs_l / nullif(s_woba_denom_vs_l, 0) as season_woba_vs_l,
    s_woba_value_vs_r / nullif(s_woba_denom_vs_r, 0) as season_woba_vs_r,
    (s_woba_value_vs_l + s_woba_value_vs_r) / nullif(s_woba_denom_vs_l + s_woba_denom_vs_r, 0) as season_woba,

    -- K%
    s_k_vs_l::double / nullif(s_pa_vs_l, 0) as season_k_pct_vs_l,
    s_k_vs_r::double / nullif(s_pa_vs_r, 0) as season_k_pct_vs_r,
    (s_k_vs_l + s_k_vs_r)::double / nullif(s_pa_vs_l + s_pa_vs_r, 0) as season_k_pct,

    -- BB%
    s_bb_vs_l::double / nullif(s_pa_vs_l, 0) as season_bb_pct_vs_l,
    s_bb_vs_r::double / nullif(s_pa_vs_r, 0) as season_bb_pct_vs_r,
    (s_bb_vs_l + s_bb_vs_r)::double / nullif(s_pa_vs_l + s_pa_vs_r, 0) as season_bb_pct,

    -- ISO
    (s_doubles_vs_l + 2 * s_triples_vs_l + 3 * s_hr_vs_l)::double / nullif(s_abs_vs_l, 0) as season_iso_vs_l,
    (s_doubles_vs_r + 2 * s_triples_vs_r + 3 * s_hr_vs_r)::double / nullif(s_abs_vs_r, 0) as season_iso_vs_r,
    ((s_doubles_vs_l + s_doubles_vs_r) + 2 * (s_triples_vs_l + s_triples_vs_r) + 3 * (s_hr_vs_l + s_hr_vs_r))::double
        / nullif(s_abs_vs_l + s_abs_vs_r, 0) as season_iso,

    -- BABIP
    (s_hits_vs_l - s_hr_vs_l)::double / nullif(s_abs_vs_l - s_k_vs_l - s_hr_vs_l + s_sf_vs_l, 0) as season_babip_vs_l,
    (s_hits_vs_r - s_hr_vs_r)::double / nullif(s_abs_vs_r - s_k_vs_r - s_hr_vs_r + s_sf_vs_r, 0) as season_babip_vs_r,
    (s_hits_vs_l + s_hits_vs_r - s_hr_vs_l - s_hr_vs_r)::double
        / nullif(s_abs_vs_l + s_abs_vs_r - s_k_vs_l - s_k_vs_r - s_hr_vs_l - s_hr_vs_r + s_sf_vs_l + s_sf_vs_r, 0) as season_babip,

    -- =========================================================================
    -- CAREER RATE STATS
    -- =========================================================================

    -- BA
    c_hits_vs_l::double / nullif(c_abs_vs_l, 0) as career_ba_vs_l,
    c_hits_vs_r::double / nullif(c_abs_vs_r, 0) as career_ba_vs_r,
    (c_hits_vs_l + c_hits_vs_r)::double / nullif(c_abs_vs_l + c_abs_vs_r, 0) as career_ba,

    -- OBP
    (c_hits_vs_l + c_bb_vs_l + c_hbp_vs_l)::double
        / nullif(c_abs_vs_l + c_bb_vs_l + c_hbp_vs_l + c_sf_vs_l, 0) as career_obp_vs_l,
    (c_hits_vs_r + c_bb_vs_r + c_hbp_vs_r)::double
        / nullif(c_abs_vs_r + c_bb_vs_r + c_hbp_vs_r + c_sf_vs_r, 0) as career_obp_vs_r,
    (c_hits_vs_l + c_hits_vs_r + c_bb_vs_l + c_bb_vs_r + c_hbp_vs_l + c_hbp_vs_r)::double
        / nullif(c_abs_vs_l + c_abs_vs_r + c_bb_vs_l + c_bb_vs_r + c_hbp_vs_l + c_hbp_vs_r + c_sf_vs_l + c_sf_vs_r, 0) as career_obp,

    -- SLG
    (c_singles_vs_l + 2 * c_doubles_vs_l + 3 * c_triples_vs_l + 4 * c_hr_vs_l)::double
        / nullif(c_abs_vs_l, 0) as career_slg_vs_l,
    (c_singles_vs_r + 2 * c_doubles_vs_r + 3 * c_triples_vs_r + 4 * c_hr_vs_r)::double
        / nullif(c_abs_vs_r, 0) as career_slg_vs_r,
    (c_singles_vs_l + c_singles_vs_r + 2 * (c_doubles_vs_l + c_doubles_vs_r)
        + 3 * (c_triples_vs_l + c_triples_vs_r) + 4 * (c_hr_vs_l + c_hr_vs_r))::double
        / nullif(c_abs_vs_l + c_abs_vs_r, 0) as career_slg,

    -- OPS
    (c_hits_vs_l + c_bb_vs_l + c_hbp_vs_l)::double / nullif(c_abs_vs_l + c_bb_vs_l + c_hbp_vs_l + c_sf_vs_l, 0)
        + (c_singles_vs_l + 2 * c_doubles_vs_l + 3 * c_triples_vs_l + 4 * c_hr_vs_l)::double / nullif(c_abs_vs_l, 0)
        as career_ops_vs_l,
    (c_hits_vs_r + c_bb_vs_r + c_hbp_vs_r)::double / nullif(c_abs_vs_r + c_bb_vs_r + c_hbp_vs_r + c_sf_vs_r, 0)
        + (c_singles_vs_r + 2 * c_doubles_vs_r + 3 * c_triples_vs_r + 4 * c_hr_vs_r)::double / nullif(c_abs_vs_r, 0)
        as career_ops_vs_r,
    (c_hits_vs_l + c_hits_vs_r + c_bb_vs_l + c_bb_vs_r + c_hbp_vs_l + c_hbp_vs_r)::double
        / nullif(c_abs_vs_l + c_abs_vs_r + c_bb_vs_l + c_bb_vs_r + c_hbp_vs_l + c_hbp_vs_r + c_sf_vs_l + c_sf_vs_r, 0)
        + (c_singles_vs_l + c_singles_vs_r + 2 * (c_doubles_vs_l + c_doubles_vs_r)
            + 3 * (c_triples_vs_l + c_triples_vs_r) + 4 * (c_hr_vs_l + c_hr_vs_r))::double
            / nullif(c_abs_vs_l + c_abs_vs_r, 0)
        as career_ops,

    -- wOBA
    c_woba_value_vs_l / nullif(c_woba_denom_vs_l, 0) as career_woba_vs_l,
    c_woba_value_vs_r / nullif(c_woba_denom_vs_r, 0) as career_woba_vs_r,
    (c_woba_value_vs_l + c_woba_value_vs_r) / nullif(c_woba_denom_vs_l + c_woba_denom_vs_r, 0) as career_woba,

    -- K%
    c_k_vs_l::double / nullif(c_pa_vs_l, 0) as career_k_pct_vs_l,
    c_k_vs_r::double / nullif(c_pa_vs_r, 0) as career_k_pct_vs_r,
    (c_k_vs_l + c_k_vs_r)::double / nullif(c_pa_vs_l + c_pa_vs_r, 0) as career_k_pct,

    -- BB%
    c_bb_vs_l::double / nullif(c_pa_vs_l, 0) as career_bb_pct_vs_l,
    c_bb_vs_r::double / nullif(c_pa_vs_r, 0) as career_bb_pct_vs_r,
    (c_bb_vs_l + c_bb_vs_r)::double / nullif(c_pa_vs_l + c_pa_vs_r, 0) as career_bb_pct,

    -- ISO
    (c_doubles_vs_l + 2 * c_triples_vs_l + 3 * c_hr_vs_l)::double / nullif(c_abs_vs_l, 0) as career_iso_vs_l,
    (c_doubles_vs_r + 2 * c_triples_vs_r + 3 * c_hr_vs_r)::double / nullif(c_abs_vs_r, 0) as career_iso_vs_r,
    ((c_doubles_vs_l + c_doubles_vs_r) + 2 * (c_triples_vs_l + c_triples_vs_r) + 3 * (c_hr_vs_l + c_hr_vs_r))::double
        / nullif(c_abs_vs_l + c_abs_vs_r, 0) as career_iso,

    -- BABIP
    (c_hits_vs_l - c_hr_vs_l)::double / nullif(c_abs_vs_l - c_k_vs_l - c_hr_vs_l + c_sf_vs_l, 0) as career_babip_vs_l,
    (c_hits_vs_r - c_hr_vs_r)::double / nullif(c_abs_vs_r - c_k_vs_r - c_hr_vs_r + c_sf_vs_r, 0) as career_babip_vs_r,
    (c_hits_vs_l + c_hits_vs_r - c_hr_vs_l - c_hr_vs_r)::double
        / nullif(c_abs_vs_l + c_abs_vs_r - c_k_vs_l - c_k_vs_r - c_hr_vs_l - c_hr_vs_r + c_sf_vs_l + c_sf_vs_r, 0) as career_babip,

    -- =========================================================================
    -- SEASON CONTACT QUALITY
    -- =========================================================================

    -- Avg Exit Velocity (BIP only)
    s_ev_sum_vs_l / nullif(s_bip_vs_l, 0) as season_avg_ev_vs_l,
    s_ev_sum_vs_r / nullif(s_bip_vs_r, 0) as season_avg_ev_vs_r,

    -- Hard Hit % (EV >= 95)
    s_hard_hit_vs_l::double / nullif(s_bip_vs_l, 0) as season_hard_hit_pct_vs_l,
    s_hard_hit_vs_r::double / nullif(s_bip_vs_r, 0) as season_hard_hit_pct_vs_r,

    -- Barrel %
    s_barrel_vs_l::double / nullif(s_bip_vs_l, 0) as season_barrel_pct_vs_l,
    s_barrel_vs_r::double / nullif(s_bip_vs_r, 0) as season_barrel_pct_vs_r,

    -- =========================================================================
    -- CAREER CONTACT QUALITY
    -- =========================================================================

    c_ev_sum_vs_l / nullif(c_bip_vs_l, 0) as career_avg_ev_vs_l,
    c_ev_sum_vs_r / nullif(c_bip_vs_r, 0) as career_avg_ev_vs_r,

    c_hard_hit_vs_l::double / nullif(c_bip_vs_l, 0) as career_hard_hit_pct_vs_l,
    c_hard_hit_vs_r::double / nullif(c_bip_vs_r, 0) as career_hard_hit_pct_vs_r,

    c_barrel_vs_l::double / nullif(c_bip_vs_l, 0) as career_barrel_pct_vs_l,
    c_barrel_vs_r::double / nullif(c_bip_vs_r, 0) as career_barrel_pct_vs_r

from cumulative
