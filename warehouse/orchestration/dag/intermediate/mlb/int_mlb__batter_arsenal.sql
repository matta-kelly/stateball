-- int_mlb__batter_arsenal.sql
-- Incremental batter wOBA + batted ball rates by pitch category: season + career.
-- Self-join pattern: prev cumulatives + bounded window within chunk.
--
-- Split by pitcher handedness (batters perform differently vs L/R).
-- CASE-selected in feat_mlb__vectors by pitch_hand.
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

{%- set categories = ['fb', 'brk', 'offspeed'] -%}
{%- set hands = ['l', 'r'] -%}
{%- set woba_cols = ['woba_value', 'woba_denom'] -%}
{%- set bb_cols = ['bip', 'gb', 'fb', 'ld'] -%}

with new_counts as (
    select * from {{ ref('int_mlb__batter_arsenal_counts') }}
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

        -- === SEASON CUMULATIVE: wOBA + batted ball counts ===
        {% for cat in categories %}
        {% for col in woba_cols + bb_cols %}
        {% for h in hands %}
        {% if is_incremental() %}coalesce(ps.s_{{ cat }}_{{ col }}_vs_{{ h }}, 0) + {% endif %}sum(n.{{ cat }}_{{ col }}_vs_{{ h }}) over w_season as s_{{ cat }}_{{ col }}_vs_{{ h }},
        {% endfor %}
        {% endfor %}
        {% endfor %}

        -- === CAREER CUMULATIVE: wOBA + batted ball counts ===
        {% for cat in categories %}
        {%- set is_last_cat = loop.last -%}
        {% for col in woba_cols + bb_cols %}
        {%- set is_last_col = loop.last -%}
        {% for h in hands %}
        {% if is_incremental() %}coalesce(pc.c_{{ cat }}_{{ col }}_vs_{{ h }}, 0) + {% endif %}sum(n.{{ cat }}_{{ col }}_vs_{{ h }}) over w_career as c_{{ cat }}_{{ col }}_vs_{{ h }}{% if not (loop.last and is_last_col and is_last_cat) %},{% endif %}
        {% endfor %}
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

    -- === CUMULATIVE COUNTS (stored for self-join on subsequent chunks) ===
    {% for cat in categories %}
    {% for col in woba_cols + bb_cols %}
    {% for h in hands %}
    s_{{ cat }}_{{ col }}_vs_{{ h }},
    {% endfor %}
    {% endfor %}
    {% endfor %}
    {% for cat in categories %}
    {% for col in woba_cols + bb_cols %}
    {% for h in hands %}
    c_{{ cat }}_{{ col }}_vs_{{ h }},
    {% endfor %}
    {% endfor %}
    {% endfor %}

    -- =========================================================================
    -- SEASON RATES
    -- =========================================================================

    -- wOBA vs fastballs (split by pitcher hand)
    s_fb_woba_value_vs_l / nullif(s_fb_woba_denom_vs_l, 0) as season_woba_vs_fb_vs_l,
    s_fb_woba_value_vs_r / nullif(s_fb_woba_denom_vs_r, 0) as season_woba_vs_fb_vs_r,

    -- wOBA vs breaking balls (split by pitcher hand)
    s_brk_woba_value_vs_l / nullif(s_brk_woba_denom_vs_l, 0) as season_woba_vs_brk_vs_l,
    s_brk_woba_value_vs_r / nullif(s_brk_woba_denom_vs_r, 0) as season_woba_vs_brk_vs_r,

    -- wOBA vs offspeed (split by pitcher hand)
    s_offspeed_woba_value_vs_l / nullif(s_offspeed_woba_denom_vs_l, 0) as season_woba_vs_offspeed_vs_l,
    s_offspeed_woba_value_vs_r / nullif(s_offspeed_woba_denom_vs_r, 0) as season_woba_vs_offspeed_vs_r,

    -- =========================================================================
    -- CAREER RATES
    -- =========================================================================

    -- wOBA vs fastballs (split by pitcher hand)
    c_fb_woba_value_vs_l / nullif(c_fb_woba_denom_vs_l, 0) as career_woba_vs_fb_vs_l,
    c_fb_woba_value_vs_r / nullif(c_fb_woba_denom_vs_r, 0) as career_woba_vs_fb_vs_r,

    -- wOBA vs breaking balls (split by pitcher hand)
    c_brk_woba_value_vs_l / nullif(c_brk_woba_denom_vs_l, 0) as career_woba_vs_brk_vs_l,
    c_brk_woba_value_vs_r / nullif(c_brk_woba_denom_vs_r, 0) as career_woba_vs_brk_vs_r,

    -- wOBA vs offspeed (split by pitcher hand)
    c_offspeed_woba_value_vs_l / nullif(c_offspeed_woba_denom_vs_l, 0) as career_woba_vs_offspeed_vs_l,
    c_offspeed_woba_value_vs_r / nullif(c_offspeed_woba_denom_vs_r, 0) as career_woba_vs_offspeed_vs_r,

    -- =========================================================================
    -- SEASON BATTED BALL RATES
    -- =========================================================================

    -- GB% by pitch category (split by pitcher hand)
    s_fb_gb_vs_l::double / nullif(s_fb_bip_vs_l, 0) as season_gb_pct_vs_fb_vs_l,
    s_fb_gb_vs_r::double / nullif(s_fb_bip_vs_r, 0) as season_gb_pct_vs_fb_vs_r,
    s_brk_gb_vs_l::double / nullif(s_brk_bip_vs_l, 0) as season_gb_pct_vs_brk_vs_l,
    s_brk_gb_vs_r::double / nullif(s_brk_bip_vs_r, 0) as season_gb_pct_vs_brk_vs_r,
    s_offspeed_gb_vs_l::double / nullif(s_offspeed_bip_vs_l, 0) as season_gb_pct_vs_offspeed_vs_l,
    s_offspeed_gb_vs_r::double / nullif(s_offspeed_bip_vs_r, 0) as season_gb_pct_vs_offspeed_vs_r,

    -- FB% by pitch category (split by pitcher hand)
    s_fb_fb_vs_l::double / nullif(s_fb_bip_vs_l, 0) as season_fb_pct_vs_fb_vs_l,
    s_fb_fb_vs_r::double / nullif(s_fb_bip_vs_r, 0) as season_fb_pct_vs_fb_vs_r,
    s_brk_fb_vs_l::double / nullif(s_brk_bip_vs_l, 0) as season_fb_pct_vs_brk_vs_l,
    s_brk_fb_vs_r::double / nullif(s_brk_bip_vs_r, 0) as season_fb_pct_vs_brk_vs_r,
    s_offspeed_fb_vs_l::double / nullif(s_offspeed_bip_vs_l, 0) as season_fb_pct_vs_offspeed_vs_l,
    s_offspeed_fb_vs_r::double / nullif(s_offspeed_bip_vs_r, 0) as season_fb_pct_vs_offspeed_vs_r,

    -- LD% by pitch category (split by pitcher hand)
    s_fb_ld_vs_l::double / nullif(s_fb_bip_vs_l, 0) as season_ld_pct_vs_fb_vs_l,
    s_fb_ld_vs_r::double / nullif(s_fb_bip_vs_r, 0) as season_ld_pct_vs_fb_vs_r,
    s_brk_ld_vs_l::double / nullif(s_brk_bip_vs_l, 0) as season_ld_pct_vs_brk_vs_l,
    s_brk_ld_vs_r::double / nullif(s_brk_bip_vs_r, 0) as season_ld_pct_vs_brk_vs_r,
    s_offspeed_ld_vs_l::double / nullif(s_offspeed_bip_vs_l, 0) as season_ld_pct_vs_offspeed_vs_l,
    s_offspeed_ld_vs_r::double / nullif(s_offspeed_bip_vs_r, 0) as season_ld_pct_vs_offspeed_vs_r,

    -- =========================================================================
    -- CAREER BATTED BALL RATES
    -- =========================================================================

    -- GB% by pitch category (split by pitcher hand)
    c_fb_gb_vs_l::double / nullif(c_fb_bip_vs_l, 0) as career_gb_pct_vs_fb_vs_l,
    c_fb_gb_vs_r::double / nullif(c_fb_bip_vs_r, 0) as career_gb_pct_vs_fb_vs_r,
    c_brk_gb_vs_l::double / nullif(c_brk_bip_vs_l, 0) as career_gb_pct_vs_brk_vs_l,
    c_brk_gb_vs_r::double / nullif(c_brk_bip_vs_r, 0) as career_gb_pct_vs_brk_vs_r,
    c_offspeed_gb_vs_l::double / nullif(c_offspeed_bip_vs_l, 0) as career_gb_pct_vs_offspeed_vs_l,
    c_offspeed_gb_vs_r::double / nullif(c_offspeed_bip_vs_r, 0) as career_gb_pct_vs_offspeed_vs_r,

    -- FB% by pitch category (split by pitcher hand)
    c_fb_fb_vs_l::double / nullif(c_fb_bip_vs_l, 0) as career_fb_pct_vs_fb_vs_l,
    c_fb_fb_vs_r::double / nullif(c_fb_bip_vs_r, 0) as career_fb_pct_vs_fb_vs_r,
    c_brk_fb_vs_l::double / nullif(c_brk_bip_vs_l, 0) as career_fb_pct_vs_brk_vs_l,
    c_brk_fb_vs_r::double / nullif(c_brk_bip_vs_r, 0) as career_fb_pct_vs_brk_vs_r,
    c_offspeed_fb_vs_l::double / nullif(c_offspeed_bip_vs_l, 0) as career_fb_pct_vs_offspeed_vs_l,
    c_offspeed_fb_vs_r::double / nullif(c_offspeed_bip_vs_r, 0) as career_fb_pct_vs_offspeed_vs_r,

    -- LD% by pitch category (split by pitcher hand)
    c_fb_ld_vs_l::double / nullif(c_fb_bip_vs_l, 0) as career_ld_pct_vs_fb_vs_l,
    c_fb_ld_vs_r::double / nullif(c_fb_bip_vs_r, 0) as career_ld_pct_vs_fb_vs_r,
    c_brk_ld_vs_l::double / nullif(c_brk_bip_vs_l, 0) as career_ld_pct_vs_brk_vs_l,
    c_brk_ld_vs_r::double / nullif(c_brk_bip_vs_r, 0) as career_ld_pct_vs_brk_vs_r,
    c_offspeed_ld_vs_l::double / nullif(c_offspeed_bip_vs_l, 0) as career_ld_pct_vs_offspeed_vs_l,
    c_offspeed_ld_vs_r::double / nullif(c_offspeed_bip_vs_r, 0) as career_ld_pct_vs_offspeed_vs_r

from cumulative
