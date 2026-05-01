-- int_mlb__batter_discipline.sql
-- Incremental batter plate discipline rates by pitch category: season + career.
-- Self-join pattern: prev cumulatives + bounded window within chunk.
--
-- Chase rate = chases / out-of-zone pitches
-- Whiff rate = whiffs / swings
-- Split by pitcher handedness AND pitch category (fb/brk/offspeed).
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
{%- set disc_cols = ['swings', 'whiffs', 'ooz', 'chases'] -%}

with new_counts as (
    select * from {{ ref('int_mlb__batter_discipline_counts') }}
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

        -- === SEASON CUMULATIVE: discipline counts ===
        {% for cat in categories %}
        {% for col in disc_cols %}
        {% for h in hands %}
        {% if is_incremental() %}coalesce(ps.s_{{ cat }}_{{ col }}_vs_{{ h }}, 0) + {% endif %}sum(n.{{ cat }}_{{ col }}_vs_{{ h }}) over w_season as s_{{ cat }}_{{ col }}_vs_{{ h }},
        {% endfor %}
        {% endfor %}
        {% endfor %}

        -- === CAREER CUMULATIVE: discipline counts ===
        {% for cat in categories %}
        {%- set is_last_cat = loop.last -%}
        {% for col in disc_cols %}
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
    {% for col in disc_cols %}
    {% for h in hands %}
    s_{{ cat }}_{{ col }}_vs_{{ h }},
    {% endfor %}
    {% endfor %}
    {% endfor %}
    {% for cat in categories %}
    {% for col in disc_cols %}
    {% for h in hands %}
    c_{{ cat }}_{{ col }}_vs_{{ h }},
    {% endfor %}
    {% endfor %}
    {% endfor %}

    -- =========================================================================
    -- SEASON CHASE RATES
    -- =========================================================================

    -- Chase rate vs fastballs (split by pitcher hand)
    s_fb_chases_vs_l::double / nullif(s_fb_ooz_vs_l, 0) as season_chase_rate_vs_fb_vs_l,
    s_fb_chases_vs_r::double / nullif(s_fb_ooz_vs_r, 0) as season_chase_rate_vs_fb_vs_r,

    -- Chase rate vs breaking balls (split by pitcher hand)
    s_brk_chases_vs_l::double / nullif(s_brk_ooz_vs_l, 0) as season_chase_rate_vs_brk_vs_l,
    s_brk_chases_vs_r::double / nullif(s_brk_ooz_vs_r, 0) as season_chase_rate_vs_brk_vs_r,

    -- Chase rate vs offspeed (split by pitcher hand)
    s_offspeed_chases_vs_l::double / nullif(s_offspeed_ooz_vs_l, 0) as season_chase_rate_vs_offspeed_vs_l,
    s_offspeed_chases_vs_r::double / nullif(s_offspeed_ooz_vs_r, 0) as season_chase_rate_vs_offspeed_vs_r,

    -- =========================================================================
    -- SEASON WHIFF RATES
    -- =========================================================================

    -- Whiff rate vs fastballs (split by pitcher hand)
    s_fb_whiffs_vs_l::double / nullif(s_fb_swings_vs_l, 0) as season_whiff_rate_vs_fb_vs_l,
    s_fb_whiffs_vs_r::double / nullif(s_fb_swings_vs_r, 0) as season_whiff_rate_vs_fb_vs_r,

    -- Whiff rate vs breaking balls (split by pitcher hand)
    s_brk_whiffs_vs_l::double / nullif(s_brk_swings_vs_l, 0) as season_whiff_rate_vs_brk_vs_l,
    s_brk_whiffs_vs_r::double / nullif(s_brk_swings_vs_r, 0) as season_whiff_rate_vs_brk_vs_r,

    -- Whiff rate vs offspeed (split by pitcher hand)
    s_offspeed_whiffs_vs_l::double / nullif(s_offspeed_swings_vs_l, 0) as season_whiff_rate_vs_offspeed_vs_l,
    s_offspeed_whiffs_vs_r::double / nullif(s_offspeed_swings_vs_r, 0) as season_whiff_rate_vs_offspeed_vs_r,

    -- =========================================================================
    -- CAREER CHASE RATES
    -- =========================================================================

    -- Chase rate vs fastballs (split by pitcher hand)
    c_fb_chases_vs_l::double / nullif(c_fb_ooz_vs_l, 0) as career_chase_rate_vs_fb_vs_l,
    c_fb_chases_vs_r::double / nullif(c_fb_ooz_vs_r, 0) as career_chase_rate_vs_fb_vs_r,

    -- Chase rate vs breaking balls (split by pitcher hand)
    c_brk_chases_vs_l::double / nullif(c_brk_ooz_vs_l, 0) as career_chase_rate_vs_brk_vs_l,
    c_brk_chases_vs_r::double / nullif(c_brk_ooz_vs_r, 0) as career_chase_rate_vs_brk_vs_r,

    -- Chase rate vs offspeed (split by pitcher hand)
    c_offspeed_chases_vs_l::double / nullif(c_offspeed_ooz_vs_l, 0) as career_chase_rate_vs_offspeed_vs_l,
    c_offspeed_chases_vs_r::double / nullif(c_offspeed_ooz_vs_r, 0) as career_chase_rate_vs_offspeed_vs_r,

    -- =========================================================================
    -- CAREER WHIFF RATES
    -- =========================================================================

    -- Whiff rate vs fastballs (split by pitcher hand)
    c_fb_whiffs_vs_l::double / nullif(c_fb_swings_vs_l, 0) as career_whiff_rate_vs_fb_vs_l,
    c_fb_whiffs_vs_r::double / nullif(c_fb_swings_vs_r, 0) as career_whiff_rate_vs_fb_vs_r,

    -- Whiff rate vs breaking balls (split by pitcher hand)
    c_brk_whiffs_vs_l::double / nullif(c_brk_swings_vs_l, 0) as career_whiff_rate_vs_brk_vs_l,
    c_brk_whiffs_vs_r::double / nullif(c_brk_swings_vs_r, 0) as career_whiff_rate_vs_brk_vs_r,

    -- Whiff rate vs offspeed (split by pitcher hand)
    c_offspeed_whiffs_vs_l::double / nullif(c_offspeed_swings_vs_l, 0) as career_whiff_rate_vs_offspeed_vs_l,
    c_offspeed_whiffs_vs_r::double / nullif(c_offspeed_swings_vs_r, 0) as career_whiff_rate_vs_offspeed_vs_r

from cumulative
