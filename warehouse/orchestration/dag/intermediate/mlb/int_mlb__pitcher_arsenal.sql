-- int_mlb__pitcher_arsenal.sql
-- Incremental pitcher arsenal stats: per-pitch-type mix rates, average velocity/spin,
-- arm angle, and release extension. Season + career timeframes.
-- Self-join pattern: prev cumulatives + bounded window within chunk.
-- Driven by pitch_types() macro.
--
-- Pitch mix is split by batter handedness (pitchers change approach vs L/R).
-- Stuff and delivery are NOT split (physical properties don't change by matchup).
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
{%- set pts = pitch_types() -%}

with new_counts as (
    select * from {{ ref('int_mlb__pitcher_arsenal_counts') }}
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

        -- === SEASON CUMULATIVE: TOTAL PITCHES ===
        {% if is_incremental() %}coalesce(ps.s_pitches_vs_l, 0) + {% endif %}sum(n.pitches_vs_l) over w_season as s_pitches_vs_l,
        {% if is_incremental() %}coalesce(ps.s_pitches_vs_r, 0) + {% endif %}sum(n.pitches_vs_r) over w_season as s_pitches_vs_r,

        -- === SEASON CUMULATIVE: PITCH MIX ===
        {% for pt in pts %}
        {% if is_incremental() %}coalesce(ps.s_{{ pt | lower }}_n_vs_l, 0) + {% endif %}sum(n.{{ pt | lower }}_n_vs_l) over w_season as s_{{ pt | lower }}_n_vs_l,
        {% if is_incremental() %}coalesce(ps.s_{{ pt | lower }}_n_vs_r, 0) + {% endif %}sum(n.{{ pt | lower }}_n_vs_r) over w_season as s_{{ pt | lower }}_n_vs_r,
        {% endfor %}

        -- === SEASON CUMULATIVE: VELOCITY ===
        {% for pt in pts %}
        {% if is_incremental() %}coalesce(ps.s_{{ pt | lower }}_velo_sum, 0) + {% endif %}sum(n.{{ pt | lower }}_velo_sum) over w_season as s_{{ pt | lower }}_velo_sum,
        {% if is_incremental() %}coalesce(ps.s_{{ pt | lower }}_velo_n, 0) + {% endif %}sum(n.{{ pt | lower }}_velo_n) over w_season as s_{{ pt | lower }}_velo_n,
        {% endfor %}

        -- === SEASON CUMULATIVE: SPIN ===
        {% for pt in pts %}
        {% if is_incremental() %}coalesce(ps.s_{{ pt | lower }}_spin_sum, 0) + {% endif %}sum(n.{{ pt | lower }}_spin_sum) over w_season as s_{{ pt | lower }}_spin_sum,
        {% if is_incremental() %}coalesce(ps.s_{{ pt | lower }}_spin_n, 0) + {% endif %}sum(n.{{ pt | lower }}_spin_n) over w_season as s_{{ pt | lower }}_spin_n,
        {% endfor %}

        -- === SEASON CUMULATIVE: DELIVERY ===
        {% if is_incremental() %}coalesce(ps.s_arm_angle_sum, 0) + {% endif %}sum(n.arm_angle_sum) over w_season as s_arm_angle_sum,
        {% if is_incremental() %}coalesce(ps.s_arm_angle_n, 0) + {% endif %}sum(n.arm_angle_n) over w_season as s_arm_angle_n,
        {% if is_incremental() %}coalesce(ps.s_extension_sum, 0) + {% endif %}sum(n.extension_sum) over w_season as s_extension_sum,
        {% if is_incremental() %}coalesce(ps.s_extension_n, 0) + {% endif %}sum(n.extension_n) over w_season as s_extension_n,

        -- === CAREER CUMULATIVE: TOTAL PITCHES ===
        {% if is_incremental() %}coalesce(pc.c_pitches_vs_l, 0) + {% endif %}sum(n.pitches_vs_l) over w_career as c_pitches_vs_l,
        {% if is_incremental() %}coalesce(pc.c_pitches_vs_r, 0) + {% endif %}sum(n.pitches_vs_r) over w_career as c_pitches_vs_r,

        -- === CAREER CUMULATIVE: PITCH MIX ===
        {% for pt in pts %}
        {% if is_incremental() %}coalesce(pc.c_{{ pt | lower }}_n_vs_l, 0) + {% endif %}sum(n.{{ pt | lower }}_n_vs_l) over w_career as c_{{ pt | lower }}_n_vs_l,
        {% if is_incremental() %}coalesce(pc.c_{{ pt | lower }}_n_vs_r, 0) + {% endif %}sum(n.{{ pt | lower }}_n_vs_r) over w_career as c_{{ pt | lower }}_n_vs_r,
        {% endfor %}

        -- === CAREER CUMULATIVE: VELOCITY ===
        {% for pt in pts %}
        {% if is_incremental() %}coalesce(pc.c_{{ pt | lower }}_velo_sum, 0) + {% endif %}sum(n.{{ pt | lower }}_velo_sum) over w_career as c_{{ pt | lower }}_velo_sum,
        {% if is_incremental() %}coalesce(pc.c_{{ pt | lower }}_velo_n, 0) + {% endif %}sum(n.{{ pt | lower }}_velo_n) over w_career as c_{{ pt | lower }}_velo_n,
        {% endfor %}

        -- === CAREER CUMULATIVE: SPIN ===
        {% for pt in pts %}
        {% if is_incremental() %}coalesce(pc.c_{{ pt | lower }}_spin_sum, 0) + {% endif %}sum(n.{{ pt | lower }}_spin_sum) over w_career as c_{{ pt | lower }}_spin_sum,
        {% if is_incremental() %}coalesce(pc.c_{{ pt | lower }}_spin_n, 0) + {% endif %}sum(n.{{ pt | lower }}_spin_n) over w_career as c_{{ pt | lower }}_spin_n,
        {% endfor %}

        -- === CAREER CUMULATIVE: DELIVERY ===
        {% if is_incremental() %}coalesce(pc.c_arm_angle_sum, 0) + {% endif %}sum(n.arm_angle_sum) over w_career as c_arm_angle_sum,
        {% if is_incremental() %}coalesce(pc.c_arm_angle_n, 0) + {% endif %}sum(n.arm_angle_n) over w_career as c_arm_angle_n,
        {% if is_incremental() %}coalesce(pc.c_extension_sum, 0) + {% endif %}sum(n.extension_sum) over w_career as c_extension_sum,
        {% if is_incremental() %}coalesce(pc.c_extension_n, 0) + {% endif %}sum(n.extension_n) over w_career as c_extension_n

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

    -- === CUMULATIVE COUNTS (stored for self-join on subsequent chunks) ===
    s_pitches_vs_l, s_pitches_vs_r,
    {% for pt in pts %}
    s_{{ pt | lower }}_n_vs_l, s_{{ pt | lower }}_n_vs_r,
    s_{{ pt | lower }}_velo_sum, s_{{ pt | lower }}_velo_n,
    s_{{ pt | lower }}_spin_sum, s_{{ pt | lower }}_spin_n,
    {% endfor %}
    s_arm_angle_sum, s_arm_angle_n,
    s_extension_sum, s_extension_n,

    c_pitches_vs_l, c_pitches_vs_r,
    {% for pt in pts %}
    c_{{ pt | lower }}_n_vs_l, c_{{ pt | lower }}_n_vs_r,
    c_{{ pt | lower }}_velo_sum, c_{{ pt | lower }}_velo_n,
    c_{{ pt | lower }}_spin_sum, c_{{ pt | lower }}_spin_n,
    {% endfor %}
    c_arm_angle_sum, c_arm_angle_n,
    c_extension_sum, c_extension_n,

    -- =========================================================================
    -- SEASON RATES
    -- =========================================================================

    -- Pitch mix (split by batter hand)
    {% for pt in pts %}
    s_{{ pt | lower }}_n_vs_l::double / nullif(s_pitches_vs_l, 0) as season_{{ pt | lower }}_pct_vs_l,
    s_{{ pt | lower }}_n_vs_r::double / nullif(s_pitches_vs_r, 0) as season_{{ pt | lower }}_pct_vs_r,
    {% endfor %}

    -- Velocity (NOT split by hand)
    {% for pt in pts %}
    s_{{ pt | lower }}_velo_sum / nullif(s_{{ pt | lower }}_velo_n, 0) as season_{{ pt | lower }}_velo,
    {% endfor %}

    -- Spin (NOT split by hand)
    {% for pt in pts %}
    s_{{ pt | lower }}_spin_sum / nullif(s_{{ pt | lower }}_spin_n, 0) as season_{{ pt | lower }}_spin,
    {% endfor %}

    -- Delivery (NOT split by hand)
    s_arm_angle_sum / nullif(s_arm_angle_n, 0) as season_arm_angle,
    s_extension_sum / nullif(s_extension_n, 0) as season_extension,

    -- =========================================================================
    -- CAREER RATES
    -- =========================================================================

    -- Pitch mix (split by batter hand)
    {% for pt in pts %}
    c_{{ pt | lower }}_n_vs_l::double / nullif(c_pitches_vs_l, 0) as career_{{ pt | lower }}_pct_vs_l,
    c_{{ pt | lower }}_n_vs_r::double / nullif(c_pitches_vs_r, 0) as career_{{ pt | lower }}_pct_vs_r,
    {% endfor %}

    -- Velocity
    {% for pt in pts %}
    c_{{ pt | lower }}_velo_sum / nullif(c_{{ pt | lower }}_velo_n, 0) as career_{{ pt | lower }}_velo,
    {% endfor %}

    -- Spin
    {% for pt in pts %}
    c_{{ pt | lower }}_spin_sum / nullif(c_{{ pt | lower }}_spin_n, 0) as career_{{ pt | lower }}_spin,
    {% endfor %}

    -- Delivery
    c_arm_angle_sum / nullif(c_arm_angle_n, 0) as career_arm_angle,
    c_extension_sum / nullif(c_extension_n, 0) as career_extension

from cumulative
