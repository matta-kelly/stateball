-- int_mlb__pitcher_arsenal_counts.sql
-- Daily pitch-level arsenal aggregations per pitcher.
-- Per-pitch-type: one set of columns per type in pitch_types() macro.
-- Operates on ALL pitches (not just terminal) — pitch mix and stuff
-- metrics require the full pitch stream.
--
-- Types: FF, SI, FC, SL, CU, KC, SV, ST, CH, FS
-- Anything else (PO, IN, KN, EP, etc.) excluded by not being in the list.
--
-- Grain: (pitcher_id, game_date)

{{
    config(
        materialized='incremental',
        unique_key=['pitcher_id', 'game_date'],
        incremental_strategy='delete+insert',
        on_schema_change='fail'
    )
}}

{%- set pts = pitch_types() -%}

with pitches as (
    select
        pitcher_id,
        game_date,
        game_year as season,
        bat_side,
        pitch_type,
        release_speed,
        release_spin_rate,
        arm_angle,
        release_extension
    from {{ ref('proc_mlb__events') }}
    where pitch_type in ('{{ pts | join("', '") }}')
    {% if is_incremental() %}
      and game_date > (select max(game_date) from {{ this }})
    {% endif %}
)

select
    pitcher_id,
    game_date,
    season,

    -- === TOTAL CLASSIFIED PITCHES (denominator for pitch mix %) ===
    count(*) filter (where bat_side = 'L') as pitches_vs_l,
    count(*) filter (where bat_side = 'R') as pitches_vs_r,

    -- === PITCH MIX COUNTS (per type x batter hand) ===
    {% for pt in pts %}
    count(*) filter (where bat_side = 'L' and pitch_type = '{{ pt }}') as {{ pt | lower }}_n_vs_l,
    count(*) filter (where bat_side = 'R' and pitch_type = '{{ pt }}') as {{ pt | lower }}_n_vs_r,
    {% endfor %}

    -- === VELOCITY SUMS + COUNTS (per type, NOT split by hand) ===
    -- Sum + count pattern: cumulative avg = SUM(sums) / SUM(counts)
    {% for pt in pts %}
    sum(release_speed) filter (where pitch_type = '{{ pt }}') as {{ pt | lower }}_velo_sum,
    count(release_speed) filter (where pitch_type = '{{ pt }}') as {{ pt | lower }}_velo_n,
    {% endfor %}

    -- === SPIN RATE SUMS + COUNTS (per type, NOT split by hand) ===
    {% for pt in pts %}
    sum(release_spin_rate) filter (where pitch_type = '{{ pt }}') as {{ pt | lower }}_spin_sum,
    count(release_spin_rate) filter (where pitch_type = '{{ pt }}') as {{ pt | lower }}_spin_n,
    {% endfor %}

    -- === DELIVERY SUMS + COUNTS (NOT per type) ===
    sum(arm_angle) as arm_angle_sum,
    count(arm_angle) as arm_angle_n,
    sum(release_extension) as extension_sum,
    count(release_extension) as extension_n

from pitches
group by pitcher_id, game_date, season
