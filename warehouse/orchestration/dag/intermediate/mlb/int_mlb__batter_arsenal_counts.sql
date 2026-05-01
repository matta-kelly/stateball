-- int_mlb__batter_arsenal_counts.sql
-- Daily batter wOBA + batted ball aggregations by pitch category and pitcher handedness.
-- Operates on TERMINAL pitches only (pa_result is not null) — wOBA is a
-- PA outcome metric tied to how the PA ended.
-- Batted ball types (bb_type) are also terminal-pitch-only (ball put in play).
--
-- Pitch type grouping (same as int_mlb__pitcher_arsenal_counts):
--   fb       = FF (4-seam), SI (sinker), FC (cutter)
--   brk      = SL (slider), CU (curve), KC (knuckle curve), SV (sweeper), ST (sweepslider)
--   offspeed = CH (changeup), FS (splitter)
--   Anything else (PO, IN, KN, EP, etc.) → NULL, excluded from counts.
--
-- Grain: (batter_id, game_date)

{{
    config(
        materialized='incremental',
        unique_key=['batter_id', 'game_date'],
        incremental_strategy='delete+insert',
        on_schema_change='fail'
    )
}}

with pa as (
    select
        batter_id,
        game_date,
        game_year as season,
        pitch_hand,
        case
            when pitch_type in ('FF', 'SI', 'FC') then 'fb'
            when pitch_type in ('SL', 'CU', 'KC', 'SV', 'ST') then 'brk'
            when pitch_type in ('CH', 'FS') then 'offspeed'
        end as pitch_group,
        woba_value,
        woba_denom,
        bb_type
    from {{ ref('proc_mlb__events') }}
    where pa_result is not null
      and pa_result != 'truncated_pa'
      and pitch_type is not null
    {% if is_incremental() %}
      and game_date > (select max(game_date) from {{ this }})
    {% endif %}
)

select
    batter_id,
    game_date,
    season,

    -- === wOBA NUMERATOR by pitch group x pitcher hand ===
    sum(woba_value) filter (where pitch_hand = 'L' and pitch_group = 'fb') as fb_woba_value_vs_l,
    sum(woba_value) filter (where pitch_hand = 'R' and pitch_group = 'fb') as fb_woba_value_vs_r,
    sum(woba_value) filter (where pitch_hand = 'L' and pitch_group = 'brk') as brk_woba_value_vs_l,
    sum(woba_value) filter (where pitch_hand = 'R' and pitch_group = 'brk') as brk_woba_value_vs_r,
    sum(woba_value) filter (where pitch_hand = 'L' and pitch_group = 'offspeed') as offspeed_woba_value_vs_l,
    sum(woba_value) filter (where pitch_hand = 'R' and pitch_group = 'offspeed') as offspeed_woba_value_vs_r,

    -- === wOBA DENOMINATOR by pitch group x pitcher hand ===
    sum(woba_denom) filter (where pitch_hand = 'L' and pitch_group = 'fb') as fb_woba_denom_vs_l,
    sum(woba_denom) filter (where pitch_hand = 'R' and pitch_group = 'fb') as fb_woba_denom_vs_r,
    sum(woba_denom) filter (where pitch_hand = 'L' and pitch_group = 'brk') as brk_woba_denom_vs_l,
    sum(woba_denom) filter (where pitch_hand = 'R' and pitch_group = 'brk') as brk_woba_denom_vs_r,
    sum(woba_denom) filter (where pitch_hand = 'L' and pitch_group = 'offspeed') as offspeed_woba_denom_vs_l,
    sum(woba_denom) filter (where pitch_hand = 'R' and pitch_group = 'offspeed') as offspeed_woba_denom_vs_r,

    -- === BALLS IN PLAY (denominator for batted ball rates) ===
    count(*) filter (where bb_type is not null and pitch_hand = 'L' and pitch_group = 'fb') as fb_bip_vs_l,
    count(*) filter (where bb_type is not null and pitch_hand = 'R' and pitch_group = 'fb') as fb_bip_vs_r,
    count(*) filter (where bb_type is not null and pitch_hand = 'L' and pitch_group = 'brk') as brk_bip_vs_l,
    count(*) filter (where bb_type is not null and pitch_hand = 'R' and pitch_group = 'brk') as brk_bip_vs_r,
    count(*) filter (where bb_type is not null and pitch_hand = 'L' and pitch_group = 'offspeed') as offspeed_bip_vs_l,
    count(*) filter (where bb_type is not null and pitch_hand = 'R' and pitch_group = 'offspeed') as offspeed_bip_vs_r,

    -- === GROUND BALLS ===
    count(*) filter (where bb_type = 'ground_ball' and pitch_hand = 'L' and pitch_group = 'fb') as fb_gb_vs_l,
    count(*) filter (where bb_type = 'ground_ball' and pitch_hand = 'R' and pitch_group = 'fb') as fb_gb_vs_r,
    count(*) filter (where bb_type = 'ground_ball' and pitch_hand = 'L' and pitch_group = 'brk') as brk_gb_vs_l,
    count(*) filter (where bb_type = 'ground_ball' and pitch_hand = 'R' and pitch_group = 'brk') as brk_gb_vs_r,
    count(*) filter (where bb_type = 'ground_ball' and pitch_hand = 'L' and pitch_group = 'offspeed') as offspeed_gb_vs_l,
    count(*) filter (where bb_type = 'ground_ball' and pitch_hand = 'R' and pitch_group = 'offspeed') as offspeed_gb_vs_r,

    -- === FLY BALLS (includes popup) ===
    count(*) filter (where bb_type in ('fly_ball', 'popup') and pitch_hand = 'L' and pitch_group = 'fb') as fb_fb_vs_l,
    count(*) filter (where bb_type in ('fly_ball', 'popup') and pitch_hand = 'R' and pitch_group = 'fb') as fb_fb_vs_r,
    count(*) filter (where bb_type in ('fly_ball', 'popup') and pitch_hand = 'L' and pitch_group = 'brk') as brk_fb_vs_l,
    count(*) filter (where bb_type in ('fly_ball', 'popup') and pitch_hand = 'R' and pitch_group = 'brk') as brk_fb_vs_r,
    count(*) filter (where bb_type in ('fly_ball', 'popup') and pitch_hand = 'L' and pitch_group = 'offspeed') as offspeed_fb_vs_l,
    count(*) filter (where bb_type in ('fly_ball', 'popup') and pitch_hand = 'R' and pitch_group = 'offspeed') as offspeed_fb_vs_r,

    -- === LINE DRIVES ===
    count(*) filter (where bb_type = 'line_drive' and pitch_hand = 'L' and pitch_group = 'fb') as fb_ld_vs_l,
    count(*) filter (where bb_type = 'line_drive' and pitch_hand = 'R' and pitch_group = 'fb') as fb_ld_vs_r,
    count(*) filter (where bb_type = 'line_drive' and pitch_hand = 'L' and pitch_group = 'brk') as brk_ld_vs_l,
    count(*) filter (where bb_type = 'line_drive' and pitch_hand = 'R' and pitch_group = 'brk') as brk_ld_vs_r,
    count(*) filter (where bb_type = 'line_drive' and pitch_hand = 'L' and pitch_group = 'offspeed') as offspeed_ld_vs_l,
    count(*) filter (where bb_type = 'line_drive' and pitch_hand = 'R' and pitch_group = 'offspeed') as offspeed_ld_vs_r

from pa
group by batter_id, game_date, season
