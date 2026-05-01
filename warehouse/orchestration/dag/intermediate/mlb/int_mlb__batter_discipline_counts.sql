-- int_mlb__batter_discipline_counts.sql
-- Daily batter plate discipline aggregations: chase rate, whiff rate
-- by pitch category and pitcher handedness.
-- Operates on ALL pitches, not just terminal — discipline is a per-pitch metric.
--
-- Pitch type grouping mirrors int_mlb__batter_arsenal_counts:
--   fb       = FF, SI, FC
--   brk      = SL, CU, KC, SV, ST
--   offspeed = CH, FS
--
-- Definitions:
--   Swing = swinging_strike, swinging_strike_blocked, foul, foul_tip,
--           hit_into_play, hit_into_play_no_out, hit_into_play_score
--   Whiff = swinging_strike, swinging_strike_blocked
--   Outside zone = zone IN (11, 12, 13, 14)
--   Chase = swing AND outside zone
--   Bunts excluded (foul_bunt, missed_bunt, bunt_foul_tip).
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

with pitches as (
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
        description in (
            'swinging_strike', 'swinging_strike_blocked',
            'foul', 'foul_tip',
            'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'
        ) as is_swing,
        description in ('swinging_strike', 'swinging_strike_blocked') as is_whiff,
        zone in (11, 12, 13, 14) as is_outside_zone
    from {{ ref('proc_mlb__events') }}
    where pitch_type is not null
      and description not in ('foul_bunt', 'missed_bunt', 'bunt_foul_tip')
    {% if is_incremental() %}
      and game_date > (select max(game_date) from {{ this }})
    {% endif %}
)

select
    batter_id,
    game_date,
    season,

    -- === SWINGS (denominator for whiff rate) ===
    count(*) filter (where pitch_hand = 'L' and pitch_group = 'fb' and is_swing) as fb_swings_vs_l,
    count(*) filter (where pitch_hand = 'R' and pitch_group = 'fb' and is_swing) as fb_swings_vs_r,
    count(*) filter (where pitch_hand = 'L' and pitch_group = 'brk' and is_swing) as brk_swings_vs_l,
    count(*) filter (where pitch_hand = 'R' and pitch_group = 'brk' and is_swing) as brk_swings_vs_r,
    count(*) filter (where pitch_hand = 'L' and pitch_group = 'offspeed' and is_swing) as offspeed_swings_vs_l,
    count(*) filter (where pitch_hand = 'R' and pitch_group = 'offspeed' and is_swing) as offspeed_swings_vs_r,

    -- === WHIFFS (swinging strikes — numerator for whiff rate) ===
    count(*) filter (where pitch_hand = 'L' and pitch_group = 'fb' and is_whiff) as fb_whiffs_vs_l,
    count(*) filter (where pitch_hand = 'R' and pitch_group = 'fb' and is_whiff) as fb_whiffs_vs_r,
    count(*) filter (where pitch_hand = 'L' and pitch_group = 'brk' and is_whiff) as brk_whiffs_vs_l,
    count(*) filter (where pitch_hand = 'R' and pitch_group = 'brk' and is_whiff) as brk_whiffs_vs_r,
    count(*) filter (where pitch_hand = 'L' and pitch_group = 'offspeed' and is_whiff) as offspeed_whiffs_vs_l,
    count(*) filter (where pitch_hand = 'R' and pitch_group = 'offspeed' and is_whiff) as offspeed_whiffs_vs_r,

    -- === OUT-OF-ZONE PITCHES (denominator for chase rate) ===
    count(*) filter (where pitch_hand = 'L' and pitch_group = 'fb' and is_outside_zone) as fb_ooz_vs_l,
    count(*) filter (where pitch_hand = 'R' and pitch_group = 'fb' and is_outside_zone) as fb_ooz_vs_r,
    count(*) filter (where pitch_hand = 'L' and pitch_group = 'brk' and is_outside_zone) as brk_ooz_vs_l,
    count(*) filter (where pitch_hand = 'R' and pitch_group = 'brk' and is_outside_zone) as brk_ooz_vs_r,
    count(*) filter (where pitch_hand = 'L' and pitch_group = 'offspeed' and is_outside_zone) as offspeed_ooz_vs_l,
    count(*) filter (where pitch_hand = 'R' and pitch_group = 'offspeed' and is_outside_zone) as offspeed_ooz_vs_r,

    -- === CHASES (swing on out-of-zone pitch — numerator for chase rate) ===
    count(*) filter (where pitch_hand = 'L' and pitch_group = 'fb' and is_swing and is_outside_zone) as fb_chases_vs_l,
    count(*) filter (where pitch_hand = 'R' and pitch_group = 'fb' and is_swing and is_outside_zone) as fb_chases_vs_r,
    count(*) filter (where pitch_hand = 'L' and pitch_group = 'brk' and is_swing and is_outside_zone) as brk_chases_vs_l,
    count(*) filter (where pitch_hand = 'R' and pitch_group = 'brk' and is_swing and is_outside_zone) as brk_chases_vs_r,
    count(*) filter (where pitch_hand = 'L' and pitch_group = 'offspeed' and is_swing and is_outside_zone) as offspeed_chases_vs_l,
    count(*) filter (where pitch_hand = 'R' and pitch_group = 'offspeed' and is_swing and is_outside_zone) as offspeed_chases_vs_r

from pitches
group by batter_id, game_date, season
