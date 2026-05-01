-- int_mlb__batter_counts.sql
-- Daily raw counting stats per batter, split by pitcher handedness.
-- Maps pa_result values to traditional counting stats.
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
        pa_result,
        woba_value,
        woba_denom,
        launch_speed,
        launch_angle,
        bb_type
    from {{ ref('proc_mlb__events') }}
    where pa_result is not null
      and pa_result != 'truncated_pa'
    {% if is_incremental() %}
      and game_date > (select max(game_date) from {{ this }})
    {% endif %}
)

select
    batter_id,
    game_date,
    season,

    -- === PLATE APPEARANCES ===
    count(*) filter (where pitch_hand = 'L') as pa_vs_l,
    count(*) filter (where pitch_hand = 'R') as pa_vs_r,

    -- === AT BATS (PA minus BB, IBB, HBP, SF, SH) ===
    count(*) filter (where pitch_hand = 'L'
        and pa_result not in ('walk', 'intent_walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt')
    ) as abs_vs_l,
    count(*) filter (where pitch_hand = 'R'
        and pa_result not in ('walk', 'intent_walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt')
    ) as abs_vs_r,

    -- === HITS ===
    count(*) filter (where pitch_hand = 'L'
        and pa_result in ('single', 'double', 'triple', 'home_run')
    ) as hits_vs_l,
    count(*) filter (where pitch_hand = 'R'
        and pa_result in ('single', 'double', 'triple', 'home_run')
    ) as hits_vs_r,

    -- === SINGLES ===
    count(*) filter (where pitch_hand = 'L' and pa_result = 'single') as singles_vs_l,
    count(*) filter (where pitch_hand = 'R' and pa_result = 'single') as singles_vs_r,

    -- === DOUBLES ===
    count(*) filter (where pitch_hand = 'L' and pa_result = 'double') as doubles_vs_l,
    count(*) filter (where pitch_hand = 'R' and pa_result = 'double') as doubles_vs_r,

    -- === TRIPLES ===
    count(*) filter (where pitch_hand = 'L' and pa_result = 'triple') as triples_vs_l,
    count(*) filter (where pitch_hand = 'R' and pa_result = 'triple') as triples_vs_r,

    -- === HOME RUNS ===
    count(*) filter (where pitch_hand = 'L' and pa_result = 'home_run') as hr_vs_l,
    count(*) filter (where pitch_hand = 'R' and pa_result = 'home_run') as hr_vs_r,

    -- === WALKS (includes IBB) ===
    count(*) filter (where pitch_hand = 'L'
        and pa_result in ('walk', 'intent_walk')
    ) as bb_vs_l,
    count(*) filter (where pitch_hand = 'R'
        and pa_result in ('walk', 'intent_walk')
    ) as bb_vs_r,

    -- === INTENTIONAL WALKS ===
    count(*) filter (where pitch_hand = 'L' and pa_result = 'intent_walk') as ibb_vs_l,
    count(*) filter (where pitch_hand = 'R' and pa_result = 'intent_walk') as ibb_vs_r,

    -- === STRIKEOUTS ===
    count(*) filter (where pitch_hand = 'L'
        and pa_result in ('strikeout', 'strikeout_double_play')
    ) as k_vs_l,
    count(*) filter (where pitch_hand = 'R'
        and pa_result in ('strikeout', 'strikeout_double_play')
    ) as k_vs_r,

    -- === HIT BY PITCH ===
    count(*) filter (where pitch_hand = 'L' and pa_result = 'hit_by_pitch') as hbp_vs_l,
    count(*) filter (where pitch_hand = 'R' and pa_result = 'hit_by_pitch') as hbp_vs_r,

    -- === SACRIFICE FLIES ===
    count(*) filter (where pitch_hand = 'L' and pa_result = 'sac_fly') as sf_vs_l,
    count(*) filter (where pitch_hand = 'R' and pa_result = 'sac_fly') as sf_vs_r,

    -- === wOBA (Savant precomputed linear weights) ===
    sum(woba_value) filter (where pitch_hand = 'L') as woba_value_vs_l,
    sum(woba_value) filter (where pitch_hand = 'R') as woba_value_vs_r,
    sum(woba_denom) filter (where pitch_hand = 'L') as woba_denom_vs_l,
    sum(woba_denom) filter (where pitch_hand = 'R') as woba_denom_vs_r,

    -- === EXIT VELOCITY SUM (BIP only — sum not avg for additive cumulation) ===
    sum(launch_speed) filter (where pitch_hand = 'L' and bb_type is not null) as ev_sum_vs_l,
    sum(launch_speed) filter (where pitch_hand = 'R' and bb_type is not null) as ev_sum_vs_r,

    -- === BALLS IN PLAY (denominator for EV, hard hit, barrel) ===
    count(*) filter (where pitch_hand = 'L' and bb_type is not null) as bip_vs_l,
    count(*) filter (where pitch_hand = 'R' and bb_type is not null) as bip_vs_r,

    -- === HARD HIT (EV >= 95 mph) ===
    count(*) filter (where pitch_hand = 'L' and bb_type is not null and launch_speed >= 95) as hard_hit_vs_l,
    count(*) filter (where pitch_hand = 'R' and bb_type is not null and launch_speed >= 95) as hard_hit_vs_r,

    -- === BARREL (Statcast approximation) ===
    count(*) filter (where pitch_hand = 'L' and bb_type is not null
        and launch_speed >= 98
        and launch_angle >= greatest(26 - (launch_speed - 98), 8)
        and launch_angle <= least(30 + (launch_speed - 98), 50)
    ) as barrel_vs_l,
    count(*) filter (where pitch_hand = 'R' and bb_type is not null
        and launch_speed >= 98
        and launch_angle >= greatest(26 - (launch_speed - 98), 8)
        and launch_angle <= least(30 + (launch_speed - 98), 50)
    ) as barrel_vs_r

from pa
group by batter_id, game_date, season
