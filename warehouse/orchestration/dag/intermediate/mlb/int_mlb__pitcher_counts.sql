-- int_mlb__pitcher_counts.sql
-- Daily raw counting stats per pitcher, split by batter handedness.
-- Maps pa_result values to traditional counting stats.
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

with pa as (
    select
        pitcher_id,
        game_date,
        game_year as season,
        bat_side,
        pa_result,
        woba_value,
        woba_denom
    from {{ ref('proc_mlb__events') }}
    where pa_result is not null
      and pa_result != 'truncated_pa'
    {% if is_incremental() %}
      and game_date > (select max(game_date) from {{ this }})
    {% endif %}
),

-- Total pitches (all pitches, not just terminal PAs)
pitch_counts as (
    select
        pitcher_id,
        game_date,
        count(*) as pitches
    from {{ ref('proc_mlb__events') }}
    {% if is_incremental() %}
    where game_date > (select max(game_date) from {{ this }})
    {% endif %}
    group by pitcher_id, game_date
)

select
    pa.pitcher_id,
    pa.game_date,
    pa.season,

    -- === BATTERS FACED ===
    count(*) filter (where bat_side = 'L') as bf_vs_l,
    count(*) filter (where bat_side = 'R') as bf_vs_r,

    -- === OUTS RECORDED ===
    -- 2 outs: double plays. 1 out: field_out, strikeout, force_out, fielders_choice_out, sac_fly, sac_bunt
    (
        count(*) filter (where bat_side = 'L'
            and pa_result in ('field_out', 'strikeout', 'force_out', 'fielders_choice_out', 'sac_fly', 'sac_bunt'))
        + 2 * count(*) filter (where bat_side = 'L'
            and pa_result in ('grounded_into_double_play', 'double_play', 'strikeout_double_play'))
    ) as outs_vs_l,
    (
        count(*) filter (where bat_side = 'R'
            and pa_result in ('field_out', 'strikeout', 'force_out', 'fielders_choice_out', 'sac_fly', 'sac_bunt'))
        + 2 * count(*) filter (where bat_side = 'R'
            and pa_result in ('grounded_into_double_play', 'double_play', 'strikeout_double_play'))
    ) as outs_vs_r,

    -- === HITS ALLOWED ===
    count(*) filter (where bat_side = 'L'
        and pa_result in ('single', 'double', 'triple', 'home_run')
    ) as hits_vs_l,
    count(*) filter (where bat_side = 'R'
        and pa_result in ('single', 'double', 'triple', 'home_run')
    ) as hits_vs_r,

    -- === SINGLES ===
    count(*) filter (where bat_side = 'L' and pa_result = 'single') as singles_vs_l,
    count(*) filter (where bat_side = 'R' and pa_result = 'single') as singles_vs_r,

    -- === DOUBLES ===
    count(*) filter (where bat_side = 'L' and pa_result = 'double') as doubles_vs_l,
    count(*) filter (where bat_side = 'R' and pa_result = 'double') as doubles_vs_r,

    -- === TRIPLES ===
    count(*) filter (where bat_side = 'L' and pa_result = 'triple') as triples_vs_l,
    count(*) filter (where bat_side = 'R' and pa_result = 'triple') as triples_vs_r,

    -- === HOME RUNS ===
    count(*) filter (where bat_side = 'L' and pa_result = 'home_run') as hr_vs_l,
    count(*) filter (where bat_side = 'R' and pa_result = 'home_run') as hr_vs_r,

    -- === WALKS (includes IBB) ===
    count(*) filter (where bat_side = 'L'
        and pa_result in ('walk', 'intent_walk')
    ) as bb_vs_l,
    count(*) filter (where bat_side = 'R'
        and pa_result in ('walk', 'intent_walk')
    ) as bb_vs_r,

    -- === INTENTIONAL WALKS ===
    count(*) filter (where bat_side = 'L' and pa_result = 'intent_walk') as ibb_vs_l,
    count(*) filter (where bat_side = 'R' and pa_result = 'intent_walk') as ibb_vs_r,

    -- === STRIKEOUTS ===
    count(*) filter (where bat_side = 'L'
        and pa_result in ('strikeout', 'strikeout_double_play')
    ) as k_vs_l,
    count(*) filter (where bat_side = 'R'
        and pa_result in ('strikeout', 'strikeout_double_play')
    ) as k_vs_r,

    -- === HIT BY PITCH ===
    count(*) filter (where bat_side = 'L' and pa_result = 'hit_by_pitch') as hbp_vs_l,
    count(*) filter (where bat_side = 'R' and pa_result = 'hit_by_pitch') as hbp_vs_r,

    -- === SACRIFICE FLIES ===
    count(*) filter (where bat_side = 'L' and pa_result = 'sac_fly') as sf_vs_l,
    count(*) filter (where bat_side = 'R' and pa_result = 'sac_fly') as sf_vs_r,

    -- === SACRIFICE BUNTS ===
    count(*) filter (where bat_side = 'L' and pa_result = 'sac_bunt') as sh_vs_l,
    count(*) filter (where bat_side = 'R' and pa_result = 'sac_bunt') as sh_vs_r,

    -- === wOBA against (Savant precomputed linear weights) ===
    sum(woba_value) filter (where bat_side = 'L') as woba_value_vs_l,
    sum(woba_value) filter (where bat_side = 'R') as woba_value_vs_r,
    sum(woba_denom) filter (where bat_side = 'L') as woba_denom_vs_l,
    sum(woba_denom) filter (where bat_side = 'R') as woba_denom_vs_r,

    -- === PITCH COUNT + APPEARANCES ===
    coalesce(pc.pitches, 0) as pitches,
    1 as games

from pa
left join pitch_counts pc
    on pa.pitcher_id = pc.pitcher_id and pa.game_date = pc.game_date
group by pa.pitcher_id, pa.game_date, pa.season, pc.pitches
