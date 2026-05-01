-- int_mlb__game_state.sql
-- In-game state features computed from pitch-level event data.
-- One row per pitch. Game state + in-game context + pitcher outing counters + target.
--
-- This is the warehouse mirror of the sim engine's state tracking:
--   GameState  → inning, outs, runners, score, count
--   OutingState → outing counters (walks, hits, k, runs, whip)
--   _build_context() → bf_game, pitch_count, tto, ab_vs_pitcher, recent_whip
--
-- Source: proc_mlb__events
-- Grain: (game_pk, at_bat_number, pitch_number)

{{
    config(
        materialized='incremental',
        unique_key=['game_pk', 'at_bat_number', 'pitch_number'],
        incremental_strategy='delete+insert',
        on_schema_change='fail'
    )
}}

with base as (
    select
        game_pk,
        at_bat_number,
        pitch_number,
        game_date,
        batter_id,
        pitcher_id,
        bat_side,
        pitch_hand,

        -- === GAME STATE ===
        inning,
        case when inning_topbot = 'Bot' then 1 else 0 end as is_bottom,
        outs_when_up as outs,
        case when on_1b is not null then 1 else 0 end as runner_1b,
        case when on_2b is not null then 1 else 0 end as runner_2b,
        case when on_3b is not null then 1 else 0 end as runner_3b,
        balls,
        strikes,
        bat_score_diff as run_diff,
        case when inning_topbot = 'Bot' then 1 else 0 end as is_home,

        -- === IN-GAME CONTEXT ===
        n_thruorder_pitcher as times_through_order,
        n_priorpa_thisgame_player_at_bat as batter_prior_pa,

        row_number() over (
            partition by game_pk, pitcher_id
            order by at_bat_number, pitch_number
        ) as pitcher_pitch_count,

        dense_rank() over (
            partition by game_pk, pitcher_id
            order by at_bat_number
        ) as pitcher_bf_game,

        dense_rank() over (
            partition by game_pk, batter_id, pitcher_id
            order by at_bat_number
        ) - 1 as batter_ab_vs_pitcher,

        -- === PITCHER OUTING COUNTERS ===
        -- pa_result is only set on terminal pitch → each PA counted exactly once.
        count(case when pa_result in ('walk', 'intent_walk') then 1 end) over (
            partition by game_pk, pitcher_id
            order by at_bat_number, pitch_number
            rows between unbounded preceding and 1 preceding
        ) as pitcher_outing_walks,

        count(case when pa_result in ('single', 'double', 'triple', 'home_run') then 1 end) over (
            partition by game_pk, pitcher_id
            order by at_bat_number, pitch_number
            rows between unbounded preceding and 1 preceding
        ) as pitcher_outing_hits,

        count(case when pa_result in ('strikeout', 'strikeout_double_play') then 1 end) over (
            partition by game_pk, pitcher_id
            order by at_bat_number, pitch_number
            rows between unbounded preceding and 1 preceding
        ) as pitcher_outing_k,

        -- Runs: score changes on every pitch (not just terminal — catches WP, balks, etc.)
        sum(coalesce(post_bat_score - bat_score, 0)) over (
            partition by game_pk, pitcher_id
            order by at_bat_number, pitch_number
            rows between unbounded preceding and 1 preceding
        ) as pitcher_outing_runs,

        -- === TARGET ===
        -- pa_result only populated on the terminal pitch of a PA.
        -- MAX propagates the single non-null value to all pitches in the PA.
        max(pa_result) over (partition by game_pk, at_bat_number) as target

    from {{ ref('proc_mlb__events') }}
    {% if var('chunk_start', none) is not none %}
    where game_date between '{{ var("chunk_start") }}' and '{{ var("chunk_end") }}'
    {% elif is_incremental() %}
    where game_pk not in (select distinct game_pk from {{ this }})
    {% endif %}
),

filtered as (
    select * from base
    where target is not null
    and target not in {{ excluded_tuple() }}
),

pitcher_pa as (
    -- One row per PA for each pitcher (dedup from pitch-level)
    select distinct game_pk, pitcher_id, at_bat_number, target
    from filtered
),

pitcher_recent as (
    -- Recent WHIP: (walks + hits) / BF over last 3 PAs faced
    select
        game_pk,
        pitcher_id,
        at_bat_number,
        count(case when target in ('walk', 'intent_walk', 'single', 'double', 'triple', 'home_run') then 1 end)
            over w_recent
            ::double
        / nullif(count(*) over w_recent, 0)
        as pitcher_recent_whip
    from pitcher_pa
    window w_recent as (
        partition by game_pk, pitcher_id
        order by at_bat_number
        rows between 3 preceding and 1 preceding
    )
)

select
    f.game_pk,
    f.at_bat_number,
    f.pitch_number,
    f.game_date,
    f.batter_id,
    f.pitcher_id,
    f.bat_side,
    f.pitch_hand,

    -- Game state
    f.inning,
    f.is_bottom,
    f.outs,
    f.runner_1b,
    f.runner_2b,
    f.runner_3b,
    f.balls,
    f.strikes,
    f.run_diff,
    f.is_home,

    -- In-game context
    f.times_through_order,
    f.batter_prior_pa,
    f.pitcher_pitch_count,
    f.pitcher_bf_game,
    f.batter_ab_vs_pitcher,

    -- Pitcher outing
    f.pitcher_outing_walks,
    f.pitcher_outing_hits,
    f.pitcher_outing_k,
    f.pitcher_outing_runs,
    (f.pitcher_outing_walks + f.pitcher_outing_hits)::double / nullif(f.pitcher_bf_game - 1, 0) as pitcher_outing_whip,
    pr.pitcher_recent_whip,

    -- Target
    f.target

from filtered f
left join pitcher_recent pr
    on f.game_pk = pr.game_pk
    and f.pitcher_id = pr.pitcher_id
    and f.at_bat_number = pr.at_bat_number
