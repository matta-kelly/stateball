-- ana_mlb__event_state.sql
-- Per-pitch state enriched with gumbo validity-window timestamps.
-- Consumer-facing analysis table for backtesting and event-study work.
--
-- Grain: (game_pk, at_bat_number, pitch_number) — one row per pitch.
-- Anchor: prev_event_end_ts (state valid-from) → event_end_ts (state valid-until).

{{
    config(
        materialized='incremental',
        unique_key=['game_pk', 'at_bat_number', 'pitch_number'],
        incremental_strategy='delete+insert',
        on_schema_change='fail'
    )
}}

with state as (
    select * from {{ ref('int_mlb__game_state') }}
),

gumbo as (
    select
        game_pk,
        at_bat_number,
        pitch_number,
        play_id,
        event_start_ts,
        event_end_ts,
        prev_event_end_ts,
        pa_start_ts,
        pa_end_ts
    from {{ ref('proc_mlb__gumbo_events') }}
    where event_type = 'pitch'
      and pitch_number is not null
),

games as (
    select
        game_pk,
        game_type,
        home_team_id,
        home_team_name,
        away_team_id,
        away_team_name
    from {{ ref('proc_mlb__games') }}
),

joined as (
    select
        -- === KEYS ===
        s.game_pk,
        s.at_bat_number,
        s.pitch_number,
        s.game_date,

        -- === TIMESTAMPS (validity window) ===
        g.prev_event_end_ts,
        g.event_end_ts,
        g.event_start_ts,
        g.pa_start_ts,
        g.pa_end_ts,
        g.play_id,

        -- === TEAMS / GAME META ===
        gm.home_team_id,
        gm.home_team_name,
        gm.away_team_id,
        gm.away_team_name,
        gm.game_type,

        -- === MATCHUP ===
        s.batter_id,
        s.pitcher_id,
        s.bat_side,
        s.pitch_hand,

        -- === GAME STATE (pre-pitch) ===
        s.inning,
        s.is_bottom,
        s.outs,
        s.runner_1b,
        s.runner_2b,
        s.runner_3b,
        s.balls,
        s.strikes,
        s.run_diff,
        s.is_home,

        -- === IN-GAME CONTEXT ===
        s.times_through_order,
        s.batter_prior_pa,
        s.pitcher_pitch_count,
        s.pitcher_bf_game,
        s.batter_ab_vs_pitcher,

        -- === PITCHER OUTING ===
        s.pitcher_outing_walks,
        s.pitcher_outing_hits,
        s.pitcher_outing_k,
        s.pitcher_outing_runs,
        s.pitcher_outing_whip,
        s.pitcher_recent_whip,

        -- === TARGET ===
        s.target
    from state s
    inner join gumbo g
        on  s.game_pk       = g.game_pk
        and s.at_bat_number = g.at_bat_number
        and s.pitch_number  = g.pitch_number
    left join games gm
        on s.game_pk = gm.game_pk
)

select * from joined
where game_pk is not null

{% if is_incremental() %}
  and game_pk not in (select distinct game_pk from {{ this }})
{% endif %}

qualify row_number() over (
    partition by game_pk, at_bat_number, pitch_number
    order by event_end_ts desc
) = 1
