-- proc_mlb__gumbo_events.sql
-- Pitch + action event data from MLB Gumbo (statsapi /feed/live).
-- One row per playEvents[j] entry (pitches + nested actions like SBs, pickoffs).
-- Primary purpose: per-event timestamps that Savant does not publish.
--
-- Grain: (game_pk, at_bat_number, play_event_index)
-- Pitch rows join to proc_mlb__events on (game_pk, at_bat_number, pitch_number).

{{
    config(
        materialized='incremental',
        unique_key=['game_pk', 'at_bat_number', 'play_event_index'],
        incremental_strategy='delete+insert',
        on_schema_change='fail'
    )
}}

-- Defensive stub for physics fields that may be absent on older games.
with schema_stub as (
    select
        NULL::DOUBLE as pitch_start_speed,
        NULL::DOUBLE as pitch_plate_x,
        NULL::DOUBLE as pitch_plate_z
    where false
),

source as (
    select *
    from {{ source('landing', 'gumbo_events') }}
    {% if var('batch_game_pks', none) is not none %}
    where game_pk in ({{ var('batch_game_pks') | join(', ') }})
    {% endif %}
    union all by name
    select * from schema_stub
),

enriched as (
    select
        -- === IDENTITY / JOIN KEYS ===
        game_pk,
        at_bat_number,
        play_event_index,
        pitch_number,
        play_id,

        -- === EVENT ===
        event_type,
        event_start_ts,
        event_end_ts,
        lag(event_end_ts) over (
            partition by game_pk
            order by at_bat_number, play_event_index
        ) as prev_event_end_ts,
        event_description,
        is_pitch,
        is_in_play,
        is_ball,
        is_strike,

        -- === POST-EVENT COUNT ===
        post_balls,
        post_strikes,
        post_outs,

        -- === PITCH PHYSICS (null for actions) ===
        pitch_start_speed,
        pitch_plate_x,
        pitch_plate_z,

        -- === PA-LEVEL CONTEXT ===
        inning,
        half_inning,
        is_top_inning,
        batter_id,
        pitcher_id,
        pa_event,
        pa_event_type,
        pa_description,
        pa_is_scoring_play,
        pa_has_out,
        pa_rbi,
        pa_away_score,
        pa_home_score,
        pa_post_on_first_id,
        pa_post_on_second_id,
        pa_post_on_third_id,
        pa_start_ts,
        pa_end_ts,

        -- === DLT METADATA ===
        _dlt_load_id,
        _dlt_id
    from source
)

select * from enriched
where game_pk is not null

{% if is_incremental() %}
  and game_pk not in (select distinct game_pk from {{ this }})
{% endif %}

qualify row_number() over (
    partition by game_pk, at_bat_number, play_event_index
    order by _dlt_load_id desc
) = 1
