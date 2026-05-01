-- proc_mlb__events.sql
-- Pitch-level event data from Baseball Savant (via pybaseball).
-- One row per pitch. All game state is pre-pitch.
--
-- Grain: (game_pk, at_bat_number, pitch_number)

{{
    config(
        materialized='incremental',
        unique_key=['game_pk', 'at_bat_number', 'pitch_number'],
        incremental_strategy='delete+insert',
        on_schema_change='fail'
    )
}}

-- Columns that may be absent from older Savant game files.
-- UNION ALL BY NAME against a zero-row stub ensures they exist as NULLs.
with schema_stub as (
    select
        NULL::DOUBLE  as spin_axis,
        NULL::DOUBLE  as arm_angle,
        NULL::DOUBLE  as bat_speed,
        NULL::DOUBLE  as swing_length,
        NULL::DOUBLE  as swing_path_tilt,
        NULL::DOUBLE  as attack_angle,
        NULL::DOUBLE  as attack_direction,
        NULL::DOUBLE  as hyper_speed,
        NULL::DOUBLE  as intercept_ball_minus_batter_pos_x_inches,
        NULL::DOUBLE  as intercept_ball_minus_batter_pos_y_inches
    where false
),

source as (
    select *
    from {{ source('landing', 'events') }}
    {% if var('batch_game_pks', none) is not none %}
    where game_pk in ({{ var('batch_game_pks') | join(', ') }})
    {% endif %}
    union all by name
    select * from schema_stub
),

renamed as (
    select
        -- === IDENTITY ===
        game_pk,
        at_bat_number,
        pitch_number,
        game_date,
        game_year,
        game_type,

        -- === MATCHUP ===
        batter                                          as batter_id,
        pitcher                                         as pitcher_id,
        player_name,
        stand                                           as bat_side,
        p_throws                                        as pitch_hand,
        home_team,
        away_team,

        -- === PRE-PITCH SITUATION ===
        inning,
        inning_topbot,
        balls,
        strikes,
        outs_when_up,
        on_1b,
        on_2b,
        on_3b,

        -- === SCORE (pre-pitch and post-pitch) ===
        home_score,
        away_score,
        bat_score,
        fld_score,
        post_home_score,
        post_away_score,
        post_bat_score,
        post_fld_score,
        home_score_diff,
        bat_score_diff,

        -- === PITCH CLASSIFICATION ===
        pitch_type,
        pitch_name,
        "type"                                          as pitch_result_type,
        zone,

        -- === PITCH PHYSICS ===
        release_speed,
        effective_speed,
        release_spin_rate,
        spin_axis,
        release_pos_x,
        release_pos_y,
        release_pos_z,
        release_extension,
        pfx_x,
        pfx_z,
        plate_x,
        plate_z,
        vx0,
        vy0,
        vz0,
        ax,
        ay,
        az,
        sz_top,
        sz_bot,
        api_break_z_with_gravity,
        api_break_x_arm,
        api_break_x_batter_in,
        arm_angle,

        -- === HIT DATA ===
        launch_speed,
        launch_angle,
        launch_speed_angle,
        hit_distance_sc,
        bb_type,
        hit_location,
        hc_x,
        hc_y,

        -- === BATTER SWING ===
        bat_speed,
        swing_length,
        swing_path_tilt,
        attack_angle,
        attack_direction,
        hyper_speed,
        intercept_ball_minus_batter_pos_x_inches,
        intercept_ball_minus_batter_pos_y_inches,

        -- === EVENT OUTCOME ===
        events                                          as pa_result,
        description,
        des,

        -- === SAVANT METRICS ===
        estimated_ba_using_speedangle,
        estimated_woba_using_speedangle,
        estimated_slg_using_speedangle,
        woba_value,
        woba_denom,
        babip_value,
        iso_value,
        delta_run_exp,
        delta_pitcher_run_exp,
        delta_home_win_exp,
        home_win_exp,
        bat_win_exp,

        -- === FIELDING ===
        fielder_2,
        fielder_3,
        fielder_4,
        fielder_5,
        fielder_6,
        fielder_7,
        fielder_8,
        fielder_9,
        if_fielding_alignment,
        of_fielding_alignment,

        -- === PLAYER CONTEXT ===
        age_pit,
        age_bat,
        n_thruorder_pitcher,
        n_priorpa_thisgame_player_at_bat,
        pitcher_days_since_prev_game,
        batter_days_since_prev_game,

        -- === DLT METADATA ===
        _dlt_load_id,
        _dlt_id

    from source
)

select * from renamed
where game_type in ('R', 'F', 'D', 'L', 'W')  -- regular season + postseason only

{% if is_incremental() %}
  and game_pk not in (select distinct game_pk from {{ this }})
{% endif %}

qualify row_number() over (
    partition by game_pk, at_bat_number, pitch_number
    order by _dlt_load_id desc
) = 1
