-- feat_mlb__vectors.sql
-- Training vectors for XGBoost PA outcome prediction.
-- One row per pitch. Features + target.
-- Pitch-level: every pitch predicts the PA outcome, conditioned on count.
--
-- Pure three-way join:
--   int_mlb__game_state    — in-game state + outing counters + target
--   int_mlb__batter_profile — ASOF on batter_id (most recent game_date < event)
--   int_mlb__pitcher_profile — ASOF on pitcher_id (most recent game_date < event)
--   Matchup-specific splits selected by opponent handedness.
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

select
    gs.game_pk,
    gs.at_bat_number,
    gs.pitch_number,
    gs.game_date,

    -- === BLOCK 1: GAME STATE (10 features) ===
    gs.inning,
    gs.is_bottom,
    gs.outs,
    gs.runner_1b,
    gs.runner_2b,
    gs.runner_3b,
    gs.balls,
    gs.strikes,
    gs.run_diff,
    gs.is_home,

    -- === BLOCK 2: BATTER PROFILE (20 features) ===
    -- Matchup-specific: batter's stats vs pitcher's throwing hand
    case when gs.pitch_hand = 'L' then bp.season_ba_vs_l else bp.season_ba_vs_r end as bat_season_ba,
    case when gs.pitch_hand = 'L' then bp.career_ba_vs_l else bp.career_ba_vs_r end as bat_career_ba,
    case when gs.pitch_hand = 'L' then bp.season_obp_vs_l else bp.season_obp_vs_r end as bat_season_obp,
    case when gs.pitch_hand = 'L' then bp.career_obp_vs_l else bp.career_obp_vs_r end as bat_career_obp,
    case when gs.pitch_hand = 'L' then bp.season_slg_vs_l else bp.season_slg_vs_r end as bat_season_slg,
    case when gs.pitch_hand = 'L' then bp.career_slg_vs_l else bp.career_slg_vs_r end as bat_career_slg,
    case when gs.pitch_hand = 'L' then bp.season_ops_vs_l else bp.season_ops_vs_r end as bat_season_ops,
    case when gs.pitch_hand = 'L' then bp.career_ops_vs_l else bp.career_ops_vs_r end as bat_career_ops,
    case when gs.pitch_hand = 'L' then bp.season_woba_vs_l else bp.season_woba_vs_r end as bat_season_woba,
    case when gs.pitch_hand = 'L' then bp.career_woba_vs_l else bp.career_woba_vs_r end as bat_career_woba,
    case when gs.pitch_hand = 'L' then bp.season_k_pct_vs_l else bp.season_k_pct_vs_r end as bat_season_k_pct,
    case when gs.pitch_hand = 'L' then bp.career_k_pct_vs_l else bp.career_k_pct_vs_r end as bat_career_k_pct,
    case when gs.pitch_hand = 'L' then bp.season_bb_pct_vs_l else bp.season_bb_pct_vs_r end as bat_season_bb_pct,
    case when gs.pitch_hand = 'L' then bp.career_bb_pct_vs_l else bp.career_bb_pct_vs_r end as bat_career_bb_pct,
    case when gs.pitch_hand = 'L' then bp.season_iso_vs_l else bp.season_iso_vs_r end as bat_season_iso,
    case when gs.pitch_hand = 'L' then bp.career_iso_vs_l else bp.career_iso_vs_r end as bat_career_iso,
    case when gs.pitch_hand = 'L' then bp.season_babip_vs_l else bp.season_babip_vs_r end as bat_season_babip,
    case when gs.pitch_hand = 'L' then bp.career_babip_vs_l else bp.career_babip_vs_r end as bat_career_babip,
    bp.s_pa as bat_s_pa,
    bp.c_pa as bat_c_pa,

    -- === BLOCK 3: PITCHER PROFILE (24 features) ===
    -- Matchup-specific: pitcher's stats vs batter's handedness
    case when gs.bat_side = 'L' then pp.season_whip_vs_l else pp.season_whip_vs_r end as pit_season_whip,
    case when gs.bat_side = 'L' then pp.career_whip_vs_l else pp.career_whip_vs_r end as pit_career_whip,
    case when gs.bat_side = 'L' then pp.season_k9_vs_l else pp.season_k9_vs_r end as pit_season_k9,
    case when gs.bat_side = 'L' then pp.career_k9_vs_l else pp.career_k9_vs_r end as pit_career_k9,
    case when gs.bat_side = 'L' then pp.season_bb9_vs_l else pp.season_bb9_vs_r end as pit_season_bb9,
    case when gs.bat_side = 'L' then pp.career_bb9_vs_l else pp.career_bb9_vs_r end as pit_career_bb9,
    case when gs.bat_side = 'L' then pp.season_hr9_vs_l else pp.season_hr9_vs_r end as pit_season_hr9,
    case when gs.bat_side = 'L' then pp.career_hr9_vs_l else pp.career_hr9_vs_r end as pit_career_hr9,
    case when gs.bat_side = 'L' then pp.season_h9_vs_l else pp.season_h9_vs_r end as pit_season_h9,
    case when gs.bat_side = 'L' then pp.career_h9_vs_l else pp.career_h9_vs_r end as pit_career_h9,
    case when gs.bat_side = 'L' then pp.season_k_pct_vs_l else pp.season_k_pct_vs_r end as pit_season_k_pct,
    case when gs.bat_side = 'L' then pp.career_k_pct_vs_l else pp.career_k_pct_vs_r end as pit_career_k_pct,
    case when gs.bat_side = 'L' then pp.season_bb_pct_vs_l else pp.season_bb_pct_vs_r end as pit_season_bb_pct,
    case when gs.bat_side = 'L' then pp.career_bb_pct_vs_l else pp.career_bb_pct_vs_r end as pit_career_bb_pct,
    case when gs.bat_side = 'L' then pp.season_fip_vs_l else pp.season_fip_vs_r end as pit_season_fip,
    case when gs.bat_side = 'L' then pp.career_fip_vs_l else pp.career_fip_vs_r end as pit_career_fip,
    case when gs.bat_side = 'L' then pp.season_babip_vs_l else pp.season_babip_vs_r end as pit_season_babip,
    case when gs.bat_side = 'L' then pp.career_babip_vs_l else pp.career_babip_vs_r end as pit_career_babip,
    case when gs.bat_side = 'L' then pp.season_woba_vs_l else pp.season_woba_vs_r end as pit_season_woba,
    case when gs.bat_side = 'L' then pp.career_woba_vs_l else pp.career_woba_vs_r end as pit_career_woba,
    case when gs.bat_side = 'L' then pp.season_ip_vs_l else pp.season_ip_vs_r end as pit_season_ip,
    case when gs.bat_side = 'L' then pp.career_ip_vs_l else pp.career_ip_vs_r end as pit_career_ip,
    pp.s_bf as pit_s_bf,
    pp.c_bf as pit_c_bf,

    -- === BLOCK 4: IN-GAME CONTEXT (6 features) ===
    gs.times_through_order,
    pp.pitcher_rest_days as pit_rest_days,
    gs.batter_prior_pa,
    gs.pitcher_pitch_count,
    gs.pitcher_bf_game,
    gs.batter_ab_vs_pitcher,

    -- === BLOCK 5: PITCHER ARSENAL (64 features) ===
    -- Pitch mix: CASE-selected by batter handedness (pitchers change approach vs L/R)
    {% for pt in pitch_types() %}
    case when gs.bat_side = 'L' then pp.season_{{ pt | lower }}_pct_vs_l else pp.season_{{ pt | lower }}_pct_vs_r end as pit_season_{{ pt | lower }}_pct,
    case when gs.bat_side = 'L' then pp.career_{{ pt | lower }}_pct_vs_l else pp.career_{{ pt | lower }}_pct_vs_r end as pit_career_{{ pt | lower }}_pct,
    {% endfor %}

    -- Stuff: velocity (NOT split by hand — physical property of the pitch)
    {% for pt in pitch_types() %}
    pp.season_{{ pt | lower }}_velo as pit_season_{{ pt | lower }}_velo,
    pp.career_{{ pt | lower }}_velo as pit_career_{{ pt | lower }}_velo,
    {% endfor %}

    -- Stuff: spin rate (NOT split by hand)
    {% for pt in pitch_types() %}
    pp.season_{{ pt | lower }}_spin as pit_season_{{ pt | lower }}_spin,
    pp.career_{{ pt | lower }}_spin as pit_career_{{ pt | lower }}_spin,
    {% endfor %}

    -- Delivery (NOT split by hand — biomechanical constants)
    pp.season_arm_angle as pit_season_arm_angle,
    pp.career_arm_angle as pit_career_arm_angle,
    pp.season_extension as pit_season_extension,
    pp.career_extension as pit_career_extension,

    -- === BLOCK 6: PITCHER OUTING PERFORMANCE (6 features) ===
    gs.pitcher_outing_walks,
    gs.pitcher_outing_hits,
    gs.pitcher_outing_k,
    gs.pitcher_outing_runs,
    gs.pitcher_outing_whip,
    gs.pitcher_recent_whip,

    -- === BLOCK 7: BATTER ARSENAL (6 features) ===
    -- Batter wOBA by pitch category, CASE-selected by pitcher handedness
    case when gs.pitch_hand = 'L' then bp.season_woba_vs_fb_vs_l else bp.season_woba_vs_fb_vs_r end as bat_season_woba_vs_fb,
    case when gs.pitch_hand = 'L' then bp.career_woba_vs_fb_vs_l else bp.career_woba_vs_fb_vs_r end as bat_career_woba_vs_fb,
    case when gs.pitch_hand = 'L' then bp.season_woba_vs_brk_vs_l else bp.season_woba_vs_brk_vs_r end as bat_season_woba_vs_brk,
    case when gs.pitch_hand = 'L' then bp.career_woba_vs_brk_vs_l else bp.career_woba_vs_brk_vs_r end as bat_career_woba_vs_brk,
    case when gs.pitch_hand = 'L' then bp.season_woba_vs_offspeed_vs_l else bp.season_woba_vs_offspeed_vs_r end as bat_season_woba_vs_offspeed,
    case when gs.pitch_hand = 'L' then bp.career_woba_vs_offspeed_vs_l else bp.career_woba_vs_offspeed_vs_r end as bat_career_woba_vs_offspeed,

    -- === BLOCK 8: BATTER BATTED BALL PROFILE (18 features) ===
    -- GB% by pitch category, CASE-selected by pitcher handedness
    case when gs.pitch_hand = 'L' then bp.season_gb_pct_vs_fb_vs_l else bp.season_gb_pct_vs_fb_vs_r end as bat_season_gb_pct_vs_fb,
    case when gs.pitch_hand = 'L' then bp.career_gb_pct_vs_fb_vs_l else bp.career_gb_pct_vs_fb_vs_r end as bat_career_gb_pct_vs_fb,
    case when gs.pitch_hand = 'L' then bp.season_gb_pct_vs_brk_vs_l else bp.season_gb_pct_vs_brk_vs_r end as bat_season_gb_pct_vs_brk,
    case when gs.pitch_hand = 'L' then bp.career_gb_pct_vs_brk_vs_l else bp.career_gb_pct_vs_brk_vs_r end as bat_career_gb_pct_vs_brk,
    case when gs.pitch_hand = 'L' then bp.season_gb_pct_vs_offspeed_vs_l else bp.season_gb_pct_vs_offspeed_vs_r end as bat_season_gb_pct_vs_offspeed,
    case when gs.pitch_hand = 'L' then bp.career_gb_pct_vs_offspeed_vs_l else bp.career_gb_pct_vs_offspeed_vs_r end as bat_career_gb_pct_vs_offspeed,

    -- FB% by pitch category, CASE-selected by pitcher handedness
    case when gs.pitch_hand = 'L' then bp.season_fb_pct_vs_fb_vs_l else bp.season_fb_pct_vs_fb_vs_r end as bat_season_fb_pct_vs_fb,
    case when gs.pitch_hand = 'L' then bp.career_fb_pct_vs_fb_vs_l else bp.career_fb_pct_vs_fb_vs_r end as bat_career_fb_pct_vs_fb,
    case when gs.pitch_hand = 'L' then bp.season_fb_pct_vs_brk_vs_l else bp.season_fb_pct_vs_brk_vs_r end as bat_season_fb_pct_vs_brk,
    case when gs.pitch_hand = 'L' then bp.career_fb_pct_vs_brk_vs_l else bp.career_fb_pct_vs_brk_vs_r end as bat_career_fb_pct_vs_brk,
    case when gs.pitch_hand = 'L' then bp.season_fb_pct_vs_offspeed_vs_l else bp.season_fb_pct_vs_offspeed_vs_r end as bat_season_fb_pct_vs_offspeed,
    case when gs.pitch_hand = 'L' then bp.career_fb_pct_vs_offspeed_vs_l else bp.career_fb_pct_vs_offspeed_vs_r end as bat_career_fb_pct_vs_offspeed,

    -- LD% by pitch category, CASE-selected by pitcher handedness
    case when gs.pitch_hand = 'L' then bp.season_ld_pct_vs_fb_vs_l else bp.season_ld_pct_vs_fb_vs_r end as bat_season_ld_pct_vs_fb,
    case when gs.pitch_hand = 'L' then bp.career_ld_pct_vs_fb_vs_l else bp.career_ld_pct_vs_fb_vs_r end as bat_career_ld_pct_vs_fb,
    case when gs.pitch_hand = 'L' then bp.season_ld_pct_vs_brk_vs_l else bp.season_ld_pct_vs_brk_vs_r end as bat_season_ld_pct_vs_brk,
    case when gs.pitch_hand = 'L' then bp.career_ld_pct_vs_brk_vs_l else bp.career_ld_pct_vs_brk_vs_r end as bat_career_ld_pct_vs_brk,
    case when gs.pitch_hand = 'L' then bp.season_ld_pct_vs_offspeed_vs_l else bp.season_ld_pct_vs_offspeed_vs_r end as bat_season_ld_pct_vs_offspeed,
    case when gs.pitch_hand = 'L' then bp.career_ld_pct_vs_offspeed_vs_l else bp.career_ld_pct_vs_offspeed_vs_r end as bat_career_ld_pct_vs_offspeed,

    -- === BLOCK 9: BATTER CONTACT QUALITY (6 features) ===
    case when gs.pitch_hand = 'L' then bp.season_avg_ev_vs_l else bp.season_avg_ev_vs_r end as bat_season_avg_ev,
    case when gs.pitch_hand = 'L' then bp.career_avg_ev_vs_l else bp.career_avg_ev_vs_r end as bat_career_avg_ev,
    case when gs.pitch_hand = 'L' then bp.season_hard_hit_pct_vs_l else bp.season_hard_hit_pct_vs_r end as bat_season_hard_hit_pct,
    case when gs.pitch_hand = 'L' then bp.career_hard_hit_pct_vs_l else bp.career_hard_hit_pct_vs_r end as bat_career_hard_hit_pct,
    case when gs.pitch_hand = 'L' then bp.season_barrel_pct_vs_l else bp.season_barrel_pct_vs_r end as bat_season_barrel_pct,
    case when gs.pitch_hand = 'L' then bp.career_barrel_pct_vs_l else bp.career_barrel_pct_vs_r end as bat_career_barrel_pct,

    -- === BLOCK 10: BATTER PLATE DISCIPLINE (12 features) ===
    -- Chase rate by pitch category, CASE-selected by pitcher handedness
    case when gs.pitch_hand = 'L' then bp.season_chase_rate_vs_fb_vs_l else bp.season_chase_rate_vs_fb_vs_r end as bat_season_chase_rate_vs_fb,
    case when gs.pitch_hand = 'L' then bp.career_chase_rate_vs_fb_vs_l else bp.career_chase_rate_vs_fb_vs_r end as bat_career_chase_rate_vs_fb,
    case when gs.pitch_hand = 'L' then bp.season_chase_rate_vs_brk_vs_l else bp.season_chase_rate_vs_brk_vs_r end as bat_season_chase_rate_vs_brk,
    case when gs.pitch_hand = 'L' then bp.career_chase_rate_vs_brk_vs_l else bp.career_chase_rate_vs_brk_vs_r end as bat_career_chase_rate_vs_brk,
    case when gs.pitch_hand = 'L' then bp.season_chase_rate_vs_offspeed_vs_l else bp.season_chase_rate_vs_offspeed_vs_r end as bat_season_chase_rate_vs_offspeed,
    case when gs.pitch_hand = 'L' then bp.career_chase_rate_vs_offspeed_vs_l else bp.career_chase_rate_vs_offspeed_vs_r end as bat_career_chase_rate_vs_offspeed,

    -- Whiff rate by pitch category, CASE-selected by pitcher handedness
    case when gs.pitch_hand = 'L' then bp.season_whiff_rate_vs_fb_vs_l else bp.season_whiff_rate_vs_fb_vs_r end as bat_season_whiff_rate_vs_fb,
    case when gs.pitch_hand = 'L' then bp.career_whiff_rate_vs_fb_vs_l else bp.career_whiff_rate_vs_fb_vs_r end as bat_career_whiff_rate_vs_fb,
    case when gs.pitch_hand = 'L' then bp.season_whiff_rate_vs_brk_vs_l else bp.season_whiff_rate_vs_brk_vs_r end as bat_season_whiff_rate_vs_brk,
    case when gs.pitch_hand = 'L' then bp.career_whiff_rate_vs_brk_vs_l else bp.career_whiff_rate_vs_brk_vs_r end as bat_career_whiff_rate_vs_brk,
    case when gs.pitch_hand = 'L' then bp.season_whiff_rate_vs_offspeed_vs_l else bp.season_whiff_rate_vs_offspeed_vs_r end as bat_season_whiff_rate_vs_offspeed,
    case when gs.pitch_hand = 'L' then bp.career_whiff_rate_vs_offspeed_vs_l else bp.career_whiff_rate_vs_offspeed_vs_r end as bat_career_whiff_rate_vs_offspeed,

    -- === TARGET ===
    gs.target

from {{ ref('int_mlb__game_state') }} gs
asof join {{ ref('int_mlb__batter_profile') }} bp
    on gs.batter_id = bp.batter_id
    and gs.game_date > bp.game_date
asof join {{ ref('int_mlb__pitcher_profile') }} pp
    on gs.pitcher_id = pp.pitcher_id
    and gs.game_date > pp.game_date
{% if is_incremental() %}
where gs.game_pk not in (select distinct game_pk from {{ this }})
{% endif %}
