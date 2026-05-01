-- int_mlb__pitcher_profile.sql
-- Wide pitcher profile: JOINs pitchers + pitcher_arsenal
-- into a single table for downstream consumers (sim, backtest, prod).
--
-- Grain: (pitcher_id, game_date)

{{ config(
    materialized='incremental',
    unique_key=['pitcher_id', 'game_date'],
    incremental_strategy='delete+insert',
    on_schema_change='fail',
) }}

select
    p.*,
    pa.* EXCLUDE (pitcher_id, game_date, season)
from {{ ref('int_mlb__pitchers') }} p
left join {{ ref('int_mlb__pitcher_arsenal') }} pa
    on p.pitcher_id = pa.pitcher_id and p.game_date = pa.game_date
{% if is_incremental() %}
where p.game_date > (select max(game_date) from {{ this }})
{% endif %}
