-- int_mlb__batter_profile.sql
-- Wide batter profile: JOINs batters + batter_arsenal + batter_discipline
-- into a single table for downstream consumers (sim, backtest, prod).
--
-- Grain: (batter_id, game_date)

{{ config(
    materialized='incremental',
    unique_key=['batter_id', 'game_date'],
    incremental_strategy='delete+insert',
    on_schema_change='fail',
) }}

select
    b.*,
    ba.* EXCLUDE (batter_id, game_date, season),
    bd.* EXCLUDE (batter_id, game_date, season)
from {{ ref('int_mlb__batters') }} b
left join {{ ref('int_mlb__batter_arsenal') }} ba
    on b.batter_id = ba.batter_id and b.game_date = ba.game_date
left join {{ ref('int_mlb__batter_discipline') }} bd
    on b.batter_id = bd.batter_id and b.game_date = bd.game_date
{% if is_incremental() %}
where b.game_date > (select max(game_date) from {{ this }})
{% endif %}
