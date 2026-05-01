{{
    config(
        materialized='incremental',
        unique_key=['game_pk', 'player_id'],
        incremental_strategy='delete+insert',
        on_schema_change='fail'
    )
}}

select
    game_pk,
    player_id,
    team_id,
    side,
    position,
    batting_order,
    is_starting_pitcher,
    _dlt_load_id

from {{ source('landing', 'boxscores') }}

{% if is_incremental() %}
where game_pk not in (select distinct game_pk from {{ this }})
{% endif %}

qualify row_number() over (
    partition by game_pk, player_id
    order by _dlt_load_id desc
) = 1
