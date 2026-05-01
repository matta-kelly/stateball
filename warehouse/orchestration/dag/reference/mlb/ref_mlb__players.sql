-- ref_mlb__players.sql
-- Player reference/lookup table. Maps player IDs to names, position,
-- handedness, and team. Sourced from MLB Stats API.
--
-- Grain: player_id

{{ config(materialized='table') }}

with source as (
    select
        *,
        row_number() over (partition by player_id order by _dlt_load_id desc) as rn
    from {{ source('landing', 'players') }}
),

deduplicated as (
    select * from source where rn = 1
)

select
    player_id,
    full_name,
    first_name,
    last_name,
    position,
    bats,
    throws,
    team_id,
    team_name,
    active,
    birth_date,
    mlb_debut_date
from deduplicated
