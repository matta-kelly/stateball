{{
    config(
        materialized='incremental',
        unique_key='game_pk',
        incremental_strategy='delete+insert',
        on_schema_change='append_new_columns'
    )
}}

with source as (
    select * from {{ source('landing', 'games') }}
),

renamed as (
    select
        game_pk,
        official_date as game_date,
        game_date as game_datetime,
        season,
        game_type,
        json_extract_string(status, '$.detailedState') as status,
        json_extract_string(status, '$.abstractGameState') as abstract_game_state,
        json_extract(teams, '$.away.team.id')::int as away_team_id,
        json_extract_string(teams, '$.away.team.name') as away_team_name,
        json_extract(teams, '$.home.team.id')::int as home_team_id,
        json_extract_string(teams, '$.home.team.name') as home_team_name,
        json_extract(teams, '$.away.score')::int as away_score,
        json_extract(teams, '$.home.score')::int as home_score,
        json_extract_string(venue, '$.name') as venue_name,
        json_extract(venue, '$.id')::int as venue_id,
        day_night,
        scheduled_innings,
        double_header,
        _dlt_load_id
    from source
)

select * from renamed
where game_type in ('R', 'F', 'D', 'L', 'W', 'S')  -- regular season + postseason + spring training

qualify row_number() over (
    partition by game_pk
    order by _dlt_load_id desc
) = 1
