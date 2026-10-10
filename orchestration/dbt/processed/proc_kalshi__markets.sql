-- proc_kalshi__markets
-- One row per Kalshi MLB market, with the ticker string decoded.
--
--   KXMLBGAME-26AUG06HOUSEA-HOU
--   └─series─┘ └─date─┘└teams┘ └─side─┘
--
-- Teams are concatenated with no delimiter, so the away abbreviation is
-- resolved by longest match against kalshi_team_map and home is whatever
-- remains. Spring training tickers (KXMLBSTGAME) carry an extra HHMM before
-- the teams; doubleheaders carry a trailing G1/G2.
--
-- No game_pk. Matching a market to a real MLB game is a later join against
-- data that is permanent and re-fetchable — this model deliberately stops at
-- date + both teams, which is everything that join needs.
--
-- Materialized as a table, not incremental: landing is merged on ticker, so it
-- already holds one current row per market and there is nothing to deduplicate.
-- A full rebuild is a few thousand rows.

{{ config(materialized='table') }}

with source as (
    select * from {{ source('kalshi', 'markets') }}
),

parsed as (
    select
        ticker,
        event_ticker,
        market_type,
        title,
        status,
        nullif(result, '') as result,
        created_time,
        updated_time,
        open_time,
        close_time,
        settlement_ts,

        case when event_ticker like 'KXMLBSTGAME%' then 'S' else 'R' end as game_type,

        -- Date and teams come off the EVENT ticker (shared by a game's two
        -- markets); which side this market pays on comes off the market ticker.
        substring(event_ticker from 'KXMLB\w+-(\d{2}\w{3}\d{2})') as date_raw,
        substring(event_ticker from 'KXMLB\w+-\d{2}\w{3}\d{2}(\w+)') as teams_raw,
        substring(ticker from 'KXMLB\w+-\w+-(\w+)$') as team_abbrev
    from source
),

cleaned as (
    select
        *,
        -- YYMONDD parses natively in Postgres: MON takes the abbreviated month
        -- name case-insensitively, and YY maps 00-69 to 2000-2069. This is what
        -- the v1 model spent a twelve-branch jinja CASE on.
        to_date(date_raw, 'YYMONDD') as game_date,

        coalesce(
            nullif(substring(teams_raw from 'G(\d)$'), '')::int,
            1
        ) as game_number,

        -- Strip the leading HHMM that spring-training tickers carry, and the
        -- trailing doubleheader marker, leaving just the two abbreviations.
        regexp_replace(
            regexp_replace(teams_raw, '^\d{4}', ''),
            'G\d$', ''
        ) as teams_clean
    from parsed
    where date_raw is not null
),

-- Longest match wins: without a delimiter, SF is a prefix of nothing but SD and
-- SEA both start with S, and CHC/CIN/CLE/CWS all start with C. Ordering by
-- length descending is what makes the split unambiguous.
resolved as (
    select
        c.*,
        away.kalshi_abbrev as away_abbrev,
        away.team_id as away_team_id,
        home.kalshi_abbrev as home_abbrev,
        home.team_id as home_team_id
    from cleaned c

    left join lateral (
        select m.kalshi_abbrev, m.team_id
        from {{ ref('kalshi_team_map') }} m
        where c.teams_clean like m.kalshi_abbrev || '%'
        order by length(m.kalshi_abbrev) desc
        limit 1
    ) away on true

    left join {{ ref('kalshi_team_map') }} home
        on home.kalshi_abbrev = substring(
            c.teams_clean from length(away.kalshi_abbrev) + 1
        )
)

select
    ticker,
    event_ticker,
    game_date,
    game_number,
    game_type,

    away_abbrev,
    away_team_id,
    home_abbrev,
    home_team_id,

    -- Which team a YES on this ticker pays out on.
    team_abbrev,
    case
        when team_abbrev = away_abbrev then 'away'
        when team_abbrev = home_abbrev then 'home'
    end as side,

    status,
    result,
    open_time,
    close_time,
    settlement_ts,
    created_time,
    updated_time,

    market_type,
    title
from resolved
