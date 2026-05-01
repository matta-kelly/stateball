import { Link } from "react-router-dom";
import type { Game } from "@/types/api";
import { formatGameTime } from "@/lib/time";
import { Bases, Outs, Count, InningLabel } from "@/components/baseball/Situation";

/** Last word of a team name — e.g. "Los Angeles Angels" → "Angels". */
function shortName(name: string): string {
  const parts = name.split(" ");
  return parts[parts.length - 1];
}

function LiveSection({ game }: { game: Game }) {
  const isTop = game.inning_half === "Top" || game.inning_half === "top";
  const pitchingTeam = shortName(isTop ? game.home_team_name : game.away_team_name);
  const battingTeam = shortName(isTop ? game.away_team_name : game.home_team_name);

  return (
    <div className="mt-3 flex items-center justify-between border-t border-border pt-3">
      <div className="flex flex-col gap-1">
        <InningLabel inning={game.inning ?? null} half={game.inning_half} />
        {game.current_pitcher_name && (
          <div className="text-xs text-muted-foreground">
            <span className={`font-medium ${isTop ? "text-info" : "text-zinc-300"}`}>
              {pitchingTeam}
            </span>{" "}
            {game.current_pitcher_name}
          </div>
        )}
        {game.current_batter_name && (
          <div className="text-xs text-muted-foreground">
            <span className={`font-medium ${isTop ? "text-zinc-300" : "text-info"}`}>
              {battingTeam}
            </span>{" "}
            {game.current_batter_name}
          </div>
        )}
      </div>

      <div className="flex items-center gap-3">
        <Bases runners={game.runners} />
        <div className="flex flex-col items-center gap-1">
          <Outs count={game.outs} />
          <Count balls={game.balls} strikes={game.strikes} />
        </div>
      </div>
    </div>
  );
}

export default function GameCard({ game }: { game: Game }) {
  const isLive = game.abstract_game_state === "Live";
  const isFinal = game.abstract_game_state === "Final";
  const showScore = isLive || isFinal;

  return (
    <Link
      to={`/game/${game.game_pk}`}
      className={`block rounded-lg border bg-card p-4 shadow-sm transition-colors hover:bg-accent/50 ${
        isLive ? "border-success/30" : "border-border"
      }`}
    >
      {/* Status row */}
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isLive ? (
            <span className="flex items-center gap-1.5 text-xs font-semibold text-success">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-success" />
              LIVE
            </span>
          ) : isFinal ? (
            <span className="text-xs font-medium text-zinc-500">FINAL</span>
          ) : (
            <span className="text-xs font-medium text-info">
              {game.status}
            </span>
          )}
        </div>
        <span className="text-xs text-muted-foreground">
          {formatGameTime(game.game_datetime)}
        </span>
      </div>

      {/* Scores */}
      <div className="space-y-1.5">
        {(() => {
          const isTop = game.inning_half === "Top" || game.inning_half === "top";
          const awayBatting = isLive && isTop;
          const homeBatting = isLive && !isTop;
          return (
            <>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span
                    className={`text-sm text-card-foreground ${
                      awayBatting ? "font-bold" : "font-medium"
                    }`}
                  >
                    {game.away_team_name}
                  </span>
                  {awayBatting && (
                    <span className="text-[10px] text-zinc-400">▲</span>
                  )}
                </span>
                {showScore && (
                  <span className="text-lg font-bold tabular-nums text-card-foreground">
                    {game.away_score ?? 0}
                  </span>
                )}
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span
                    className={`text-sm text-card-foreground ${
                      homeBatting ? "font-bold" : "font-medium"
                    }`}
                  >
                    {game.home_team_name}
                  </span>
                  {homeBatting && (
                    <span className="text-[10px] text-zinc-400">▼</span>
                  )}
                </span>
                {showScore && (
                  <span className="text-lg font-bold tabular-nums text-card-foreground">
                    {game.home_score ?? 0}
                  </span>
                )}
              </div>
            </>
          );
        })()}
      </div>

      {/* Live detail section */}
      {isLive && <LiveSection game={game} />}

      {/* WE baseline */}
      {game.sim_we_baseline != null && (
        <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
          <span>WE Table</span>
          <span className="font-mono tabular-nums text-foreground">{(game.sim_we_baseline * 100).toFixed(1)}%</span>
        </div>
      )}

      {/* Venue for non-live */}
      {!isLive && game.venue_name && (
        <p className="mt-3 text-xs text-muted-foreground">{game.venue_name}</p>
      )}

      {/* Last play */}
      {isLive && game.last_play && (
        <p
          className="mt-2 truncate text-xs text-zinc-500"
          title={game.last_play}
        >
          {game.last_play}
        </p>
      )}
    </Link>
  );
}
