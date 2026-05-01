import { Link } from "react-router-dom";
import type { Game } from "@/types/api";
import { formatGameTime } from "@/lib/time";
import StatusBadge from "./StatusBadge";

interface GameRowProps {
  game: Game;
}

export default function GameRow({ game }: GameRowProps) {
  const isLive = game.abstract_game_state === "Live";
  const isFinal = game.abstract_game_state === "Final";
  const showScore = isLive || isFinal;
  const isTop = game.inning_half === "Top" || game.inning_half === "top";

  return (
    <Link
      to={`/game/${game.game_pk}`}
      className="flex items-center gap-4 rounded-lg border border-border bg-card px-4 py-3 shadow-sm transition-colors hover:bg-accent/50"
    >
      {/* Status + time on same line */}
      <div className="flex shrink-0 items-center gap-2">
        <StatusBadge abstractState={game.abstract_game_state} status={game.status} />
        <span className="whitespace-nowrap text-xs text-muted-foreground">
          {formatGameTime(game.game_datetime)}
        </span>
      </div>

      {/* Teams + scores + inning arrow on batting team */}
      <div className="min-w-0 space-y-0.5">
        <div className="flex items-center gap-2">
          <span className={`text-sm text-card-foreground ${isLive && isTop ? "font-bold" : "font-medium"}`}>
            {game.away_team_name}
          </span>
          {showScore && (
            <span className="text-sm font-bold tabular-nums text-card-foreground">
              {game.away_score ?? 0}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-sm text-card-foreground ${isLive && !isTop ? "font-bold" : "font-medium"}`}>
            {game.home_team_name}
          </span>
          {showScore && (
            <span className="text-sm font-bold tabular-nums text-card-foreground">
              {game.home_score ?? 0}
            </span>
          )}
        </div>
      </div>

      {/* Inning + outs — centered, next to scores */}
      {isLive && game.inning != null && (
        <span className="shrink-0 whitespace-nowrap text-xs text-muted-foreground">
          {isTop ? "▲" : "▼"} {game.inning}
        </span>
      )}
      {isLive && game.outs != null && (
        <span className="shrink-0 text-xs text-muted-foreground">
          {game.outs} out
        </span>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Venue */}
      {game.venue_name && (
        <span className="hidden shrink-0 text-xs text-muted-foreground lg:block">
          {game.venue_name}
        </span>
      )}
    </Link>
  );
}
