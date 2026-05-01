import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Activity } from "lucide-react";
import type { Game } from "@/types/api";
import { toDateString } from "@/lib/date";
import { useGames } from "@/hooks/queries";
import { useGameStream, type StreamStatus } from "@/hooks/useGameStream";
import { PageLoading, ErrorBanner } from "@/components/ui/states";
import GameCard from "@/components/GameCard";
import GameRow from "@/components/GameRow";

type ViewMode = "cards" | "rows";
type StateFilter = "all" | "Preview" | "Live" | "Final";

const STATUS_CONFIG: Record<StreamStatus, { color: string; dotClass: string; label: string }> = {
  connected:    { color: "bg-success/10 text-success",  dotClass: "bg-success",                  label: "Live" },
  reconnecting: { color: "bg-warning/10 text-warning",  dotClass: "bg-warning animate-pulse",    label: "Reconnecting..." },
  stale:        { color: "bg-orange-500/10 text-orange-500", dotClass: "bg-orange-500",           label: "Stale" },
  disconnected: { color: "bg-gray-500/10 text-gray-500",    dotClass: "bg-gray-500",             label: "Offline" },
};

export default function LiveGame() {
  const [date, setDate] = useState(() => toDateString(new Date()));
  const [view, setView] = useState<ViewMode>("cards");
  const [filter, setFilter] = useState<StateFilter>("all");

  const isToday = date === toDateString(new Date());

  // REST baseline with polling
  const { data: restGames = [], isLoading, error } = useGames(date, { refetchInterval: 60_000 });

  // SSE stream for today's games
  const { games: streamGames, status: streamStatus, lastEventAt } = useGameStream(isToday);

  // Filter stream games to selected date
  const todayGames = useMemo(
    () => streamGames.filter((g) => g.game_date === date),
    [streamGames, date],
  );

  // Merge: SSE games override REST games by game_pk
  const games = useMemo(() => {
    if (!isToday) return restGames;
    if (todayGames.length > 0) {
      const merged = new Map(restGames.map((g: Game) => [g.game_pk, g]));
      for (const g of todayGames) merged.set(g.game_pk, g);
      return [...merged.values()].sort(
        (a, b) => a.game_datetime.localeCompare(b.game_datetime),
      );
    }
    return restGames;
  }, [isToday, todayGames, restGames]);

  const shiftDate = (days: number) => {
    const d = new Date(date + "T00:00:00");
    d.setDate(d.getDate() + days);
    setDate(toDateString(d));
  };

  const filtered = filter === "all"
    ? games
    : games.filter((g) => g.abstract_game_state === filter);

  const counts = {
    all: games.length,
    Preview: games.filter((g) => g.abstract_game_state === "Preview").length,
    Live: games.filter((g) => g.abstract_game_state === "Live").length,
    Final: games.filter((g) => g.abstract_game_state === "Final").length,
  };

  return (
    <div className="mx-auto w-full max-w-5xl space-y-4 p-4 sm:space-y-6 sm:p-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 sm:gap-4">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold text-foreground sm:text-2xl">Games</h1>
          {isToday && (() => {
            const cfg = STATUS_CONFIG[streamStatus];
            const staleAgo = streamStatus === "stale" && lastEventAt
              ? `${Math.round((Date.now() - lastEventAt) / 1000)}s ago`
              : null;
            return (
              <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs ${cfg.color}`}>
                <span className={`h-1.5 w-1.5 rounded-full ${cfg.dotClass}`} />
                {staleAgo ? `Stale \u2014 ${staleAgo}` : cfg.label}
              </span>
            );
          })()}
        </div>

        {/* Date nav */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => shiftDate(-1)}
            className="rounded-md border border-border bg-card px-3 py-1.5 text-sm text-card-foreground hover:bg-accent"
          >
            Prev
          </button>
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            className="rounded-md border border-border bg-card px-3 py-1.5 text-sm text-card-foreground"
          />
          <button
            onClick={() => shiftDate(1)}
            className="rounded-md border border-border bg-card px-3 py-1.5 text-sm text-card-foreground hover:bg-accent"
          >
            Next
          </button>
          <button
            onClick={() => setDate(toDateString(new Date()))}
            className="rounded-md border border-border bg-card px-2 py-1.5 text-xs text-muted-foreground hover:bg-accent"
          >
            Today
          </button>
        </div>
      </div>

      {/* Quick links */}
      <div className="flex items-center gap-2">
        <Link
          to="/feed-monitor"
          className="inline-flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-xs text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
        >
          <Activity className="h-3.5 w-3.5" />
          Feed Monitor
        </Link>
      </div>

      {/* Filters + view toggle */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex gap-2">
          {(["all", "Preview", "Live", "Final"] as StateFilter[]).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                filter === f
                  ? "bg-primary text-primary-foreground"
                  : "border border-border text-muted-foreground hover:bg-accent"
              }`}
            >
              {f === "all" ? "All" : f} ({counts[f]})
            </button>
          ))}
        </div>

        <div className="flex gap-1 rounded-md border border-border p-0.5">
          <button
            onClick={() => setView("cards")}
            className={`rounded px-2 py-1 text-xs ${
              view === "cards" ? "bg-accent text-foreground" : "text-muted-foreground"
            }`}
          >
            Cards
          </button>
          <button
            onClick={() => setView("rows")}
            className={`rounded px-2 py-1 text-xs ${
              view === "rows" ? "bg-accent text-foreground" : "text-muted-foreground"
            }`}
          >
            List
          </button>
        </div>
      </div>

      {/* Content */}
      {isLoading ? (
        <PageLoading />
      ) : error ? (
        <ErrorBanner message={error.message} />
      ) : filtered.length === 0 ? (
        <p className="text-center text-sm text-muted-foreground">
          No games {filter !== "all" ? `(${filter}) ` : ""}for {date}
        </p>
      ) : view === "cards" ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filtered.map((g) => (
            <GameCard key={g.game_pk} game={g} />
          ))}
        </div>
      ) : (
        <div className="space-y-2">
          {filtered.map((g) => (
            <GameRow key={g.game_pk} game={g} />
          ))}
        </div>
      )}
    </div>
  );
}
