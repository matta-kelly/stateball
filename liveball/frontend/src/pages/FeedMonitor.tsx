import { useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import type { FeedAlert, FeedHealth, PollHistoryDetail, PollHistoryGame } from "@/types/api";
import { useFeedHealth, useFeedHistory, useFeedGameHistory } from "@/hooks/queries";
import { MetricCard } from "@/components/MetricCard";
import { ErrorBanner } from "@/components/ui/states";

// --- Health panel ---

function SummaryBar({ health }: { health: FeedHealth }) {
  const { summary } = health;
  const errorColor = summary.error_rate_pct > 5 ? "text-destructive" : "text-foreground";
  return (
    <div className="flex flex-wrap gap-6 rounded-lg border border-border bg-card px-6 py-3">
      <MetricCard label="Active Trackers" value={String(summary.active_trackers)} color="text-success" />
      <MetricCard label="Total Trackers" value={String(summary.total_trackers)} />
      <MetricCard label="Total Polls" value={summary.total_polls.toLocaleString()} />
      <MetricCard label="Error Rate" value={`${summary.error_rate_pct}%`} color={errorColor} />
    </div>
  );
}

function AlertList({ alerts }: { alerts: FeedAlert[] }) {
  if (alerts.length === 0) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-border bg-card px-6 py-4 text-sm text-muted-foreground">
        <span className="h-2 w-2 rounded-full bg-success" />
        All trackers healthy
      </div>
    );
  }
  return (
    <div className="space-y-2">
      {alerts.map((a, i) => (
        <div
          key={i}
          className={`flex items-start gap-3 rounded-lg border px-4 py-3 text-sm ${
            a.level === "error"
              ? "border-destructive/50 bg-destructive/10 text-destructive"
              : "border-warning/50 bg-warning/10 text-warning"
          }`}
        >
          <span className="mt-0.5 font-mono text-xs uppercase opacity-70">{a.level}</span>
          <div>
            <span className="font-mono text-xs text-muted-foreground mr-2">{a.game_pk}</span>
            {a.message}
          </div>
        </div>
      ))}
    </div>
  );
}

// --- History panel ---

function HistoryTable({
  games,
  selectedPk,
  onSelect,
}: {
  games: PollHistoryGame[];
  selectedPk: number | null;
  onSelect: (g: PollHistoryGame) => void;
}) {
  if (games.length === 0) {
    return (
      <p className="py-4 text-center text-sm text-muted-foreground">
        No completed game logs yet — logs are written when a tracked game finalizes.
      </p>
    );
  }
  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border bg-muted/40 text-xs text-muted-foreground">
            <th className="px-4 py-2 text-left">Date</th>
            <th className="px-4 py-2 text-left">Game PK</th>
          </tr>
        </thead>
        <tbody>
          {games.map((g) => (
            <tr
              key={`${g.game_date}-${g.game_pk}`}
              onClick={() => onSelect(g)}
              className={`cursor-pointer border-b border-border transition-colors last:border-0 hover:bg-muted/30 ${
                selectedPk === g.game_pk ? "bg-muted/50" : ""
              }`}
            >
              <td className="px-4 py-2 font-mono text-xs text-muted-foreground">{g.game_date}</td>
              <td className="px-4 py-2 font-mono">{g.game_pk}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EventBreakdown({ detail }: { detail: PollHistoryDetail }) {
  const events = Object.entries(detail.polls_per_change_by_event).sort((a, b) => b[1] - a[1]);

  return (
    <div className="rounded-lg border border-border bg-card p-4 space-y-4">
      <div className="flex flex-wrap gap-6">
        <MetricCard label="Total Polls" value={detail.total_polls.toLocaleString()} />
        <MetricCard
          label="Change Rate"
          value={`${(detail.change_rate * 100).toFixed(1)}%`}
          color="text-success"
        />
        <MetricCard
          label="Wasted Polls"
          value={`${detail.wasted_poll_pct.toFixed(1)}%`}
          color={detail.wasted_poll_pct > 80 ? "text-warning" : "text-foreground"}
        />
        <MetricCard
          label="Error Rate"
          value={`${(detail.error_rate * 100).toFixed(1)}%`}
          color={detail.error_rate > 0.05 ? "text-destructive" : "text-foreground"}
        />
        <MetricCard label="Avg Response" value={`${detail.avg_response_ms}ms`} />
      </div>

      {events.length > 0 && (
        <div>
          <p className="mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wide">
            Median idle polls after event type
          </p>
          <div className="space-y-1.5">
            {events.map(([et, median]) => (
              <div key={et} className="flex items-center gap-3">
                <span className="w-48 truncate text-xs font-mono text-muted-foreground">{et}</span>
                <div className="flex-1 rounded-full bg-muted h-1.5">
                  <div
                    className="h-1.5 rounded-full bg-info"
                    style={{ width: `${Math.min((median / 20) * 100, 100)}%` }}
                  />
                </div>
                <span className="w-10 text-right text-xs tabular-nums">{median}</span>
              </div>
            ))}
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            Higher = more polls wasted waiting for a change after this event. Target for adaptive interval tuning.
          </p>
        </div>
      )}
    </div>
  );
}

// --- Main ---

export default function FeedMonitor() {
  const { data: health, error: healthError } = useFeedHealth();
  const { data: historyGames = [], isLoading: historyLoading } = useFeedHistory();

  const [selectedGame, setSelectedGame] = useState<PollHistoryGame | null>(null);
  const { data: detail, isLoading: detailLoading, error: detailError } = useFeedGameHistory(
    selectedGame?.game_pk ?? 0,
    selectedGame?.game_date ?? "",
  );

  return (
    <div className="mx-auto w-full max-w-5xl space-y-6 p-4 sm:p-6">
      <Link to="/" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="h-4 w-4" /> Back to games
      </Link>
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold sm:text-2xl">Feed Monitor</h1>
        <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-success" />
          Health every 10s
        </span>
      </div>

      {/* Health section */}
      <section className="space-y-3">
        <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
          System Health
        </h2>
        {healthError ? (
          <ErrorBanner message={healthError.message} />
        ) : health ? (
          <>
            <SummaryBar health={health} />
            <AlertList alerts={health.alerts} />
          </>
        ) : (
          <p className="text-sm text-muted-foreground">Loading...</p>
        )}
      </section>

      {/* History section */}
      <section className="space-y-3">
        <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
          Completed Game Logs
        </h2>
        {historyLoading ? (
          <p className="text-sm text-muted-foreground">Loading...</p>
        ) : (
          <div className="grid gap-4 lg:grid-cols-2">
            <HistoryTable
              games={historyGames}
              selectedPk={selectedGame?.game_pk ?? null}
              onSelect={setSelectedGame}
            />
            <div>
              {detailLoading && (
                <p className="text-sm text-muted-foreground">Loading...</p>
              )}
              {detailError && (
                <ErrorBanner message={detailError.message} />
              )}
              {detail && !detailLoading && (
                <EventBreakdown detail={detail} />
              )}
              {!detail && !detailLoading && !detailError && (
                <p className="text-sm text-muted-foreground py-4">
                  Select a game to see poll analysis.
                </p>
              )}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
