import { useState } from "react";
import { cn } from "@/lib/utils";
import { useStatsSeasons, useStatsBatters, useStatsPitchers } from "@/hooks/queries";
import BatterStatsTable from "./stats/BatterStatsTable";
import PitcherStatsTable from "./stats/PitcherStatsTable";

const PAGE_SIZE = 50;

type Tab = "batters" | "pitchers";

export default function Stats() {
  const [tab, setTab] = useState<Tab>("batters");
  const [season, setSeason] = useState(() => new Date().getFullYear());
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);

  const { data: seasons } = useStatsSeasons();
  const batters = useStatsBatters(tab === "batters" ? season : undefined, page * PAGE_SIZE, PAGE_SIZE);
  const pitchers = useStatsPitchers(tab === "pitchers" ? season : undefined, page * PAGE_SIZE, PAGE_SIZE);

  const isLoading = tab === "batters" ? batters.isLoading : pitchers.isLoading;
  const error = tab === "batters" ? batters.error : pitchers.error;
  const dataLen = tab === "batters" ? (batters.data?.length ?? 0) : (pitchers.data?.length ?? 0);

  return (
    <div className="mx-auto w-full space-y-4">
      {/* Header row */}
      <div className="flex flex-wrap items-center gap-4">
        <h2 className="text-xl font-semibold">Stats</h2>

        {/* Batters / Pitchers toggle */}
        <div className="flex items-center gap-0.5 rounded-md bg-muted p-0.5">
          <button
            onClick={() => { setTab("batters"); setPage(0); }}
            className={cn(
              "rounded px-2.5 py-1 text-xs font-medium transition-colors",
              tab === "batters"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            Batters
          </button>
          <button
            onClick={() => { setTab("pitchers"); setPage(0); }}
            className={cn(
              "rounded px-2.5 py-1 text-xs font-medium transition-colors",
              tab === "pitchers"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            Pitchers
          </button>
        </div>

        {/* Season selector */}
        <select
          value={season}
          onChange={(e) => { setSeason(Number(e.target.value)); setPage(0); }}
          className="rounded-md border border-border bg-card px-2 py-1 text-xs text-foreground"
        >
          {(seasons ?? []).map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>

        {/* Search */}
        <input
          type="text"
          placeholder="Search player..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="rounded-md border border-border bg-card px-3 py-1 text-xs text-foreground placeholder:text-muted-foreground"
        />

        {isLoading && (
          <span className="text-xs text-muted-foreground">Loading...</span>
        )}
      </div>

      {error && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
          Failed to load stats: {error.message}
        </div>
      )}

      {tab === "batters" && batters.data && (
        <BatterStatsTable data={batters.data} search={search} />
      )}
      {tab === "pitchers" && pitchers.data && (
        <PitcherStatsTable data={pitchers.data} search={search} />
      )}

      {/* Pagination */}
      <div className="flex items-center gap-3 text-xs text-muted-foreground">
        <button
          onClick={() => setPage((p) => Math.max(0, p - 1))}
          disabled={page === 0}
          className="rounded border border-border px-2 py-1 disabled:opacity-30"
        >
          Prev
        </button>
        <span>
          {page * PAGE_SIZE + 1}–{page * PAGE_SIZE + dataLen}
        </span>
        <button
          onClick={() => setPage((p) => p + 1)}
          disabled={dataLen < PAGE_SIZE}
          className="rounded border border-border px-2 py-1 disabled:opacity-30"
        >
          Next
        </button>
      </div>
    </div>
  );
}
