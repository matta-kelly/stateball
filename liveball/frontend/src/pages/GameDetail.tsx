import { useMemo, useState, type ReactNode } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, ChevronLeft, ChevronRight } from "lucide-react";
import ReactECharts from "echarts-for-react";
import type {
  Game,
  GameDetailData,
  GameStateEvent,
  SimResult,
  BenchPlayer,
} from "@/types/api";
import {
  useGame,
  useGameDetail as useGameDetailQuery,
  useGameStateEvents,
  useSimResults,
} from "@/hooks/queries";
import { useGameStream } from "@/hooks/useGameStream";
import { formatGameTime } from "@/lib/time";
import { cn } from "@/lib/utils";
import { getChartTheme } from "@/lib/chartTheme";
import { Bases, Outs } from "@/components/baseball/Situation";
import { PageLoading, ErrorBanner } from "@/components/ui/states";

/* ── Win Probability ── */

interface HoverData {
  time: string;
  sim: number | null;
}

type PresetView = "half" | "inning" | "3inn" | "full" | "custom";

// Marker color tokens — hierarchy promotion is server-side, frontend just reads the trigger
const MARKER_COLORS = {
  out_recorded: "#ef4444",  // red
  half_start:   "#60a5fa",  // blue
  inning_start: "#22c55e",  // green
} as const;

const SIM_COLOR = "#e4e4e7";

function WinProbability({
  game,
  events,
  sim,
  chartPill,
}: {
  game: Game;
  events: GameStateEvent[];
  sim: SimResult[];
  chartPill?: React.ReactNode;
}) {
  const [hover, setHover] = useState<HoverData | null>(null);
  const [view, setView] = useState<PresetView>("full");
  // 0 = current (latest) inning, -1 = previous, etc.
  const [anchorOffset, setAnchorOffset] = useState(0);

  // NOTE: all hooks below this line must run on every render, no matter
  // what market/sim/events look like. Conditional hooks corrupt memoization
  // slots and cause view/anchor changes to silently not propagate.

  const pHome = game.sim_p_home_win ?? 0;
  const se = game.sim_p_home_win_se ?? 0;
  const fmtPct = (v: number) => `${(v * 100).toFixed(1)}%`;

  // Inning boundaries: just inning_start.
  // Half boundaries: inning_start ∪ half_start — an inning_start is also
  // the start of a top half, not just a new inning.
  const inningStarts = useMemo(
    () => events.filter((e) => e.trigger === "inning_start"),
    [events],
  );
  const halfBoundaries = useMemo(
    () => events.filter(
      (e) => e.trigger === "inning_start" || e.trigger === "half_start",
    ),
    [events],
  );

  // Clamp anchor offset against available boundaries when view changes.
  const boundaries = view === "half" ? halfBoundaries : inningStarts;
  const maxIndex = boundaries.length - 1;
  const anchorIndex = Math.max(0, maxIndex + anchorOffset);

  // Resolve x-range from preset view.
  const [xMin, xMax] = useMemo<[number | null, number | null]>(() => {
    const nowMs = Date.now();
    let firstTs = Infinity;
    let lastTs = -Infinity;
    for (const s of sim) {
      const t = new Date(s.ts).getTime();
      if (t < firstTs) firstTs = t;
      if (t > lastTs) lastTs = t;
    }
    if (!Number.isFinite(firstTs)) firstTs = nowMs;
    if (!Number.isFinite(lastTs)) lastTs = nowMs;

    if (view === "full") return [firstTs, lastTs];

    if (view === "inning" && inningStarts.length > 0) {
      const lo = new Date(inningStarts[anchorIndex].ts).getTime();
      const hi = anchorIndex + 1 < inningStarts.length
        ? new Date(inningStarts[anchorIndex + 1].ts).getTime()
        : lastTs;
      return [lo, hi];
    }

    if (view === "half" && halfBoundaries.length > 0) {
      const lo = new Date(halfBoundaries[anchorIndex].ts).getTime();
      const hi = anchorIndex + 1 < halfBoundaries.length
        ? new Date(halfBoundaries[anchorIndex + 1].ts).getTime()
        : lastTs;
      return [lo, hi];
    }

    if (view === "3inn" && inningStarts.length > 0) {
      const startIdx = Math.max(0, anchorIndex - 2);
      const lo = new Date(inningStarts[startIdx].ts).getTime();
      const hi = anchorIndex + 1 < inningStarts.length
        ? new Date(inningStarts[anchorIndex + 1].ts).getTime()
        : lastTs;
      return [lo, hi];
    }

    // custom — ECharts dataZoom controls x; leave bounds unset
    return [null, null];
  }, [view, anchorIndex, inningStarts, halfBoundaries]);

  // Values from sim series that fall inside the current x-window.
  // Drives both the adaptive y-axis and the range readout.
  const windowValues = useMemo(() => {
    const vals: number[] = [];
    const lo = xMin, hi = xMax;
    const inWindow = (ts: string) => {
      if (lo == null || hi == null) return true;
      const t = new Date(ts).getTime();
      return t >= lo && t <= hi;
    };
    for (const s of sim) {
      if (inWindow(s.ts)) vals.push(s.p_home_win);
    }
    return vals;
  }, [xMin, xMax, sim]);

  // Adaptive y-axis: fit to the visible window. Recomputes on view/anchor
  // change and when new data extends the window. 10% padding on the range,
  // minimum 2pp, clamped to [0, 1]. Falls back to [0, 1] pre-data.
  const [yMin, yMax] = useMemo<[number, number]>(() => {
    if (windowValues.length < 2) return [0, 1];
    const lo = Math.min(...windowValues);
    const hi = Math.max(...windowValues);
    const range = hi - lo;
    const pad = Math.max(range * 0.1, 0.02);
    return [Math.max(0, lo - pad), Math.min(1, hi + pad)];
  }, [windowValues]);

  const rangeReadout = useMemo(() => {
    if (windowValues.length < 2) return null;
    const lo = Math.min(...windowValues);
    const hi = Math.max(...windowValues);
    return { lo, hi, swing: hi - lo };
  }, [windowValues]);

  // Hover readout — resolve from axis pointer timestamp to nearest market/sim entries.
  const onAxisPointer = (params: { axesInfo?: Array<{ value: number }> }) => {
    const axisValue = params.axesInfo?.[0]?.value;
    if (axisValue == null) return;

    const nearest = <T extends { ts: string }>(arr: T[]) => {
      if (arr.length === 0) return null;
      let best = arr[0];
      let bestDist = Math.abs(new Date(best.ts).getTime() - axisValue);
      for (const e of arr) {
        const d = Math.abs(new Date(e.ts).getTime() - axisValue);
        if (d < bestDist) { bestDist = d; best = e; }
      }
      return best;
    };

    const nearSim = nearest(sim);
    setHover({
      time: new Date(axisValue).toLocaleTimeString(),
      sim: nearSim?.p_home_win ?? null,
    });
  };

  // Marker lines as a separate series per color group.
  const markerSeries = (["out_recorded", "half_start", "inning_start"] as const).map((trig) => {
    let entries = events.filter((e) => e.trigger === trig);
    // Defensive: older recorded rows may have trigger=out_recorded with outs==3.
    // New writes absorb the 3rd out into the half/inning marker — filter
    // the stale ones here so they don't double-draw next to a blue/green.
    if (trig === "out_recorded") {
      entries = entries.filter((e) => e.outs !== 3);
    }
    const style = trig === "inning_start"
      ? { width: 1.5, type: "solid" as const, opacity: 0.55 }
      : trig === "half_start"
        ? { width: 1, type: "dashed" as const, opacity: 0.4 }
        : { width: 1, type: "dashed" as const, opacity: 0.25 };

    return {
      type: "line" as const,
      data: [],
      markLine: {
        silent: true,
        symbol: "none",
        animation: false,
        lineStyle: { color: MARKER_COLORS[trig], ...style },
        label: { show: false },
        data: entries.map((e) => ({ xAxis: e.ts })),
      },
    };
  });

  const simDense = sim.length > 40;

  const chartOption = {
    animation: false,
    grid: { top: 8, right: 12, bottom: view === "custom" ? 48 : 24, left: 40 },
    xAxis: {
      type: "time" as const,
      min: xMin ?? undefined,
      max: xMax ?? undefined,
      axisLabel: { fontSize: 10 },
      splitLine: { show: false },
      axisPointer: {
        show: true,
        snap: true,
        lineStyle: { color: SIM_COLOR, opacity: 0.25, type: "dotted", width: 1 },
        label: { show: false },
      },
    },
    yAxis: {
      type: "value" as const,
      min: yMin,
      max: yMax,
      axisLabel: { fontSize: 10, formatter: (v: number) => `${(v * 100).toFixed(1)}%` },
      splitLine: { lineStyle: { opacity: 0.15 } },
      axisPointer: { show: false },
    },
    dataZoom: view === "custom" ? [
      { type: "slider" as const, show: true, height: 20, bottom: 4 },
      { type: "inside" as const },
    ] : undefined,
    series: [
      // Sim CI band — two stacked invisible lines with the top one filled.
      {
        type: "line" as const,
        name: "ci_lower",
        data: sim.map((e) => [e.ts, Math.max(0, e.p_home_win - e.p_home_win_se)]),
        lineStyle: { opacity: 0 },
        symbol: "none" as const,
        stack: "ci",
        areaStyle: { opacity: 0 },
        emphasis: { disabled: true },
      },
      {
        type: "line" as const,
        name: "ci_width",
        data: sim.map((e) => [
          e.ts,
          Math.min(1, e.p_home_win + e.p_home_win_se) - Math.max(0, e.p_home_win - e.p_home_win_se),
        ]),
        lineStyle: { opacity: 0 },
        symbol: "none" as const,
        stack: "ci",
        areaStyle: { color: SIM_COLOR, opacity: 0.08 },
        emphasis: { disabled: true },
      },
      // Sim line — dashed gray
      {
        type: "line" as const,
        name: "Sim",
        data: sim.map((e) => [e.ts, e.p_home_win]),
        smooth: false,
        symbol: simDense ? "none" : "circle",
        symbolSize: simDense ? 0 : 2.5,
        lineStyle: { color: SIM_COLOR, width: 1.5, type: "dashed" as const },
        itemStyle: { color: SIM_COLOR },
        emphasis: { disabled: true },
      },
      // 50% reference line
      {
        type: "line" as const,
        data: [],
        markLine: {
          silent: true,
          symbol: "none",
          lineStyle: { color: getChartTheme().axisLabel.color, type: "dashed", opacity: 0.4 },
          data: [{ yAxis: 0.5 }],
          label: { show: false },
        },
      },
      // Event marker lines (red outs, blue half-innings, green innings)
      ...markerSeries,
    ],
    tooltip: { show: false },
    axisPointer: { link: [{ xAxisIndex: "all" }], triggerOn: "mousemove" },
  };

  const chartEvents = useMemo(() => ({
    updateaxispointer: onAxisPointer,
  }), [sim]);

  // Early return AFTER all hook calls. Doing this above (where the data
  // check naturally sits) is a Rules of Hooks violation — React misaligns
  // memo slots on the empty→populated transition, producing stale
  // xMin/xMax and empty marker series.
  if (sim.length === 0) {
    return (
      <div className="flex flex-col rounded-lg border border-dashed border-border bg-card p-4">
        {chartPill ?? <h3 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Win Probability</h3>}
        <div className="flex items-center justify-center py-4 text-xs text-muted-foreground/50">
          Waiting for sim data
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <div className="mb-2 flex items-center justify-between">
        {chartPill ?? <h3 className="text-sm font-medium text-muted-foreground">Win Probability</h3>}
        {game.sim_n_sims != null && (
          <span className="text-xs text-muted-foreground">
            {game.sim_n_sims} sims
            {game.sim_duration_ms ? ` · ${(game.sim_duration_ms / 1000).toFixed(1)}s` : ""}
          </span>
        )}
      </div>

      <div className="mb-1.5 flex items-baseline gap-3">
        <div>
          <span className="text-xl font-bold tabular-nums text-foreground">
            {(pHome * 100).toFixed(1)}%
          </span>
          <span className="ml-1 text-xs text-muted-foreground">
            ±{(se * 100).toFixed(1)}%
          </span>
        </div>
        <span className="text-xs text-muted-foreground">
          {game.home_team_name} win probability
        </span>
      </div>

      {/* Hover readout — fixed row, replaces floating tooltip */}
      <div className="mb-1 flex items-center gap-3 text-[11px] tabular-nums h-4">
        {hover ? (
          <>
            <span className="text-muted-foreground">{hover.time}</span>
            {hover.sim != null && (
              <span style={{ color: SIM_COLOR }}>Sim {fmtPct(hover.sim)}</span>
            )}
          </>
        ) : rangeReadout ? (
          <span className="text-muted-foreground">
            Range: <span className="font-mono text-foreground">{fmtPct(rangeReadout.lo)}–{fmtPct(rangeReadout.hi)}</span>
            <span className="ml-1.5">swing <span className="font-mono text-foreground">{fmtPct(rangeReadout.swing)}</span></span>
          </span>
        ) : null}
      </div>

      <div onMouseLeave={() => setHover(null)}>
        <ReactECharts
          option={chartOption}
          style={{ height: view === "custom" ? 220 : 200 }}
          notMerge
          onEvents={chartEvents}
        />
      </div>

      {/* Preset view pills + inning pager */}
      <div className="mt-2 flex items-center justify-between gap-2 border-t border-border pt-2">
        <div className="flex items-center gap-0.5 rounded-md bg-muted p-0.5">
          {(["half", "inning", "3inn", "full", "custom"] as PresetView[]).map((v) => (
            <button
              key={v}
              onClick={() => { setView(v); setAnchorOffset(0); }}
              className={cn(
                "rounded px-2 py-0.5 text-[10px] font-medium transition-colors",
                view === v ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground",
              )}
            >
              {v === "3inn" ? "3 Inn" : v === "full" ? "Full" : v === "custom" ? "Custom" : v === "inning" ? "Inning" : "Half"}
            </button>
          ))}
        </div>

        {(view === "inning" || view === "half" || view === "3inn") && boundaries.length > 1 && (
          <div className="flex items-center gap-1">
            <button
              onClick={() => setAnchorOffset((o) => Math.max(-maxIndex, o - 1))}
              disabled={anchorIndex <= 0}
              className="rounded p-0.5 text-muted-foreground hover:text-foreground disabled:opacity-30"
            >
              <ChevronLeft className="h-3 w-3" />
            </button>
            <span className="text-[10px] tabular-nums text-muted-foreground">
              {anchorOffset === 0 ? "Current" : `${anchorOffset}`}
            </span>
            <button
              onClick={() => setAnchorOffset((o) => Math.min(0, o + 1))}
              disabled={anchorOffset >= 0}
              className="rounded p-0.5 text-muted-foreground hover:text-foreground disabled:opacity-30"
            >
              <ChevronRight className="h-3 w-3" />
            </button>
          </div>
        )}

        {/* Legend */}
        <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
          <span className="flex items-center gap-1">
            <span className="inline-block h-0 w-3 border-t border-dashed" style={{ borderColor: SIM_COLOR }} />
            Sim
          </span>
        </div>
      </div>
    </div>
  );
}

/* ── Team Pill Toggle ── */

function TeamPill({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
        active
          ? "bg-zinc-200 text-zinc-900 dark:bg-zinc-700 dark:text-zinc-100"
          : "text-muted-foreground hover:text-card-foreground"
      }`}
    >
      {children}
    </button>
  );
}

/* ── Current AB: pitch plot + sequence ── */

function CurrentABPanel({ ab, matchup }: { ab: NonNullable<Game["current_ab"]>; matchup?: React.ReactNode }) {
  const pitches = ab.pitches ?? [];
  const szTop = ab.sz_top ?? 3.5;
  const szBot = ab.sz_bottom ?? 1.5;

  const dotColor = (p: { is_strike: boolean; is_ball: boolean; is_in_play: boolean }) => {
    if (p.is_in_play) return "#22c55e";
    if (p.is_strike) return "#ef4444";
    return "#3b82f6";
  };

  const scatterData = pitches
    .filter((p) => p.px != null && p.pz != null)
    .map((p, i) => ({
      value: [p.px!, p.pz!],
      itemStyle: {
        color: dotColor(p),
        opacity: i === pitches.length - 1 ? 1 : 0.65,
        borderColor: i === pitches.length - 1 ? "#fff" : "transparent",
        borderWidth: i === pitches.length - 1 ? 1.5 : 0,
      },
      symbolSize: i === pitches.length - 1 ? 12 : 9,
    }));

  const chartOption = {
    animation: false,
    backgroundColor: "transparent",
    grid: { top: 8, right: 8, bottom: 8, left: 8 },
    xAxis: {
      type: "value" as const,
      min: -1.5,
      max: 1.5,
      show: false,
    },
    yAxis: {
      type: "value" as const,
      min: 0.5,
      max: 5,
      show: false,
    },
    series: [
      {
        type: "scatter" as const,
        data: scatterData,
        markArea: {
          silent: true,
          itemStyle: { color: "transparent", borderColor: getChartTheme().axisLine.lineStyle.color, borderWidth: 1.5 },
          data: [
            [{ coord: [-0.708, szBot] }, { coord: [0.708, szTop] }],
          ],
        },
      },
    ],
  };

  const resultLabel = (p: typeof pitches[0]) => {
    if (p.result) return p.result;
    if (p.is_in_play) return "In Play";
    if (p.is_strike) return "Strike";
    if (p.is_ball) return "Ball";
    return "—";
  };

  return (
    <div className="grid border-t border-border" style={{ gridTemplateColumns: "170px 1fr" }}>
      {/* Pitch plot */}
      <div className="flex items-center justify-center border-r border-border p-2">
        <ReactECharts option={chartOption} style={{ height: 160, width: 160 }} />
      </div>

      {/* Sequence table */}
      <div className="px-4 py-3">
        <div className="mb-2 text-[10px] uppercase tracking-wide text-zinc-500">
          Pitch Sequence
        </div>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-muted-foreground">
              <th className="pb-1 text-left w-5">#</th>
              <th className="pb-1 text-left">Type</th>
              <th className="pb-1 text-right w-12">MPH</th>
              <th className="pb-1 text-left pl-3">Result</th>
            </tr>
          </thead>
          <tbody>
            {[...pitches].reverse().map((p) => (
              <tr key={p.num} className="border-t border-border/50">
                <td className="py-0.5 tabular-nums text-muted-foreground">{p.num}</td>
                <td className="py-0.5">{p.type_name ?? p.type_code ?? "—"}</td>
                <td className="py-0.5 text-right tabular-nums text-muted-foreground">
                  {p.speed != null ? p.speed.toFixed(1) : "—"}
                </td>
                <td
                  className={`py-0.5 pl-3 ${
                    p.is_in_play
                      ? "text-success"
                      : p.is_strike
                        ? "text-destructive"
                        : "text-info"
                  }`}
                >
                  {resultLabel(p)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {matchup}
      </div>
    </div>
  );
}


/* ── Bullpen / Bench toggle panel ── */

function BullpenBenchPanel({
  bullpen,
  bench,
}: {
  bullpen: { id: number; name: string; jersey: string }[];
  bench: BenchPlayer[];
}) {
  const [tab, setTab] = useState<"bullpen" | "bench">("bullpen");

  return (
    <div className="mt-3 border-t border-border pt-3">
      <div className="mb-2 flex items-center gap-1">
        <TeamPill active={tab === "bullpen"} onClick={() => setTab("bullpen")}>
          Bullpen
        </TeamPill>
        <TeamPill active={tab === "bench"} onClick={() => setTab("bench")}>
          Bench
        </TeamPill>
      </div>

      {tab === "bullpen" ? (
        <div className="flex flex-wrap gap-x-3 gap-y-1">
          {bullpen.length > 0 ? (
            bullpen.map((bp) => (
              <span key={bp.id} className="text-xs text-card-foreground">
                {bp.name}
              </span>
            ))
          ) : (
            <span className="text-xs text-muted-foreground">No available relievers</span>
          )}
        </div>
      ) : (
        <div className="space-y-1">
          {bench.length > 0 ? (
            bench.map((p) => (
              <div key={p.id} className="flex items-center justify-between text-xs">
                <span className="text-card-foreground">
                  {p.name}
                  {p.position && (
                    <span className="ml-1.5 text-muted-foreground">{p.position}</span>
                  )}
                </span>
                {p.stats.ab > 0 && (
                  <span className="tabular-nums text-muted-foreground">
                    {p.stats.h}/{p.stats.ab}
                    {p.stats.rbi > 0 && ` · ${p.stats.rbi} RBI`}
                  </span>
                )}
              </div>
            ))
          ) : (
            <span className="text-xs text-muted-foreground">No bench data</span>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Roster Panel (Lineup + Pitching + Bullpen, one team at a time) ── */

function RosterPanel({
  detail,
  currentBatterId,
}: {
  detail: GameDetailData | null;
  currentBatterId: number | null;
}) {
  const [side, setSide] = useState<"away" | "home">("away");

  if (!detail) {
    return (
      <div className="rounded-lg border border-dashed border-border bg-card p-6">
        <h3 className="text-sm font-medium text-muted-foreground">Roster</h3>
        <p className="mt-2 text-xs text-muted-foreground">
          Waiting for live data
        </p>
      </div>
    );
  }

  const team = side === "away" ? detail.away : detail.home;
  const s = team.pitcher.stats;

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      {/* Team toggle */}
      <div className="mb-3 flex items-center gap-1">
        <TeamPill active={side === "away"} onClick={() => setSide("away")}>
          <span className={side === "away" ? "text-zinc-200" : ""}>
            {detail.away.abbreviation || "Away"}
          </span>
        </TeamPill>
        <TeamPill active={side === "home"} onClick={() => setSide("home")}>
          <span className={side === "home" ? "text-info" : ""}>
            {detail.home.abbreviation || "Home"}
          </span>
        </TeamPill>
      </div>

      {/* Lineup */}
      <table className="w-full text-xs">
        <thead>
          <tr className="text-muted-foreground">
            <th className="w-6 pb-1 text-left">#</th>
            <th className="w-8 pb-1 text-left">Pos</th>
            <th className="pb-1 text-left">Name</th>
            <th className="w-8 pb-1 text-right">AB</th>
            <th className="w-8 pb-1 text-right">H</th>
            <th className="w-8 pb-1 text-right">R</th>
            <th className="w-8 pb-1 text-right">RBI</th>
          </tr>
        </thead>
        <tbody>
          {team.lineup.map((p) => {
            const isActive = p.id === currentBatterId;
            return (
              <tr
                key={p.id}
                className={
                  isActive
                    ? "bg-warning/10 font-semibold text-warning"
                    : "text-card-foreground"
                }
              >
                <td className="py-0.5 tabular-nums">{p.batting_order}</td>
                <td className="py-0.5 text-muted-foreground">{p.position}</td>
                <td className="py-0.5">
                  {p.name}
                  {isActive && (
                    <span className="ml-1.5 text-[10px] text-warning">AB</span>
                  )}
                </td>
                <td className="py-0.5 text-right tabular-nums">{p.stats.ab}</td>
                <td className="py-0.5 text-right tabular-nums">{p.stats.h}</td>
                <td className="py-0.5 text-right tabular-nums">{p.stats.r}</td>
                <td className="py-0.5 text-right tabular-nums">{p.stats.rbi}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {/* Current pitcher */}
      {team.pitcher.name && (
        <div className="mt-4 border-t border-border pt-3">
          <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">
            Pitching
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-card-foreground">
              {team.pitcher.name}
            </span>
            {team.pitcher.jersey && (
              <span className="text-xs text-muted-foreground">#{team.pitcher.jersey}</span>
            )}
          </div>
          <div className="mt-1 flex gap-3 text-xs text-muted-foreground">
            <span>{s.ip} IP</span>
            <span>{s.so} K</span>
            <span>{s.bb} BB</span>
            <span>{s.h} H</span>
            <span>{s.er} ER</span>
            <span>{s.pitches} P</span>
          </div>
        </div>
      )}

      {/* Pitchers used (not including current) */}
      {team.pitchers_used.length > 0 && (
        <div className="mt-3 border-t border-border pt-3">
          <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">
            Pitchers Used
          </div>
          <div className="space-y-1">
            {team.pitchers_used.map((p) => (
              <div key={p.id}>
                <div className="flex items-center gap-1.5">
                  <span className="text-xs text-zinc-500">{p.name}</span>
                  {p.jersey && <span className="text-[10px] text-zinc-600">#{p.jersey}</span>}
                </div>
                <div className="flex gap-2 text-[10px] text-zinc-600">
                  <span>{p.stats.ip} IP</span>
                  <span>{p.stats.so} K</span>
                  <span>{p.stats.bb} BB</span>
                  <span>{p.stats.h} H</span>
                  <span>{p.stats.er} ER</span>
                  <span>{p.stats.pitches} P</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Bullpen / Bench toggle */}
      {(team.bullpen.length > 0 || team.bench.length > 0) && (
        <BullpenBenchPanel bullpen={team.bullpen} bench={team.bench} />
      )}
    </div>
  );
}

/* ── Box Score (Linescore) ── */

function BoxScore({ detail, game }: { detail: GameDetailData | null; game: Game }) {
  if (!detail?.linescore) return null;
  const { linescore } = detail;
  const maxInnings = Math.max(9, linescore.innings.length);
  const inningNums = Array.from({ length: maxInnings }, (_, i) => i + 1);

  const cellClass = "px-1.5 py-1 text-center tabular-nums text-xs";
  const headerClass = "px-1.5 py-1 text-center text-[10px] font-medium text-muted-foreground";

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border">
              <th className="px-2 py-1 text-left text-[10px] font-medium text-muted-foreground" />
              {inningNums.map((n) => (
                <th key={n} className={headerClass}>{n}</th>
              ))}
              <th className={`${headerClass} border-l border-border font-semibold`}>R</th>
              <th className={`${headerClass} font-semibold`}>H</th>
              <th className={`${headerClass} font-semibold`}>E</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-border">
              <td className="px-2 py-1 text-xs font-semibold text-zinc-300">
                {detail.away.abbreviation || game.away_team_name}
              </td>
              {inningNums.map((n) => {
                const inn = linescore.innings.find((i) => i.num === n);
                return (
                  <td key={n} className={cellClass}>
                    {inn?.away != null ? inn.away : ""}
                  </td>
                );
              })}
              <td className={`${cellClass} border-l border-border font-bold`}>{linescore.away.runs}</td>
              <td className={`${cellClass} font-bold`}>{linescore.away.hits}</td>
              <td className={`${cellClass} font-bold`}>{linescore.away.errors}</td>
            </tr>
            <tr>
              <td className="px-2 py-1 text-xs font-semibold text-info">
                {detail.home.abbreviation || game.home_team_name}
              </td>
              {inningNums.map((n) => {
                const inn = linescore.innings.find((i) => i.num === n);
                return (
                  <td key={n} className={`${cellClass} ${inn?.home != null && inn.home > 0 ? "font-semibold" : ""}`}>
                    {inn?.home != null ? inn.home : ""}
                  </td>
                );
              })}
              <td className={`${cellClass} border-l border-border font-bold`}>{linescore.home.runs}</td>
              <td className={`${cellClass} font-bold`}>{linescore.home.hits}</td>
              <td className={`${cellClass} font-bold`}>{linescore.home.errors}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── Main page ── */

export default function GameDetail() {
  const { gamePk } = useParams<{ gamePk: string }>();
  const gamePkNum = Number(gamePk);

  // React Query data fetching
  const { data: restGame, isLoading: gameLoading, error: gameError } = useGame(gamePkNum);
  const { data: detailData } = useGameDetailQuery(gamePkNum);
  const detail = detailData ?? null;
  const { data: events = [] } = useGameStateEvents(gamePkNum, { refetchInterval: 10_000 });
  const { data: sim = [] } = useSimResults(gamePkNum, { refetchInterval: 10_000 });

  // SSE stream for live updates
  const { games: streamGames } = useGameStream(true);
  const sseGame = useMemo(
    () => streamGames.find((g) => g.game_pk === gamePkNum) ?? null,
    [streamGames, gamePkNum],
  );

  // Use SSE game if available, otherwise REST
  const game = sseGame ?? restGame ?? null;

  if (gameLoading) {
    return (
      <div className="mx-auto w-full max-w-5xl p-4 sm:p-6">
        <PageLoading />
      </div>
    );
  }

  if (gameError) {
    return (
      <div className="mx-auto w-full max-w-5xl p-4 sm:p-6">
        <ErrorBanner message={gameError.message} />
      </div>
    );
  }

  if (!game) {
    return (
      <div className="mx-auto w-full max-w-6xl p-4 sm:p-6">
        <Link
          to="/"
          className="mb-4 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" /> Back to games
        </Link>
        <p className="text-center text-sm text-destructive">
          Game not found
        </p>
      </div>
    );
  }

  const isLive = game.abstract_game_state === "Live";
  const isFinal = game.abstract_game_state === "Final";
  const showScore = isLive || isFinal;

  return (
    <div className="mx-auto w-full max-w-6xl space-y-4 p-4 sm:p-6">
      {/* Back link */}
      <Link
        to="/"
        className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-4 w-4" /> Back to games
      </Link>

      {/* ── Top row: two equal-height columns ── */}
      <div className="grid gap-4 lg:grid-cols-[1fr_380px] lg:items-start">

        {/* Left column: Game Header + Projections placeholder */}
        <div className="flex flex-col gap-4">
          {/* Game Header */}
          <div
            className={`rounded-lg border bg-card p-4 shadow-sm ${
              isLive ? "border-success/30" : "border-border"
            }`}
          >
            {/* Status + time */}
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                {isLive ? (
                  <span className="flex items-center gap-1.5 text-sm font-semibold text-success">
                    <span className="h-2 w-2 animate-pulse rounded-full bg-success" />
                    LIVE
                  </span>
                ) : isFinal ? (
                  <span className="text-sm font-medium text-zinc-500">FINAL</span>
                ) : (
                  <span className="text-sm font-medium text-info">
                    {game.status}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-3 text-xs text-muted-foreground">
                <span>{formatGameTime(game.game_datetime)}</span>
                {game.venue_name && <span>{game.venue_name}</span>}
              </div>
            </div>

            {/* Scoreboard */}
            <div className="flex items-center gap-4">
              {/* Teams + scores */}
              {(() => {
                const isTop = game.inning_half === "Top" || game.inning_half === "top";
                const awayBatting = isLive && isTop;
                const homeBatting = isLive && !isTop;
                return (
                  <div className="flex-1 space-y-1.5">
                    <div className="flex items-center justify-between">
                      <span className="flex items-center gap-2">
                        <span
                          className={`text-base text-card-foreground ${
                            awayBatting ? "font-bold" : "font-semibold"
                          }`}
                        >
                          {game.away_team_name}
                        </span>
                        {awayBatting && (
                          <span className="text-xs text-zinc-400">▲</span>
                        )}
                      </span>
                      {showScore && (
                        <span className="text-2xl font-bold tabular-nums text-card-foreground">
                          {game.away_score ?? 0}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="flex items-center gap-2">
                        <span
                          className={`text-base text-card-foreground ${
                            homeBatting ? "font-bold" : "font-semibold"
                          }`}
                        >
                          {game.home_team_name}
                        </span>
                        {homeBatting && (
                          <span className="text-xs text-zinc-400">▼</span>
                        )}
                      </span>
                      {showScore && (
                        <span className="text-2xl font-bold tabular-nums text-card-foreground">
                          {game.home_score ?? 0}
                        </span>
                      )}
                    </div>
                  </div>
                );
              })()}

              {/* Live situation */}
              {isLive && (
                <div className="flex flex-col items-center gap-1.5 border-l border-border pl-4">
                  {game.inning != null && (
                    <span className="text-xs font-semibold uppercase tracking-wide text-success">
                      {game.inning_half === "Top" || game.inning_half === "top"
                        ? "TOP"
                        : "BOT"}{" "}
                      {game.inning}
                    </span>
                  )}
                  <Bases runners={game.runners} size="md" />
                  <Outs count={game.outs} size="md" />
                  <span className="text-xs tabular-nums text-muted-foreground">
                    {game.balls ?? 0}-{game.strikes ?? 0}
                  </span>
                </div>
              )}
            </div>

            {/* Current AB: pitch plot + sequence + matchup */}
            {game.current_ab && game.current_ab.pitches.length > 0 && (
              <CurrentABPanel
                ab={game.current_ab}
                matchup={isLive && (game.current_pitcher_name || game.current_batter_name) ? (() => {
                  const isTop = game.inning_half === "Top" || game.inning_half === "top";
                  const pitchTeam = detail
                    ? (isTop ? detail.home.abbreviation : detail.away.abbreviation)
                    : null;
                  const batTeam = detail
                    ? (isTop ? detail.away.abbreviation : detail.home.abbreviation)
                    : null;
                  return (
                    <div className="mt-2 flex gap-6 border-t border-border/50 pt-2 text-sm">
                      {game.current_pitcher_name && (
                        <div>
                          <div className="flex items-center gap-1.5">
                            <span className="text-[10px] uppercase tracking-wide text-zinc-500">Pitching</span>
                            {pitchTeam && (
                              <span className={`text-[10px] font-medium ${isTop ? "text-info" : "text-zinc-300"}`}>
                                {pitchTeam}
                              </span>
                            )}
                          </div>
                          <p className="text-xs font-medium text-card-foreground">
                            {game.current_pitcher_name}
                          </p>
                        </div>
                      )}
                      {game.current_batter_name && (
                        <div>
                          <div className="flex items-center gap-1.5">
                            <span className="text-[10px] uppercase tracking-wide text-zinc-500">At Bat</span>
                            {batTeam && (
                              <span className={`text-[10px] font-medium ${isTop ? "text-zinc-300" : "text-info"}`}>
                                {batTeam}
                              </span>
                            )}
                          </div>
                          <p className="text-xs font-medium text-card-foreground">
                            {game.current_batter_name}
                          </p>
                        </div>
                      )}
                    </div>
                  );
                })() : undefined}
              />
            )}

            {/* Last play */}
            {game.last_play && (
              <div className="mt-3 border-t border-border pt-3">
                <span className="text-[10px] uppercase tracking-wide text-zinc-500">Last Play</span>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  {game.last_play}
                </p>
              </div>
            )}
          </div>

          <WinProbability game={game} events={events} sim={sim} />
        </div>

        {/* Right column: Box Score + Roster */}
        <div className="flex flex-col gap-4">
          <BoxScore detail={detail} game={game} />
          <RosterPanel detail={detail} currentBatterId={game.current_batter_id} />
        </div>
      </div>
    </div>
  );
}
