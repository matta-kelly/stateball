import { Link, useSearchParams } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import ReactECharts from "echarts-for-react";
import { useQueries } from "@tanstack/react-query";
import { getSimEval } from "@/lib/api";
import { queryKeys } from "@/hooks/queries";
import { PageLoading, ErrorBanner } from "@/components/ui/states";
import { getChartColors } from "@/lib/chartTheme";
import type { SimEvalDetail } from "@/types/api";

let _chartColors: string[] | null = null;
function evalColor(i: number) {
  if (!_chartColors) _chartColors = [...getChartColors(), "#ec4899"];
  return _chartColors[i % _chartColors.length];
}

function shortId(id: string) {
  return id.length > 15 ? id.slice(0, 15) : id;
}

/* ── Section wrapper ── */

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <h3 className="mb-3 text-sm font-semibold text-card-foreground">{title}</h3>
      {children}
    </div>
  );
}

/* ── Summary comparison table ── */

function SummaryTable({ evals }: { evals: SimEvalDetail[] }) {
  const metrics: { label: string; get: (e: SimEvalDetail) => number | null; fmt: (v: number) => string; lower?: boolean }[] = [
    { label: "Accuracy", get: (e) => e.accuracy, fmt: (v) => `${(v * 100).toFixed(1)}%` },
    { label: "WP Brier", get: (e) => e.win_probability?.brier ?? null, fmt: (v) => v.toFixed(4), lower: true },
    { label: "WP Skill", get: (e) => e.win_probability?.brier_skill ?? null, fmt: (v) => `${(v * 100).toFixed(1)}%` },
    { label: "Score MAE", get: (e) => e.scores?.mean_abs_error ?? null, fmt: (v) => v.toFixed(2), lower: true },
    { label: "Time/Game", get: (e) => e.mean_mc_time, fmt: (v) => `${v.toFixed(2)}s`, lower: true },
    { label: "Games", get: (e) => e.n_games, fmt: (v) => v.toLocaleString() },
    { label: "Sims", get: (e) => e.n_sims, fmt: (v) => v.toLocaleString() },
    { label: "Prune Rate", get: (e) => e.prune_rate, fmt: (v) => `${(v * 100).toFixed(1)}%` },
    { label: "Mean P(Home)", get: (e) => e.mean_p_home, fmt: (v) => v.toFixed(4) },
  ];

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="px-3 py-2 text-left font-medium text-muted-foreground">Metric</th>
            {evals.map((ev, i) => (
              <th key={ev.eval_id} className="px-3 py-2 text-right font-mono text-xs" style={{ color: evalColor(i) }}>
                {shortId(ev.eval_id)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {metrics.map((m) => {
            const values = evals.map((e) => m.get(e));
            const validValues = values.filter((v): v is number => v != null);
            const best = validValues.length > 0
              ? (m.lower ? Math.min(...validValues) : Math.max(...validValues))
              : null;
            return (
              <tr key={m.label} className="border-b border-border last:border-0">
                <td className="px-3 py-1.5 text-muted-foreground">{m.label}</td>
                {values.map((v, i) => (
                  <td
                    key={evals[i].eval_id}
                    className={`px-3 py-1.5 text-right font-mono tabular-nums ${
                      v != null && v === best && evals.length > 1 ? "text-success font-semibold" : ""
                    }`}
                  >
                    {v != null ? m.fmt(v) : "—"}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ── Accuracy by Inning chart ── */

function AccuracyByInningChart({ evals }: { evals: SimEvalDetail[] }) {
  const innings = ["1", "2", "3", "4", "5", "6", "7", "8", "9"];
  const option = {
    animation: false,
    grid: { top: 32, right: 16, bottom: 32, left: 48 },
    legend: { data: evals.map((e) => shortId(e.eval_id)), top: 0, textStyle: { fontSize: 11 } },
    xAxis: { type: "category", data: innings, name: "Inning", nameLocation: "center", nameGap: 22 },
    yAxis: { type: "value", name: "Accuracy %", min: 40, max: 100, axisLabel: { formatter: "{value}%" } },
    tooltip: { trigger: "axis", formatter: (params: { seriesName: string; value: number; axisValue: string }[]) => {
      return params.map((p) => `${p.seriesName}: ${p.value.toFixed(1)}% (inn ${p.axisValue})`).join("<br/>");
    }},
    series: evals.map((ev, i) => ({
      name: shortId(ev.eval_id),
      type: "line",
      data: innings.map((inn) => {
        const d = ev.accuracy_by_inning?.[inn];
        return d ? +(d.accuracy * 100).toFixed(1) : null;
      }),
      color: evalColor(i),
      symbolSize: 6,
    })),
  };
  return <ReactECharts option={option} style={{ height: 280 }} notMerge />;
}

/* ── Accuracy by Phase chart ── */

function AccuracyByPhaseChart({ evals }: { evals: SimEvalDetail[] }) {
  const phases = ["early", "mid", "late"];
  const phaseLabels = ["Early (1-3)", "Mid (4-6)", "Late (7-9)"];
  const option = {
    animation: false,
    grid: { top: 32, right: 16, bottom: 24, left: 48 },
    legend: { data: evals.map((e) => shortId(e.eval_id)), top: 0, textStyle: { fontSize: 11 } },
    xAxis: { type: "category", data: phaseLabels },
    yAxis: { type: "value", name: "Accuracy %", min: 50, max: 100, axisLabel: { formatter: "{value}%" } },
    tooltip: { trigger: "axis" },
    series: evals.map((ev, i) => ({
      name: shortId(ev.eval_id),
      type: "bar",
      data: phases.map((ph) => {
        const d = ev.accuracy_by_phase?.[ph];
        return d ? +(d.accuracy * 100).toFixed(1) : null;
      }),
      color: evalColor(i),
      barMaxWidth: 40,
    })),
  };
  return <ReactECharts option={option} style={{ height: 240 }} notMerge />;
}

/* ── Brier Reliability Bins chart ── */

function BrierBinsChart({ evals }: { evals: SimEvalDetail[] }) {
  const anyBins = evals.some((e) => e.win_probability?.bins?.length);
  if (!anyBins) return <p className="text-xs text-muted-foreground">No bin data available</p>;

  const refBins = evals.find((e) => e.win_probability?.bins?.length)?.win_probability?.bins ?? [];
  const labels = refBins.map((b) => `${b.bin_lo.toFixed(1)}-${b.bin_hi.toFixed(1)}`);

  const option = {
    animation: false,
    grid: { top: 32, right: 16, bottom: 32, left: 56 },
    legend: { data: evals.map((e) => shortId(e.eval_id)), top: 0, textStyle: { fontSize: 11 } },
    xAxis: { type: "category", data: labels, name: "P(home_win) bin", nameLocation: "center", nameGap: 22 },
    yAxis: { type: "value", name: "Reliability contrib", axisLabel: { fontSize: 10 } },
    tooltip: { trigger: "axis" },
    series: evals.map((ev, i) => ({
      name: shortId(ev.eval_id),
      type: "bar",
      data: (ev.win_probability?.bins ?? []).map((b) => +b.reliability_contrib.toFixed(5)),
      color: evalColor(i),
      barMaxWidth: 24,
    })),
  };
  return <ReactECharts option={option} style={{ height: 280 }} notMerge />;
}

/* ── Brier Decomposition table ── */

function BrierDecompTable({ evals }: { evals: SimEvalDetail[] }) {
  const rows = [
    { label: "Brier Score", get: (wp: SimEvalDetail["win_probability"]) => wp?.brier, lower: true },
    { label: "Reliability", get: (wp: SimEvalDetail["win_probability"]) => wp?.reliability, lower: true },
    { label: "Resolution", get: (wp: SimEvalDetail["win_probability"]) => wp?.resolution },
    { label: "Uncertainty", get: (wp: SimEvalDetail["win_probability"]) => wp?.uncertainty },
    { label: "Brier Skill", get: (wp: SimEvalDetail["win_probability"]) => wp?.brier_skill },
  ];
  return (
    <div className="mt-3 overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="px-3 py-1.5 text-left text-xs text-muted-foreground">Component</th>
            {evals.map((ev, i) => (
              <th key={ev.eval_id} className="px-3 py-1.5 text-right font-mono text-xs" style={{ color: evalColor(i) }}>
                {shortId(ev.eval_id)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const values = evals.map((e) => r.get(e.win_probability));
            const valid = values.filter((v): v is number => v != null);
            const best = valid.length > 0 ? (r.lower ? Math.min(...valid) : Math.max(...valid)) : null;
            return (
              <tr key={r.label} className="border-b border-border last:border-0">
                <td className="px-3 py-1 text-xs text-muted-foreground">{r.label}</td>
                {values.map((v, i) => (
                  <td
                    key={evals[i].eval_id}
                    className={`px-3 py-1 text-right font-mono tabular-nums text-xs ${
                      v != null && v === best && evals.length > 1 ? "text-success font-semibold" : ""
                    }`}
                  >
                    {v != null ? v.toFixed(4) : "—"}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ── Score Prediction table ── */

function ScoresTable({ evals }: { evals: SimEvalDetail[] }) {
  const rows = [
    { label: "Home Error", get: (s: SimEvalDetail["scores"]) => s?.mean_error_home },
    { label: "Away Error", get: (s: SimEvalDetail["scores"]) => s?.mean_error_away },
    { label: "Abs Error", get: (s: SimEvalDetail["scores"]) => s?.mean_abs_error, lower: true },
  ];
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="px-3 py-1.5 text-left text-xs text-muted-foreground">Metric</th>
            {evals.map((ev, i) => (
              <th key={ev.eval_id} className="px-3 py-1.5 text-right font-mono text-xs" style={{ color: evalColor(i) }}>
                {shortId(ev.eval_id)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const values = evals.map((e) => r.get(e.scores));
            const valid = values.filter((v): v is number => v != null);
            const best = valid.length > 0 && r.lower ? Math.min(...valid) : null;
            return (
              <tr key={r.label} className="border-b border-border last:border-0">
                <td className="px-3 py-1 text-xs text-muted-foreground">{r.label}</td>
                {values.map((v, i) => (
                  <td
                    key={evals[i].eval_id}
                    className={`px-3 py-1 text-right font-mono tabular-nums text-xs ${
                      v != null && v === best && evals.length > 1 ? "text-success font-semibold" : ""
                    }`}
                  >
                    {v != null ? v.toFixed(3) : "—"}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ── Main page ── */

export default function EvalAnalysis() {
  const [searchParams] = useSearchParams();
  const ids = (searchParams.get("ids") ?? "").split(",").filter(Boolean);

  const evalQueries = useQueries({
    queries: ids.map((id) => ({
      queryKey: queryKeys.simEval(id),
      queryFn: () => getSimEval(id),
      enabled: !!id,
    })),
  });
  const isLoading = evalQueries.some((q) => q.isLoading);
  const error = evalQueries.find((q) => q.error)?.error;
  const evals = evalQueries.filter((q) => q.data).map((q) => q.data!);

  if (isLoading) return <PageLoading />;

  if (error) {
    return (
      <div className="space-y-4">
        <Link to="/workshop" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
          <ArrowLeft className="h-4 w-4" /> Back to evals
        </Link>
        <ErrorBanner message={error.message} />
      </div>
    );
  }

  if (ids.length === 0 || evals.length === 0) {
    return (
      <div className="space-y-4">
        <Link to="/workshop" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
          <ArrowLeft className="h-4 w-4" /> Back to evals
        </Link>
        <p className="text-sm text-muted-foreground">Select evals from the table to analyze.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Link to="/workshop" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
          <ArrowLeft className="h-4 w-4" /> Back to evals
        </Link>
        <div className="flex gap-2">
          {evals.map((ev, i) => (
            <span
              key={ev.eval_id}
              className="rounded-full px-2.5 py-0.5 font-mono text-xs font-medium text-white"
              style={{ backgroundColor: evalColor(i) }}
            >
              {shortId(ev.eval_id)}
            </span>
          ))}
        </div>
      </div>

      <h2 className="text-xl font-semibold">
        Eval Analysis {evals.length > 1 && `(${evals.length} runs)`}
      </h2>

      <Section title="Summary">
        <SummaryTable evals={evals} />
      </Section>

      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Accuracy by Inning">
          <AccuracyByInningChart evals={evals} />
        </Section>
        <Section title="Accuracy by Phase">
          <AccuracyByPhaseChart evals={evals} />
        </Section>
      </div>

      <Section title="Win Probability Calibration">
        <BrierBinsChart evals={evals} />
        <BrierDecompTable evals={evals} />
      </Section>

      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Score Prediction">
          <ScoresTable evals={evals} />
        </Section>
      </div>
    </div>
  );
}
