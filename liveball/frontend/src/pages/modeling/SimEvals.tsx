import { useState } from "react";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import { ChevronDown, ChevronRight, Loader2 } from "lucide-react";
import { useSimEvals, useEvalDiagnostics } from "@/hooks/queries";
import { SortIcon } from "@/components/ui/sort-icon";
import { PageLoading, ErrorBanner } from "@/components/ui/states";
import type { SimEval } from "@/types/api";

const col = createColumnHelper<SimEval>();

const nullablePercent = (v: number | null) =>
  v != null ? (
    <span className="font-mono tabular-nums">{(v * 100).toFixed(1)}%</span>
  ) : (
    <span className="text-muted-foreground/50">—</span>
  );

const nullableFixed = (v: number | null, digits: number) =>
  v != null ? (
    <span className="font-mono tabular-nums">{v.toFixed(digits)}</span>
  ) : (
    <span className="text-muted-foreground/50">—</span>
  );

function DiagnosticsPanel({ evalId }: { evalId: string }) {
  const [inning, setInning] = useState("all");
  const { data, isLoading } = useEvalDiagnostics(evalId, inning === "all" ? undefined : inning);
  const horizons = data?.horizons ?? [];

  return (
    <div className="space-y-3 p-4">
      <div className="flex items-center gap-3">
        <span className="text-sm font-medium text-muted-foreground">Entry Inning:</span>
        <select
          value={inning}
          onChange={(e) => setInning(e.target.value)}
          className="rounded border border-border bg-background px-2 py-1 text-sm"
        >
          <option value="all">All</option>
          {[1, 2, 3, 4, 5, 6, 7, 8, 9].map((i) => (
            <option key={i} value={String(i)}>Inning {i}</option>
          ))}
        </select>
        {isLoading && <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />}
      </div>

      {horizons.length > 0 ? (
        <div className="overflow-x-auto rounded border border-border">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border bg-muted/30">
                <th className="px-2 py-1.5 text-left font-medium">Horizon</th>
                <th className="px-2 py-1.5 text-right font-medium">N</th>
                <th className="px-2 py-1.5 text-right font-medium" title="mean(|pred_we - actual_we|)">Pred MAE</th>
                <th className="px-2 py-1.5 text-right font-medium" title="mean(|entry_we - actual_we|)">Entry MAE</th>
                <th className="px-2 py-1.5 text-right font-medium" title="(entry_mae - pred_mae) / entry_mae">Improve</th>
                <th className="px-2 py-1.5 text-right font-medium" title="mean(|predicted_move - actual_move|)">Move MAE</th>
                <th className="px-2 py-1.5 text-right font-medium" title="mean(predicted_move - actual_move)">Move Bias</th>
                <th className="px-2 py-1.5 text-right font-medium" title="Brier Skill Score">BSS</th>
                <th className="px-2 py-1.5 text-right font-medium" title="Brier reliability (lower = better)">Reliab.</th>
              </tr>
            </thead>
            <tbody>
              {horizons.map((h) => (
                <tr key={h.horizon} className="border-b border-border last:border-0 hover:bg-muted/20">
                  <td className="px-2 py-1 font-mono">{h.horizon}</td>
                  <td className="px-2 py-1 text-right">{h.n}</td>
                  <td className="px-2 py-1 text-right font-mono">{nullableFixed(h.pred_mae, 4)}</td>
                  <td className="px-2 py-1 text-right font-mono">{nullableFixed(h.entry_mae, 4)}</td>
                  <td className="px-2 py-1 text-right font-mono">
                    {h.improvement != null ? (
                      <span className={h.improvement > 0 ? "text-success" : h.improvement < 0 ? "text-destructive" : ""}>
                        {(h.improvement * 100).toFixed(1)}%
                      </span>
                    ) : "—"}
                  </td>
                  <td className="px-2 py-1 text-right font-mono">{nullableFixed(h.move_mae, 4)}</td>
                  <td className="px-2 py-1 text-right font-mono">
                    {h.move_bias != null ? (
                      <span className={h.move_bias > 0.005 ? "text-destructive" : h.move_bias < -0.005 ? "text-info" : ""}>
                        {h.move_bias > 0 ? "+" : ""}{h.move_bias.toFixed(4)}
                      </span>
                    ) : "—"}
                  </td>
                  <td className="px-2 py-1 text-right font-mono">{nullableFixed(h.bss, 3)}</td>
                  <td className="px-2 py-1 text-right font-mono">{nullableFixed(h.reliability, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : !isLoading ? (
        <p className="text-sm text-muted-foreground">No horizon data for this eval.</p>
      ) : null}
    </div>
  );
}

export default function SimEvals() {
  const { data = [], isLoading, error } = useSimEvals();
  const [sorting, setSorting] = useState<SortingState>([
    { id: "created_at", desc: true },
  ]);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const columns = [
    col.display({
      id: "expand",
      header: "",
      cell: (info) => {
        const isExpanded = expandedId === info.row.original.eval_id;
        return isExpanded
          ? <ChevronDown className="h-4 w-4 text-muted-foreground" />
          : <ChevronRight className="h-4 w-4 text-muted-foreground" />;
      },
    }),
    col.accessor("eval_id", {
      header: "Eval",
      cell: (info) => (
        <span className="font-mono text-xs">{info.getValue()}</span>
      ),
    }),
    col.accessor("created_at", {
      header: "Created",
      cell: (info) => {
        const d = new Date(info.getValue());
        return d.toLocaleDateString("en-US", {
          month: "short",
          day: "numeric",
          year: "numeric",
          hour: "2-digit",
          minute: "2-digit",
        });
      },
    }),
    col.accessor("accuracy", {
      header: "Accuracy",
      cell: (info) => (
        <span className="font-mono tabular-nums">
          {(info.getValue() * 100).toFixed(1)}%
        </span>
      ),
    }),
    col.accessor("score_mae", {
      header: "Score MAE",
      cell: (info) => nullableFixed(info.getValue(), 2),
    }),
    col.accessor("mean_mc_time", {
      header: "Time/Game",
      cell: (info) => (
        <span className="font-mono tabular-nums">
          {info.getValue().toFixed(2)}s
        </span>
      ),
    }),
    col.accessor("n_games", {
      header: "Games",
      cell: (info) => info.getValue().toLocaleString(),
    }),
    col.accessor("n_sims", {
      header: "Sims",
      cell: (info) => info.getValue().toLocaleString(),
    }),
    col.accessor("prune_rate", {
      header: "Prune %",
      cell: (info) => nullablePercent(info.getValue()),
    }),
  ];

  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  if (isLoading) return <PageLoading />;
  if (error) return <ErrorBanner message={error.message} />;

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-xl font-semibold">Production Evaluations</h2>
        <p className="text-sm text-muted-foreground">
          {data.length} eval run{data.length !== 1 && "s"}
        </p>
      </div>

      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-sm">
          <thead>
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id} className="border-b border-border bg-muted/50">
                {hg.headers.map((header) => (
                  <th
                    key={header.id}
                    className={`select-none px-3 py-2 text-left font-medium text-muted-foreground ${
                      header.column.getCanSort() ? "cursor-pointer hover:text-foreground" : ""
                    }`}
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    {flexRender(header.column.columnDef.header, header.getContext())}
                    {header.column.getCanSort() && (
                      <SortIcon sorted={header.column.getIsSorted()} />
                    )}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => {
              const evalId = row.original.eval_id;
              const isExpanded = expandedId === evalId;
              return (
                <>
                  <tr
                    key={row.id}
                    className="cursor-pointer border-b border-border last:border-0 hover:bg-muted/30"
                    onClick={() => setExpandedId(isExpanded ? null : evalId)}
                  >
                    {row.getVisibleCells().map((cell) => (
                      <td key={cell.id} className="px-3 py-2">
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </td>
                    ))}
                  </tr>
                  {isExpanded && (
                    <tr key={`${row.id}-diag`}>
                      <td colSpan={columns.length} className="bg-muted/10 p-0">
                        <DiagnosticsPanel evalId={evalId} />
                      </td>
                    </tr>
                  )}
                </>
              );
            })}
            {data.length === 0 && (
              <tr>
                <td
                  colSpan={columns.length}
                  className="px-3 py-8 text-center text-muted-foreground"
                >
                  No production evaluations yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
