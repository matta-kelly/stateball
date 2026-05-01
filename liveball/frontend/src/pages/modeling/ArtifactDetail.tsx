import { useState } from "react";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import { Loader2, Trash2 } from "lucide-react";
import { useArtifacts, useSetArtifactSlot, useDeleteArtifact } from "@/hooks/queries";
import { SortIcon } from "@/components/ui/sort-icon";
import { SlotBadges } from "@/components/SlotBadges";
import { PageLoading, ErrorBanner } from "@/components/ui/states";
import type {
  Artifact,
  BaserunningMetrics,
  PitcherExitMetrics,
  WinExpectancyMetrics,
} from "@/types/api";

// ---------------------------------------------------------------------------
// Type-specific column configs
// ---------------------------------------------------------------------------

interface TypeConfig {
  label: string;
  columns: { header: string; accessor: (m: Record<string, unknown>) => string }[];
}

const typeConfigs: Record<string, TypeConfig> = {
  baserunning: {
    label: "Baserunning Tables",
    columns: [
      { header: "Source PAs", accessor: (m) => ((m as unknown as BaserunningMetrics).n_pas ?? 0).toLocaleString() },
      { header: "Lookup Keys", accessor: (m) => ((m as unknown as BaserunningMetrics).n_keys ?? 0).toLocaleString() },
      { header: "Transitions", accessor: (m) => ((m as unknown as BaserunningMetrics).n_transition_entries ?? 0).toLocaleString() },
      { header: "Source", accessor: (m) => (m as unknown as BaserunningMetrics).source ?? "—" },
    ],
  },
  pitcher_exit: {
    label: "Pitcher Exit Models",
    columns: [
      { header: "Training Rows", accessor: (m) => ((m as unknown as PitcherExitMetrics).n_training_rows ?? 0).toLocaleString() },
      { header: "AUC", accessor: (m) => ((m as unknown as PitcherExitMetrics).model_metrics?.auc ?? 0).toFixed(4) },
      { header: "Brier", accessor: (m) => ((m as unknown as PitcherExitMetrics).model_metrics?.brier ?? 0).toFixed(4) },
      { header: "Log Loss", accessor: (m) => ((m as unknown as PitcherExitMetrics).model_metrics?.log_loss ?? 0).toFixed(4) },
      { header: "Pos Rate", accessor: (m) => {
        const rate = (m as unknown as PitcherExitMetrics).model_metrics?.pos_rate;
        return rate != null ? `${(rate * 100).toFixed(1)}%` : "—";
      }},
    ],
  },
  win_expectancy: {
    label: "Win Expectancy Tables",
    columns: [
      { header: "Games", accessor: (m) => ((m as unknown as WinExpectancyMetrics).n_games ?? 0).toLocaleString() },
      { header: "PAs", accessor: (m) => ((m as unknown as WinExpectancyMetrics).n_pas ?? 0).toLocaleString() },
      { header: "Full Keys", accessor: (m) => ((m as unknown as WinExpectancyMetrics).n_full_keys ?? 0).toLocaleString() },
      { header: "Coarse Keys", accessor: (m) => ((m as unknown as WinExpectancyMetrics).n_coarse_keys ?? 0).toLocaleString() },
      { header: "Run Diff", accessor: (m) => {
        const range = (m as unknown as WinExpectancyMetrics).run_diff_range;
        return range ? `[${range[0]}, ${range[1]}]` : "—";
      }},
    ],
  },
};

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

const col = createColumnHelper<Artifact>();

export default function ArtifactDetail({
  artifactType,
}: {
  artifactType: string;
}) {
  const config = typeConfigs[artifactType];
  const { data = [], isLoading, error } = useArtifacts(artifactType);
  const setSlot = useSetArtifactSlot();
  const deleteArt = useDeleteArtifact();

  const promoting = setSlot.isPending ? (setSlot.variables?.artifactId ?? null) : null;

  const [sorting, setSorting] = useState<SortingState>([
    { id: "created_at", desc: true },
  ]);

  const handleSetSlot = (artifactId: string, slot: string) => {
    setSlot.mutate({ artifactId, slot });
  };

  const handleDelete = (artifactId: string) => {
    if (!confirm("Delete this artifact? This cannot be undone.")) return;
    deleteArt.mutate(artifactId);
  };

  const columns = [
    col.accessor("run_id", {
      header: "Run",
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
    col.display({
      id: "slots",
      header: "Slot",
      cell: (info) => <SlotBadges isProd={info.row.original.is_prod ?? false} isTest={info.row.original.is_test ?? false} />,
    }),
    ...(config?.columns.map((c, i) =>
      col.display({
        id: `metric_${i}`,
        header: c.header,
        cell: (info) => (
          <span className="font-mono tabular-nums">
            {c.accessor(info.row.original.metrics)}
          </span>
        ),
      }),
    ) ?? []),
    col.display({
      id: "actions",
      header: "",
      cell: (info) => {
        const row = info.row.original;
        const isPromoting = promoting === row.artifact_id;
        const inSlot = row.is_prod || row.is_test;
        return (
          <div className="flex gap-1">
            <button
              disabled={row.is_prod || isPromoting}
              onClick={() => handleSetSlot(row.artifact_id, "prod")}
              className="rounded border border-border px-2 py-1 text-xs hover:bg-muted disabled:opacity-30"
            >
              {isPromoting ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                "Prod"
              )}
            </button>
            <button
              disabled={row.is_test || isPromoting}
              onClick={() => handleSetSlot(row.artifact_id, "test")}
              className="rounded border border-border px-2 py-1 text-xs hover:bg-muted disabled:opacity-30"
            >
              {isPromoting ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                "Test"
              )}
            </button>
            <button
              disabled={inSlot}
              onClick={() => handleDelete(row.artifact_id)}
              className="rounded border border-border px-2 py-1 text-xs text-destructive hover:bg-destructive/10 disabled:opacity-30"
              title={inSlot ? "Unassign from slot first" : "Delete artifact"}
            >
              <Trash2 className="h-3 w-3" />
            </button>
          </div>
        );
      },
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

  if (!config) {
    return (
      <div className="p-4 text-sm text-destructive">
        Unknown artifact type: {artifactType}
      </div>
    );
  }

  if (isLoading) return <PageLoading />;

  if (error) return <ErrorBanner message={error.message} />;

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-xl font-semibold">{config.label}</h2>
        <p className="text-sm text-muted-foreground">
          {data.length} artifact{data.length !== 1 && "s"} registered
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
                    className="cursor-pointer select-none px-3 py-2 text-left font-medium text-muted-foreground hover:text-foreground"
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    {flexRender(
                      header.column.columnDef.header,
                      header.getContext(),
                    )}
                    {header.column.getCanSort() && (
                      <SortIcon sorted={header.column.getIsSorted()} />
                    )}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr
                key={row.id}
                className={`border-b border-border last:border-0 hover:bg-muted/30 ${
                  row.original.is_prod || row.original.is_test ? "bg-muted/10" : ""
                }`}
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-3 py-2">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
            {data.length === 0 && (
              <tr>
                <td
                  colSpan={columns.length}
                  className="px-3 py-8 text-center text-muted-foreground"
                >
                  No artifacts registered yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
