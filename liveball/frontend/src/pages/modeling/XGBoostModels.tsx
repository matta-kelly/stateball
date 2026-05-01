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
import type { Artifact, XGBoostMetrics } from "@/types/api";

interface ModelRow {
  artifact_id: string;
  artifact_type: string;
  run_id: string;
  created_at: string;
  is_prod: boolean;
  is_test: boolean;
  brier_skill: number;
  brier_ci_lower: number;
  brier_ci_upper: number;
  n_features: number;
  total_pas: number;
  seasons: string;
  inference_ms: number;
  sweep_mode: string | null;
}

function toRows(artifacts: Artifact[]): ModelRow[] {
  return artifacts.map((a) => {
    const m = a.metrics as unknown as XGBoostMetrics;
    return {
      artifact_id: a.artifact_id,
      artifact_type: a.artifact_type,
      run_id: a.run_id,
      created_at: a.created_at,
      is_prod: a.is_prod ?? false,
      is_test: a.is_test ?? false,
      brier_skill: m.brier_skill ?? 0,
      brier_ci_lower: m.brier_ci_lower ?? 0,
      brier_ci_upper: m.brier_ci_upper ?? 0,
      n_features: m.n_features ?? 0,
      total_pas: m.total_pas ?? 0,
      seasons: m.seasons?.length ? `${m.seasons[0]}–${m.seasons[m.seasons.length - 1]}` : "",
      inference_ms: m.inference_ms_per_row ?? 0,
      sweep_mode: m.sweep_mode ?? null,
    };
  });
}

function ModelTable({
  title,
  data,
  promoting,
  onSetSlot,
  onDelete,
}: {
  title: string;
  data: ModelRow[];
  promoting: string | null;
  onSetSlot: (artifactId: string, slot: string) => void;
  onDelete: (artifactId: string) => void;
}) {
  const [sorting, setSorting] = useState<SortingState>([
    { id: "created_at", desc: true },
  ]);

  const col = createColumnHelper<ModelRow>();

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
        });
      },
    }),
    col.display({
      id: "slots",
      header: "Slot",
      cell: (info) => <SlotBadges isProd={info.row.original.is_prod} isTest={info.row.original.is_test} />,
    }),
    col.accessor("brier_skill", {
      header: "Brier Skill",
      cell: (info) => (
        <span className="font-mono tabular-nums">
          {(info.getValue() * 100).toFixed(2)}%
        </span>
      ),
    }),
    col.display({
      id: "brier_ci",
      header: "Brier CI",
      cell: (info) => {
        const row = info.row.original;
        if (!row.brier_ci_lower && !row.brier_ci_upper) return <span className="text-muted-foreground/50">—</span>;
        const halfWidth = ((row.brier_ci_upper - row.brier_ci_lower) / 2) * 100;
        return (
          <span className="font-mono tabular-nums text-xs">
            &plusmn;{halfWidth.toFixed(2)}%
          </span>
        );
      },
    }),
    col.accessor("n_features", {
      header: "Features",
      cell: (info) => info.getValue(),
    }),
    col.accessor("total_pas", {
      header: "PAs",
      cell: (info) => info.getValue().toLocaleString(),
    }),
    col.accessor("seasons", { header: "Seasons" }),
    col.accessor("inference_ms", {
      header: "Inference",
      cell: (info) => {
        const v = info.getValue();
        return v ? (
          <span className="font-mono tabular-nums text-xs">{v.toFixed(1)}ms</span>
        ) : (
          <span className="text-muted-foreground/50">—</span>
        );
      },
    }),
    col.accessor("sweep_mode", {
      header: "Sweep",
      cell: (info) => {
        const v = info.getValue();
        return v ? (
          <span className="rounded-full bg-violet-500/15 px-2 py-0.5 text-xs font-medium text-violet-500">
            {v}
          </span>
        ) : null;
      },
    }),
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
              onClick={() => onSetSlot(row.artifact_id, "prod")}
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
              onClick={() => onSetSlot(row.artifact_id, "test")}
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
              onClick={() => onDelete(row.artifact_id)}
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

  return (
    <div className="space-y-2">
      <div>
        <h3 className="text-lg font-semibold">{title}</h3>
        <p className="text-sm text-muted-foreground">
          {data.length} training run{data.length !== 1 && "s"} registered
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
                  No models registered yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function XGBoostModels() {
  const { data: simArtifacts = [], isLoading: simLoading, error: simError } = useArtifacts("xgboost_sim");
  const { data: liveArtifacts = [], isLoading: liveLoading, error: liveError } = useArtifacts("xgboost_live");
  const setSlot = useSetArtifactSlot();
  const deleteArt = useDeleteArtifact();

  const loading = simLoading || liveLoading;
  const error = simError || liveError;

  if (loading) return <PageLoading />;
  if (error) return <ErrorBanner message={error.message} />;

  const simData = toRows(simArtifacts);
  const liveData = toRows(liveArtifacts);
  const promoting = setSlot.isPending ? (setSlot.variables?.artifactId ?? null) : null;

  const handleSetSlot = (artifactId: string, slot: string) => {
    setSlot.mutate({ artifactId, slot });
  };

  const handleDelete = (artifactId: string) => {
    if (!confirm("Delete this artifact? This cannot be undone.")) return;
    deleteArt.mutate(artifactId);
  };

  return (
    <div className="space-y-8">
      <ModelTable
        title="XGBoost Sim"
        data={simData}
        promoting={promoting}
        onSetSlot={handleSetSlot}
        onDelete={handleDelete}
      />
      <ModelTable
        title="XGBoost Live"
        data={liveData}
        promoting={promoting}
        onSetSlot={handleSetSlot}
        onDelete={handleDelete}
      />
    </div>
  );
}
