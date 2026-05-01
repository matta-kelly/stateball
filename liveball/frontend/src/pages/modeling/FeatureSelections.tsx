import { useState } from "react";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import { ChevronDown, ChevronRight, Loader2, Trash2 } from "lucide-react";
import { useArtifacts, useSetArtifactSlot, useDeleteArtifact } from "@/hooks/queries";
import { SortIcon } from "@/components/ui/sort-icon";
import { SlotBadges } from "@/components/SlotBadges";
import { PageLoading, ErrorBanner } from "@/components/ui/states";
import type { Artifact, FeatureManifestMetrics } from "@/types/api";

interface ManifestRow {
  artifact_id: string;
  run_id: string;
  created_at: string;
  is_prod: boolean;
  is_test: boolean;
  method: string;
  n_features: number;
  n_sim_features: number;
  features: string[];
  sim_features: string[];
}

function toRows(artifacts: Artifact[]): ManifestRow[] {
  return artifacts.map((a) => {
    const m = a.metrics as unknown as FeatureManifestMetrics;
    return {
      artifact_id: a.artifact_id,
      run_id: a.run_id,
      created_at: a.created_at,
      is_prod: a.is_prod ?? false,
      is_test: a.is_test ?? false,
      method: m.method ?? "—",
      n_features: m.n_features ?? 0,
      n_sim_features: m.n_sim_features ?? 0,
      features: m.features ?? [],
      sim_features: m.sim_features ?? [],
    };
  });
}

function ExpandedFeatures({ row }: { row: ManifestRow }) {
  const simSet = new Set(row.sim_features);
  const liveOnly = row.features.filter((f) => !simSet.has(f));

  return (
    <tr className="bg-muted/20">
      <td colSpan={8} className="px-4 py-4">
        <div className="grid grid-cols-2 gap-6">
          <div>
            <p className="mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              Live Features ({row.n_features})
            </p>
            <div className="flex flex-wrap gap-1">
              {row.features.map((f) => (
                <span
                  key={f}
                  className={`rounded px-2 py-0.5 font-mono text-xs ${
                    liveOnly.includes(f)
                      ? "bg-amber-500/15 text-amber-400"
                      : "bg-muted text-foreground"
                  }`}
                >
                  {f}
                </span>
              ))}
            </div>
            {liveOnly.length > 0 && (
              <p className="mt-2 text-xs text-muted-foreground">
                <span className="rounded bg-amber-500/15 px-1 text-amber-400">amber</span> = live-only (count state, excluded from sim)
              </p>
            )}
          </div>
          <div>
            <p className="mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              Sim Features ({row.n_sim_features})
            </p>
            <div className="flex flex-wrap gap-1">
              {row.sim_features.map((f) => (
                <span key={f} className="rounded bg-muted px-2 py-0.5 font-mono text-xs text-foreground">
                  {f}
                </span>
              ))}
            </div>
          </div>
        </div>
      </td>
    </tr>
  );
}

function ManifestTable({
  data,
  promoting,
  onSetSlot,
  onDelete,
}: {
  data: ManifestRow[];
  promoting: string | null;
  onSetSlot: (artifactId: string, slot: string) => void;
  onDelete: (artifactId: string) => void;
}) {
  const [sorting, setSorting] = useState<SortingState>([
    { id: "created_at", desc: true },
  ]);
  const [expandedRow, setExpandedRow] = useState<string | null>(null);

  const col = createColumnHelper<ManifestRow>();

  const columns = [
    col.display({
      id: "expand",
      header: "",
      cell: (info) => {
        const id = info.row.original.artifact_id;
        return expandedRow === id
          ? <ChevronDown className="h-3 w-3 text-muted-foreground" />
          : <ChevronRight className="h-3 w-3 text-muted-foreground" />;
      },
    }),
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
      cell: (info) => <SlotBadges isProd={info.row.original.is_prod} isTest={info.row.original.is_test} />,
    }),
    col.accessor("method", {
      header: "Method",
      cell: (info) => (
        <span className="font-mono text-xs text-muted-foreground">{info.getValue()}</span>
      ),
    }),
    col.accessor("n_features", {
      header: "Live Features",
      cell: (info) => info.getValue(),
    }),
    col.accessor("n_sim_features", {
      header: "Sim Features",
      cell: (info) => info.getValue(),
    }),
    col.display({
      id: "actions",
      header: "",
      cell: (info) => {
        const row = info.row.original;
        const isPromoting = promoting === row.artifact_id;
        const inSlot = row.is_prod || row.is_test;
        return (
          <div className="flex gap-1" onClick={(e) => e.stopPropagation()}>
            <button
              disabled={row.is_prod || isPromoting}
              onClick={() => onSetSlot(row.artifact_id, "prod")}
              className="rounded border border-border px-2 py-1 text-xs hover:bg-muted disabled:opacity-30"
            >
              {isPromoting ? <Loader2 className="h-3 w-3 animate-spin" /> : "Prod"}
            </button>
            <button
              disabled={row.is_test || isPromoting}
              onClick={() => onSetSlot(row.artifact_id, "test")}
              className="rounded border border-border px-2 py-1 text-xs hover:bg-muted disabled:opacity-30"
            >
              {isPromoting ? <Loader2 className="h-3 w-3 animate-spin" /> : "Test"}
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
          {table.getRowModel().rows.map((row) => (
            <>
              <tr
                key={row.id}
                className={`cursor-pointer border-b border-border hover:bg-muted/30 ${
                  row.original.is_prod || row.original.is_test ? "bg-muted/10" : ""
                } ${expandedRow === row.original.artifact_id ? "border-b-0" : "last:border-0"}`}
                onClick={() =>
                  setExpandedRow(
                    expandedRow === row.original.artifact_id ? null : row.original.artifact_id
                  )
                }
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-3 py-2">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
              {expandedRow === row.original.artifact_id && (
                <ExpandedFeatures key={`${row.id}-expand`} row={row.original} />
              )}
            </>
          ))}
          {data.length === 0 && (
            <tr>
              <td colSpan={columns.length} className="px-3 py-8 text-center text-muted-foreground">
                No feature selections registered yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

export default function FeatureSelections() {
  const { data: artifacts = [], isLoading, error } = useArtifacts("feature_manifest");
  const setSlot = useSetArtifactSlot();
  const deleteArt = useDeleteArtifact();
  const data = toRows(artifacts);
  const promoting = setSlot.isPending ? (setSlot.variables?.artifactId ?? null) : null;

  const handleSetSlot = (artifactId: string, slot: string) => {
    setSlot.mutate({ artifactId, slot });
  };

  const handleDelete = (artifactId: string) => {
    if (!confirm("Delete this feature selection? This cannot be undone.")) return;
    deleteArt.mutate(artifactId);
  };

  if (isLoading) return <PageLoading />;

  if (error) return <ErrorBanner message={error.message} />;

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold">Feature Selections</h3>
        <p className="text-sm text-muted-foreground">
          {data.length} selection{data.length !== 1 && "s"} registered — click a row to see features
        </p>
      </div>
      <ManifestTable
        data={data}
        promoting={promoting}
        onSetSlot={handleSetSlot}
        onDelete={handleDelete}
      />
    </div>
  );
}
