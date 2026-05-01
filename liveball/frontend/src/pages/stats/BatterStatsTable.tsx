import { useState } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import { cn } from "@/lib/utils";
import type { BatterStats } from "@/types/api";

const fmt3 = (v: number | null) => (v != null ? v.toFixed(3) : "--");
const fmtPct = (v: number | null) => (v != null ? (v * 100).toFixed(1) + "%" : "--");
const fmtEv = (v: number | null) => (v != null ? v.toFixed(1) : "--");
const fmtInt = (v: number | null) => (v != null ? String(v) : "--");

const columns: ColumnDef<BatterStats, unknown>[] = [
  // Player info
  { accessorKey: "full_name", header: "Player", size: 160 },
  { accessorKey: "team_name", header: "Team", size: 80 },
  { accessorKey: "position", header: "Pos", size: 50 },
  // Season
  { accessorKey: "season_pa", header: "PA", cell: (i) => fmtInt(i.getValue() as number) },
  { accessorKey: "season_ba", header: "BA", cell: (i) => fmt3(i.getValue() as number) },
  { accessorKey: "season_obp", header: "OBP", cell: (i) => fmt3(i.getValue() as number) },
  { accessorKey: "season_slg", header: "SLG", cell: (i) => fmt3(i.getValue() as number) },
  { accessorKey: "season_ops", header: "OPS", cell: (i) => fmt3(i.getValue() as number) },
  { accessorKey: "season_woba", header: "wOBA", cell: (i) => fmt3(i.getValue() as number) },
  { accessorKey: "season_k_pct", header: "K%", cell: (i) => fmtPct(i.getValue() as number) },
  { accessorKey: "season_bb_pct", header: "BB%", cell: (i) => fmtPct(i.getValue() as number) },
  { accessorKey: "season_avg_ev", header: "EV", cell: (i) => fmtEv(i.getValue() as number) },
  { accessorKey: "season_barrel_pct", header: "Brl%", cell: (i) => fmtPct(i.getValue() as number) },
  // Career
  { accessorKey: "career_pa", header: "PA", cell: (i) => fmtInt(i.getValue() as number), meta: { group: "career" } },
  { accessorKey: "career_ba", header: "BA", cell: (i) => fmt3(i.getValue() as number) },
  { accessorKey: "career_obp", header: "OBP", cell: (i) => fmt3(i.getValue() as number) },
  { accessorKey: "career_slg", header: "SLG", cell: (i) => fmt3(i.getValue() as number) },
  { accessorKey: "career_ops", header: "OPS", cell: (i) => fmt3(i.getValue() as number) },
  { accessorKey: "career_woba", header: "wOBA", cell: (i) => fmt3(i.getValue() as number) },
  { accessorKey: "career_k_pct", header: "K%", cell: (i) => fmtPct(i.getValue() as number) },
  { accessorKey: "career_bb_pct", header: "BB%", cell: (i) => fmtPct(i.getValue() as number) },
  { accessorKey: "career_avg_ev", header: "EV", cell: (i) => fmtEv(i.getValue() as number) },
  { accessorKey: "career_barrel_pct", header: "Brl%", cell: (i) => fmtPct(i.getValue() as number) },
];

// Indices for column group headers
const SEASON_START = 3;
const SEASON_END = 13; // exclusive
const CAREER_START = 13;
const CAREER_END = 23;

interface Props {
  data: BatterStats[];
  search: string;
}

export default function BatterStatsTable({ data, search }: Props) {
  const [sorting, setSorting] = useState<SortingState>([]);

  const table = useReactTable({
    data,
    columns,
    state: { sorting, globalFilter: search },
    onSortingChange: setSorting,
    onGlobalFilterChange: () => {},
    globalFilterFn: (row, _columnId, filterValue) => {
      const q = filterValue.toLowerCase();
      for (const key of ["full_name", "team_name", "position"] as const) {
        const v = row.getValue(key) as string | null;
        if (v && v.toLowerCase().includes(q)) return true;
      }
      return false;
    },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  });

  return (
    <div className="overflow-x-auto rounded-md border border-border">
      <table className="w-full text-sm">
        {/* Column group headers */}
        <thead>
          <tr className="border-b border-border bg-muted/60 text-xs text-muted-foreground">
            <th colSpan={SEASON_START} className="px-3 py-1.5 text-left font-medium" />
            <th
              colSpan={SEASON_END - SEASON_START}
              className="border-l border-border px-3 py-1.5 text-center font-medium"
            >
              Season
            </th>
            <th
              colSpan={CAREER_END - CAREER_START}
              className="border-l border-border px-3 py-1.5 text-center font-medium"
            >
              Career
            </th>
          </tr>
          {table.getHeaderGroups().map((hg) => (
            <tr key={hg.id} className="border-b border-border bg-muted/40 text-xs text-muted-foreground">
              {hg.headers.map((h, idx) => (
                <th
                  key={h.id}
                  onClick={h.column.getToggleSortingHandler()}
                  className={cn(
                    "cursor-pointer select-none whitespace-nowrap px-3 py-1.5 text-left font-medium",
                    idx === SEASON_START && "border-l border-border",
                    idx === CAREER_START && "border-l border-border",
                  )}
                >
                  {flexRender(h.column.columnDef.header, h.getContext())}
                  {{ asc: " \u25B2", desc: " \u25BC" }[h.column.getIsSorted() as string] ?? ""}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id} className="border-b border-border transition-colors last:border-0 hover:bg-muted/20">
              {row.getVisibleCells().map((cell, idx) => (
                <td
                  key={cell.id}
                  className={cn(
                    "whitespace-nowrap px-3 py-1.5 font-mono text-xs",
                    idx < SEASON_START ? "text-foreground" : "text-foreground/80",
                    idx === SEASON_START && "border-l border-border",
                    idx === CAREER_START && "border-l border-border",
                  )}
                >
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {table.getRowModel().rows.length === 0 && (
        <div className="p-6 text-center text-sm text-muted-foreground">No results</div>
      )}
    </div>
  );
}
