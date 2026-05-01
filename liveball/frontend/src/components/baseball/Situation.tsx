type SituationSize = "sm" | "md";

const baseSizes = {
  sm: { base: "h-3 w-3", gap: "gap-0.5", row: "gap-2.5" },
  md: { base: "h-4 w-4 border-2", gap: "gap-1", row: "gap-3" },
};

const outSizes = {
  sm: { dot: "h-2 w-2", gap: "gap-1" },
  md: { dot: "h-2.5 w-2.5 border-2", gap: "gap-1.5" },
};

export function Bases({ runners, size = "sm" }: { runners: string | null; size?: SituationSize }) {
  let occupied = { first: false, second: false, third: false };
  if (runners) {
    try {
      const parsed = JSON.parse(runners);
      occupied = {
        first: !!parsed.first,
        second: !!parsed.second,
        third: !!parsed.third,
      };
    } catch {
      // malformed
    }
  }

  const s = baseSizes[size];
  const base = (on: boolean) =>
    `${s.base} rotate-45 ${size === "sm" ? "border" : ""} ${
      on
        ? "border-warning bg-warning"
        : "border-zinc-600 bg-transparent"
    }`;

  return (
    <div className={`flex flex-col items-center ${s.gap}`}>
      <div className={base(occupied.second)} />
      <div className={`flex ${s.row}`}>
        <div className={base(occupied.third)} />
        <div className={base(occupied.first)} />
      </div>
    </div>
  );
}

export function Outs({ count, size = "sm" }: { count: number | null; size?: SituationSize }) {
  const n = count ?? 0;
  const s = outSizes[size];
  return (
    <div className={`flex ${s.gap}`}>
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className={`${s.dot} rounded-full ${size === "sm" ? "border" : ""} ${
            i < n ? "border-warning bg-warning" : "border-zinc-600 bg-transparent"
          }`}
        />
      ))}
    </div>
  );
}

export function Count({ balls, strikes }: { balls: number | null; strikes: number | null }) {
  if (balls == null && strikes == null) return null;
  return (
    <span className="text-xs tabular-nums text-muted-foreground">
      {balls ?? 0}-{strikes ?? 0}
    </span>
  );
}

export function InningLabel({ inning, half }: { inning: number | null; half: string | null }) {
  if (inning == null) return <span className="text-xs font-semibold text-success">--</span>;
  const prefix = half === "Top" || half === "top" ? "TOP" : "BOT";
  return (
    <span className="text-xs font-semibold uppercase tracking-wide text-success">
      {prefix} {inning}
    </span>
  );
}
