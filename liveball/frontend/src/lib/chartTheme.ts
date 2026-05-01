function cssVar(name: string): string {
  return getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
}

export function getChartColors(): string[] {
  return [
    cssVar("--chart-1"),
    cssVar("--chart-2"),
    cssVar("--chart-3"),
    cssVar("--chart-4"),
    cssVar("--chart-5"),
  ].map((c) => c || "#3b82f6");
}

export function getChartTheme() {
  const fg = cssVar("--foreground") || "#fafafa";
  const muted = cssVar("--muted-foreground") || "#71717a";
  const border = cssVar("--border") || "#27272a";
  const card = cssVar("--card") || "#18181b";

  return {
    textStyle: { color: fg, fontSize: 11 },
    axisLabel: { color: muted, fontSize: 10 },
    splitLine: { lineStyle: { color: border, opacity: 0.3 } },
    axisLine: { lineStyle: { color: border } },
    tooltip: {
      backgroundColor: card,
      borderColor: border,
      textStyle: { color: fg, fontSize: 11 },
    },
  };
}
