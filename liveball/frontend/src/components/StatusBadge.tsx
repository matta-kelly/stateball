import { Badge } from "@/components/ui/badge";

interface StatusBadgeProps {
  abstractState: string;
  status: string;
}

const stateStyles: Record<string, string> = {
  Live: "bg-success/20 text-success border-success/30",
  Final: "bg-muted text-muted-foreground border-muted",
  Preview: "bg-info/20 text-info border-info/30",
};

export default function StatusBadge({ abstractState }: StatusBadgeProps) {
  const style = stateStyles[abstractState] ?? stateStyles.Preview;
  return (
    <Badge className={style}>
      {abstractState === "Live" && (
        <span className="mr-0.5 h-1.5 w-1.5 animate-pulse rounded-full bg-success" />
      )}
      {abstractState}
    </Badge>
  );
}
