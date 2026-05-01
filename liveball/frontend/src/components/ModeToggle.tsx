import { useLocation, useNavigate } from "react-router-dom";
import { cn } from "@/lib/utils";

type Mode = "ops" | "workshop" | "stats";

const MODE_DEFAULTS: Record<Mode, string> = {
  ops: "/",
  workshop: "/workshop",
  stats: "/stats",
};

function detectMode(pathname: string): Mode {
  if (pathname.startsWith("/stats")) return "stats";
  if (pathname.startsWith("/workshop")) return "workshop";
  return "ops";
}

export default function ModeToggle({ className }: { className?: string }) {
  const location = useLocation();
  const navigate = useNavigate();
  const mode = detectMode(location.pathname);

  const handleSwitch = (target: Mode) => {
    sessionStorage.setItem(`mode:${mode}:path`, location.pathname);
    const saved = sessionStorage.getItem(`mode:${target}:path`);
    navigate(saved || MODE_DEFAULTS[target]);
  };

  const pill = (target: Mode, label: string) => (
    <button
      onClick={() => handleSwitch(target)}
      className={cn(
        "rounded px-2.5 py-1 text-xs font-medium transition-colors",
        mode === target
          ? "bg-background text-foreground shadow-sm"
          : "text-muted-foreground hover:text-foreground",
      )}
    >
      {label}
    </button>
  );

  return (
    <div className={cn("flex items-center gap-0.5 rounded-md bg-muted p-0.5", className)}>
      {pill("ops", "Operations")}
      {pill("workshop", "Workshop")}
      {pill("stats", "Stats")}
    </div>
  );
}
