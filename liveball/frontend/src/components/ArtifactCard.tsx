import { useState } from "react";
import { cn } from "@/lib/utils";
import type { Artifact } from "@/types/api";

interface ArtifactCardProps {
  title: string;
  prodArtifact: Artifact | null;
  testArtifact: Artifact | null;
  renderMetrics: (artifact: Artifact) => React.ReactNode;
  onClick: () => void;
}

export default function ArtifactCard({
  title,
  prodArtifact,
  testArtifact,
  renderMetrics,
  onClick,
}: ArtifactCardProps) {
  const hasTest = testArtifact != null;
  const [slot, setSlot] = useState<"prod" | "test">("prod");
  const artifact = slot === "prod" ? prodArtifact : testArtifact;

  return (
    <div
      onClick={onClick}
      className="flex min-w-[260px] flex-1 cursor-pointer flex-col rounded-lg border border-border bg-card p-5 transition-colors hover:border-foreground/20"
    >
      {/* Header: title + slot toggle */}
      <div className="mb-3 flex items-start justify-between gap-2">
        <h4 className="text-base font-semibold text-foreground">{title}</h4>
        {hasTest && (
          <div
            className="flex shrink-0 items-center gap-0.5 rounded bg-muted p-0.5"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setSlot("prod")}
              className={cn(
                "rounded px-1.5 py-0.5 text-[10px] font-medium transition-colors",
                slot === "prod"
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground",
              )}
            >
              prod
            </button>
            <button
              onClick={() => setSlot("test")}
              className={cn(
                "rounded px-1.5 py-0.5 text-[10px] font-medium transition-colors",
                slot === "test"
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground",
              )}
            >
              test
            </button>
          </div>
        )}
        {!hasTest && prodArtifact && (
          <span className="shrink-0 rounded bg-success/15 px-1.5 py-0.5 text-[10px] font-medium text-success">
            prod
          </span>
        )}
      </div>

      {/* Content */}
      {artifact ? (
        <div className="space-y-1.5">
          <p className="truncate font-mono text-sm text-muted-foreground">
            {artifact.run_id}
          </p>
          <div className="text-sm text-foreground">{renderMetrics(artifact)}</div>
        </div>
      ) : (
        <p className="text-sm text-muted-foreground/50">Not assigned</p>
      )}
    </div>
  );
}
