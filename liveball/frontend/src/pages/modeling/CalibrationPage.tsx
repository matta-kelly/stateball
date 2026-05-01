import { Loader2, ShieldCheck } from "lucide-react";
import { useCalibrations, useArtifactsByType, usePromoteArtifact } from "@/hooks/queries";
import { PageLoading } from "@/components/ui/states";


export const CALIBRATION_ARTIFACT_TYPES = ["n_lookup", "stopping_thresholds", "gamma_schedule", "horizon_weights"] as const;

export function CalibrationArtifactTable({ artifactType }: { artifactType: string }) {
  const { data: versions = [], isLoading } = useArtifactsByType(artifactType);
  const promoteArt = usePromoteArtifact();

  const handlePromote = (artifactId: string) => {
    promoteArt.mutate({ artifactId });
  };

  if (isLoading) {
    return <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />;
  }

  if (versions.length === 0) {
    return <p className="text-sm text-muted-foreground">No artifacts built yet.</p>;
  }

  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="border-b border-border bg-muted/30">
          <th className="px-2 py-1.5 text-left font-medium">Eval ID</th>
          <th className="px-2 py-1.5 text-left font-medium">Created</th>
          <th className="px-2 py-1.5 text-right font-medium">States</th>
          <th className="px-2 py-1.5 text-center font-medium">Action</th>
        </tr>
      </thead>
      <tbody>
        {versions.map((v) => (
          <tr key={`${v.artifact_id}-${v.created_at}`} className="border-b border-border last:border-0 hover:bg-muted/20">
            <td className="px-2 py-1 font-mono">{v.run_id}</td>
            <td className="px-2 py-1 text-muted-foreground">
              {new Date(v.created_at).toLocaleDateString()}
            </td>
            <td className="px-2 py-1 text-right">
              {(v.metrics as Record<string, number>)?.n_states ?? "—"}
            </td>
            <td className="px-2 py-1 text-center">
              {v.is_prod ? (
                <ShieldCheck className="inline h-4 w-4 text-success" />
              ) : (
                <button
                  onClick={() => handlePromote(v.artifact_id)}
                  disabled={promoteArt.isPending}
                  className="rounded bg-primary px-2 py-0.5 text-xs font-medium text-primary-foreground hover:bg-primary/80 disabled:opacity-50"
                >
                  {promoteArt.isPending && promoteArt.variables?.artifactId === v.artifact_id ? (
                    <Loader2 className="inline h-3 w-3 animate-spin" />
                  ) : (
                    "Promote"
                  )}
                </button>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function CalibrationRuns() {
  const { data: runs = [], isLoading } = useCalibrations();

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold">Calibration Runs</h2>
        <p className="text-sm text-muted-foreground">
          Full-completion naive MC evals that produce calibration data.
        </p>
      </div>

      {isLoading ? (
        <PageLoading />
      ) : (
        <div className="overflow-x-auto rounded border border-border">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-muted/50">
                <th className="px-3 py-2 text-left font-medium">Eval ID</th>
                <th className="px-3 py-2 text-left font-medium">Date</th>
                <th className="px-3 py-2 text-right font-medium">Games</th>
                <th className="px-3 py-2 text-right font-medium">Accuracy</th>
                <th className="px-3 py-2 text-right font-medium">Time/Game</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.eval_id} className="border-b border-border last:border-0 hover:bg-muted/30">
                  <td className="px-3 py-2 font-mono text-xs">{run.eval_id}</td>
                  <td className="px-3 py-2 text-xs text-muted-foreground">
                    {new Date(run.created_at).toLocaleDateString()}
                  </td>
                  <td className="px-3 py-2 text-right">{run.n_games?.toLocaleString()}</td>
                  <td className="px-3 py-2 text-right">
                    {run.accuracy != null ? `${(run.accuracy * 100).toFixed(1)}%` : "—"}
                  </td>
                  <td className="px-3 py-2 text-right">
                    {run.mean_mc_time != null ? `${run.mean_mc_time.toFixed(2)}s` : "—"}
                  </td>
                </tr>
              ))}
              {runs.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-3 py-8 text-center text-muted-foreground">
                    No calibration runs found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default function CalibrationPage() {
  return (
    <div className="space-y-6">
      <CalibrationRuns />
    </div>
  );
}
