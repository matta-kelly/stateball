import { useState } from "react";
import { useModelingOverview, useArtifactsByType } from "@/hooks/queries";
import ArtifactCard from "@/components/ArtifactCard";
import ArtifactModal from "@/components/ArtifactModal";
import { PageLoading } from "@/components/ui/states";
import type {
  Artifact,
  XGBoostMetrics,
  BaserunningMetrics,
  PitcherExitMetrics,
  WinExpectancyMetrics,
  FeatureManifestMetrics,
} from "@/types/api";
import XGBoostModels from "./XGBoostModels";
import ArtifactDetail from "./ArtifactDetail";
import FeatureSelections from "./FeatureSelections";
import { CALIBRATION_ARTIFACT_TYPES, CalibrationArtifactTable } from "./CalibrationPage";

// ── Metric renderers ──

function xgboostMetrics(a: Artifact) {
  const m = a.metrics as unknown as XGBoostMetrics;
  const seasons = m.seasons?.length
    ? `${m.seasons[0]}–${m.seasons[m.seasons.length - 1]}`
    : "";
  return (
    <>
      <div className="font-mono tabular-nums">
        Skill: {((m.brier_skill ?? 0) * 100).toFixed(1)}%
      </div>
      <div className="text-muted-foreground">
        {(m.total_pas ?? 0).toLocaleString()} PAs{seasons && ` · ${seasons}`}
      </div>
    </>
  );
}

function featureMetrics(a: Artifact) {
  const m = a.metrics as unknown as FeatureManifestMetrics;
  return (
    <>
      <div>{m.n_features ?? 0} live · {m.n_sim_features ?? 0} sim</div>
      <div className="text-muted-foreground">{m.method ?? "—"}</div>
    </>
  );
}

function baserunningMetrics(a: Artifact) {
  const m = a.metrics as unknown as BaserunningMetrics;
  return (
    <>
      <div>{(m.n_pas ?? 0).toLocaleString()} PAs</div>
      <div className="text-muted-foreground">{(m.n_keys ?? 0).toLocaleString()} keys</div>
    </>
  );
}

function pitcherExitMetrics(a: Artifact) {
  const m = a.metrics as unknown as PitcherExitMetrics;
  return (
    <>
      <div className="font-mono tabular-nums">
        AUC: {(m.model_metrics?.auc ?? 0).toFixed(3)} · Brier: {(m.model_metrics?.brier ?? 0).toFixed(3)}
      </div>
      <div className="text-muted-foreground">{(m.n_training_rows ?? 0).toLocaleString()} rows</div>
    </>
  );
}

function winExpMetrics(a: Artifact) {
  const m = a.metrics as unknown as WinExpectancyMetrics;
  return (
    <>
      <div>{(m.n_games ?? 0).toLocaleString()} games</div>
      <div className="text-muted-foreground">{(m.n_full_keys ?? 0).toLocaleString()} keys</div>
    </>
  );
}

function calibrationMetrics(a: Artifact) {
  const m = a.metrics as Record<string, number>;
  return <div>{m?.n_states ?? "—"} states</div>;
}

// ── Calibration row ──

const CAL_TYPES = ["n_lookup", "stopping_thresholds", "gamma_schedule", "horizon_weights"] as const;
const CAL_LABELS: Record<string, string> = {
  n_lookup: "N Lookup",
  stopping_thresholds: "Stop Thresholds",
  gamma_schedule: "Gamma Schedule",
  horizon_weights: "Horizon Weights",
};

function CalibrationCard({
  artifactType,
  label,
  onClick,
}: {
  artifactType: string;
  label: string;
  onClick: () => void;
}) {
  const { data: versions = [] } = useArtifactsByType(artifactType);
  const prodArtifact = versions.find((v) => v.is_prod) ?? null;
  return (
    <ArtifactCard
      title={label}
      prodArtifact={prodArtifact as Artifact | null}
      testArtifact={null}
      renderMetrics={calibrationMetrics}
      onClick={onClick}
    />
  );
}

// ── Modal content mapping ──

const MODAL_TITLES: Record<string, string> = {
  xgboost_sim: "XGBoost Sim",
  xgboost_live: "XGBoost Live",
  feature_manifest: "Feature Selections",
  baserunning: "Baserunning",
  pitcher_exit: "Pitcher Exit",
  win_expectancy: "Win Expectancy",
  calibration: "Calibration Artifacts",
};

function ModalContent({ type }: { type: string }) {
  switch (type) {
    case "xgboost_sim":
    case "xgboost_live":
      return <XGBoostModels />;
    case "feature_manifest":
      return <FeatureSelections />;
    case "baserunning":
    case "pitcher_exit":
    case "win_expectancy":
      return <ArtifactDetail artifactType={type} />;
    case "calibration":
      return (
        <div className="grid gap-4 md:grid-cols-2">
          {CALIBRATION_ARTIFACT_TYPES.map((atype) => (
            <div key={atype} className="rounded-lg border border-border p-3">
              <h3 className="mb-2 text-sm font-semibold">{atype.replace(/_/g, " ")}</h3>
              <CalibrationArtifactTable artifactType={atype} />
            </div>
          ))}
        </div>
      );
    default:
      return null;
  }
}

// ── Main ──

export default function WorkshopCards() {
  const { data: overview, isLoading } = useModelingOverview();
  const [modalType, setModalType] = useState<string | null>(null);

  if (isLoading) return <PageLoading />;

  const prod = overview?.prod.artifacts ?? {};
  const test = overview?.test.artifacts ?? {};

  return (
    <>
      <div className="space-y-6">
        {/* Models row */}
        <div className="rounded-lg border border-border bg-card/50 p-4">
          <h3 className="mb-3 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Models
          </h3>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <ArtifactCard
              title="XGBoost Sim"
              prodArtifact={prod.xgboost_sim ?? null}
              testArtifact={test.xgboost_sim ?? null}
              renderMetrics={xgboostMetrics}
              onClick={() => setModalType("xgboost_sim")}
            />
            <ArtifactCard
              title="XGBoost Live"
              prodArtifact={prod.xgboost_live ?? null}
              testArtifact={test.xgboost_live ?? null}
              renderMetrics={xgboostMetrics}
              onClick={() => setModalType("xgboost_live")}
            />
            <ArtifactCard
              title="Features"
              prodArtifact={prod.feature_manifest ?? null}
              testArtifact={test.feature_manifest ?? null}
              renderMetrics={featureMetrics}
              onClick={() => setModalType("feature_manifest")}
            />
          </div>
        </div>

        {/* Tables row */}
        <div className="rounded-lg border border-border bg-card/50 p-4">
          <h3 className="mb-3 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Tables
          </h3>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <ArtifactCard
              title="Baserunning"
              prodArtifact={prod.baserunning ?? null}
              testArtifact={test.baserunning ?? null}
              renderMetrics={baserunningMetrics}
              onClick={() => setModalType("baserunning")}
            />
            <ArtifactCard
              title="Pitcher Exit"
              prodArtifact={prod.pitcher_exit ?? null}
              testArtifact={test.pitcher_exit ?? null}
              renderMetrics={pitcherExitMetrics}
              onClick={() => setModalType("pitcher_exit")}
            />
            <ArtifactCard
              title="Win Expectancy"
              prodArtifact={prod.win_expectancy ?? null}
              testArtifact={test.win_expectancy ?? null}
              renderMetrics={winExpMetrics}
              onClick={() => setModalType("win_expectancy")}
            />
          </div>
        </div>

        {/* Calibration row */}
        <div className="rounded-lg border border-border bg-card/50 p-4">
          <h3 className="mb-3 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Calibration
          </h3>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            {CAL_TYPES.map((t) => (
              <CalibrationCard
                key={t}
                artifactType={t}
                label={CAL_LABELS[t]}
                onClick={() => setModalType("calibration")}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Detail modal */}
      <ArtifactModal
        open={modalType !== null}
        onClose={() => setModalType(null)}
        title={modalType ? MODAL_TITLES[modalType] ?? modalType : ""}
      >
        {modalType && <ModalContent type={modalType} />}
      </ArtifactModal>
    </>
  );
}
