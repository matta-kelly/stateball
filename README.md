# stateball

End-to-end MLB modeling system: data ingestion, warehouse, ML training,
in-game Monte Carlo simulation, and a real-time dashboard.

## What it does

Predicts plate-appearance outcomes with an XGBoost model (22-class
multi-class, ~172 features), uses those probabilities to drive a
Markov chain Monte Carlo simulator, and surfaces the resulting
win-probability estimates on a live dashboard during games.

The pipeline is grain-correct end-to-end: pitch-level for the model,
PA-level for the sim transitions, game-level for win probability.

## Repo layout

| Path | What's there |
|---|---|
| `packages/simulator/xg/` | XGBoost training pipeline — data prep, train, sweep, validate, calibration |
| `packages/simulator/sim/` | Monte Carlo simulation engine — game state, batch-vectorized PA stepper, baserunning + pitcher-exit lookup tables, native XGBoost inference |
| `warehouse/orchestration/` | Dagster assets + sensors + dbt project — Statcast / Stats API ingestion, processing, feature vectors |
| `warehouse/orchestration/dag/` | dbt models (raw → processed → intermediate → analysis → feat) |
| `warehouse/deploy/` | k8s manifests for the warehouse side (Dagster, MinIO, DuckLake catalog) |
| `liveball/backend/` | FastAPI service — game state, sim results, model artifact registry |
| `liveball/data_feed/` | Live ingestion — schedule mirror, MLB Stats API live polling, sim job dispatcher, Redis-backed snapshot |
| `liveball/frontend/` | React + Vite SPA — live game cards, win probability charts, model workshop |
| `liveball/deploy/` | k8s manifests for the dashboard + sim workers |
| `.forgejo/workflows/` | CI: image build + manifest tag bump |

## How the pieces fit

```
                    ┌───────────────────────────────────────────────┐
                    │  warehouse/  —  Dagster + dbt                 │
   MLB Stats API ─► │  raw  →  proc  →  int  →  feat  →  artifacts │
                    └────────────────────┬──────────────────────────┘
                                         │  features, models, lookup tables
                                         ▼
                    ┌──────────────────────────────────────────────┐
                    │  packages/simulator/                         │
                    │  xg/   train + calibrate XGBoost             │
                    │  sim/  Markov chain + Monte Carlo (batched)  │
                    └────────────────────┬─────────────────────────┘
                                         │  P(home wins) ± SE
                                         ▼
                    ┌──────────────────────────────────────────────┐
                    │  liveball/                                   │
                    │  data_feed   live MLB poll → snapshot        │
                    │  backend     FastAPI                          │
                    │  frontend    React dashboard                  │
                    └──────────────────────────────────────────────┘
```

## Where to start reading

If you're skimming the code:

- **Dagster asset graph** — `warehouse/orchestration/definitions.py`
  Top-level wiring: jobs, schedules, sensors, all assets registered.
- **dbt models** — `warehouse/orchestration/dag/`
  Layered: `raw/` → `processed/` → `intermediate/` → `analysis/` → `feat/`.
- **XGBoost training** — `packages/simulator/xg/train.py` and `xg/configs/`
  Plus `xg/sweep.py` for hyperparameter sweeps and `xg/validate.py` for the
  validation harness.
- **Sim engine** — `packages/simulator/sim/core/batch_engine.py`
  Batch-vectorized PA stepper. The scalar version (`engine.py`) is preserved
  as the reference implementation. Decomposed-model inference avoids the
  sklearn calibration wrapper for ~5x speedup vs ONNX in batch mode.
- **Live data feed** — `liveball/data_feed/__main__.py`
  Async lifespan that runs the live MLB game manager + sim dispatcher +
  Redis-backed snapshot.
- **Dashboard** — `liveball/frontend/src/pages/GameDetail.tsx`
  Win-probability chart, current at-bat panel, lineup, box score, roster.

## Running it

This is wired for production deployment via the manifests in `*/deploy/`
(Kubernetes + Flux + ExternalSecrets in the original setup). For local dev:

```bash
# Install workspace
uv sync

# Local infra: MinIO + Postgres
go-task minio postgres

# Dagster UI
go-task dev

# XGBoost training
go-task train CONFIG=packages/simulator/xg/configs/default.toml

# Sim eval harness
go-task sim
```

The `liveball/` dashboard runs against MinIO + DuckLake + Redis — see
`liveball/.env.example` for the env vars it needs and `liveball/deploy/`
for how it's wired in production.

## Stack

- **Orchestration**: Dagster (assets, sensors, schedules)
- **Warehouse**: DuckDB via DuckLake (S3 + Postgres catalog), dbt for transforms
- **Modeling**: XGBoost + isotonic calibration, decomposed for inference
- **Simulator**: NumPy-vectorized Markov chain Monte Carlo, baserunning + pitcher-exit lookup tables
- **Backend**: Python 3.12, FastAPI, Redis (live state), SQLite (schedule)
- **Frontend**: React + Vite + TypeScript, TanStack Query, Apache ECharts, shadcn/ui + Tailwind
- **Deploy**: Kubernetes + Flux (GitOps), ExternalSecrets, image registry-driven CI

## License

MIT — see [LICENSE](LICENSE).
