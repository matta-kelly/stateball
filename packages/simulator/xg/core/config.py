"""
Experiment configuration: loading, validation, and feature resolution.

Feature blocks mirror feat_mlb__vectors.sql exactly. Config is TOML,
parsed with stdlib tomllib (Python 3.11+).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Feature block definitions — canonical source of truth
# ---------------------------------------------------------------------------

# Canonical pitch type list for per-pitch-type arsenal features.
# SQL mirror: orchestration/macros/pitch_types.sql::pitch_types()
# Adding a new pitch type: update both, run --full-refresh on counts.
PITCH_TYPES: list[str] = ["FF", "SI", "FC", "SL", "CU", "KC", "SV", "ST", "CH", "FS"]

# PA outcome taxonomy — loaded from xg/outcomes.yaml (single source of truth).
# dbt mirror: orchestration/macros/outcomes.sql (validated by tests/test_outcome_sync.py).
# Training filters rare ones via min_samples. Models predict a subset.
# Baserunning tables cover what the data supports. Each artifact
# declares which subset it handles.
_OUTCOMES_PATH = Path(__file__).resolve().parent.parent / "outcomes.yaml"


def _load_outcomes() -> dict:
    with open(_OUTCOMES_PATH) as f:
        return yaml.safe_load(f)


_OUTCOMES = _load_outcomes()

OUTCOME_CLASSES: list[str] = _OUTCOMES["classes"]
EXCLUDED_OUTCOMES: list[str] = _OUTCOMES["excluded"]

FEATURE_BLOCKS: dict[str, list[str]] = {
    "game_state": [
        "inning", "is_bottom", "outs", "runner_1b", "runner_2b", "runner_3b",
        "balls", "strikes", "run_diff", "is_home",
        "times_through_order", "batter_prior_pa",
        "pitcher_pitch_count", "pitcher_bf_game", "batter_ab_vs_pitcher",
        "pitcher_outing_walks", "pitcher_outing_hits",
        "pitcher_outing_k", "pitcher_outing_runs",
        "pitcher_outing_whip", "pitcher_recent_whip",
    ],
    "game_state_no_count": [
        "inning", "outs", "runner_1b", "runner_2b", "runner_3b",
        "times_through_order", "batter_prior_pa",
        "pitcher_bf_game", "batter_ab_vs_pitcher",
        "pitcher_outing_walks", "pitcher_outing_hits",
        "pitcher_outing_k", "pitcher_outing_runs",
        "pitcher_outing_whip", "pitcher_recent_whip",
    ],
    "batter_profile": [
        "bat_season_ba", "bat_career_ba", "bat_season_obp", "bat_career_obp",
        "bat_season_slg", "bat_career_slg", "bat_season_ops", "bat_career_ops",
        "bat_season_woba", "bat_career_woba", "bat_season_k_pct", "bat_career_k_pct",
        "bat_season_bb_pct", "bat_career_bb_pct", "bat_season_iso", "bat_career_iso",
        "bat_season_babip", "bat_career_babip", "bat_s_pa", "bat_c_pa",
    ],
    "pitcher_profile": [
        "pit_season_whip", "pit_career_whip", "pit_season_k9", "pit_career_k9",
        "pit_season_bb9", "pit_career_bb9", "pit_season_hr9", "pit_career_hr9",
        "pit_season_h9", "pit_career_h9", "pit_season_k_pct", "pit_career_k_pct",
        "pit_season_bb_pct", "pit_career_bb_pct", "pit_season_fip", "pit_career_fip",
        "pit_season_babip", "pit_career_babip", "pit_season_woba", "pit_career_woba",
        "pit_season_ip", "pit_career_ip", "pit_s_bf", "pit_c_bf",
        "pit_rest_days",
    ],
    "batter_arsenal": [
        "bat_season_woba_vs_fb", "bat_career_woba_vs_fb",
        "bat_season_woba_vs_brk", "bat_career_woba_vs_brk",
        "bat_season_woba_vs_offspeed", "bat_career_woba_vs_offspeed",
    ],
    "batter_batted_ball": [
        "bat_season_gb_pct_vs_fb", "bat_career_gb_pct_vs_fb",
        "bat_season_gb_pct_vs_brk", "bat_career_gb_pct_vs_brk",
        "bat_season_gb_pct_vs_offspeed", "bat_career_gb_pct_vs_offspeed",
        "bat_season_fb_pct_vs_fb", "bat_career_fb_pct_vs_fb",
        "bat_season_fb_pct_vs_brk", "bat_career_fb_pct_vs_brk",
        "bat_season_fb_pct_vs_offspeed", "bat_career_fb_pct_vs_offspeed",
        "bat_season_ld_pct_vs_fb", "bat_career_ld_pct_vs_fb",
        "bat_season_ld_pct_vs_brk", "bat_career_ld_pct_vs_brk",
        "bat_season_ld_pct_vs_offspeed", "bat_career_ld_pct_vs_offspeed",
    ],
    "batter_contact_quality": [
        "bat_season_avg_ev", "bat_career_avg_ev",
        "bat_season_hard_hit_pct", "bat_career_hard_hit_pct",
        "bat_season_barrel_pct", "bat_career_barrel_pct",
    ],
    "batter_discipline": [
        "bat_season_chase_rate_vs_fb", "bat_career_chase_rate_vs_fb",
        "bat_season_chase_rate_vs_brk", "bat_career_chase_rate_vs_brk",
        "bat_season_chase_rate_vs_offspeed", "bat_career_chase_rate_vs_offspeed",
        "bat_season_whiff_rate_vs_fb", "bat_career_whiff_rate_vs_fb",
        "bat_season_whiff_rate_vs_brk", "bat_career_whiff_rate_vs_brk",
        "bat_season_whiff_rate_vs_offspeed", "bat_career_whiff_rate_vs_offspeed",
    ],
    "pitcher_arsenal": [
        *[f"pit_{tf}_{pt.lower()}_pct" for tf in ("season", "career") for pt in PITCH_TYPES],
        *[f"pit_{tf}_{pt.lower()}_velo" for tf in ("season", "career") for pt in PITCH_TYPES],
        *[f"pit_{tf}_{pt.lower()}_spin" for tf in ("season", "career") for pt in PITCH_TYPES],
        "pit_season_arm_angle", "pit_career_arm_angle",
        "pit_season_extension", "pit_career_extension",
    ],
}

# Columns in the feature table that are NOT model features.
# Used by mode="auto" to discover features from the dataframe.
METADATA_COLUMNS: set[str] = {"game_pk", "at_bat_number", "pitch_number", "game_date", "target"}


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    db_path: str = "stateball.duckdb"
    table: str = "main.feat_mlb__vectors"
    connection_mode: str = "duckdb"  # "duckdb" (local file) or "ducklake" (cluster)
    seasons: list[int] = field(default_factory=list)
    date_range: list[str] = field(default_factory=list)
    min_game_date: str | None = None
    max_games: int | None = None


@dataclass
class FeatureConfig:
    mode: str = "explicit"  # "explicit" (use blocks) or "auto" (discover from table)
    blocks: list[str] = field(default_factory=lambda: list(FEATURE_BLOCKS.keys()))
    add: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    manifest_path: str = ""  # empty = use blocks/auto; S3 or local path = use manifest


@dataclass
class SamplingConfig:
    enabled: bool = True
    seed: int = 42


@dataclass
class SplitConfig:
    test_fraction: float = 0.20
    calibrate_fraction: float = 0.25
    seed: int = 42


@dataclass
class XGBoostConfig:
    objective: str = "multi:softprob"
    eval_metric: str = "mlogloss"
    max_depth: int = 4
    min_child_weight: int = 100
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    learning_rate: float = 0.05
    n_estimators: int = 2000
    reg_lambda: float = 10
    reg_alpha: float = 1
    max_delta_step: int = 1
    random_state: int = 42
    n_jobs: int = -1
    tree_method: str = "hist"
    grow_policy: str = "depthwise"
    max_leaves: int = 0
    max_bin: int = 256


@dataclass
class TrainingConfig:
    early_stopping_rounds: int = 50
    early_stopping_validation_fraction: float = 0.10


@dataclass
class CalibrationConfig:
    method: str = "isotonic"
    cv: int = 5


@dataclass
class OutputConfig:
    dir: str = "xg/experiments"
    name: str | None = None


@dataclass
class SweepConfig:
    n_trials: int = 50
    timeout: int | None = None
    study_name: str = "stateball_xgb"
    storage: str | None = None
    pruner_startup_trials: int = 10
    pruner_warmup_steps: int = 20
    # Search space bounds — each is [low, high]
    n_estimators_range: list[int] = field(default_factory=lambda: [200, 2000])
    max_depth_range: list[int] = field(default_factory=lambda: [3, 8])
    min_child_weight_range: list[int] = field(default_factory=lambda: [10, 300])
    learning_rate_range: list[float] = field(default_factory=lambda: [0.01, 0.3])
    subsample_range: list[float] = field(default_factory=lambda: [0.5, 1.0])
    colsample_bytree_range: list[float] = field(default_factory=lambda: [0.5, 1.0])
    reg_lambda_range: list[float] = field(default_factory=lambda: [0.1, 100.0])
    reg_alpha_range: list[float] = field(default_factory=lambda: [0.0, 10.0])
    max_leaves_range: list[int] = field(default_factory=lambda: [16, 128])
    feature_blocks_sweep: bool = False
    objectives: list[str] = field(default_factory=lambda: ["reliability", "resolution", "inference_ms"])
    sweep_eval_fraction: float = 0.40
    sweep_sample_fraction: float = 1.0
    max_inference_ms: float | None = None
    inference_speed_weight: float = 0.003  # λ in score = brier + λ*log(inference_ms)


@dataclass
class SelectionConfig:
    mrmr_n_select: int = 75
    # Stage 2: SAGE importance ranking
    sage_n_select: int = 25        # features passed from stage 2 to stage 3
    sage_n_eval: int = 8000        # eval samples for SAGE estimation
    sage_background_n: int = 512   # MarginalImputer background samples (keep ≤512)
    # Stage 3: CV cutoff on SAGE-ranked subsets
    cv_n_folds: int = 5
    cv_min_features: int = 10      # minimum subset size to evaluate
    target: str = "live"           # "live" | "sim" — pre-filter SIM_EXCLUDED before selection when "sim"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> ExperimentConfig:
    """Load and validate experiment config from TOML file."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    cfg = ExperimentConfig(
        data=DataConfig(**raw.get("data", {})),
        features=FeatureConfig(**raw.get("features", {})),
        sampling=SamplingConfig(**raw.get("sampling", {})),
        split=SplitConfig(**raw.get("split", {})),
        xgboost=XGBoostConfig(**raw.get("xgboost", {})),
        training=TrainingConfig(**raw.get("training", {})),
        calibration=CalibrationConfig(**raw.get("calibration", {})),
        output=OutputConfig(**raw.get("output", {})),
        sweep=SweepConfig(**raw.get("sweep", {})),
        selection=SelectionConfig(**raw.get("selection", {})),
    )

    _validate(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(cfg: ExperimentConfig):
    """Validate config consistency. Fails loud on bad input."""
    # Connection mode
    if cfg.data.connection_mode not in ("duckdb", "ducklake"):
        raise ValueError(
            f"[data] connection_mode must be 'duckdb' or 'ducklake', "
            f"got {cfg.data.connection_mode!r}"
        )

    # Split fractions
    if not 0 < cfg.split.test_fraction < 1:
        raise ValueError(f"[split] test_fraction must be in (0, 1), got {cfg.split.test_fraction}")
    if not 0 < cfg.split.calibrate_fraction < 1:
        raise ValueError(f"[split] calibrate_fraction must be in (0, 1), got {cfg.split.calibrate_fraction}")

    # Feature mode
    if cfg.features.mode not in ("explicit", "auto"):
        raise ValueError(
            f"[features] mode must be 'explicit' or 'auto', got {cfg.features.mode!r}"
        )

    # Feature blocks (only validated in explicit mode)
    if cfg.features.mode == "explicit":
        for block in cfg.features.blocks:
            if block not in FEATURE_BLOCKS:
                raise ValueError(
                    f"[features] unknown block: {block!r}. "
                    f"Known: {list(FEATURE_BLOCKS.keys())}"
                )

    # Calibration method
    if cfg.calibration.method not in ("isotonic", "sigmoid"):
        raise ValueError(f"[calibration] unknown method: {cfg.calibration.method!r}")

    # Tree growth policy
    if cfg.xgboost.grow_policy not in ("depthwise", "lossguide"):
        raise ValueError(
            f"[xgboost] grow_policy must be 'depthwise' or 'lossguide', "
            f"got {cfg.xgboost.grow_policy!r}"
        )
    if cfg.xgboost.grow_policy == "lossguide" and cfg.xgboost.max_depth != 0:
        import warnings
        warnings.warn(
            "[xgboost] grow_policy='lossguide' with max_depth > 0 — "
            "depth still constrains tree growth. Set max_depth=0 for leaf-only control."
        )

    # Sweep search space bounds — each must be [low, high]
    s = cfg.sweep
    for name, rng in [
        ("n_estimators_range", s.n_estimators_range),
        ("max_depth_range", s.max_depth_range),
        ("min_child_weight_range", s.min_child_weight_range),
        ("learning_rate_range", s.learning_rate_range),
        ("subsample_range", s.subsample_range),
        ("colsample_bytree_range", s.colsample_bytree_range),
        ("reg_lambda_range", s.reg_lambda_range),
        ("reg_alpha_range", s.reg_alpha_range),
        ("max_leaves_range", s.max_leaves_range),
    ]:
        if len(rng) != 2 or rng[0] > rng[1]:
            raise ValueError(f"[sweep] {name} must be [low, high], got {rng}")

    if not 0 < s.sweep_eval_fraction < 1:
        raise ValueError(
            f"[sweep] sweep_eval_fraction must be in (0, 1), got {s.sweep_eval_fraction}"
        )

    # Selection config
    sel = cfg.selection
    if sel.mrmr_n_select < 1:
        raise ValueError(f"[selection] mrmr_n_select must be >= 1, got {sel.mrmr_n_select}")
    if sel.sage_n_select < 1:
        raise ValueError(f"[selection] sage_n_select must be >= 1, got {sel.sage_n_select}")
    if sel.sage_n_eval < 1:
        raise ValueError(f"[selection] sage_n_eval must be >= 1, got {sel.sage_n_eval}")
    if sel.sage_background_n < 1:
        raise ValueError(f"[selection] sage_background_n must be >= 1, got {sel.sage_background_n}")
    if sel.cv_n_folds < 2:
        raise ValueError(f"[selection] cv_n_folds must be >= 2, got {sel.cv_n_folds}")
    if sel.cv_min_features < 1:
        raise ValueError(f"[selection] cv_min_features must be >= 1, got {sel.cv_min_features}")
    if sel.target not in ("live", "sim"):
        raise ValueError(f"[selection] target must be 'live' or 'sim', got {sel.target!r}")


# ---------------------------------------------------------------------------
# Feature resolution
# ---------------------------------------------------------------------------

def resolve_features(cfg: FeatureConfig, df_columns: list[str] | None = None) -> list[str]:
    """Resolve final feature column list.

    In explicit mode (default): resolves from FEATURE_BLOCKS via blocks + add + exclude.
    In auto mode: discovers features from df_columns, subtracting METADATA_COLUMNS.
    """
    if cfg.mode == "auto":
        if df_columns is None:
            raise ValueError(
                "[features] mode='auto' requires df_columns — "
                "load data before resolving features"
            )
        cols = [c for c in df_columns if c not in METADATA_COLUMNS]
    else:
        cols = []
        for block_name in cfg.blocks:
            cols.extend(FEATURE_BLOCKS[block_name])

        for col in cfg.add:
            if col not in cols:
                cols.append(col)

    for col in cfg.exclude:
        if col in cols:
            cols.remove(col)
        else:
            raise ValueError(
                f"[features] cannot exclude {col!r} — not in resolved feature list. "
                f"Resolved columns: {cols}"
            )

    if not cols:
        raise ValueError("[features] resolved feature list is empty")

    return cols


# ---------------------------------------------------------------------------
# Sim feature slicing
# ---------------------------------------------------------------------------

# Features in game_state but NOT in game_state_no_count are sim-excluded:
# count features (balls, strikes, pitcher_pitch_count) and situational
# features (is_home, is_bottom, run_diff) that introduce home bias when
# compounded over a full game simulation.
SIM_EXCLUDED_FEATURES: set[str] = (
    set(FEATURE_BLOCKS["game_state"]) - set(FEATURE_BLOCKS["game_state_no_count"])
)


def resolve_sim_features(feature_cols: list[str]) -> list[str]:
    """Drop sim-excluded features from a live feature list."""
    return [c for c in feature_cols if c not in SIM_EXCLUDED_FEATURES]


def slice_sim_data(data: dict) -> dict:
    """Create a sim variant of a data dict by column-slicing X matrices.

    Takes a data dict (from prepare_data()) with the live feature superset
    and returns a copy with sim-excluded columns removed from X_train,
    X_calib, X_test, and feature_cols.
    """
    sim_feature_cols = resolve_sim_features(data["feature_cols"])
    keep_indices = [i for i, col in enumerate(data["feature_cols"]) if col in sim_feature_cols]

    return {
        **data,
        "X_train": data["X_train"][:, keep_indices],
        "X_calib": data["X_calib"][:, keep_indices],
        "X_test": data["X_test"][:, keep_indices],
        "feature_cols": sim_feature_cols,
    }
