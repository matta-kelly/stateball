"""Dagster Config classes — surfaced in Dagster Launchpad UI."""
from dagster import Config


class RawEventsConfig(Config):
    batch_size: int = 200


class RawBoxscoresConfig(Config):
    batch_size: int = 500


class RawGumboEventsConfig(Config):
    batch_size: int = 200


class ProcGumboEventsConfig(Config):
    batch_size: int = 500  # cap to avoid ARG_MAX on dbt --vars during fresh backfill
    full_refresh: bool = False


class ProcEventsConfig(Config):
    full_refresh: bool = False


class DbtConfig(Config):
    full_refresh: bool = False


class CalibrationConfig(Config):
    """Full-completion naive MC — builds convergence + horizon cutoff artifacts."""
    n_per_inning: int = 500
    n_sims: int = 1000
    seed: int = 42
    slot: str = "prod"


class ProductionConfig(Config):
    """Fast estimator for eval use — consumes calibration artifacts."""
    n_per_inning: int = 20
    n_sims: int = 1000
    seed: int = 42
    slot: str = "prod"
    estimator: str = "truncated_mc"
    adaptive_n: bool = True
    enable_pruning: bool = True
    max_horizon: int = 0  # 0 = use artifact as-is; >0 = cap + renormalize


class CalibrationArtifactConfig(Config):
    """Build a calibration artifact from a specific eval run."""
    eval_id: str = ""


class SelectFeaturesConfig(Config):
    mrmr_n_select: int = 0       # 0 = use TOML default
    boruta_n_iterations: int = 0 # 0 = use TOML default
    rfecv_min_features: int = 0  # 0 = use TOML default
    target: str = ""             # "sim" | "live" | "" = use TOML default


class TrainXgboostConfig(Config):
    max_games: int = 0             # 0 = use TOML default; override to cap training games
    learning_rate: float = 0.0     # 0 = use TOML default
    n_estimators: int = 0          # 0 = use TOML default
    early_stopping_rounds: int = 0 # 0 = use TOML default
    max_leaves: int = 0            # 0 = use TOML default
    subsample: float = 0.0         # 0 = use TOML default
    colsample_bytree: float = 0.0  # 0 = use TOML default
    max_bin: int = 0               # 0 = use TOML default
    manifest_path: str = ""        # S3 path to manifest; empty = resolve from registry or use all features


class PitcherExitConfig(Config):
    sweep: bool = False


class AnalyzeFeaturesConfig(Config):
    artifact_path: str = ""  # S3 path override; empty = resolve from registry
    variant: str = "xgboost_sim"  # xgboost_sim or xgboost_live
    slot: str = "test"  # registry slot: test or prod
    n_shap_samples: int = 2000  # SHAP sample count (10k default OOMs in prod)
