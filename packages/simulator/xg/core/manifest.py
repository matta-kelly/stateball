"""Feature manifest: schema, I/O, and sim derivation.

A feature manifest is a versioned artifact that records which features
were selected, why, and how. It travels with the model from selection
through training, validation, and analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
logger = logging.getLogger("xg.manifest")


@dataclass
class FeatureManifest:
    """Selected feature set with full provenance."""

    run_id: str
    method: str  # e.g. "mrmr_boruta_shap_rfecv"
    source_feature_count: int
    warehouse_columns: list[str]
    features: list[str]  # selected for live model
    blocks: dict[str, list[str]]  # selected features grouped by block
    dropped: dict[str, dict]  # reason per dropped feature
    stage_results: dict[str, dict]  # per-stage metadata
    sim_excluded: list[str]  # structural exclusions (can't simulate)
    sim_features: list[str]  # features minus sim_excluded
    metadata: dict = field(default_factory=dict)
    target: str = "live"  # "live" | "sim" — how selection was run


def manifest_to_dict(m: FeatureManifest) -> dict:
    """Serialize manifest to JSON-safe dict."""
    return asdict(m)


def manifest_from_dict(d: dict) -> FeatureManifest:
    """Deserialize manifest from dict."""
    return FeatureManifest(**d)


def build_blocks(features: list[str]) -> dict[str, list[str]]:
    """Map selected features back to their FEATURE_BLOCKS groups."""
    from xg.core.config import FEATURE_BLOCKS

    block_lookup: dict[str, str] = {}
    for block_name, block_features in FEATURE_BLOCKS.items():
        for f in block_features:
            block_lookup[f] = block_name

    blocks: dict[str, list[str]] = {}
    for f in features:
        block = block_lookup.get(f, "unknown")
        blocks.setdefault(block, []).append(f)
    return blocks


def derive_sim_manifest(manifest: FeatureManifest) -> FeatureManifest:
    """Create sim variant by removing SIM_EXCLUDED_FEATURES.

    For both target='live' and target='sim', manifest.features is the live set
    (sim features + sim-excluded added back). Strip SIM_EXCLUDED to get the sim
    variant regardless of how selection was run.
    """
    from xg.core.config import SIM_EXCLUDED_FEATURES

    sim_excluded = sorted(SIM_EXCLUDED_FEATURES)
    sim_features = [f for f in manifest.features if f not in SIM_EXCLUDED_FEATURES]
    sim_blocks = build_blocks(sim_features)

    return FeatureManifest(
        run_id=manifest.run_id,
        method=manifest.method,
        source_feature_count=manifest.source_feature_count,
        warehouse_columns=manifest.warehouse_columns,
        features=sim_features,
        blocks=sim_blocks,
        dropped={
            **manifest.dropped,
            **{f: {"reason": "sim_excluded"} for f in sim_excluded if f in manifest.features},
        },
        stage_results=manifest.stage_results,
        sim_excluded=sim_excluded,
        sim_features=sim_features,
        metadata=manifest.metadata,
        target=manifest.target,
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def save_manifest(manifest: FeatureManifest, path: str) -> None:
    """Write manifest JSON to local path or S3."""
    data = json.dumps(manifest_to_dict(manifest), indent=2, default=str)

    if path.startswith("s3://"):
        from xg.core.io import get_s3fs
        fs = get_s3fs()
        with fs.open(path, "w") as f:
            f.write(data)
    else:
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data)

    logger.info("Manifest saved: %s (%d features)", path, len(manifest.features))


def load_manifest(path: str) -> FeatureManifest:
    """Read manifest JSON from local path or S3."""
    if path.startswith("s3://"):
        from xg.core.io import get_s3fs
        fs = get_s3fs()
        with fs.open(path) as f:
            d = json.load(f)
    else:
        with open(path) as f:
            d = json.load(f)

    manifest = manifest_from_dict(d)
    logger.info("Manifest loaded: %s (%d features)", path, len(manifest.features))
    return manifest


def try_load_artifact_manifest(artifact_s3_path: str) -> FeatureManifest | None:
    """Try to load manifest.json from a model artifact directory. Returns None if not found."""
    manifest_path = f"{artifact_s3_path.rstrip('/')}/manifest.json"
    try:
        return load_manifest(manifest_path)
    except (FileNotFoundError, OSError, KeyError):
        return None
