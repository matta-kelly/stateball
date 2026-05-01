"""
Feature analysis for trained XGBoost PA outcome models.

Post-hoc analysis of a saved model artifact: booster feature importance,
SHAP attribution, and feature correlation/redundancy. No retraining needed.

Usage:
    .venv/bin/python -m xg.analyze s3://dazoo/stateball/artifacts/xgboost/<run_id>
    .venv/bin/python -m xg.analyze s3://... --config xg/configs/default.toml
    .venv/bin/python -m xg.analyze s3://... --n-shap-samples 5000
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from xg.core.config import FEATURE_BLOCKS, ExperimentConfig, load_config
from xg.core.data import prepare_data
from xg.core.io import get_s3fs, log_memory

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("xg.analyze")


# ---------------------------------------------------------------------------
# Feature → block mapping
# ---------------------------------------------------------------------------


def _build_feature_to_block(
    feature_names: list[str],
    manifest_blocks: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Map each feature name to its block name.

    When manifest_blocks is provided, uses the manifest as the source of
    truth for block assignments. Otherwise falls back to FEATURE_BLOCKS.
    Features not found in any block are mapped to "unknown".
    """
    source = manifest_blocks if manifest_blocks is not None else FEATURE_BLOCKS
    name_to_block: dict[str, str] = {}
    for block_name, features in source.items():
        for f in features:
            if f not in name_to_block:
                name_to_block[f] = block_name

    return {f: name_to_block.get(f, "unknown") for f in feature_names}


def _aggregate_by_block(
    per_feature: dict[str, float],
    feature_to_block: dict[str, str],
) -> dict[str, float]:
    """Sum per-feature values into per-block totals."""
    by_block: dict[str, float] = {}
    for feat, val in per_feature.items():
        block = feature_to_block.get(feat, "unknown")
        by_block[block] = by_block.get(block, 0.0) + val
    return dict(sorted(by_block.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------
# Analysis 1: Booster feature importance
# ---------------------------------------------------------------------------


def _analyze_importance(
    booster,
    feature_names: list[str],
    feature_to_block: dict[str, str],
) -> dict:
    """Extract and normalize booster feature importance (gain, cover, weight)."""
    result = {}
    by_block = {}

    for imp_type in ("gain", "cover", "weight"):
        raw = booster.get_score(importance_type=imp_type)

        # Map f0/f1/... → real feature names
        mapped: dict[str, float] = {}
        for key, val in raw.items():
            idx = int(key[1:])  # "f42" → 42
            if idx < len(feature_names):
                mapped[feature_names[idx]] = float(val)

        # Normalize to sum to 1.0
        total = sum(mapped.values())
        if total > 0:
            mapped = {k: v / total for k, v in mapped.items()}

        # Include features with zero importance
        for f in feature_names:
            if f not in mapped:
                mapped[f] = 0.0

        # Sort descending
        mapped = dict(sorted(mapped.items(), key=lambda kv: -kv[1]))
        result[imp_type] = mapped
        by_block[imp_type] = _aggregate_by_block(mapped, feature_to_block)

    result["by_block"] = by_block
    return result


# ---------------------------------------------------------------------------
# Analysis 2: SHAP values
# ---------------------------------------------------------------------------


def _analyze_shap(
    booster,
    X_sample: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    class_weights: np.ndarray,
    feature_to_block: dict[str, str],
) -> dict:
    """Compute SHAP values via XGBoost's native pred_contribs.

    Uses booster.predict(pred_contribs=True) directly instead of the shap
    library's TreeExplainer — same Shapley values (XGBoost's C++ tree
    traversal), no shap C extension involved. Avoids heap corruption from
    shap/xgboost version mismatch (shap 0.51 + xgboost 3.x).
    """
    import xgboost as xgb

    log_memory("shap: start")
    logger.info("Computing SHAP values on %d samples...", X_sample.shape[0])

    dmat = xgb.DMatrix(X_sample, feature_names=feature_names)
    # pred_contribs returns (n_samples, n_classes, n_features+1)
    # last column per class is the bias term
    contribs = booster.predict(dmat, pred_contribs=True)
    shap_values = contribs[:, :, :-1]  # drop bias → (n_samples, n_classes, n_features)
    log_memory("shap: values computed")

    n_classes = shap_values.shape[1]
    n_features = shap_values.shape[2]

    # Mean |SHAP| per class per feature: (n_classes, n_features)
    shap_abs = np.abs(shap_values).mean(axis=0)

    # Mean |SHAP| per feature per class
    mean_abs_by_class: dict[str, dict[str, float]] = {}
    for c in range(n_classes):
        cls_name = class_names[c] if c < len(class_names) else str(c)
        per_feat = {}
        for f in range(n_features):
            per_feat[feature_names[f]] = round(float(shap_abs[c, f]), 8)
        mean_abs_by_class[cls_name] = dict(
            sorted(per_feat.items(), key=lambda kv: -kv[1])
        )

    # Overall mean |SHAP| weighted by class frequency
    weighted = (shap_abs * class_weights[:n_classes, None]).sum(axis=0)
    mean_abs = {
        feature_names[f]: round(float(weighted[f]), 8)
        for f in range(n_features)
    }
    mean_abs = dict(sorted(mean_abs.items(), key=lambda kv: -kv[1]))

    # Per-block aggregated
    by_block = _aggregate_by_block(mean_abs, feature_to_block)

    # Top interaction pairs: correlation between class-weighted |SHAP| columns
    # (n_samples, n_features) — weighted sum across classes
    shap_combined = (np.abs(shap_values) * class_weights[None, :n_classes, None]).sum(axis=1)

    shap_corr = np.corrcoef(shap_combined, rowvar=False)
    np.fill_diagonal(shap_corr, 0.0)

    # Top 20 pairs by absolute correlation
    top_interactions = []
    n_f = len(feature_names)
    flat_idx = np.argsort(np.abs(shap_corr).ravel())[::-1]
    seen = set()
    for idx in flat_idx:
        i, j = divmod(int(idx), n_f)
        if i >= j:
            continue
        pair = (feature_names[i], feature_names[j])
        if pair in seen:
            continue
        seen.add(pair)
        top_interactions.append({
            "feature_a": pair[0],
            "feature_b": pair[1],
            "correlation": round(float(shap_corr[i, j]), 4),
        })
        if len(top_interactions) >= 20:
            break

    log_memory("shap: complete")

    return {
        "mean_abs": mean_abs,
        "mean_abs_by_class": mean_abs_by_class,
        "by_block": by_block,
        "top_interactions": top_interactions,
    }


# ---------------------------------------------------------------------------
# Analysis 3: Feature correlation
# ---------------------------------------------------------------------------


def _analyze_correlation(
    X: np.ndarray,
    feature_names: list[str],
    feature_to_block: dict[str, str],
    high_threshold: float = 0.9,
) -> dict:
    """Compute feature correlation matrix and summarize redundancy."""
    logger.info("Computing correlation matrix on %d samples...", X.shape[0])

    corr = np.corrcoef(X, rowvar=False)
    # Handle NaN (constant features)
    corr = np.nan_to_num(corr, nan=0.0)
    n_f = len(feature_names)

    # Within-block mean absolute correlation
    blocks = sorted(set(feature_to_block.values()))
    within_block: dict[str, float] = {}
    for block in blocks:
        idxs = [i for i, f in enumerate(feature_names) if feature_to_block.get(f) == block]
        if len(idxs) < 2:
            within_block[block] = 0.0
            continue
        sub = corr[np.ix_(idxs, idxs)]
        # Mean of upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(sub, dtype=bool), k=1)
        within_block[block] = round(float(np.abs(sub[mask]).mean()), 4)

    # Cross-block mean absolute correlation
    cross_block: dict[str, float] = {}
    for i, b1 in enumerate(blocks):
        for j, b2 in enumerate(blocks):
            if j <= i:
                continue
            idxs1 = [k for k, f in enumerate(feature_names) if feature_to_block.get(f) == b1]
            idxs2 = [k for k, f in enumerate(feature_names) if feature_to_block.get(f) == b2]
            if not idxs1 or not idxs2:
                continue
            sub = corr[np.ix_(idxs1, idxs2)]
            cross_block[f"{b1}|{b2}"] = round(float(np.abs(sub).mean()), 4)

    cross_block = dict(sorted(cross_block.items(), key=lambda kv: -kv[1]))

    # High-correlation pairs
    high_pairs = []
    for i in range(n_f):
        for j in range(i + 1, n_f):
            r = corr[i, j]
            if abs(r) >= high_threshold:
                high_pairs.append({
                    "a": feature_names[i],
                    "b": feature_names[j],
                    "r": round(float(r), 4),
                    "blocks": f"{feature_to_block.get(feature_names[i], '?')}|{feature_to_block.get(feature_names[j], '?')}",
                })

    high_pairs.sort(key=lambda p: -abs(p["r"]))

    return {
        "within_block": within_block,
        "cross_block": cross_block,
        "high_pairs": high_pairs,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def analyze(
    artifact_path: str,
    cfg: ExperimentConfig,
    n_shap_samples: int = 10_000,
) -> dict:
    """Run full feature analysis on a trained model artifact.

    Args:
        artifact_path: S3 path to the model run directory.
        cfg: Experiment config (for data loading — same config reproduces the split).
        n_shap_samples: Number of test samples to use for SHAP (default 10k).

    Returns:
        Analysis report dict (also saved as feature_analysis.json in the artifact dir).
    """
    logger.info("Feature analysis: %s", artifact_path)

    # Load model
    from sim.artifacts import load_model
    bundle = load_model(artifact_path)
    if bundle.booster is None:
        raise ValueError("Feature analysis requires a decomposed model (model.ubj), not legacy joblib")

    feature_names = bundle.feature_names
    class_names = bundle.outcome_labels
    n_features = len(feature_names)
    n_classes = len(class_names)
    logger.info("Model: %d features, %d classes", n_features, n_classes)

    # Try to load manifest from artifact directory
    from xg.core.manifest import try_load_artifact_manifest
    manifest = try_load_artifact_manifest(artifact_path)
    manifest_blocks = manifest.blocks if manifest else None
    if manifest:
        logger.info("Manifest loaded: %s (%d features)", manifest.run_id, len(manifest.features))

    # Load test data
    log_memory("analyze: loading data")
    data = prepare_data(cfg)
    X_test = data["X_test"]
    y_test = data["y_test"]
    log_memory("analyze: data loaded")

    # Align data columns to model's feature list
    data_cols = data["feature_cols"]
    if data_cols != feature_names:
        col_idx = [data_cols.index(f) for f in feature_names]
        X_test = X_test[:, col_idx]
        dropped = set(data_cols) - set(feature_names)
        logger.info(
            "Aligned data (%d cols) to model (%d features), dropped: %s",
            len(data_cols), n_features, sorted(dropped),
        )

    # Class frequency weights (for SHAP averaging)
    class_counts = np.bincount(y_test, minlength=n_classes).astype(np.float64)
    class_weights = class_counts / class_counts.sum()

    # Feature → block mapping
    feature_to_block = _build_feature_to_block(feature_names, manifest_blocks=manifest_blocks)

    # Block definitions for metadata (only blocks that have features in this model)
    active_blocks: dict[str, list[str]] = {}
    for f in feature_names:
        block = feature_to_block[f]
        active_blocks.setdefault(block, []).append(f)

    # --- Analysis 1: Importance ---
    logger.info("=== Booster Feature Importance ===")
    importance = _analyze_importance(bundle.booster, feature_names, feature_to_block)

    # Log top 15 by gain
    logger.info("Top 15 features by gain:")
    for i, (feat, val) in enumerate(list(importance["gain"].items())[:15]):
        logger.info("  %2d. %-35s %.4f  [%s]", i + 1, feat, val, feature_to_block.get(feat, "?"))

    logger.info("Block importance (gain):")
    for block, val in importance["by_block"]["gain"].items():
        logger.info("  %-30s %.4f", block, val)

    # --- Analysis 2: SHAP ---
    logger.info("=== SHAP Values ===")
    rng = np.random.RandomState(42)
    n_sample = min(n_shap_samples, len(X_test))
    sample_idx = rng.choice(len(X_test), n_sample, replace=False)
    X_sample = X_test[sample_idx]

    shap_result = _analyze_shap(
        bundle.booster, X_sample, feature_names,
        class_names, class_weights, feature_to_block,
    )

    logger.info("Top 15 features by mean |SHAP|:")
    for i, (feat, val) in enumerate(list(shap_result["mean_abs"].items())[:15]):
        logger.info("  %2d. %-35s %.6f  [%s]", i + 1, feat, val, feature_to_block.get(feat, "?"))

    logger.info("Block SHAP:")
    for block, val in shap_result["by_block"].items():
        logger.info("  %-30s %.6f", block, val)

    if shap_result["top_interactions"]:
        logger.info("Top 5 SHAP interaction pairs:")
        for pair in shap_result["top_interactions"][:5]:
            logger.info("  %s ↔ %s  r=%.3f", pair["feature_a"], pair["feature_b"], pair["correlation"])

    # --- Analysis 3: Correlation ---
    logger.info("=== Feature Correlation ===")
    correlation = _analyze_correlation(X_test, feature_names, feature_to_block)

    logger.info("Within-block mean |correlation|:")
    for block, val in sorted(correlation["within_block"].items(), key=lambda kv: -kv[1]):
        logger.info("  %-30s %.3f", block, val)

    if correlation["high_pairs"]:
        logger.info("High-correlation pairs (|r| >= 0.9):")
        for pair in correlation["high_pairs"][:10]:
            logger.info("  %s ↔ %s  r=%.3f  [%s]", pair["a"], pair["b"], pair["r"], pair["blocks"])

    # --- Assemble report ---
    report = {
        "metadata": {
            "artifact_path": artifact_path,
            "n_features": n_features,
            "n_classes": n_classes,
            "n_test_samples": len(X_test),
            "n_shap_samples": n_sample,
            "feature_names": feature_names,
            "class_names": class_names,
            "feature_blocks": active_blocks,
            "class_frequencies": {
                class_names[i]: round(float(class_weights[i]), 6)
                for i in range(n_classes)
            },
            "manifest": {
                "run_id": manifest.run_id,
                "method": manifest.method,
                "n_features": len(manifest.features),
            } if manifest else None,
        },
        "importance": importance,
        "shap": shap_result,
        "correlation": correlation,
    }

    # Save to S3
    try:
        fs = get_s3fs()
        report_path = f"{artifact_path.rstrip('/')}/feature_analysis.json"
        with fs.open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Report saved: %s", report_path)
    except Exception:
        logger.warning("Could not save to S3 — saving locally", exc_info=True)
        local_path = REPO_ROOT / "xg" / "experiments" / "feature_analysis.json"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Report saved locally: %s", local_path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("xg").setLevel(logging.DEBUG)

    if len(sys.argv) < 2:
        print("Usage: python -m xg.analyze <artifact_s3_path> [--config path] [--n-shap-samples N]")
        sys.exit(1)

    artifact_path = sys.argv[1]

    config_path = Path(__file__).parent.parent / "configs" / "default.toml"
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            config_path = Path(sys.argv[idx + 1])

    n_shap = 10_000
    if "--n-shap-samples" in sys.argv:
        idx = sys.argv.index("--n-shap-samples")
        if idx + 1 < len(sys.argv):
            n_shap = int(sys.argv[idx + 1])

    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    cfg = load_config(config_path)
    analyze(artifact_path, cfg, n_shap_samples=n_shap)
