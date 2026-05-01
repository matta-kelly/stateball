"""SAGE feature importance — model-agnostic loss-based importance estimation.

Measures how much each feature contributes to model performance (cross-entropy
loss) by computing the expected loss increase when the feature is removed.
Unlike SHAP attribution, SAGE is not anchored by dominant features — each
feature competes against its own marginal contribution to the loss, not
against other features' shadow copies.

Usage:
    from xg.pa.sage import compute_sage_importance

    values, stds = compute_sage_importance(
        model_fn=predict_fn,      # (N, d) → (N, K) probabilities
        X_eval=X_test[:8000],
        y_eval=y_test[:8000],
        X_background=X_bg[:512], # keep ≤512 rows
        feature_names=feature_names,
    )
    ranked = sorted(feature_names, key=lambda f: values[f], reverse=True)
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger("xg.sage_importance")


def compute_sage_importance(
    model_fn: Callable,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    X_background: np.ndarray,
    feature_names: list[str],
    n_permutations: int | None = None,
    thresh: float = 0.025,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute SAGE global feature importance via permutation estimator.

    Args:
        model_fn: callable (N, d) → (N, K) float ndarray of probabilities,
            summing to 1 per row. Renormalize inside the callable if needed.
        X_eval: evaluation data, shape (n_eval, n_features).
        y_eval: integer class labels 0..K-1, shape (n_eval,).
        X_background: background samples for MarginalImputer,
            shape (n_bg, n_features). Keep ≤512 rows to avoid slow runtime.
        feature_names: list of feature names, length == n_features.
        n_permutations: cap on permutation iterations. None = run until
            convergence (thresh criterion).
        thresh: convergence threshold — stops when
            max(std) / (max(values) - min(values)) < thresh.
            Default 0.025. Tighten to 0.01 for more precision.

    Returns:
        (values, stds): dicts mapping feature_name → float.
        Values can be negative (feature adds noise — rank last).
        Stds are per-feature uncertainty estimates (Welford online algorithm).
    """
    import sage

    n_bg = X_background.shape[0]
    if n_bg > 512:
        logger.warning(
            "X_background has %d rows — MarginalImputer is slow above 512, "
            "consider slicing to X_background[:512]",
            n_bg,
        )

    imputer = sage.MarginalImputer(model_fn, X_background)
    estimator = sage.PermutationEstimator(imputer, "cross entropy")

    explanation = estimator(
        X_eval,
        y_eval.astype(int),
        n_permutations=n_permutations,
        thresh=thresh,
        verbose=False,
        bar=False,
    )

    values = dict(zip(feature_names, explanation.values.tolist()))
    stds = dict(zip(feature_names, explanation.std.tolist()))

    n_positive = sum(1 for v in values.values() if v > 0)
    logger.info(
        "SAGE: %d features, %d positive, %d negative/zero. Range [%.5f, %.5f]",
        len(feature_names),
        n_positive,
        len(feature_names) - n_positive,
        min(values.values()),
        max(values.values()),
    )
    return values, stds
