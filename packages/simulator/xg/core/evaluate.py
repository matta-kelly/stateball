"""
Evaluation utilities for the XGBoost PA outcome model.

Includes the original 4 functions (brier_score, per_class_brier,
calibration_by_class, plot_calibration_curves) plus upgraded metrics:
brier_decomposition, stratified_calibration, bootstrap_ci, inference_timing,
and full_evaluate.
"""

from __future__ import annotations

import math
import time

import numpy as np
from sklearn.preprocessing import LabelBinarizer


def brier_score(y_true, y_proba, classes):
    """Multiclass Brier score."""
    lb = LabelBinarizer()
    lb.classes_ = classes
    y_onehot = lb.transform(y_true)
    return float(np.mean(np.sum((y_onehot - y_proba) ** 2, axis=1)))


def per_class_brier(y_true, y_proba, classes):
    """Brier score broken down by class."""
    lb = LabelBinarizer()
    lb.classes_ = classes
    y_onehot = lb.transform(y_true)
    scores = {}
    for i, cls in enumerate(classes):
        scores[cls] = float(np.mean((y_onehot[:, i] - y_proba[:, i]) ** 2))
    return scores


def calibration_by_class(y_true, y_proba, classes, n_bins=10):
    """Per-class calibration: predicted vs actual frequency in probability bins."""
    lb = LabelBinarizer()
    lb.classes_ = classes
    y_onehot = lb.transform(y_true)
    results = {}
    for i, cls in enumerate(classes):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba[:, i], bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        predicted = []
        actual = []
        counts = []
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() == 0:
                continue
            predicted.append(float(y_proba[mask, i].mean()))
            actual.append(float(y_onehot[mask, i].mean()))
            counts.append(int(mask.sum()))
        results[cls] = {"predicted": predicted, "actual": actual, "counts": counts}
    return results


# ---------------------------------------------------------------------------
# Brier decomposition (Murphy 1973)
# ---------------------------------------------------------------------------

def brier_decomposition(y_true, y_proba, classes, n_bins=20) -> dict:
    """Decompose multiclass Brier score into reliability, resolution, uncertainty.

    Murphy (1973) decomposition applied per-class then averaged weighted by
    class frequency.

    - reliability: calibration error (lower is better)
    - resolution: discriminative power (higher is better)
    - uncertainty: irreducible, depends only on class base rates
    - brier = reliability - resolution + uncertainty

    Returns dict with float values for each component.
    """
    lb = LabelBinarizer()
    lb.classes_ = classes
    y_onehot = lb.transform(y_true)
    n = len(y_true)

    total_rel = 0.0
    total_res = 0.0
    total_unc = 0.0

    for k in range(len(classes)):
        p_k = y_proba[:, k]
        o_k = y_onehot[:, k]
        bar_o = o_k.mean()  # base rate for class k

        # Bin predictions
        bins = np.linspace(0, 1, n_bins + 1)
        bin_idx = np.clip(np.digitize(p_k, bins) - 1, 0, n_bins - 1)

        rel_k = 0.0
        res_k = 0.0

        for b in range(n_bins):
            mask = bin_idx == b
            n_b = mask.sum()
            if n_b == 0:
                continue
            bar_p_b = p_k[mask].mean()  # mean predicted prob in bin
            bar_o_b = o_k[mask].mean()  # observed frequency in bin
            rel_k += n_b * (bar_p_b - bar_o_b) ** 2
            res_k += n_b * (bar_o_b - bar_o) ** 2

        rel_k /= n
        res_k /= n
        unc_k = bar_o * (1 - bar_o)

        # Weight by class frequency
        total_rel += rel_k
        total_res += res_k
        total_unc += unc_k

    return {
        "reliability": float(total_rel),
        "resolution": float(total_res),
        "uncertainty": float(total_unc),
        "brier_score": float(total_rel - total_res + total_unc),
    }


# ---------------------------------------------------------------------------
# Stratified calibration
# ---------------------------------------------------------------------------

def stratified_calibration(
    y_true, y_proba, classes, groups: dict[str, np.ndarray], n_bins=20,
) -> dict:
    """Run brier_decomposition per stratum defined by boolean masks.

    Args:
        groups: Dict mapping stratum name → boolean index array (same length as y_true).

    Returns dict mapping stratum name → decomposition dict + sample count.
    """
    results = {}
    for name, mask in groups.items():
        if mask.sum() == 0:
            continue
        decomp = brier_decomposition(y_true[mask], y_proba[mask], classes, n_bins=n_bins)
        decomp["n"] = int(mask.sum())
        results[name] = decomp
    return results


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true, y_proba, classes, metric_fn=None, n_bootstrap=1000, seed=42,
) -> dict:
    """Resample test set and compute 95% percentile CI for a metric.

    Args:
        metric_fn: Callable(y_true, y_proba, classes) → float.
                   Defaults to brier_score if not provided.

    Returns dict with mean, ci_lower, ci_upper, std.
    """
    if metric_fn is None:
        metric_fn = brier_score

    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        scores[i] = metric_fn(y_true[idx], y_proba[idx], classes)

    return {
        "mean": float(scores.mean()),
        "ci_lower": float(np.percentile(scores, 2.5)),
        "ci_upper": float(np.percentile(scores, 97.5)),
        "std": float(scores.std()),
    }


# ---------------------------------------------------------------------------
# Inference timing
# ---------------------------------------------------------------------------

def inference_timing(model_bundle, X_sample, n_repeats=10) -> dict:
    """Time model inference via the decomposed booster path (production-equivalent).

    Extracts the raw XGBoost booster from whatever wrapper is passed and
    benchmarks booster.predict() directly — the same path the sim engine uses.

    Args:
        model_bundle: CalibratedClassifierCV, XGBClassifier, a dict with
                      'booster' key, or a raw xgb.Booster.
        X_sample: Feature matrix to predict on.
        n_repeats: Number of timing repeats (takes median).

    Returns dict with ms_per_batch, ms_per_row, batch_size.
    """
    import xgboost as xgb

    booster = _extract_booster(model_bundle)
    dmat = xgb.DMatrix(X_sample)

    # Warm up (first call has overhead)
    booster.predict(dmat)

    batch_size = len(X_sample)
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        booster.predict(dmat)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    median_ms = float(np.median(times))

    return {
        "ms_per_batch": median_ms,
        "ms_per_row": median_ms / batch_size if batch_size > 0 else 0.0,
        "batch_size": batch_size,
    }


def _extract_booster(model_bundle):
    """Extract raw xgb.Booster from any model wrapper."""
    import xgboost as xgb

    # Already a booster
    if isinstance(model_bundle, xgb.Booster):
        return model_bundle

    # Dict with booster key
    if isinstance(model_bundle, dict) and "booster" in model_bundle:
        return model_bundle["booster"]

    # XGBClassifier
    if isinstance(model_bundle, xgb.XGBClassifier):
        return model_bundle.get_booster()

    # CalibratedClassifierCV(FrozenEstimator(XGBClassifier))
    if hasattr(model_bundle, "calibrated_classifiers_"):
        inner = model_bundle.calibrated_classifiers_[0].estimator
        if hasattr(inner, "estimator"):
            inner = inner.estimator  # unwrap FrozenEstimator
        if isinstance(inner, xgb.XGBClassifier):
            return inner.get_booster()

    raise TypeError(f"Cannot extract booster from {type(model_bundle)}")


# ---------------------------------------------------------------------------
# Full evaluate (superset of old evaluate_model)
# ---------------------------------------------------------------------------

def full_evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    strata: dict[str, np.ndarray] | None = None,
    model_bundle=None,
) -> dict:
    """Comprehensive evaluation: Brier, decomposition, stratified, timing.

    Backward-compatible with old evaluate_model() output keys.

    Args:
        model: Calibrated model with predict_proba().
        X_test, y_test: Test data (y_test is integer-encoded).
        class_names: Maps index → string name.
        strata: Optional dict of boolean masks for stratified evaluation.
        model_bundle: Optional — passed to inference_timing if provided.
    """
    y_proba = model.predict_proba(X_test)
    int_classes = list(range(len(class_names)))

    # Core metrics (backward-compatible)
    bs = brier_score(y_test, y_proba, int_classes)
    pcb = per_class_brier(y_test, y_proba, int_classes)
    cbc = calibration_by_class(y_test, y_proba, int_classes)

    # Remap integer keys → string class names
    pcb = {class_names[k]: v for k, v in pcb.items()}
    cbc = {class_names[k]: v for k, v in cbc.items()}

    # Naive baseline
    class_freqs = np.bincount(y_test, minlength=len(class_names)) / len(y_test)
    naive_bs = float(np.sum(class_freqs * (1 - class_freqs)))
    brier_skill = 1.0 - (bs / naive_bs) if naive_bs > 0 else 0.0

    class_counts_test = {
        class_names[i]: int(ct)
        for i, ct in enumerate(np.bincount(y_test, minlength=len(class_names)))
    }

    results = {
        "brier_score_test": bs,
        "naive_brier_test": naive_bs,
        "brier_skill": brier_skill,
        "class_counts_test": class_counts_test,
        "per_class_brier": pcb,
        "calibration_by_class": cbc,
    }

    # Brier decomposition
    decomp = brier_decomposition(y_test, y_proba, int_classes)
    results["brier_reliability"] = decomp["reliability"]
    results["brier_resolution"] = decomp["resolution"]
    results["brier_uncertainty"] = decomp["uncertainty"]

    # Bootstrap CI
    ci = bootstrap_ci(y_test, y_proba, int_classes)
    results["brier_ci_lower"] = ci["ci_lower"]
    results["brier_ci_upper"] = ci["ci_upper"]
    results["brier_ci_std"] = ci["std"]

    # Stratified calibration
    if strata:
        results["stratified_calibration"] = stratified_calibration(
            y_test, y_proba, int_classes, strata,
        )

    # Inference timing
    if model_bundle is not None:
        timing = inference_timing(model_bundle, X_test[:1000])
        results["inference_ms_per_batch"] = timing["ms_per_batch"]
        results["inference_ms_per_row"] = timing["ms_per_row"]
        results["inference_batch_size"] = timing["batch_size"]

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_calibration_curves(
    calibration_data: dict[str, dict],
    save_path: str | None = None,
    top_n: int | None = None,
    return_fig: bool = False,
):
    """Plot calibration curves (predicted vs actual) per class.

    Args:
        calibration_data: Output of calibration_by_class(). Keys are class
            names, values have "predicted", "actual", "counts".
        save_path: If provided, save PNG to this path instead of showing.
        top_n: Limit to the N classes with the most total samples.
        return_fig: If True, return the matplotlib Figure instead of saving/showing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = list(calibration_data.keys())

    # Optionally limit to top_n most frequent classes
    if top_n and top_n < len(classes):
        totals = {c: sum(calibration_data[c]["counts"]) for c in classes}
        classes = sorted(totals, key=totals.get, reverse=True)[:top_n]

    n = len(classes)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)

    for idx, cls in enumerate(classes):
        ax = axes[idx // cols][idx % cols]
        data = calibration_data[cls]
        pred = data["predicted"]
        act = data["actual"]
        counts = data["counts"]

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="perfect")
        ax.plot(pred, act, "o-", markersize=4)

        # Annotate bin counts
        for p, a, c in zip(pred, act, counts):
            ax.annotate(str(c), (p, a), textcoords="offset points",
                        xytext=(0, 6), fontsize=6, ha="center", alpha=0.7)

        ax.set_title(cls, fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        if idx // cols == rows - 1:
            ax.set_xlabel("predicted", fontsize=8)
        if idx % cols == 0:
            ax.set_ylabel("observed", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Calibration Curves by Class", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if return_fig:
        return fig
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
