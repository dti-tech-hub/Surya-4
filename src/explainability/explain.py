"""Explainability for the classical and quantum models.

What is provided
----------------
Classical:
    * permutation importance on the reduced feature space (model-agnostic),
    * PCA loading attribution mapping reduced-feature importance back to the
      original encoded features,
    * SHAP explanation of the CNN+LSTM (when the ``shap`` package is available).

Quantum:
    * circuit-level feature sensitivity: the change in output probability when
      a single encoded feature is perturbed.  This is an *empirical sensitivity*
      analysis, not a Shapley-value attribution and not a claim of "quantum
      explainability".  We label it accordingly.

Nothing here is fabricated: if a method fails or is inapplicable, the result
records the failure reason instead of an invented figure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

LOGGER = get_logger("qeids.explain")


def permutation_importance(
    predict_fn,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    metric_fn=None,
    repeats: int = 10,
    seed: int = 42,
    max_samples: int = 500,
) -> pd.DataFrame:
    """Model-agnostic permutation importance on the reduced features."""
    from sklearn.metrics import roc_auc_score

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int)
    if len(X) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, y = X[idx], y[idx]

    metric_fn = metric_fn or (lambda yt, yp: roc_auc_score(yt, yp))
    baseline_scores = predict_fn(X)
    try:
        baseline = float(metric_fn(y, baseline_scores))
    except ValueError:
        return pd.DataFrame({"feature": feature_names, "importance_mean": np.nan,
                             "importance_std": np.nan, "baseline_metric": np.nan})

    rng = np.random.default_rng(seed)
    importances = np.zeros((repeats, X.shape[1]), dtype=float)
    for r in range(repeats):
        for j in range(X.shape[1]):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            try:
                score = float(metric_fn(y, predict_fn(X_perm)))
            except ValueError:
                score = float("nan")
            importances[r, j] = baseline - score

    return (
        pd.DataFrame({
            "feature": feature_names,
            "importance_mean": importances.mean(axis=0),
            "importance_std": importances.std(axis=0, ddof=1) if repeats > 1 else 0.0,
            "baseline_metric": baseline,
            "metric": "roc_auc_drop",
            "n_samples": len(X),
            "repeats": repeats,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def pca_attribution(
    reducer,
    reduced_importance: pd.Series | np.ndarray,
    top_k: int = 15,
) -> pd.DataFrame:
    """Distribute reduced-feature importance back through the PCA loadings.

    ``attribution_j = sum_c |loading[c, j]| * importance_c``

    This is a linear, first-order attribution.  It indicates which original
    encoded features influence the components the model relies on; it is not a
    causal explanation.
    """
    if not hasattr(reducer, "loadings"):
        return pd.DataFrame()
    loadings = np.asarray(reducer.loadings())          # (n_components, n_original)
    importance = np.asarray(reduced_importance, dtype=float).ravel()
    if len(importance) != loadings.shape[0]:
        raise ValueError(
            f"importance length {len(importance)} != n_components {loadings.shape[0]}"
        )
    weights = np.abs(loadings) * np.abs(importance)[:, None]
    score = weights.sum(axis=0)
    names = reducer.feature_names_in_ or [str(i) for i in range(len(score))]
    df = pd.DataFrame({
        "original_feature": names,
        "attributed_importance": score,
        "normalized_attribution": score / max(score.sum(), 1e-12),
    }).sort_values("attributed_importance", ascending=False)
    return df.head(top_k).reset_index(drop=True)


def shap_explain_cnn(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    out_dir: str | Path,
    max_background: int = 50,
    max_samples: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """KernelSHAP on the CNN+LSTM probability function (optional)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import shap

        from src.classical.cnn_lstm import predict_proba as cnn_predict_proba

        rng = np.random.default_rng(seed)
        bg_idx = rng.choice(len(X_background), size=min(max_background, len(X_background)), replace=False)
        ex_idx = rng.choice(len(X_explain), size=min(max_samples, len(X_explain)), replace=False)
        background = np.asarray(X_background, dtype=np.float32)[bg_idx]
        explain = np.asarray(X_explain, dtype=np.float32)[ex_idx]

        def f(x):
            return cnn_predict_proba(model, np.asarray(x, dtype=np.float32))

        explainer = shap.KernelExplainer(f, background)
        values = explainer.shap_values(explain, nsamples=200, silent=True)
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[..., 1]
        mean_abs = np.abs(values).mean(axis=0)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap as shap_pkg

        shap_pkg.summary_plot(
            values, explain, feature_names=feature_names,
            show=False, plot_type="bar",
            rng=np.random.default_rng(seed),
        )
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary.png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close()

        df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False)
        df.to_csv(out_dir / "shap_importance.csv", index=False)
        return {
            "performed": True,
            "n_explained": int(len(explain)),
            "n_background": int(len(background)),
            "figure": str(out_dir / "shap_summary.png"),
            "importance": df.to_dict(orient="records"),
        }
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("SHAP explanation failed: %s", exc)
        return {
            "performed": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "note": "Permutation importance remains available as the primary explanation.",
        }


def quantum_feature_sensitivity(
    predict_fn,
    X: np.ndarray,
    feature_names: list[str],
    delta: float = 0.25,
    max_samples: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Empirical sensitivity of the quantum model output to each encoded feature.

    For each feature, the value is replaced by ``x + delta`` (clipped to the
    encoding range ``[0, pi]``) and the mean absolute change in the predicted
    probability is recorded.  This characterises the *encoded* input-output map.
    It is not a Shapley attribution.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    if len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X[idx]
    base = np.asarray(predict_fn(X), dtype=float)
    rows = []
    for j, name in enumerate(feature_names):
        Xp = X.copy()
        Xp[:, j] = np.clip(Xp[:, j] + delta, 0.0, np.pi)
        shifted = np.asarray(predict_fn(Xp), dtype=float)
        change = np.abs(shifted - base)
        rows.append({
            "feature": name,
            "mean_abs_probability_change": float(change.mean()),
            "std_probability_change": float(change.std()),
            "max_abs_probability_change": float(change.max()),
            "delta": float(delta),
            "n_samples": int(len(X)),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("mean_abs_probability_change", ascending=False)
        .reset_index(drop=True)
    )


def example_explanation(
    predict_fn,
    sample: np.ndarray,
    feature_names: list[str],
    baseline: Optional[np.ndarray] = None,
    delta: float = 0.25,
) -> pd.DataFrame:
    """Per-feature local sensitivity for a single prediction."""
    sample = np.asarray(sample, dtype=np.float32).ravel()
    base = float(np.asarray(predict_fn(sample.reshape(1, -1))).ravel()[0])
    rows = []
    for j, name in enumerate(feature_names):
        perturbed = sample.copy()
        perturbed[j] = float(np.clip(perturbed[j] + delta, 0.0, np.pi))
        value = float(np.asarray(predict_fn(perturbed.reshape(1, -1))).ravel()[0])
        rows.append({
            "feature": name,
            "input_value": float(sample[j]),
            "baseline_probability": base,
            "perturbed_probability": value,
            "delta_probability": value - base,
        })
    return pd.DataFrame(rows).sort_values("delta_probability", key=np.abs, ascending=False)
