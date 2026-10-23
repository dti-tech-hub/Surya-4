"""Controlled distribution-shift ("concept drift") experiment.

Important scope statement
-------------------------
NSL-KDD has no timestamps, so this module does **not** perform temporal drift
detection on a real network stream.  It performs two controlled experiments:

1. Covariate shift: Gaussian noise of increasing magnitude and random feature
   permutation are injected into the model input, and degradation is measured.
2. Simulated ordered windows: the official KDDTest+ file is split into
   contiguous chunks in file order and scored per chunk.  The ordering is a
   *convention*, not a real capture timeline.

Both are clearly labelled as simulation in the results.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats as st

from src.evaluation.metrics import compute_metrics
from src.pipeline.training import ModelBundle


def _ks_shift(X_ref: np.ndarray, X_shift: np.ndarray) -> dict[str, float]:
    stats = []
    for j in range(X_ref.shape[1]):
        try:
            stats.append(float(st.ks_2samp(X_ref[:, j], X_shift[:, j]).statistic))
        except Exception:
            continue
    if not stats:
        return {"ks_mean": float("nan"), "ks_max": float("nan")}
    return {"ks_mean": float(np.mean(stats)), "ks_max": float(np.max(stats))}


def noise_shift_experiment(
    bundle: ModelBundle,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_reference: np.ndarray,
    noise_levels: Sequence[float],
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scale = np.std(X_reference, axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    rows = []
    for level in noise_levels:
        X_shift = X_test + rng.normal(0.0, 1.0, size=X_test.shape) * scale * float(level)
        scores = bundle.predict_proba(X_shift)
        metrics = compute_metrics(y_test, scores, threshold=bundle.threshold)
        rows.append({
            "shift_type": "gaussian_noise",
            "severity": float(level),
            "model": bundle.kind,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "fpr": metrics["fpr"],
            "fnr": metrics["fnr"],
            **_ks_shift(X_reference, X_shift),
        })
    return pd.DataFrame(rows)


def permutation_shift_experiment(
    bundle: ModelBundle,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_reference: np.ndarray,
    fractions: Sequence[float],
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_features = X_test.shape[1]
    rows = []
    for frac in fractions:
        X_shift = X_test.copy()
        k = int(round(frac * n_features))
        if k > 0:
            cols = rng.choice(n_features, size=k, replace=False)
            for j in cols:
                X_shift[:, j] = rng.permutation(X_shift[:, j])
        scores = bundle.predict_proba(X_shift)
        metrics = compute_metrics(y_test, scores, threshold=bundle.threshold)
        rows.append({
            "shift_type": "feature_permutation",
            "severity": float(frac),
            "n_permuted_features": int(k),
            "model": bundle.kind,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "fpr": metrics["fpr"],
            "fnr": metrics["fnr"],
            **_ks_shift(X_reference, X_shift),
        })
    return pd.DataFrame(rows)


def ordered_window_experiment(
    bundle: ModelBundle,
    X_external: np.ndarray,
    y_external: np.ndarray,
    windows: int = 5,
) -> pd.DataFrame:
    """Score contiguous file-order chunks (simulated stream, not real time)."""
    n = len(X_external)
    if n == 0:
        return pd.DataFrame()
    boundaries = np.linspace(0, n, windows + 1).astype(int)
    rows = []
    for w in range(windows):
        lo, hi = boundaries[w], boundaries[w + 1]
        if hi <= lo:
            continue
        Xw, yw = X_external[lo:hi], y_external[lo:hi]
        if len(np.unique(yw)) < 2:
            continue
        scores = bundle.predict_proba(Xw)
        metrics = compute_metrics(yw, scores, threshold=bundle.threshold)
        rows.append({
            "window": w + 1,
            "start_index": int(lo),
            "end_index": int(hi),
            "n_samples": int(hi - lo),
            "attack_rate": float(yw.mean()),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "fpr": metrics["fpr"],
            "fnr": metrics["fnr"],
        })
    return pd.DataFrame(rows)


def drift_summary(drift_df: pd.DataFrame) -> pd.DataFrame:
    if drift_df.empty:
        return drift_df
    baseline = drift_df[drift_df["severity"] == 0.0]
    rows = []
    for (shift_type, model), group in drift_df.groupby(["shift_type", "model"]):
        base = baseline[(baseline["shift_type"] == shift_type) & (baseline["model"] == model)]
        if base.empty:
            continue
        base_f1 = float(base["f1"].iloc[0])
        base_auc = float(base["roc_auc"].iloc[0])
        worst = group.sort_values("severity").iloc[-1]
        rows.append({
            "shift_type": shift_type,
            "model": model,
            "baseline_f1": base_f1,
            "worst_f1": float(worst["f1"]),
            "f1_degradation": base_f1 - float(worst["f1"]),
            "baseline_roc_auc": base_auc,
            "worst_roc_auc": float(worst["roc_auc"]),
            "roc_auc_degradation": base_auc - float(worst["roc_auc"]),
            "worst_severity": float(worst["severity"]),
            "worst_ks_mean": float(worst["ks_mean"]),
        })
    return pd.DataFrame(rows)
