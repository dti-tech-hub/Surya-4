"""Feature-reduction factory and validation-only configuration selection."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.features.mutual_information import MIReducer
from src.features.pca import PCAReducer
from src.features.scaling import FeatureScaler
from src.utils.logging import get_logger

LOGGER = get_logger("qeids.features")


def build_reducer(method: str, k: int, seed: int = 42):
    method = method.lower()
    if method in ("pca", "principal_component_analysis"):
        return PCAReducer(n_components=k, random_state=seed)
    if method in ("mi", "mutual_info", "mutual-information"):
        return MIReducer(k=k, random_state=seed)
    raise ValueError(f"Unknown feature-reduction method: {method!r}")


def fit_reducer(reducer, X_train: np.ndarray, y_train: np.ndarray,
                feature_names: list[str] | None = None):
    if isinstance(reducer, PCAReducer):
        return reducer.fit(X_train, feature_names=feature_names)
    if isinstance(reducer, MIReducer):
        return reducer.fit(X_train, y_train, feature_names=feature_names)
    raise TypeError(f"Unsupported reducer type {type(reducer)}")


def evaluate_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
) -> dict[str, float]:
    """Cheap linear probe used to rank feature configurations on validation."""
    model = LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced", random_state=seed, n_jobs=None
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_val, pred)),
        "f1": float(f1_score(y_val, pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_val, proba))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def select_configuration(
    X_train_full: np.ndarray,
    y_train: np.ndarray,
    X_val_full: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    methods: list[str],
    dimensions: list[int],
    metric: str = "f1",
    seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Rank (method, k) configurations using VALIDATION data only.

    Returns the winning configuration and the full ablation table.
    """
    rows: list[dict[str, Any]] = []
    for method in methods:
        for k in dimensions:
            if k > X_train_full.shape[1]:
                continue
            try:
                reducer = build_reducer(method, k, seed)
                fit_reducer(reducer, X_train_full, y_train, feature_names)
                Xtr = reducer.transform(X_train_full)
                Xva = reducer.transform(X_val_full)
                # Rescale per configuration so the probe sees a bounded space.
                scaler = FeatureScaler().fit(Xtr)
                scores = evaluate_probe(
                    scaler.transform(Xtr), y_train, scaler.transform(Xva), y_val, seed
                )
                row = {
                    "method": method,
                    "n_features": k,
                    **scores,
                    "status": "ok",
                }
                if isinstance(reducer, PCAReducer):
                    row["explained_variance"] = float(
                        reducer.cumulative_variance()[-1]
                    )
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Configuration %s/%s failed: %s", method, k, exc)
                row = {
                    "method": method,
                    "n_features": k,
                    "accuracy": float("nan"),
                    "f1": float("nan"),
                    "roc_auc": float("nan"),
                    "status": f"failed: {exc}",
                }
            rows.append(row)

    valid = [r for r in rows if r["status"] == "ok" and np.isfinite(r.get(metric, np.nan))]
    if not valid:
        raise RuntimeError("Every feature-reduction configuration failed validation.")
    best = max(valid, key=lambda r: (r[metric], r.get("roc_auc", 0.0), -r["n_features"]))
    return best, rows


def apply_reduction(
    method: str,
    k: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, FeatureScaler]:
    """Fit reducer + scaler on train, apply to all splits.

    The returned scaler is shared by every model so the reduced representation
    is identical for the classical and quantum pipelines.
    """
    reducer = build_reducer(method, k, seed)
    fit_reducer(reducer, X_train, y_train, feature_names)
    Xtr = reducer.transform(X_train)
    Xva = reducer.transform(X_val)
    Xte = reducer.transform(X_test)

    scaler = FeatureScaler().fit(Xtr)
    return (
        scaler.transform(Xtr),
        scaler.transform(Xva),
        scaler.transform(Xte),
        reducer,
        scaler,
    )


def save_reducer(reducer, cfg, tag: str) -> str:
    out_dir = Path(cfg.get_path("paths.data_processed")) / "reducers"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.joblib"
    reducer.save(path)
    return str(path)


def load_reducer(path: str | Path):
    path = Path(path)
    import joblib

    return joblib.load(path)
