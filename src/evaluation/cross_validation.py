"""k-fold cross-validation with in-fold preprocessing.

Every fold refits the imputer, one-hot vocabulary, scaler, PCA/MI selector and
post-reduction scaler on that fold's training data only.  The held-out fold is
never seen during fitting, early stopping or threshold selection.
"""
from __future__ import annotations

import time
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.evaluation.metrics import compute_metrics
from src.evaluation.statistical_tests import summarize_cv
from src.pipeline.training import evaluate_bundle, fit_model, fit_preprocessing
from src.utils.logging import get_logger

LOGGER = get_logger("qeids.cv")


def run_cross_validation(
    pool_df: pd.DataFrame,
    cfg,
    n_features: int,
    method: str,
    models: Sequence[str] = ("cnn_lstm", "qsvm", "vqc"),
    folds: Optional[int] = None,
    max_samples: Optional[int] = None,
    quantum_train_size: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Stratified k-fold CV on the non-test pool.

    Returns
    -------
    cv_df : per-fold metric table
    summary : mean/std/CI per model
    details : metadata (folds, sizes, leakage policy)
    """
    seed = int(seed if seed is not None else cfg.get_path("project.seed", 42))
    folds = int(folds or cfg.get_path("cv.folds", 5))
    max_samples = max_samples if max_samples is not None else cfg.get_path("experiment.quick.cv_max_samples")

    data = pool_df
    if max_samples is not None and len(data) > int(max_samples):
        data, _ = train_test_split(
            data, train_size=int(max_samples),
            stratify=data["is_attack"], random_state=seed, shuffle=True,
        )
        data = data.reset_index(drop=True)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    y_all = data["is_attack"].to_numpy()

    rows: list[dict[str, Any]] = []
    fold_meta: list[dict[str, Any]] = []
    t_cv = time.perf_counter()

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(data, y_all), start=1):
        fold_train_full = data.iloc[tr_idx].reset_index(drop=True)
        fold_test = data.iloc[te_idx].reset_index(drop=True)

        # Internal validation for early stopping only.
        fold_train, internal_val = train_test_split(
            fold_train_full, test_size=0.12, stratify=fold_train_full["is_attack"],
            random_state=seed + fold_idx, shuffle=True,
        )
        fold_train = fold_train.reset_index(drop=True)
        internal_val = internal_val.reset_index(drop=True)

        if min(fold_train["is_attack"].value_counts().min(),
               internal_val["is_attack"].value_counts().min()) < 2:
            LOGGER.warning("Fold %d is too small / single-class; skipping", fold_idx)
            continue

        t_fold = time.perf_counter()
        prep = fit_preprocessing(
            fold_train, internal_val, fold_test, cfg,
            n_features=n_features, method=method, tag=f"cv{fold_idx}",
        )

        for kind in models:
            if verbose:
                LOGGER.info("CV fold %d/%d — training %s", fold_idx, folds, kind)
            Xtr, ytr = prep.X_train, fold_train["is_attack"].to_numpy()
            Xva, yva = prep.X_val, internal_val["is_attack"].to_numpy()

            train_size = len(Xtr)
            if kind in ("qsvm", "vqc") and quantum_train_size:
                rng = np.random.default_rng(seed + fold_idx)
                n_q = min(int(quantum_train_size), len(Xtr))
                # stratified subsample
                pos = np.flatnonzero(ytr == 1)
                neg = np.flatnonzero(ytr == 0)
                n_pos = max(1, int(round(n_q * len(pos) / len(ytr))))
                n_neg = max(1, n_q - n_pos)
                sel = np.concatenate([
                    rng.choice(pos, size=min(n_pos, len(pos)), replace=False),
                    rng.choice(neg, size=min(n_neg, len(neg)), replace=False),
                ])
                sel = np.sort(sel)
                Xtr, ytr = Xtr[sel], ytr[sel]
                train_size = len(sel)

            try:
                t0 = time.perf_counter()
                bundle = fit_model(
                    kind, Xtr, ytr, Xva, yva, cfg,
                    epochs_override=int(cfg.get_path("experiment.quick.epochs", 25))
                    if kind == "cnn_lstm" else None,
                    verbose=False,
                )
                train_time = time.perf_counter() - t0
                t1 = time.perf_counter()
                proba = bundle.predict_proba(prep.X_test)
                inference_time = time.perf_counter() - t1
                metrics = compute_metrics(
                    fold_test["is_attack"].to_numpy(), proba,
                    threshold=float(cfg.get_path("evaluation.threshold", 0.5)),
                )
                rows.append({
                    "fold": fold_idx,
                    "model": kind,
                    "n_train": int(train_size),
                    "n_test": int(len(fold_test)),
                    "train_time_s": train_time,
                    "inference_time_s": inference_time,
                    **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
                })
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.error("CV fold %d model %s failed: %s", fold_idx, kind, exc)
                rows.append({
                    "fold": fold_idx, "model": kind, "status": f"failed: {exc}",
                    "n_train": int(train_size), "n_test": int(len(fold_test)),
                })

        fold_meta.append({
            "fold": fold_idx,
            "n_train": int(len(fold_train)),
            "n_internal_val": int(len(internal_val)),
            "n_test": int(len(fold_test)),
            "fold_seconds": time.perf_counter() - t_fold,
        })

    cv_df = pd.DataFrame(rows)
    summary: list[dict[str, Any]] = []
    for metric in ("f1", "accuracy", "roc_auc", "recall", "precision"):
        if metric in cv_df.columns:
            summary.extend(summarize_cv(cv_df, metric, alpha=float(cfg.get_path("evaluation.alpha", 0.05))))

    details = {
        "folds_requested": folds,
        "folds_completed": int(cv_df["fold"].nunique()) if not cv_df.empty else 0,
        "pool_size": int(len(data)),
        "metrics": ["f1", "accuracy", "roc_auc", "recall", "precision"],
        "total_time_s": time.perf_counter() - t_cv,
        "fold_details": fold_meta,
        "leakage_policy": (
            "Preprocessing, feature reduction and quantum rescaling are refit "
            "inside every fold on fold-training data only. Early stopping uses "
            "an internal validation split carved out of the fold-training data. "
            "The held-out fold is used only for scoring."
        ),
    }
    return cv_df, summary, details


def paired_fold_scores(cv_df: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """Pivot to folds x models for paired tests."""
    if cv_df.empty or metric not in cv_df.columns:
        return pd.DataFrame()
    return cv_df.pivot_table(index="fold", columns="model", values=metric)
