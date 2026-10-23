"""Unseen attack-category generalisation experiment.

Terminology
-----------
We deliberately avoid the phrase "zero-day detection".  Holding an attack
category out of training measures *unseen-category generalisation*: how well a
model flags a family it has never been trained on.  A true zero-day claim would
require a genuinely novel exploit captured after deployment, which an offline
benchmark cannot provide.

Two complementary settings are implemented:

A. Category holdout (primary)
   R2L and U2R are removed from the training pool.  Models are retrained on
   Normal + DoS + Probe and evaluated on Normal + R2L + U2R.  This is the
   hardest realistic case: both categories are rare and behaviourally distinct
   from the volumetric attacks used for training.

B. External novel attack types (secondary)
   The official KDDTest+ file contains attack types that never appear in
   KDDTrain+ (e.g. apache2, mscan, snmpgetattack, worm).  The main model —
   trained without these types — is scored on them.  This is a distribution
   shift as well as an unseen-type test, and is reported as such.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.splitting import build_unseen_category_split, verify_no_category_leakage
from src.evaluation.metrics import compute_metrics, per_category_detection
from src.pipeline.training import ModelBundle, fit_model, fit_preprocessing
from src.utils.logging import get_logger

LOGGER = get_logger("qeids.zero_day")


def run_unseen_category_experiment(
    pool_df: pd.DataFrame,
    cfg,
    models: Sequence[str] = ("cnn_lstm", "qsvm", "vqc"),
    n_features: Optional[int] = None,
    quantum_train_size: Optional[int] = None,
    holdout_categories: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Retrain on known categories and evaluate on held-out categories."""
    seed = int(cfg.get_path("project.seed", 42))
    holdout = list(holdout_categories or cfg.get_path("zero_day.holdout_categories", ["R2L", "U2R"]))
    n_features = int(n_features or cfg.get_path("feature_reduction.n_features", 8))
    method = str(cfg.get_path("feature_reduction.method", "pca"))

    split = build_unseen_category_split(
        pool_df, holdout, seed=seed,
        normal_ratio=float(cfg.get_path("zero_day.normal_ratio_in_test", 1.0)),
    )
    leakage = verify_no_category_leakage(split["train"], split["test"], holdout)
    if not leakage["passed"]:
        raise RuntimeError(f"Category leakage detected: {leakage}")

    train_df = split["train"]
    val_df = split["validation"]
    test_df = split["test"]

    if verbose:
        LOGGER.info(
            "Unseen-category split: train=%d val=%d test=%d (holdout=%s)",
            len(train_df), len(val_df), len(test_df), holdout,
        )

    prep = fit_preprocessing(train_df, val_df, test_df, cfg, n_features, method, tag="unseen")

    y_train = train_df["is_attack"].to_numpy()
    y_val = val_df["is_attack"].to_numpy()
    y_test = test_df["is_attack"].to_numpy()

    rows: list[dict[str, Any]] = []
    bundles: dict[str, ModelBundle] = {}
    for kind in models:
        if verbose:
            LOGGER.info("Unseen-category experiment — training %s", kind)
        Xtr, ytr = prep.X_train, y_train
        if kind in ("qsvm", "vqc") and quantum_train_size:
            rng = np.random.default_rng(seed)
            n_q = min(int(quantum_train_size), len(Xtr))
            pos = np.flatnonzero(ytr == 1)
            neg = np.flatnonzero(ytr == 0)
            n_pos = max(1, int(round(n_q * len(pos) / len(ytr))))
            sel = np.concatenate([
                rng.choice(pos, size=min(n_pos, len(pos)), replace=False),
                rng.choice(neg, size=max(1, n_q - n_pos), replace=False),
            ])
            sel = np.sort(sel)
            Xtr, ytr = Xtr[sel], ytr[sel]

        bundle = fit_model(kind, Xtr, ytr, prep.X_val, y_val, cfg, verbose=verbose)
        from src.pipeline.training import tune_threshold

        tune_threshold(bundle, prep.X_val, y_val, cfg)
        scores = bundle.predict_proba(prep.X_test)
        metrics = compute_metrics(y_test, scores, threshold=bundle.threshold)
        bundles[kind] = bundle

        attack_mask = y_test == 1
        normal_mask = y_test == 0
        rows.append({
            "model": kind,
            "encoding": bundle.encoding,
            "holdout_categories": "+".join(holdout),
            "n_train": int(len(Xtr)),
            "n_test": int(len(y_test)),
            "threshold": float(bundle.threshold),
            **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            "unseen_attack_recall": float(
                (scores[attack_mask] >= bundle.threshold).mean()
            ) if attack_mask.any() else float("nan"),
            "unseen_attack_miss_rate": float(
                1 - (scores[attack_mask] >= bundle.threshold).mean()
            ) if attack_mask.any() else float("nan"),
            "normal_false_positive_rate": float(
                (scores[normal_mask] >= bundle.threshold).mean()
            ) if normal_mask.any() else float("nan"),
            "per_category": per_category_detection(
                test_df["attack_category"].to_numpy(), y_test,
                (scores >= bundle.threshold).astype(int), scores,
            ).to_dict(orient="records"),
        })

    details = {
        "holdout_categories": holdout,
        "held_out_counts": split["held_out_counts"],
        "test_label_counts": split["test_label_counts"],
        "split_sizes": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "leakage_audit": leakage,
        "train_categories": leakage["train_categories"],
        "test_categories": leakage["test_categories"],
        "terminology": "unseen attack-category generalisation (not a zero-day claim)",
        "feature_method": method,
        "n_features": n_features,
    }
    return pd.DataFrame(rows), details, {"prep": prep, "bundles": bundles}


def evaluate_external_novel_attacks(
    bundles: dict[str, ModelBundle],
    prep,
    external_df: pd.DataFrame,
    train_df: pd.DataFrame,
    seed: int = 42,
    reference_types: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score novel attack types in KDDTest+ with the already-trained main models.

    The bundles and ``prep`` must come from the same main experiment: the
    external rows are transformed with that experiment's preprocessing,
    feature reduction and scaling, and scored at each model's original
    validation-selected threshold.  Do not mix unseen-category bundles with the
    main preprocessing pipeline.

    ``reference_types`` should contain the attack-type vocabulary of the *full*
    KDDTrain+ file.  Quick mode trains on a 12,000-row subsample, so using only
    the sampled rows would mislabel rare training types as novel.  When not
    supplied, the sampled training types are used and the report says so.
    """
    external = external_df.copy()
    external["attack_type"] = external["attack_type"].astype(str).str.lower()
    if reference_types is None:
        train_types = set(train_df["attack_type"].astype(str).str.lower().unique())
        novelty_reference = "sampled training pool"
    else:
        train_types = {str(t).lower() for t in reference_types}
        novelty_reference = "full KDDTrain+ label set"
    novel_types = sorted(set(external["attack_type"].unique()) - train_types - {"normal"})

    novel_mask = external["attack_type"].isin(novel_types) | (external["is_attack"] == 0)
    subset = external[novel_mask].reset_index(drop=True)
    if subset.empty:
        return pd.DataFrame(), {"novel_types": novel_types, "n_samples": 0}

    X_ext = prep.preprocessor.transform(subset)
    X_ext = prep.scaler.transform(prep.reducer.transform(X_ext))
    y_ext = subset["is_attack"].to_numpy()

    rows = []
    for kind, bundle in bundles.items():
        scores = bundle.predict_proba(X_ext)
        metrics = compute_metrics(y_ext, scores, threshold=bundle.threshold)
        attack_mask = y_ext == 1
        rows.append({
            "model": kind,
            "encoding": bundle.encoding,
            "n_samples": int(len(subset)),
            "n_novel_attack_samples": int(attack_mask.sum()),
            "novel_attack_types": len(novel_types),
            **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            "unseen_attack_recall": float((scores[attack_mask] >= bundle.threshold).mean())
            if attack_mask.any() else float("nan"),
        })

    details = {
        "novel_types": novel_types,
        "n_novel_types": len(novel_types),
        "novelty_reference": novelty_reference,
        "n_samples": int(len(subset)),
        "n_novel_attack_samples": int(y_ext.sum()),
        "n_normal_samples": int((y_ext == 0).sum()),
        "terminology": "external novel attack types + normal (KDDTest+)",
        "caveat": (
            "KDDTest+ also differs in feature distribution from KDDTrain+, so "
            "this measures novel-type generalisation under covariate shift, "
            "not an isolated zero-day effect."
        ),
    }
    return pd.DataFrame(rows), details
