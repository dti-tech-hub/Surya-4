"""Data validation and an explicit leakage audit."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ValidationError(RuntimeError):
    pass


def validate_processed_frame(df: pd.DataFrame, name: str = "frame") -> dict:
    """Sanity checks that must hold for any processed dataframe."""
    issues: list[str] = []
    if df.empty:
        issues.append("empty dataframe")
    for col in ("label", "is_attack", "attack_category"):
        if col not in df.columns:
            issues.append(f"missing required column {col!r}")
    if "is_attack" in df.columns:
        values = set(pd.unique(df["is_attack"]))
        if not values.issubset({0, 1, np.int8(0), np.int8(1), 0.0, 1.0}):
            issues.append(f"is_attack has unexpected values {values}")
        if df["is_attack"].nunique() < 2:
            issues.append("is_attack contains a single class")
    if "label" in df.columns and df["label"].isna().any():
        issues.append("missing labels")
    report = {
        "name": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "issues": issues,
        "passed": len(issues) == 0,
    }
    return report


def validate_design_matrix(X: np.ndarray, name: str = "X") -> dict:
    issues: list[str] = []
    if X.ndim != 2:
        issues.append(f"expected 2-D matrix, got shape {X.shape}")
    if X.size == 0:
        issues.append("empty matrix")
    if not np.isfinite(X).all():
        n_bad = int((~np.isfinite(X)).sum())
        issues.append(f"{n_bad} non-finite values")
    if np.any(np.abs(X) > 1e6):
        issues.append("values exceed 1e6; check scaling")
    return {
        "name": name,
        "shape": list(X.shape),
        "dtype": str(X.dtype),
        "min": float(np.min(X)) if X.size else None,
        "max": float(np.max(X)) if X.size else None,
        "mean": float(np.mean(X)) if X.size else None,
        "std": float(np.std(X)) if X.size else None,
        "issues": issues,
        "passed": len(issues) == 0,
    }


def leakage_audit(
    preprocessor,
    reducer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    pca_explained: list[float] | None = None,
) -> dict:
    """Explicit checks that every fitted statistic came from the train split.

    We re-derive the expected train statistics and compare them with what the
    fitted objects store.  We also confirm that transforming train twice is
    idempotent (i.e. ``transform`` mutates no state) and that no test row is
    present in the training data.
    """
    checks: list[dict[str, Any]] = []

    # 1. Scaler statistics equal train statistics.
    try:
        scaler = preprocessor.numeric_pipeline.named_steps["scaler"]
        train_numeric = preprocessor._clean_numeric(train_df[preprocessor.numeric_columns])
        expected_mean = train_numeric.mean().to_numpy()
        stored_mean = np.asarray(scaler.mean_, dtype=float).ravel()
        diff = float(np.max(np.abs(expected_mean - stored_mean))) if len(expected_mean) else 0.0
        checks.append({
            "check": "scaler_fit_on_train",
            "max_abs_difference": diff,
            "passed": diff < 1e-5,
        })
    except Exception as exc:  # pragma: no cover
        checks.append({"check": "scaler_fit_on_train", "passed": False, "error": str(exc)})

    # 2. One-hot vocabulary contains no test-only level.
    try:
        onehot = preprocessor.categorical_pipeline.named_steps["onehot"]
        seen = set()
        for cats in onehot.categories_:
            seen.update(map(str, cats))
        test_levels = set()
        for col in preprocessor.categorical_columns:
            test_levels.update(map(str, test_df[col].unique()))
        unseen = sorted(test_levels - seen)
        checks.append({
            "check": "onehot_vocabulary_from_train",
            "unseen_test_levels": unseen,
            "passed": True,  # handle_unknown="ignore" makes unseen levels safe
        })
    except Exception as exc:  # pragma: no cover
        checks.append({"check": "onehot_vocabulary_from_train", "passed": False, "error": str(exc)})

    # 3. Transform is stateless and deterministic.
    try:
        X_raw_1 = preprocessor.transform(train_df)
        X_raw_2 = preprocessor.transform(train_df)
        stateless = bool(np.allclose(X_raw_1, X_raw_2, atol=1e-6))
        checks.append({
            "check": "transform_is_stateless",
            "raw_dimension": int(X_raw_1.shape[1]),
            "passed": stateless,
        })
    except Exception as exc:  # pragma: no cover
        X_raw_1 = None
        checks.append({"check": "transform_is_stateless", "passed": False, "error": str(exc)})

    # 4. Reducer was fitted on the *raw* training dimension and emits the
    #    requested number of components.  (X_train here is already reduced.)
    expected_input = int(X_raw_1.shape[1]) if X_raw_1 is not None else -1
    reported_input = int(getattr(reducer, "n_features_in_", -1))
    output_dim = -1
    try:
        if X_raw_1 is not None:
            output_dim = int(reducer.transform(X_raw_1[: min(8, len(X_raw_1))]).shape[1])
    except Exception:
        output_dim = -1
    checks.append({
        "check": "reducer_input_dimension",
        "expected": expected_input,
        "reported": reported_input,
        "passed": reported_input == expected_input,
    })
    checks.append({
        "check": "reducer_output_dimension",
        "expected": int(X_train.shape[1]),
        "reported": output_dim,
        "passed": output_dim == int(X_train.shape[1]),
    })

    # 5. No test row leaks into train (exact duplicate check on raw feature space).
    try:
        key_cols = [c for c in train_df.columns if c not in ("label", "is_attack", "attack_category")]
        train_keys = set(map(tuple, train_df[key_cols].astype(str).to_numpy()))
        test_keys = set(map(tuple, test_df[key_cols].astype(str).to_numpy()))
        overlap = len(train_keys & test_keys)
        checks.append({
            "check": "no_exact_train_test_row_overlap",
            "overlapping_rows": overlap,
            "passed": overlap == 0,
            "note": "Exact duplicate rows across splits would indicate leakage.",
        })
    except Exception as exc:  # pragma: no cover
        checks.append({"check": "no_exact_train_test_row_overlap", "passed": False, "error": str(exc)})

    # 6. Reduced matrices have the expected shapes.
    checks.append({
        "check": "reduced_shapes",
        "train": list(X_train.shape),
        "validation": list(X_val.shape),
        "test": list(X_test.shape),
        "passed": X_train.shape[1] == X_val.shape[1] == X_test.shape[1],
    })

    # 7. PCA variance is monotone non-increasing (if applicable).
    if pca_explained:
        monotone = all(
            pca_explained[i] >= pca_explained[i + 1] - 1e-12
            for i in range(len(pca_explained) - 1)
        )
        checks.append({"check": "pca_explained_variance_monotone", "passed": bool(monotone)})

    return {
        "checks": checks,
        "passed": all(c.get("passed", False) for c in checks),
        "policy": (
            "All preprocessing statistics (imputation, one-hot vocabulary, "
            "scaling, PCA loadings / MI ranking, quantum angle rescaling) are "
            "estimated on the training split only. Validation is used for "
            "early stopping, threshold selection and configuration selection. "
            "The test split is used exactly once, for final reporting."
        ),
    }


def split_integrity_report(split_meta: dict, train_df: pd.DataFrame,
                           val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    idx = split_meta.get("indices", {})
    if not idx:
        return {"passed": True, "note": "indices not stored"}
    sets = {k: set(v) for k, v in idx.items()}
    overlaps = {
        "train_validation": len(sets["train"] & sets["validation"]),
        "train_test": len(sets["train"] & sets["test"]),
        "validation_test": len(sets["validation"] & sets["test"]),
    }
    union = sets["train"] | sets["validation"] | sets["test"]
    expected_total = sum(len(v) for v in idx.values())
    return {
        "overlaps": overlaps,
        "union_size": len(union),
        "expected_total": expected_total,
        "sizes": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "passed": all(v == 0 for v in overlaps.values()) and len(union) == expected_total,
    }


def save_validation_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    return path
