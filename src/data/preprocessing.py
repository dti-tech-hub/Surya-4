"""Leakage-safe preprocessing.

Design rule: every fitted object in this module (imputer values, constant-column
list, one-hot vocabulary, scaler statistics) is estimated on the TRAINING split
only.  ``transform`` never updates state.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler

from src.data.loader import get_feature_columns


@dataclass
class Preprocessor:
    """Fit-on-train encoder + imputer + scaler for tabular IDS data."""

    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    numeric_pipeline: Optional[Pipeline] = None
    categorical_pipeline: Optional[Pipeline] = None
    missing_strategy: str = "median"
    scale_method: str = "standard"
    min_category_frequency: int = 2
    drop_constant: bool = True
    n_features_in_: int = 0
    fitted: bool = False

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        feature_cols = get_feature_columns(df)
        self.n_features_in_ = len(feature_cols)

        numeric_cols = [
            c for c in feature_cols
            if c not in ("protocol_type", "service", "flag")
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        categorical_cols = [c for c in feature_cols if c not in numeric_cols]

        # Constant columns detected on TRAIN only.
        constant = []
        if self.drop_constant:
            for col in numeric_cols:
                values = pd.to_numeric(df[col], errors="coerce")
                if values.nunique(dropna=True) <= 1:
                    constant.append(col)
            numeric_cols = [c for c in numeric_cols if c not in constant]

        self.constant_columns = constant
        self.numeric_columns = numeric_cols
        self.categorical_columns = categorical_cols

        # -- numeric branch -------------------------------------------------
        numeric_steps = [
            ("imputer", SimpleImputer(strategy=self.missing_strategy)),
        ]
        scaler = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }.get(self.scale_method)
        if scaler is None:
            raise ValueError(f"Unknown scale method {self.scale_method!r}")
        numeric_steps.append(("scaler", scaler))
        self.numeric_pipeline = Pipeline(numeric_steps)
        self.numeric_pipeline.fit(self._clean_numeric(df[numeric_cols]))

        # -- categorical branch ---------------------------------------------
        self.categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                min_frequency=self.min_category_frequency if self.min_category_frequency > 1 else None,
                dtype=np.float32,
            )),
        ])
        self.categorical_pipeline.fit(
            df[categorical_cols].astype(str) if categorical_cols else pd.DataFrame(index=df.index)
        )

        names: list[str] = []
        if numeric_cols:
            names.extend(numeric_cols)
        if categorical_cols:
            onehot = self.categorical_pipeline.named_steps["onehot"]
            names.extend(list(onehot.get_feature_names_out(categorical_cols)))
        self.feature_names = names
        self.fitted = True
        return self

    @staticmethod
    def _clean_numeric(frame: pd.DataFrame) -> pd.DataFrame:
        cleaned = frame.apply(pd.to_numeric, errors="coerce")
        return cleaned.replace([np.inf, -np.inf], np.nan)

    # -------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Preprocessor.transform called before fit().")
        blocks: list[np.ndarray] = []
        if self.numeric_columns:
            blocks.append(self.numeric_pipeline.transform(
                self._clean_numeric(df[self.numeric_columns])
            ))
        if self.categorical_columns:
            blocks.append(self.categorical_pipeline.transform(
                df[self.categorical_columns].astype(str)
            ))
        if not blocks:
            raise RuntimeError("Preprocessor has no features to transform.")
        X = np.hstack([np.asarray(b, dtype=np.float32) for b in blocks])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------- serialise
    def metadata(self) -> dict:
        meta: dict[str, Any] = {
            "type": "Preprocessor",
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "constant_columns": self.constant_columns,
            "feature_names": self.feature_names,
            "n_features_out": len(self.feature_names),
            "missing_strategy": self.missing_strategy,
            "scale_method": self.scale_method,
            "min_category_frequency": self.min_category_frequency,
            "drop_constant": self.drop_constant,
            "fitted": self.fitted,
            "leakage_policy": "fit on training split only",
        }
        if self.numeric_pipeline is not None:
            imputer = self.numeric_pipeline.named_steps["imputer"]
            scaler = self.numeric_pipeline.named_steps["scaler"]
            meta["imputation_values"] = dict(
                zip(self.numeric_columns, np.asarray(imputer.statistics_).ravel().tolist())
            )
            meta["scaler"] = {
                "class": type(scaler).__name__,
                "mean": np.asarray(getattr(scaler, "mean_", [])).ravel().tolist(),
                "scale": np.asarray(getattr(scaler, "scale_", [])).ravel().tolist(),
            }
        return meta

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as fh:
            json.dump(self.metadata(), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a Preprocessor")
        return obj


def clean_dataframe(df: pd.DataFrame, cfg=None) -> tuple[pd.DataFrame, dict]:
    """Basic cleaning *before* the split (unsupervised only, no leakage risk).

    Anything that depends on the target or on distributional statistics must
    happen after the split and is handled by ``Preprocessor``.
    """
    report: dict[str, Any] = {"rows_in": int(len(df))}
    df = df.copy()

    # Normalise whitespace in label / categorical columns.
    for col in ("label", "attack_type", "attack_category", "protocol_type", "service", "flag"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Bring every declared feature to a numeric or string representation.
    numeric = [c for c in get_feature_columns(df) if c not in ("protocol_type", "service", "flag")]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    report["missing_cells"] = int(df.isna().sum().sum())
    report["missing_by_column"] = {
        str(k): int(v) for k, v in df.isna().sum().items() if int(v) > 0
    }

    if cfg is not None and bool(cfg.get_path("preprocessing.remove_duplicates", False)):
        before = len(df)
        df = df.drop_duplicates(subset=get_feature_columns(df) + ["label"], keep="first")
        report["duplicates_removed"] = int(before - len(df))
    else:
        report["duplicates_removed"] = 0

    report["rows_out"] = int(len(df))
    report["binary_attack_rate"] = float(df["is_attack"].mean()) if "is_attack" in df else None
    report["label_counts"] = {str(k): int(v) for k, v in df["label"].value_counts().items()}
    return df.reset_index(drop=True), report


def build_class_distribution_report(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    def _counts(df: pd.DataFrame) -> dict:
        return {str(k): int(v) for k, v in df["label"].value_counts().items()}

    return {
        "pool": {"rows": int(len(train) + len(val) + len(test)), "labels": _counts(pd.concat([train, val, test]))},
        "train": {"rows": int(len(train)), "labels": _counts(train),
                  "attack_rate": float(train["attack_category"].ne("Normal").mean())},
        "validation": {"rows": int(len(val)), "labels": _counts(val),
                       "attack_rate": float(val["attack_category"].ne("Normal").mean())},
        "test": {"rows": int(len(test)), "labels": _counts(test),
                 "attack_rate": float(test["attack_category"].ne("Normal").mean())},
        "category_counts": {
            split: {str(k): int(v) for k, v in df["attack_category"].value_counts().items()}
            for split, df in (("train", train), ("validation", val), ("test", test))
        },
    }
