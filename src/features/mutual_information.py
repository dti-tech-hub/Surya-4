"""Mutual-information feature selection (fit on train only)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_selection import mutual_info_classif


class MIReducer:
    """Top-k mutual-information selector on the encoded feature matrix.

    Notes
    -----
    MI is estimated on the *encoded* matrix (numeric columns scaled, categorical
    columns one-hot).  Consequently a selected feature may be a single one-hot
    indicator.  This is reported honestly in the feature-importance table.
    """

    def __init__(self, k: int = 8, random_state: int = 42, discrete_features: str | bool = "auto") -> None:
        self.k = int(k)
        self.random_state = int(random_state)
        self.discrete_features = discrete_features
        self.scores_: np.ndarray | None = None
        self.selected_indices_: np.ndarray | None = None
        self.feature_names_in_: list[str] | None = None
        self.n_features_in_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> "MIReducer":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        if self.k > X.shape[1]:
            raise ValueError(f"k={self.k} exceeds the number of available features {X.shape[1]}")
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = list(feature_names) if feature_names else list(map(str, range(X.shape[1])))
        self.scores_ = mutual_info_classif(
            X, y, random_state=self.random_state, discrete_features=self.discrete_features
        )
        # Stable sort: ties broken by original column order for reproducibility.
        order = np.argsort(-self.scores_, kind="stable")
        self.selected_indices_ = np.sort(order[: self.k])
        self._ranked_order = order
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_indices_ is None:
            raise RuntimeError("MIReducer.transform called before fit().")
        X = np.asarray(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"MIReducer expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        return X[:, self.selected_indices_].astype(np.float32)

    def fit_transform(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
        return self.fit(X, y, feature_names).transform(X)

    def selected_features(self) -> list[str]:
        if self.selected_indices_ is None or self.feature_names_in_ is None:
            return []
        return [self.feature_names_in_[i] for i in self.selected_indices_]

    def ranking(self) -> list[dict[str, Any]]:
        if self.scores_ is None or self.feature_names_in_ is None:
            return []
        return [
            {
                "rank": rank + 1,
                "index": int(i),
                "feature": self.feature_names_in_[i],
                "mutual_information": float(self.scores_[i]),
                "selected": bool(i in set(self.selected_indices_.tolist())),
            }
            for rank, i in enumerate(self._ranked_order)
        ]

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "MIReducer",
            "k": self.k,
            "n_features_in": self.n_features_in_,
            "selected_indices": self.selected_indices_.tolist() if self.selected_indices_ is not None else [],
            "selected_features": self.selected_features(),
            "scores": self.scores_.tolist() if self.scores_ is not None else [],
            "ranking": self.ranking(),
            "leakage_policy": "mutual information estimated on training split only",
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as fh:
            json.dump(self.metadata(), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MIReducer":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain an MIReducer")
        return obj
