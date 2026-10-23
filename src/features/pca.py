"""PCA-based dimensionality reduction (fit on train only)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA


class PCAReducer:
    def __init__(self, n_components: int = 8, whiten: bool = False, random_state: int = 42) -> None:
        self.n_components = int(n_components)
        self.whiten = bool(whiten)
        self.random_state = int(random_state)
        self.pca_: PCA | None = None
        self.feature_names_in_: list[str] | None = None
        self.n_features_in_: int = 0

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> "PCAReducer":
        X = np.asarray(X, dtype=np.float64)
        max_components = min(X.shape[0], X.shape[1])
        if self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} exceeds the maximum "
                f"{max_components} for data shape {X.shape}"
            )
        self.pca_ = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            svd_solver="full",
            random_state=self.random_state,
        )
        self.pca_.fit(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = list(feature_names) if feature_names else list(
            map(str, range(X.shape[1]))
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca_ is None:
            raise RuntimeError("PCAReducer.transform called before fit().")
        return self.pca_.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)

    def fit_transform(self, X: np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
        return self.fit(X, feature_names).transform(X)

    # --------------------------------------------------------------- reports
    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        if self.pca_ is None:
            return np.array([])
        return self.pca_.explained_variance_ratio_

    def cumulative_variance(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio_)

    def loadings(self) -> np.ndarray:
        """Component vectors, shape (n_components, n_original_features)."""
        if self.pca_ is None:
            raise RuntimeError("PCA not fitted")
        return self.pca_.components_

    def feature_contributions(self, top_k: int = 10) -> dict[str, list[dict[str, Any]]]:
        """Map each PCA component back to its strongest original features."""
        if self.pca_ is None:
            raise RuntimeError("PCA not fitted")
        names = self.feature_names_in_ or [str(i) for i in range(self.pca_.n_features_in_)]
        out: dict[str, list[dict[str, Any]]] = {}
        for comp_idx, component in enumerate(self.pca_.components_):
            order = np.argsort(np.abs(component))[::-1][:top_k]
            out[f"PC{comp_idx + 1}"] = [
                {
                    "feature": names[i] if i < len(names) else str(i),
                    "loading": float(component[i]),
                    "abs_loading": float(abs(component[i])),
                }
                for i in order
            ]
        return out

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "PCAReducer",
            "n_components": self.n_components,
            "whiten": self.whiten,
            "n_features_in": self.n_features_in_,
            "explained_variance_ratio": self.explained_variance_ratio_.tolist(),
            "cumulative_explained_variance": self.cumulative_variance().tolist(),
            "total_explained_variance": float(self.cumulative_variance()[-1])
            if len(self.explained_variance_ratio_) else 0.0,
            "feature_contributions": self.feature_contributions(),
            "leakage_policy": "fit on training split only",
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as fh:
            json.dump(self.metadata(), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PCAReducer":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a PCAReducer")
        return obj
