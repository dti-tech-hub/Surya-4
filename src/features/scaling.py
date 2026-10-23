"""Feature scaling.

The same post-reduction scaler is applied to *every* model so that the
representation entering the CNN+LSTM, QSVM and VQC is numerically identical.
The range ``[0, pi]`` is chosen because both angle encoding and the ZZ feature
map expect angles in that interval.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class FeatureScaler:
    """Percentile-clipped min-max scaling to ``[0, pi]`` (fit on train only)."""

    def __init__(
        self,
        feature_range: tuple[float, float] = (0.0, float(np.pi)),
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
    ) -> None:
        self.feature_range = (float(feature_range[0]), float(feature_range[1]))
        self.lower_percentile = float(lower_percentile)
        self.upper_percentile = float(upper_percentile)
        self.lo_: np.ndarray | None = None
        self.hi_: np.ndarray | None = None
        self.n_features_in_: int = 0
        self.n_clipped_: dict[str, int] = {}
        self.fitted = False

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D input, got {X.shape}")
        self.n_features_in_ = X.shape[1]
        self.lo_ = np.percentile(X, self.lower_percentile, axis=0)
        self.hi_ = np.percentile(X, self.upper_percentile, axis=0)
        # Guard against degenerate (constant) columns.
        span = self.hi_ - self.lo_
        degenerate = span < 1e-12
        span[degenerate] = 1.0
        self._degenerate = degenerate
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("FeatureScaler.transform called before fit().")
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"FeatureScaler expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        below = (X < self.lo_).sum(axis=0)
        above = (X > self.hi_).sum(axis=0)
        self.n_clipped_ = {
            "below": int(below.sum()),
            "above": int(above.sum()),
            "total_values": int(X.size),
        }
        clipped = np.clip(X, self.lo_, self.hi_)
        span = self.hi_ - self.lo_
        span = np.where(span < 1e-12, 1.0, span)
        unit = (clipped - self.lo_) / span
        low, high = self.feature_range
        out = unit * (high - low) + low
        return out.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        low, high = self.feature_range
        unit = (np.asarray(X_scaled, dtype=np.float64) - low) / (high - low)
        span = np.where((self.hi_ - self.lo_) < 1e-12, 1.0, self.hi_ - self.lo_)
        return unit * span + self.lo_

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "FeatureScaler",
            "feature_range": list(self.feature_range),
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
            "n_features_in": self.n_features_in_,
            "lo": self.lo_.tolist() if self.lo_ is not None else None,
            "hi": self.hi_.tolist() if self.hi_ is not None else None,
            "clipped_values_last_transform": self.n_clipped_,
            "leakage_policy": "fit on training split only",
            "rationale": "angle / ZZ feature maps require bounded rotation angles",
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as fh:
            json.dump(self.metadata(), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a FeatureScaler")
        return obj
