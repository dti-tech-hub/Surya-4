"""Quantum Support Vector Machine (fidelity quantum kernel + soft-margin SVM)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.svm import SVC

from src.quantum.circuits import (
    circuit_resources,
    fidelity_kernel,
    kernel_diagnostics,
    validate_kernel_against_qiskit,
)
from src.quantum.encoding import EncodingConfig, build_encoder


class QuantumSVM:
    """QSVM: classical reduced features -> quantum feature map -> fidelity kernel -> SVM."""

    def __init__(
        self,
        encoding: EncodingConfig,
        C: float = 1.0,
        class_weight: str | dict | None = "balanced",
        probability: bool = True,
        seed: int = 42,
        validate_kernel: bool = True,
        validation_shots: int = 100_000,
    ) -> None:
        self.encoding = encoding
        self.C = float(C)
        self.class_weight = class_weight
        self.probability = bool(probability)
        self.seed = int(seed)
        self.validate_kernel = bool(validate_kernel)
        self.validation_shots = int(validation_shots)

        self.encoder = build_encoder(encoding)
        self.model: Optional[SVC] = None
        self.train_states_: Optional[np.ndarray] = None
        self.train_y_: Optional[np.ndarray] = None
        self.timings_: dict[str, float] = {}
        self.kernel_diagnostics_: dict[str, Any] = {}
        self.validation_: dict[str, Any] = {}
        self.circuit_: Any = None

    # ------------------------------------------------------------------ fit
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "QuantumSVM":
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train).astype(int).ravel()

        t0 = time.perf_counter()
        states = self.encoder.states(X_train)
        kernel_time = time.perf_counter() - t0
        self.train_states_ = states
        self.train_y_ = y_train

        t0 = time.perf_counter()
        K_train = fidelity_kernel(states)
        gram_time = time.perf_counter() - t0

        self.model = SVC(
            kernel="precomputed",
            C=self.C,
            class_weight=self.class_weight,
            probability=self.probability,
            random_state=self.seed,
            cache_size=500,
        )
        t0 = time.perf_counter()
        self.model.fit(K_train, y_train)
        fit_time = time.perf_counter() - t0

        self.timings_ = {
            "state_preparation_time_s": kernel_time,
            "gram_matrix_time_s": gram_time,
            "svm_fit_time_s": fit_time,
            "training_time_s": kernel_time + gram_time + fit_time,
        }
        self.kernel_diagnostics_ = kernel_diagnostics(K_train)

        if self.validate_kernel and len(X_train) > 2:
            rng = np.random.default_rng(self.seed)
            n_val = min(len(X_train), 16)
            idx = rng.choice(len(X_train), size=n_val, replace=False)
            self.validation_ = validate_kernel_against_qiskit(
                self.encoder.circuit(), X_train[idx], shots=self.validation_shots
            )
        return self

    # -------------------------------------------------------------- predict
    def _kernel_to_train(self, X: np.ndarray) -> np.ndarray:
        if self.train_states_ is None:
            raise RuntimeError("QSVM.predict called before fit().")
        t0 = time.perf_counter()
        states = self.encoder.states(np.asarray(X, dtype=np.float64))
        t_states = time.perf_counter() - t0
        t0 = time.perf_counter()
        K = fidelity_kernel(states, self.train_states_)
        t_kernel = time.perf_counter() - t0
        self.timings_["test_state_preparation_time_s"] = t_states
        self.timings_["test_kernel_time_s"] = t_kernel
        return K

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        K = self._kernel_to_train(X)
        t0 = time.perf_counter()
        scores = self.model.decision_function(K)
        self.timings_["prediction_time_s"] = time.perf_counter() - t0
        return np.asarray(scores, dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        K = self._kernel_to_train(X)
        t0 = time.perf_counter()
        if self.probability:
            proba = self.model.predict_proba(K)[:, 1]
        else:
            scores = self.model.decision_function(K)
            proba = 1.0 / (1.0 + np.exp(-scores))
        self.timings_["prediction_time_s"] = time.perf_counter() - t0
        return np.asarray(proba, dtype=float)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # -------------------------------------------------------------- reporting
    def resources(self) -> dict[str, Any]:
        resources = circuit_resources(self.encoder.circuit())
        resources.update({
            "model": "QSVM",
            "encoding": self.encoding.method,
            "feature_map": self.encoding.to_dict().get("map_name"),
            "classical_svm": "sklearn.svm.SVC(kernel='precomputed')",
            "C": self.C,
            "shots": None,
            "backend": "qiskit.quantum_info.Statevector (exact fidelity kernel)",
            "kernel_calls_training": int(len(self.train_y_)) if self.train_y_ is not None else 0,
        })
        return resources

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "encoding": self.encoding,
            "train_states": self.train_states_,
            "train_y": self.train_y_,
            "C": self.C,
            "class_weight": self.class_weight,
            "probability": self.probability,
            "seed": self.seed,
            "timings": self.timings_,
            "kernel_diagnostics": self.kernel_diagnostics_,
            "validation": self.validation_,
            "resources": self.resources(),
        }
        joblib.dump(payload, path, compress=3)
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "encoding": self.encoding.to_dict(),
                    "resources": self.resources(),
                    "timings": self.timings_,
                    "kernel_diagnostics": self.kernel_diagnostics_,
                    "validation": self.validation_,
                    "serialization_note": (
                        "The fitted sklearn SVC, quantum kernel training states "
                        "and feature-map configuration are stored together so the "
                        "kernel can be recomputed deterministically at inference."
                    ),
                },
                fh,
                indent=2,
                default=str,
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "QuantumSVM":
        payload = joblib.load(Path(path))
        obj = cls(
            encoding=payload["encoding"],
            C=payload["C"],
            class_weight=payload.get("class_weight"),
            probability=payload.get("probability", True),
            seed=payload.get("seed", 42),
            validate_kernel=False,
        )
        obj.model = payload["model"]
        obj.train_states_ = payload["train_states"]
        obj.train_y_ = payload["train_y"]
        obj.timings_ = payload.get("timings", {})
        obj.kernel_diagnostics_ = payload.get("kernel_diagnostics", {})
        obj.validation_ = payload.get("validation", {})
        return obj
