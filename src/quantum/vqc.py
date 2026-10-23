"""Variational Quantum Classifier (VQC).

Implementation strategy
-----------------------
The classifier is a genuine Qiskit circuit: a data-dependent feature map
(angle / ZZ or amplitude state preparation) followed by a trainable
``RealAmplitudes`` ansatz.  The decision is the probability of measuring
``|1>`` on qubit 0.

For efficiency we exploit an exact identity that holds for statevector
simulation:

    |psi(x; theta)> = U_ansatz(theta) |phi(x)>

so the feature states ``|phi(x)>`` are computed **once** with Qiskit's
``Statevector`` and each optimizer step only applies the ansatz unitary
``U_ansatz(theta)`` (obtained with ``qiskit.quantum_info.Operator``) to the
batch of feature states.  This is mathematically identical to running the full
circuit per sample, but ~100x faster.  ``tests/test_models.py`` verifies the
identity against direct full-circuit simulation.

The optimizer, ansatz and feature map all come from Qiskit / Qiskit Machine
Learning; no external quantum simulator is used.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

import joblib
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes
from qiskit.quantum_info import Operator
from qiskit_machine_learning.optimizers import COBYLA, SPSA, ADAM, SLSQP

from src.quantum.circuits import circuit_resources
from src.quantum.encoding import EncodingConfig, build_encoder

EPS = 1e-9


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


class VariationalQuantumClassifier:
    """Binary VQC with an exact-statevector training loop."""

    def __init__(
        self,
        encoding: EncodingConfig,
        ansatz_reps: int = 2,
        ansatz: str = "real_amplitudes",
        entanglement: str = "linear",
        optimizer: str = "COBYLA",
        maxiter: int = 150,
        learning_rate: float = 0.05,
        perturbation: float = 0.1,
        initial_point_scale: float = 0.1,
        l2: float = 1e-4,
        seed: int = 42,
        shots: Optional[int] = None,
    ) -> None:
        self.encoding = encoding
        self.ansatz_reps = int(ansatz_reps)
        self.ansatz_name = str(ansatz).lower()
        self.entanglement = str(entanglement)
        self.optimizer_name = str(optimizer).upper()
        self.maxiter = int(maxiter)
        self.learning_rate = float(learning_rate)
        self.perturbation = float(perturbation)
        self.initial_point_scale = float(initial_point_scale)
        self.l2 = float(l2)
        self.seed = int(seed)
        self.shots = shots

        self.encoder = build_encoder(encoding)
        self.ansatz: Optional[QuantumCircuit] = None
        self.weights_: Optional[np.ndarray] = None
        self.feature_states_train_: Optional[np.ndarray] = None
        self.history_: list[dict[str, float]] = []
        self.timings_: dict[str, float] = {}
        self.optimizer_result_: dict[str, Any] = {}
        self.scaler_: Optional[tuple[np.ndarray, np.ndarray]] = None

    # ------------------------------------------------------------ circuits
    def _build_ansatz(self) -> QuantumCircuit:
        if self.ansatz_name not in ("real_amplitudes", "realamplitudes"):
            raise ValueError(f"Unsupported ansatz {self.ansatz_name!r}")
        return real_amplitudes(
            num_qubits=self.encoding.num_qubits,
            reps=self.ansatz_reps,
            entanglement=self.entanglement,
            parameter_prefix="theta",
        )

    def full_circuit(self, sample: Optional[np.ndarray] = None, measure: bool = True) -> QuantumCircuit:
        """Human-inspectable full circuit for one sample."""
        n = self.encoding.num_qubits
        if sample is None:
            sample = np.linspace(0.1, 0.2, self.encoding.n_features)
        circuit = QuantumCircuit(n, name="VQC")
        if self.encoding.method == "angle":
            fm = self.encoder.circuit().assign_parameters(
                np.asarray(sample, dtype=float)
            )
            circuit.compose(fm, inplace=True)
        else:
            circuit.compose(self.encoder.circuit(sample), inplace=True)
        circuit.compose(self.ansatz or self._build_ansatz(), inplace=True)
        if measure:
            circuit.measure_all()
        return circuit

    # ------------------------------------------------------------------ fit
    def _make_optimizer(self):
        name = self.optimizer_name
        if name == "COBYLA":
            return COBYLA(maxiter=self.maxiter, tol=1e-4)
        if name == "SPSA":
            return SPSA(
                maxiter=self.maxiter,
                learning_rate=self.learning_rate,
                perturbation=self.perturbation,
            )
        if name == "ADAM":
            return ADAM(maxiter=self.maxiter, lr=self.learning_rate)
        if name == "SLSQP":
            return SLSQP(maxiter=self.maxiter, tol=1e-4)
        raise ValueError(
            f"Unsupported optimizer {self.optimizer_name!r}. "
            "Supported: COBYLA, SPSA, ADAM, SLSQP."
        )

    def _probabilities(self, states: np.ndarray, weights: np.ndarray) -> np.ndarray:
        ansatz = self.ansatz or self._build_ansatz()
        unitary = Operator(ansatz.assign_parameters(weights)).data
        evolved = states @ unitary.T
        # Qiskit uses little-endian ordering: qubit 0 is the least significant bit.
        return np.sum(np.abs(evolved.reshape(len(states), -1, 2)[:, :, 1]) ** 2, axis=1)

    def _loss(self, weights: np.ndarray, states: np.ndarray, y: np.ndarray) -> float:
        p = np.clip(self._probabilities(states, weights), EPS, 1 - EPS)
        bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        reg = self.l2 * float(np.sum(weights ** 2))
        return float(bce + reg)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> "VariationalQuantumClassifier":
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=float).ravel()

        t0 = time.perf_counter()
        states = self.encoder.states(X_train)
        state_time = time.perf_counter() - t0
        self.feature_states_train_ = states

        self.ansatz = self._build_ansatz()
        n_params = int(self.ansatz.num_parameters)
        rng = np.random.default_rng(self.seed)
        x0 = rng.uniform(
            -self.initial_point_scale, self.initial_point_scale, size=n_params
        )

        history: list[dict[str, float]] = []
        best = {"loss": np.inf, "weights": x0.copy(), "eval": 0}
        eval_counter = {"n": 0}

        def objective(w: np.ndarray) -> float:
            eval_counter["n"] += 1
            value = self._loss(w, states, y_train)
            history.append({
                "evaluation": float(eval_counter["n"]),
                "loss": float(value),
                "loss_plus_reg": float(value),
            })
            if value < best["loss"]:
                best["loss"] = value
                best["weights"] = np.asarray(w, dtype=float).copy()
                best["eval"] = eval_counter["n"]
            return value

        optimizer = self._make_optimizer()
        t0 = time.perf_counter()
        result = optimizer.minimize(objective, x0)
        optimization_time = time.perf_counter() - t0

        self.weights_ = np.asarray(getattr(result, "x", x0), dtype=float)
        self.history_ = history
        final_loss = float(getattr(result, "fun", objective(self.weights_)))
        # Prefer the best evaluated point (COBYLA's final point is usually best).
        if best["loss"] < final_loss:
            self.weights_ = best["weights"]
            final_loss = best["loss"]

        self.optimizer_result_ = {
            "optimizer": self.optimizer_name,
            "maxiter": self.maxiter,
            "num_parameters": n_params,
            "iterations": int(
                getattr(result, "nit", 0) or getattr(result, "nfev", 0)
                or eval_counter["n"]
            ),
            "objective_evaluations": int(eval_counter["n"]),
            "final_loss": final_loss,
            "best_evaluation": int(best["eval"]),
            "converged": bool(getattr(result, "success", False)),
            "message": str(getattr(result, "message", "")),
        }
        self.timings_ = {
            "state_preparation_time_s": state_time,
            "optimization_time_s": optimization_time,
            "training_time_s": state_time + optimization_time,
            "time_per_objective_evaluation_s": (
                optimization_time / max(eval_counter["n"], 1)
            ),
        }

        if X_val is not None and y_val is not None:
            t0 = time.perf_counter()
            val_proba = self.predict_proba(X_val)
            self.timings_["validation_inference_time_s"] = time.perf_counter() - t0
            p = np.clip(val_proba, EPS, 1 - EPS)
            yv = np.asarray(y_val, dtype=float)
            self.optimizer_result_["validation_loss"] = float(
                -np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p))
            )
        if verbose:
            print(
                f"VQC {self.optimizer_name}: {eval_counter['n']} evals, "
                f"loss={final_loss:.4f}, {optimization_time:.2f}s"
            )
        return self

    # -------------------------------------------------------------- predict
    def predict_proba(self, X: np.ndarray, shots: Optional[int] = None) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("VQC.predict_proba called before fit().")
        t0 = time.perf_counter()
        states = self.encoder.states(np.asarray(X, dtype=np.float64))
        p = self._probabilities(states, self.weights_)
        self.timings_["inference_time_s"] = time.perf_counter() - t0
        self.timings_["inference_time_per_sample_s"] = self.timings_["inference_time_s"] / max(len(X), 1)
        if shots:
            rng = np.random.default_rng(self.seed)
            counts = rng.binomial(int(shots), np.clip(p, 0, 1))
            p = counts / float(shots)
        return np.asarray(p, dtype=float)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        p = np.clip(self.predict_proba(X), EPS, 1 - EPS)
        return np.log(p / (1 - p))

    # ------------------------------------------------------------ reporting
    def convergence(self) -> list[dict[str, float]]:
        return self.history_

    def resources(self) -> dict[str, Any]:
        ansatz = self.ansatz or self._build_ansatz()
        full = self.full_circuit(measure=False)
        info = circuit_resources(full)
        ansatz_info = circuit_resources(ansatz)
        info.update({
            "model": "VQC",
            "encoding": self.encoding.method,
            "feature_map": self.encoding.to_dict().get("map_name"),
            "ansatz": "RealAmplitudes",
            "ansatz_reps": self.ansatz_reps,
            "ansatz_depth": ansatz_info["depth"],
            "num_ansatz_parameters": ansatz_info["num_parameters"],
            "optimizer": self.optimizer_name,
            "maxiter": self.maxiter,
            "shots": self.shots,
            "backend": "qiskit.quantum_info.Statevector (analytic probabilities)",
            "measurement": "P(qubit 0 = |1>)",
            "objective_evaluations": self.optimizer_result_.get("objective_evaluations"),
        })
        return info

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "encoding": self.encoding,
                "ansatz_reps": self.ansatz_reps,
                "ansatz_name": self.ansatz_name,
                "entanglement": self.entanglement,
                "optimizer": self.optimizer_name,
                "maxiter": self.maxiter,
                "learning_rate": self.learning_rate,
                "perturbation": self.perturbation,
                "initial_point_scale": self.initial_point_scale,
                "l2": self.l2,
                "seed": self.seed,
                "shots": self.shots,
                "weights": self.weights_,
                "history": self.history_,
                "timings": self.timings_,
                "optimizer_result": self.optimizer_result_,
                "resources": self.resources(),
            },
            path,
            compress=3,
        )
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "encoding": self.encoding.to_dict(),
                    "resources": self.resources(),
                    "timings": self.timings_,
                    "optimizer_result": self.optimizer_result_,
                    "weights": self.weights_.tolist() if self.weights_ is not None else None,
                },
                fh,
                indent=2,
                default=str,
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "VariationalQuantumClassifier":
        payload = joblib.load(Path(path))
        obj = cls(
            encoding=payload["encoding"],
            ansatz_reps=payload["ansatz_reps"],
            ansatz=payload.get("ansatz_name", "real_amplitudes"),
            entanglement=payload.get("entanglement", "linear"),
            optimizer=payload.get("optimizer", "COBYLA"),
            maxiter=payload.get("maxiter", 150),
            learning_rate=payload.get("learning_rate", 0.05),
            perturbation=payload.get("perturbation", 0.1),
            initial_point_scale=payload.get("initial_point_scale", 0.1),
            l2=payload.get("l2", 1e-4),
            seed=payload.get("seed", 42),
            shots=payload.get("shots"),
        )
        obj.weights_ = payload["weights"]
        obj.ansatz = obj._build_ansatz()
        obj.history_ = payload.get("history", [])
        obj.timings_ = payload.get("timings", {})
        obj.optimizer_result_ = payload.get("optimizer_result", {})
        return obj
