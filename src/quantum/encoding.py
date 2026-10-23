"""Quantum feature encodings.

Two encodings are implemented:

``angle``
    Each reduced feature ``x_i`` is a rotation angle.  With ``map_name='zz'`` the
    circuit is a second-order Pauli-Z (ZZ) feature map, which introduces
    two-qubit phase interactions.  With ``map_name='angle'`` the circuit is a
    product of ``RY`` rotations followed by optional CNOT entanglement layers.

``amplitude``
    The (L2-normalised, zero-padded) feature vector is prepared directly as a
    quantum state via ``QuantumCircuit.initialize``.  This requires a
    power-of-two dimension, so 8 features -> 3 qubits, 12 or 16 -> 4 qubits.

Scientific note
---------------
For pure amplitude encoding the fidelity kernel reduces to the squared cosine
similarity between the padded feature vectors, i.e. a classical polynomial
kernel.  This is a genuine encoding experiment, but it cannot by itself create
a computational separation from classical methods.  We state this explicitly in
the results and limitations instead of implying an advantage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector


class EncodingError(ValueError):
    pass


@dataclass
class EncodingConfig:
    method: str = "angle"              # angle | amplitude
    n_features: int = 8
    map_name: str = "zz"               # zz | angle  (angle encoding only)
    reps: int = 2
    entanglement: str = "linear"
    name: str = ""

    def __post_init__(self) -> None:
        self.method = str(self.method).lower()
        if self.method not in ("angle", "amplitude"):
            raise EncodingError(
                f"Unknown encoding {self.method!r}. Supported: 'angle', 'amplitude'."
            )
        if self.n_features < 1:
            raise EncodingError("n_features must be >= 1")
        if not self.name:
            self.name = f"{self.method}-{self.n_features}f"

    @property
    def num_qubits(self) -> int:
        if self.method == "angle":
            return int(self.n_features)
        dim = 1
        qubits = 0
        while dim < self.n_features:
            dim *= 2
            qubits += 1
        return max(1, qubits)

    @property
    def padded_dimension(self) -> int:
        return int(2 ** self.num_qubits)

    @property
    def padding(self) -> int:
        return int(self.padded_dimension - self.n_features) if self.method == "amplitude" else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_features": self.n_features,
            "map_name": self.map_name if self.method == "angle" else "state_preparation",
            "reps": self.reps if self.method == "angle" else 0,
            "entanglement": self.entanglement if self.method == "angle" else "none",
            "num_qubits": self.num_qubits,
            "padded_dimension": self.padded_dimension,
            "padding": self.padding,
            "name": self.name,
        }


# ---------------------------------------------------------------------------
# Angle encoding
# ---------------------------------------------------------------------------
def angle_feature_map(
    n_qubits: int,
    reps: int = 1,
    entanglement: str = "linear",
    parameter_prefix: str = "x",
    name: str = "AngleFeatureMap",
) -> QuantumCircuit:
    """RY angle encoding with ``reps`` entangling layers (no trainable params)."""
    params = ParameterVector(parameter_prefix, n_qubits)
    circuit = QuantumCircuit(n_qubits, name=name)
    for i in range(n_qubits):
        circuit.ry(params[i], i)
    for _ in range(max(1, reps)):
        if entanglement in ("linear", "ring"):
            pairs = [(i, i + 1) for i in range(n_qubits - 1)]
            if entanglement == "ring" and n_qubits > 2:
                pairs.append((n_qubits - 1, 0))
        elif entanglement == "full":
            pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
        elif entanglement == "none":
            pairs = []
        else:
            raise EncodingError(f"Unknown entanglement {entanglement!r}")
        for i, j in pairs:
            circuit.cx(i, j)
    return circuit


def build_feature_map_circuit(
    map_name: str,
    n_qubits: int,
    reps: int = 2,
    entanglement: str = "linear",
) -> QuantumCircuit:
    map_name = str(map_name).lower()
    if map_name == "zz":
        # Qiskit >= 2.1 function form (the ZZFeatureMap class is deprecated).
        return zz_feature_map(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement=entanglement,
            name="ZZFeatureMap",
        )
    if map_name == "angle":
        return angle_feature_map(n_qubits, reps=reps, entanglement=entanglement)
    if map_name in ("pauli", "pauli_feature_map"):
        from qiskit.circuit.library import pauli_feature_map

        return pauli_feature_map(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement=entanglement,
            paulis=["Z", "ZZ"],
            name="PauliFeatureMap",
        )
    raise EncodingError(
        f"Unknown feature map {map_name!r}. Supported: 'zz', 'angle', 'pauli'."
    )


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
class BaseEncoder:
    """Common interface: bounded inputs -> statevectors + circuit resources."""

    def __init__(self, config: EncodingConfig) -> None:
        self.config = config

    @property
    def num_qubits(self) -> int:
        return self.config.num_qubits

    def circuit(self) -> QuantumCircuit:  # pragma: no cover - interface
        raise NotImplementedError

    def states(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return self.config.to_dict()


class AngleEncoder(BaseEncoder):
    """Rotation-angle encoding (ZZ or plain angle feature map)."""

    def __init__(self, config: EncodingConfig) -> None:
        super().__init__(config)
        if config.n_features > 16:
            raise EncodingError(
                f"Angle encoding with {config.n_features} qubits needs "
                f"{2 ** config.n_features} statevector amplitudes and is not "
                "supported on a laptop. Reduce the feature dimension."
            )
        self._circuit = build_feature_map_circuit(
            config.map_name,
            config.num_qubits,
            reps=config.reps,
            entanglement=config.entanglement,
        )

    def circuit(self) -> QuantumCircuit:
        return self._circuit

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise EncodingError(f"Expected 2-D input, got shape {X.shape}")
        if X.shape[1] != self.config.n_features:
            raise EncodingError(
                f"Encoder expects {self.config.n_features} features, got {X.shape[1]}"
            )
        if not np.isfinite(X).all():
            raise EncodingError("Input contains NaN/inf values")
        return X

    def states(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X)
        out = np.empty((len(X), 2 ** self.num_qubits), dtype=np.complex128)
        for i, row in enumerate(X):
            bound = self._circuit.assign_parameters(row)
            out[i] = Statevector.from_instruction(bound).data
        return out

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        circuit = self._circuit
        info.update({
            "feature_map": self.config.map_name,
            "rotation": "RY (angle encoding)" if self.config.map_name == "angle" else "Pauli-Z / ZZ phase",
            "circuit_depth": int(circuit.depth()),
            "circuit_size": int(circuit.size()),
            "num_parameters": int(circuit.num_parameters),
        })
        return info


class AmplitudeEncoder(BaseEncoder):
    """Amplitude encoding with L2 normalisation and zero padding."""

    def __init__(self, config: EncodingConfig) -> None:
        super().__init__(config)
        if config.num_qubits > 10:
            raise EncodingError(
                f"Amplitude encoding needs {config.num_qubits} qubits "
                f"({2 ** config.num_qubits} amplitudes). Reduce the dimension."
            )
        self._dim = config.padded_dimension

    def circuit(self, sample: Optional[np.ndarray] = None) -> QuantumCircuit:
        circuit = QuantumCircuit(self.num_qubits, name="AmplitudeEncoding")
        if sample is None:
            sample = np.linspace(1.0, 2.0, self.config.n_features)
        circuit.initialize(prepare_amplitudes(sample, self._dim), range(self.num_qubits))
        return circuit

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise EncodingError(f"Expected 2-D input, got shape {X.shape}")
        if X.shape[1] != self.config.n_features:
            raise EncodingError(
                f"Encoder expects {self.config.n_features} features, got {X.shape[1]}"
            )
        if not np.isfinite(X).all():
            raise EncodingError("Input contains NaN/inf values")
        return X

    def states(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X)
        amps = prepare_amplitudes_batch(X, self._dim)
        out = np.empty((len(X), self._dim), dtype=np.complex128)
        for i, vec in enumerate(amps):
            circuit = QuantumCircuit(self.num_qubits)
            circuit.initialize(vec, range(self.num_qubits))
            out[i] = Statevector.from_instruction(circuit).data
        return out

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        circuit = self.circuit()
        info.update({
            "feature_map": "amplitude state preparation",
            "rotation": "none (direct amplitude initialisation)",
            "circuit_depth": int(circuit.depth()),
            "circuit_size": int(circuit.size()),
            "num_parameters": 0,
            "normalisation": "L2",
            "padding": self.config.padding,
        })
        return info


def prepare_amplitudes(x: np.ndarray, dim: int) -> np.ndarray:
    """L2-normalise, zero-pad to ``dim`` (a power of two) and fix the global phase."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size > dim:
        raise EncodingError(
            f"Cannot amplitude-encode {x.size} features into {dim} amplitudes."
        )
    padded = np.zeros(dim, dtype=np.float64)
    padded[: x.size] = x
    norm = float(np.linalg.norm(padded))
    if norm < 1e-12:
        padded = np.zeros(dim, dtype=np.float64)
        padded[0] = 1.0
        return padded
    padded = padded / norm
    # Global phase convention: make the largest-magnitude amplitude positive.
    idx = int(np.argmax(np.abs(padded)))
    if padded[idx] < 0:
        padded = -padded
    return padded


def prepare_amplitudes_batch(X: np.ndarray, dim: int) -> np.ndarray:
    return np.vstack([prepare_amplitudes(row, dim) for row in np.asarray(X)])


def build_encoder(config: EncodingConfig) -> BaseEncoder:
    if config.method == "angle":
        return AngleEncoder(config)
    if config.method == "amplitude":
        return AmplitudeEncoder(config)
    raise EncodingError(f"Unknown encoding {config.method!r}")


def encoding_config_from(cfg, model_key: str, n_features: int) -> EncodingConfig:
    """Build an ``EncodingConfig`` from the YAML config for QSVM or VQC."""
    section = cfg.get_path(f"quantum.{model_key}", {}) or {}
    quantum = cfg.get("quantum", {}) or {}
    encoding = str(section.get("encoding", "angle"))
    map_name = str(section.get("feature_map", "zz"))
    reps = int(section.get("feature_map_reps", 2))
    entanglement = str(section.get("entanglement", "linear"))
    if encoding == "amplitude":
        map_name = "amplitude"
        reps = 0
        entanglement = "none"
    return EncodingConfig(
        method=encoding,
        n_features=int(n_features),
        map_name=map_name,
        reps=reps,
        entanglement=entanglement,
        name=f"{encoding}-{n_features}f",
    )
