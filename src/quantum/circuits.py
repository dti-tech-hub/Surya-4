"""Quantum circuit utilities: fidelity kernels, resources, drawing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

TWO_QUBIT_GATES = {
    "cx", "cz", "cy", "ch", "swap", "ecr", "rzz", "rxx", "ryy", "crx", "cry", "crz",
    "cu", "cu1", "cu3", "cp", "cswap", "ccx",
}


class KernelValidationError(RuntimeError):
    pass


def fidelity_kernel(states_a: np.ndarray, states_b: Optional[np.ndarray] = None) -> np.ndarray:
    """Exact fidelity quantum kernel ``K_ij = |<psi_i|phi_j>|^2``.

    Computed from simulated statevectors, which is mathematically identical to
    the circuit-based ComputeUncompute estimate in the infinite-shot limit.
    """
    A = np.asarray(states_a, dtype=np.complex128)
    B = A if states_b is None else np.asarray(states_b, dtype=np.complex128)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("state matrices must be 2-D")
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"dimension mismatch: {A.shape} vs {B.shape}")
    overlaps = A.conj() @ B.T
    return np.abs(overlaps) ** 2


def statevectors_for_circuit(circuit: QuantumCircuit, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    out = np.empty((len(X), 2 ** circuit.num_qubits), dtype=np.complex128)
    for i, row in enumerate(X):
        out[i] = Statevector.from_instruction(circuit.assign_parameters(row)).data
    return out


def circuit_resources(circuit: QuantumCircuit) -> dict[str, Any]:
    ops = circuit.count_ops()
    gate_counts = {str(k): int(v) for k, v in ops.items()}
    two_qubit = sum(v for k, v in gate_counts.items() if k in TWO_QUBIT_GATES)
    info = {
        "num_qubits": int(circuit.num_qubits),
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "num_parameters": int(circuit.num_parameters),
        "gate_counts": gate_counts,
        "num_two_qubit_gates": int(two_qubit),
        "num_gates_total": int(sum(gate_counts.values())),
    }
    # Amplitude state preparation appears as a single 'initialize' instruction
    # at the abstract level; report the decomposed cost too so the resource
    # table is not misleading.
    try:
        decomposed = circuit.decompose(reps=4)
        dec_ops = decomposed.count_ops()
        dec_gates = {str(k): int(v) for k, v in dec_ops.items()}
        info["decomposed"] = {
            "depth": int(decomposed.depth()),
            "size": int(decomposed.size()),
            "num_gates_total": int(sum(dec_gates.values())),
            "num_two_qubit_gates": int(
                sum(v for k, v in dec_gates.items() if k in TWO_QUBIT_GATES)
            ),
            "gate_counts": dec_gates,
        }
    except Exception:  # pragma: no cover - defensive
        info["decomposed"] = None
    return info


def draw_circuit(
    circuit: QuantumCircuit,
    path: Optional[str | Path] = None,
    style: Optional[str] = None,
    fold: int = 60,
) -> str:
    """Save a PNG (if possible) and return a text rendering."""
    text = str(circuit.draw(output="text", fold=fold))
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig = circuit.draw(output="mpl", style=style, fold=fold)
            fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        except Exception:
            # Fall back to a text rendering so the artifact still exists.
            path.with_suffix(".txt").write_text(text, encoding="utf-8")
    return text


def save_circuit_artifacts(
    circuit: QuantumCircuit,
    out_dir: str | Path,
    tag: str,
    sample: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bound = circuit
    if sample is not None and circuit.num_parameters:
        try:
            bound = circuit.assign_parameters(np.asarray(sample, dtype=float))
        except Exception:
            bound = circuit
    text = draw_circuit(bound, out_dir / f"{tag}_circuit.png")
    (out_dir / f"{tag}_circuit.txt").write_text(text, encoding="utf-8")
    resources = circuit_resources(circuit)
    with open(out_dir / f"{tag}_resources.json", "w", encoding="utf-8") as fh:
        json.dump(resources, fh, indent=2)
    return resources


def validate_kernel_against_qiskit(
    circuit: QuantumCircuit,
    X: np.ndarray,
    tolerance: Optional[float] = None,
    shots: int = 100_000,
) -> dict[str, Any]:
    """Cross-check our exact kernel against Qiskit's ComputeUncompute estimator.

    Qiskit's ``StatevectorSampler`` is a *finite-shot* sampler (default 1024
    shots), so the circuit-based estimate carries binomial sampling noise.  The
    comparison tolerance is therefore derived from shot noise, not set
    arbitrarily:

        sigma_max ~ sqrt(0.25 / shots),  tolerance = 4 * sigma_max

    This validates that the fast statevector path implements the same
    mathematical object as the circuit-based ``FidelityQuantumKernel``.
    """
    X = np.asarray(X, dtype=np.float64)
    if len(X) > 32:
        X = X[:32]
    shots = int(shots)
    sigma = float(np.sqrt(0.25 / shots))
    tol = float(tolerance if tolerance is not None else 4.0 * sigma)
    try:
        from qiskit.primitives import StatevectorSampler
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_machine_learning.state_fidelities import ComputeUncompute

        sampler = StatevectorSampler(default_shots=shots, seed=42)
        fidelity = ComputeUncompute(sampler=sampler)
        fqk = FidelityQuantumKernel(feature_map=circuit, fidelity=fidelity, enforce_psd=False)
        K_qiskit = np.asarray(fqk.evaluate(X), dtype=float)
    except Exception as exc:  # pragma: no cover - version dependent
        return {
            "performed": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "passed": False,
        }

    states = statevectors_for_circuit(circuit, X)
    K_exact = fidelity_kernel(states)
    max_diff = float(np.max(np.abs(K_qiskit - K_exact)))
    diag = np.diag(K_exact)
    return {
        "performed": True,
        "n_samples": int(len(X)),
        "max_abs_difference": max_diff,
        "shot_noise_sigma": sigma,
        "tolerance": tol,
        "passed": bool(max_diff <= tol),
        "validation_shots": shots,
        "exact_kernel_diagonal_max_deviation": float(np.max(np.abs(diag - 1.0))),
        "method": (
            "FidelityQuantumKernel + ComputeUncompute(StatevectorSampler, "
            f"{shots} shots) vs exact statevector overlap"
        ),
        "note": (
            "The circuit-based reference is finite-shot; the production kernel "
            "used for all reported results is the exact statevector overlap."
        ),
    }


def kernel_diagnostics(K: np.ndarray) -> dict[str, Any]:
    K = np.asarray(K)
    diag = np.diag(K) if K.shape[0] == K.shape[1] else np.array([])
    return {
        "shape": list(K.shape),
        "min": float(np.min(K)),
        "max": float(np.max(K)),
        "mean": float(np.mean(K)),
        "diagonal_mean": float(np.mean(diag)) if diag.size else None,
        "diagonal_min": float(np.min(diag)) if diag.size else None,
        "off_diagonal_mean": float(
            (K.sum() - diag.sum()) / max(K.size - diag.size, 1)
        ) if diag.size else float(np.mean(K)),
    }
