"""Reproducibility helpers: global seeds and environment capture."""
from __future__ import annotations

import json
import os
import platform
import random
import socket
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

_TRACKED_PACKAGES = [
    "numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "seaborn",
    "torch", "torchvision", "qiskit", "qiskit-aer", "qiskit-machine-learning",
    "shap", "streamlit", "pyarrow", "joblib", "pytest", "PyYAML", "tabulate",
]


def set_global_seed(seed: int = 42) -> None:
    """Seed every RNG we rely on and pin deterministic behaviour where possible."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(False)
    except Exception:  # pragma: no cover - torch optional for some modules
        pass


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover
        return None


def _cpu_name() -> str:
    try:
        if sys.platform.startswith("win"):
            out = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            if len(lines) > 1:
                return lines[1]
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            if out:
                return out
    except Exception:
        pass
    return platform.processor() or "unknown"


def _ram_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        return None


def _gpu_info() -> dict:
    info: dict[str, Any] = {"available": False, "devices": []}
    try:
        import torch

        if torch.cuda.is_available():
            info["available"] = True
            info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["cuda_version"] = torch.version.cuda
            info["torch_cuda_capable"] = True
    except Exception:
        pass
    return info


def capture_environment() -> dict:
    """Collect everything needed for an academic reproducibility statement."""
    packages = {name: _pkg_version(name) for name in _TRACKED_PACKAGES}
    env = {
        "timestamp_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(timespec="seconds"),
        "python": {
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": _cpu_name(),
            "hostname": socket.gethostname(),
        },
        "cpu_count_logical": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "gpu": _gpu_info(),
        "packages": packages,
        "quantum_simulation": {
            "backend": "qiskit.quantum_info.Statevector (analytic)",
            "note": (
                "No quantum hardware was used. All quantum results are "
                "simulation-based and shot-noise free unless finite_shot_"
                "evaluation is enabled."
            ),
        },
        "environment_variables": {
            k: os.environ.get(k)
            for k in ("PYTHONHASHSEED", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
            if k in os.environ
        },
    }
    return env


def save_environment(path: str | Path) -> dict:
    from src.utils.io import save_json

    env = capture_environment()
    save_json(env, path)
    return env


def environment_summary(env: dict | None = None) -> str:
    env = env or capture_environment()
    pkg = env.get("packages", {})
    lines = [
        f"Python {'.'.join(map(str, env['python']['version_info']))} on "
        f"{env['platform']['system']} {env['platform']['release']} "
        f"({env['platform']['machine']})",
        f"CPU: {env['platform']['processor']} | logical cores: {env['cpu_count_logical']} "
        f"| RAM: {env.get('ram_gb')} GB",
        f"GPU: {env['gpu'].get('devices') or 'none (CPU only)'}",
        "Key packages: " + ", ".join(
            f"{k}=={v}" for k, v in pkg.items() if v
        ),
    ]
    return "\n".join(lines)
