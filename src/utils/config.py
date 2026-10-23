"""Configuration loading with dotted access, YAML overrides and path resolution."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

# Repository root = two levels up from src/utils/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    """Dictionary supporting attribute access and dotted lookups."""

    def __init__(self, data: Optional[Mapping[str, Any]] = None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = _wrap(value)

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = _wrap(value)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Config = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], Mapping):
                node[part] = Config()
            node = node[part]  # type: ignore[assignment]
        node[parts[-1]] = _wrap(value)

    def to_dict(self) -> dict:
        return _unwrap(self)

    def copy(self) -> "Config":  # type: ignore[override]
        return Config(copy.deepcopy(self.to_dict()))


def _wrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, Mapping):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, Config):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, Mapping):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_overrides(pairs: list[str]) -> dict:
    """Parse ``key.path=value`` strings into a nested dict (YAML value parsing)."""
    overrides: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Invalid override '{pair}'. Expected key=value.")
        key, raw = pair.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        _set_nested(overrides, key.strip().split("."), value)
    return overrides


def _set_nested(node: dict, parts: list[str], value: Any) -> None:
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def resolve_paths(cfg: Config, root: Optional[Path] = None) -> Config:
    """Convert every ``paths.*`` entry into an absolute path string."""
    root = Path(root) if root is not None else PROJECT_ROOT
    paths = cfg.get("paths", {})
    for key, value in list(paths.items()):
        p = Path(str(value))
        if not p.is_absolute():
            p = root / p
        paths[key] = str(p)
    cfg["paths"] = paths
    return cfg


def load_config(
    path: Optional[str | Path] = None,
    overrides: Optional[Mapping[str, Any] | list[str]] = None,
    mode: Optional[str] = None,
    root: Optional[Path] = None,
) -> Config:
    """Load ``config/config.yaml``, apply overrides and resolve paths.

    Parameters
    ----------
    path:
        Optional alternative config file.
    overrides:
        Either a mapping or a list of ``key=value`` strings.
    mode:
        ``quick`` / ``full``.  Stored on the config and used to pick the
        experiment profile.
    root:
        Repository root (defaults to the project root).
    """
    root = Path(root) if root is not None else PROJECT_ROOT
    cfg_path = Path(path) if path is not None else root / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {cfg_path}. "
            "Run from the project root or pass --config."
        )
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if isinstance(overrides, list):
        overrides = parse_overrides(overrides)
    if overrides:
        raw = deep_merge(raw, dict(overrides))

    cfg = Config(raw)
    if mode:
        cfg["experiment"]["mode"] = mode
    cfg["project_root"] = str(root)
    cfg = resolve_paths(cfg, root)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    exp = cfg.get("experiment", {})
    mode = exp.get("mode", "quick")
    if mode not in ("quick", "full"):
        raise ValueError(f"experiment.mode must be 'quick' or 'full', got {mode!r}")
    if mode not in exp:
        raise ValueError(f"Missing experiment profile for mode={mode!r}")

    split = cfg.get("split", {})
    total = sum(
        float(split.get(k, 0.0))
        for k in ("train_size", "validation_size", "test_size")
    )
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    n_feat = int(cfg.get("feature_reduction", {}).get("quantum_n_features", 8))
    max_q = int(cfg.get("feature_reduction", {}).get("max_quantum_qubits", 8))
    if n_feat > max_q:
        raise ValueError(
            f"quantum_n_features={n_feat} exceeds max_quantum_qubits={max_q}. "
            "Increase max_quantum_qubits explicitly if you accept the cost."
        )


def profile(cfg: Config) -> Config:
    """Return the active quick/full experiment profile."""
    mode = cfg.get_path("experiment.mode", "quick")
    prof = cfg.get_path(f"experiment.{mode}")
    if prof is None:
        raise KeyError(f"No profile for experiment mode {mode!r}")
    return prof


def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False, allow_unicode=True)


def load_experiment_matrix(root: Optional[Path] = None) -> dict:
    root = Path(root) if root is not None else PROJECT_ROOT
    path = root / "config" / "experiments.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")
