"""Filesystem + serialisation helpers."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(cfg) -> None:
    for key in (
        "data_raw", "data_processed", "data_splits", "data_metadata", "models",
        "results", "metrics", "figures", "tables", "predictions", "logs",
        "circuits", "docs", "presentation",
    ):
        value = cfg.get_path(f"paths.{key}")
        if value:
            ensure_dir(value)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    try:
        import pandas as pd  # local import to avoid a hard dependency cycle

        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        if isinstance(value, pd.Series):
            return value.tolist()
    except Exception:  # pragma: no cover
        pass
    return str(value)


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(jsonable(data), fh, indent=indent, allow_nan=False)
    return path


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=index)
    elif suffix in (".parquet", ".pq"):
        df.to_parquet(path, index=index)
    elif suffix in (".md", ".txt"):
        path.write_text(df.to_markdown(index=index), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported table format: {suffix}")
    return path


def append_manifest(path: str | Path, key: str, value: Any) -> dict:
    manifest = load_json(path, default={}) or {}
    manifest[key] = jsonable(value)
    save_json(manifest, path)
    return manifest


class Timer:
    """Context manager measuring wall-clock seconds."""

    def __init__(self) -> None:
        self.seconds: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.seconds = time.perf_counter() - self._t0

    @property
    def ms(self) -> float:
        return self.seconds * 1000.0


def timed(func, *args, **kwargs):
    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    return out, time.perf_counter() - t0


def human_duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.0f}s"
