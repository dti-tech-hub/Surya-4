"""Structured logging for experiments."""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_CONFIGURED: bool = False
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    console: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Configure the root logger once per process."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED and not force:
        if log_file:
            _attach_file_handler(root, log_file)
        return root

    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        root.addHandler(stream)

    if log_file:
        _attach_file_handler(root, log_file, formatter)

    # Quieten noisy third-party libraries.
    for noisy in ("matplotlib", "qiskit", "urllib3", "PIL", "shap"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return root


def _attach_file_handler(
    logger: logging.Logger,
    log_file: str | Path,
    formatter: Optional[logging.Formatter] = None,
) -> None:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    target = str(log_file.resolve())
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == target:
            return
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(formatter or logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)


def get_logger(name: str = "qeids") -> logging.Logger:
    return logging.getLogger(name)


class ExperimentLogger:
    """Append-only JSONL record of experiment events.

    Every record carries a UTC timestamp, experiment name, model and stage so
    that a run can be reconstructed after the fact.
    """

    def __init__(
        self,
        path: str | Path,
        experiment: str = "qeids",
        model: str = "-",
        echo: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.experiment = experiment
        self.model = model
        self.echo = echo
        self.logger = get_logger(f"qeids.{experiment}")
        self._start = time.perf_counter()

    def log(self, stage: str, status: str = "info", **fields: Any) -> dict:
        record = {
            "timestamp": utc_now(),
            "experiment": self.experiment,
            "model": self.model,
            "stage": stage,
            "status": status,
            "elapsed_s": round(time.perf_counter() - self._start, 6),
            **{k: _jsonable(v) for k, v in fields.items()},
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        if self.echo:
            msg = f"[{self.experiment}/{self.model}] {stage}: {status}"
            extra = {k: v for k, v in fields.items() if k not in ("config",)}
            if extra:
                msg += " | " + ", ".join(f"{k}={v}" for k, v in extra.items())
            self.logger.info(msg)
        return record

    def error(self, stage: str, error: BaseException, **fields: Any) -> dict:
        return self.log(
            stage,
            status="error",
            error=f"{type(error).__name__}: {error}",
            **fields,
        )

    def close(self, **fields: Any) -> None:
        self.log("finish", status="success", **fields)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:  # pragma: no cover
        pass
    return str(value)
