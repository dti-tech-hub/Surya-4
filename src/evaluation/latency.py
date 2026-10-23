"""Latency and throughput measurement."""
from __future__ import annotations

import time
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from src.pipeline.training import ModelBundle


def _time_call(fn, repeats: int = 5) -> float:
    best = float("inf")
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def measure_latency(
    prep,
    bundles: Sequence[ModelBundle],
    raw_sample: pd.DataFrame,
    X_reduced_sample: np.ndarray,
    y_sample: np.ndarray,
    cfg,
) -> pd.DataFrame:
    """Stage-wise latency on identical inputs.

    ``raw_sample`` is a dataframe of raw rows (already cleaned); it is pushed
    through the saved preprocessing chain so the measurement includes the
    transformer overhead a deployed system would pay.
    """
    repeats = int(cfg.get_path("latency.repeats", 5))
    batch_size = int(cfg.get_path("latency.batch_size", 256))

    X_batch = np.asarray(X_reduced_sample, dtype=np.float32)
    n = len(X_batch)
    if n == 0:
        raise ValueError("Latency measurement received an empty sample")

    # Stage 1: raw -> engineered matrix (per batch, then normalised per sample).
    X_raw = prep.preprocessor.transform(raw_sample)
    t_pre = _time_call(lambda: prep.preprocessor.transform(raw_sample), repeats)
    t_red = _time_call(
        lambda: prep.scaler.transform(prep.reducer.transform(
            prep.preprocessor.transform(raw_sample)
        )),
        repeats,
    )
    t_red_only = _time_call(
        lambda: prep.scaler.transform(prep.reducer.transform(X_raw)), repeats
    )
    pre_ms = t_pre / n * 1000.0
    red_ms = t_red_only / n * 1000.0

    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        model = bundle.model
        kind = bundle.kind

        # Single-sample latency (includes per-call overhead).
        def single():
            bundle.predict_proba(X_batch[:1])

        t_single = _time_call(single, repeats)

        # Batch throughput at the configured batch size.
        idx = np.arange(min(batch_size, n))
        Xb = X_batch[idx]

        def batch():
            bundle.predict_proba(Xb)

        t_batch = _time_call(batch, repeats)

        row = {
            "model": kind,
            "encoding": bundle.encoding,
            "n_samples": int(n),
            "preprocessing_ms_per_sample": pre_ms,
            "feature_reduction_ms_per_sample": red_ms,
            "inference_ms_per_sample": t_single * 1000.0,
            "batch_inference_ms_per_sample": t_batch / max(len(Xb), 1) * 1000.0,
            "throughput_samples_per_s": len(Xb) / max(t_batch, 1e-12),
            "end_to_end_ms_per_sample": pre_ms + red_ms + t_single * 1000.0,
            "batch_size": int(len(Xb)),
            "repeats": repeats,
        }

        # Quantum-specific components, if recorded by the model.
        timings = getattr(model, "timings_", {}) or {}
        if kind == "qsvm":
            row["quantum_state_prep_ms_per_sample"] = (
                timings.get("test_state_preparation_time_s", float("nan")) / max(n, 1) * 1000.0
            )
            row["quantum_kernel_ms_per_sample"] = (
                timings.get("test_kernel_time_s", float("nan")) / max(n, 1) * 1000.0
            )
        if kind == "vqc":
            row["quantum_inference_ms_per_sample"] = (
                timings.get("inference_time_per_sample_s", float("nan")) * 1000.0
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    base = df.loc[df["model"] == "cnn_lstm", "inference_ms_per_sample"]
    if not base.empty:
        df["slowdown_vs_classical"] = df["inference_ms_per_sample"] / float(base.iloc[0])
    return df


def measure_training_latency(bundles: Sequence[ModelBundle]) -> pd.DataFrame:
    rows = []
    for bundle in bundles:
        row = {
            "model": bundle.kind,
            "encoding": bundle.encoding,
            "train_size": int(bundle.info.get("train_size", 0)) or None,
            "training_time_s": float(bundle.timings.get("training_time_s", float("nan"))),
            "resources": bundle.resources,
        }
        for key, value in bundle.timings.items():
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)
