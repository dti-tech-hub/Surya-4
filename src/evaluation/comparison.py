"""Master comparison tables."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

MODEL_ORDER = ["cnn_lstm", "qsvm", "vqc"]
MODEL_DISPLAY = {"cnn_lstm": "CNN+LSTM", "qsvm": "QSVM", "vqc": "VQC"}

COLUMNS = [
    "model", "model_label", "encoding", "feature_method", "n_features",
    "accuracy", "precision", "recall", "f1", "roc_auc", "fpr", "fnr",
    "training_time_s", "inference_time_s", "inference_ms_per_sample",
    "qubits", "circuit_depth", "num_parameters", "notes",
]


def _note_for(kind: str, encoding: str) -> str:
    if kind == "cnn_lstm":
        return (
            "Classical baseline; pseudo-temporal reshaping of the reduced "
            "feature vector (not real packet timing)."
        )
    if kind == "qsvm":
        return (
            f"Fidelity quantum kernel ({encoding} encoding), exact statevector "
            "simulation; SVM trained on the precomputed kernel."
        )
    return (
        f"Variational circuit ({encoding} encoding), RealAmplitudes ansatz, "
        "exact statevector probabilities."
    )


def build_master_table(
    metrics_by_model: Mapping[str, dict],
    resources_by_model: Optional[Mapping[str, dict]] = None,
    timings_by_model: Optional[Mapping[str, dict]] = None,
) -> pd.DataFrame:
    resources_by_model = resources_by_model or {}
    timings_by_model = timings_by_model or {}
    rows = []
    for kind in MODEL_ORDER:
        if kind not in metrics_by_model:
            continue
        m = metrics_by_model[kind]
        res = resources_by_model.get(kind, {}) or {}
        timings = timings_by_model.get(kind, {}) or {}
        rows.append({
            "model": kind,
            "model_label": MODEL_DISPLAY.get(kind, kind),
            "encoding": m.get("encoding", "none"),
            "feature_method": m.get("feature_method", ""),
            "n_features": m.get("n_features", None),
            "accuracy": m.get("accuracy"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "f1": m.get("f1"),
            "roc_auc": m.get("roc_auc"),
            "fpr": m.get("fpr"),
            "fnr": m.get("fnr"),
            "training_time_s": timings.get("training_time_s", res.get("training_time_s")),
            "inference_time_s": m.get("inference_time_s"),
            "inference_ms_per_sample": m.get("inference_ms_per_sample"),
            "qubits": res.get("num_qubits", 0),
            "circuit_depth": res.get("depth", 0),
            "num_parameters": res.get("num_parameters", res.get("num_ansatz_parameters", 0)),
            "notes": _note_for(kind, m.get("encoding", "none")),
        })
    df = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[COLUMNS]


def format_table_for_markdown(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, fmt in {
        "accuracy": "{:.4f}", "precision": "{:.4f}", "recall": "{:.4f}",
        "f1": "{:.4f}", "roc_auc": "{:.4f}", "fpr": "{:.4f}", "fnr": "{:.4f}",
    }.items():
        if col in out.columns:
            out[col] = out[col].map(lambda v: fmt.format(v) if pd.notna(v) else "n/a")
    for col, fmt in {
        "training_time_s": "{:.2f}", "inference_time_s": "{:.4f}",
        "inference_ms_per_sample": "{:.4f}",
    }.items():
        if col in out.columns:
            out[col] = out[col].map(lambda v: fmt.format(v) if pd.notna(v) else "n/a")
    return out


def save_master_table(df: pd.DataFrame, tables_dir: str | Path) -> dict[str, str]:
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "master_comparison.csv"
    md_path = tables_dir / "master_comparison.md"
    df.to_csv(csv_path, index=False)
    formatted = format_table_for_markdown(df)
    md = formatted.to_markdown(index=False)
    md_path.write_text("# Master comparison — common test set\n\n" + md + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "markdown": str(md_path)}


def metric_table(df: pd.DataFrame, metrics: list[str] | None = None,
                 index: str = "model_label") -> pd.DataFrame:
    metrics = metrics or ["accuracy", "precision", "recall", "f1", "roc_auc", "fpr", "fnr"]
    cols = [c for c in metrics if c in df.columns]
    out = df[[index] + cols].copy()
    return out.round(4)
