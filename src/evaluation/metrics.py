"""Classification metrics, threshold analysis and operating-point selection."""
from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

EPS = 1e-12


def _safe(fn, default: float = float("nan")):
    try:
        value = fn()
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    prefix: str = "",
) -> dict[str, Any]:
    """All metrics reported in the paper, at a fixed operating threshold."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if len(y_true) != len(y_score):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_score)}")
    y_pred = (y_score >= threshold).astype(int)
    counts = confusion_counts(y_true, y_pred)

    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")

    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
        "accuracy": _safe(lambda: accuracy_score(y_true, y_pred)),
        "precision": _safe(lambda: precision_score(y_true, y_pred, zero_division=0)),
        "recall": _safe(lambda: recall_score(y_true, y_pred, zero_division=0)),
        "f1": _safe(lambda: f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tnr),
        "sensitivity": float(tpr),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "npv": float(npv),
        "ppv": float(ppv),
        "balanced_accuracy": _safe(lambda: balanced_accuracy_score(y_true, y_pred)),
        "mcc": _safe(lambda: matthews_corrcoef(y_true, y_pred)),
        "roc_auc": _safe(lambda: roc_auc_score(y_true, y_score)),
        "average_precision": _safe(lambda: average_precision_score(y_true, y_score)),
        "brier_score": _safe(lambda: brier_score_loss(y_true, np.clip(y_score, 0, 1))),
        "log_loss": _safe(lambda: log_loss(y_true, np.clip(y_score, EPS, 1 - EPS), labels=[0, 1])),
        **counts,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
    if prefix:
        metrics = {f"{prefix}{k}" if k not in ("confusion_matrix",) else k: v
                   for k, v in metrics.items()}
    return metrics


def threshold_analysis(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Optional[Iterable[float]] = None,
    false_negative_cost: float = 10.0,
    false_positive_cost: float = 1.0,
) -> pd.DataFrame:
    """Sweep thresholds and report the FPR/FNR/recall/precision trade-off."""
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)
    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, y_score, threshold=float(t))
        cost = false_negative_cost * m["fn"] + false_positive_cost * m["fp"]
        rows.append({
            "threshold": float(t),
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "fpr": m["fpr"],
            "fnr": m["fnr"],
            "specificity": m["specificity"],
            "tp": m["tp"],
            "fp": m["fp"],
            "tn": m["tn"],
            "fn": m["fn"],
            "expected_cost": float(cost),
            "cost_per_sample": float(cost) / max(len(y_true), 1),
        })
    return pd.DataFrame(rows)


def select_operating_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    objective: str = "f1",
    thresholds: Optional[Iterable[float]] = None,
    false_negative_cost: float = 10.0,
    false_positive_cost: float = 1.0,
) -> dict[str, Any]:
    """Choose a threshold on VALIDATION data only.

    ``objective='cost'`` minimises ``FN_COST * FN + FP_COST * FP``, i.e. it
    encodes the security preference for recall over precision.
    """
    table = threshold_analysis(
        y_true, y_score, thresholds,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    objective = objective.lower()
    if objective == "f1":
        best = table.sort_values(["f1", "recall"], ascending=[False, False]).iloc[0]
    elif objective in ("cost", "expected_cost"):
        best = table.sort_values(["expected_cost", "fnr"], ascending=[True, True]).iloc[0]
    elif objective == "youden":
        table = table.assign(youden=table["recall"] + table["specificity"] - 1)
        best = table.sort_values("youden", ascending=False).iloc[0]
    else:
        raise ValueError(f"Unknown threshold objective {objective!r}")
    return {
        "objective": objective,
        "threshold": float(best["threshold"]),
        "validation_f1": float(best["f1"]),
        "validation_recall": float(best["recall"]),
        "validation_precision": float(best["precision"]),
        "validation_fpr": float(best["fpr"]),
        "validation_fnr": float(best["fnr"]),
        "validation_cost_per_sample": float(best["cost_per_sample"]),
        "table": table,
    }


def classification_report_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[dict[int, str]] = None,
) -> pd.DataFrame:
    labels = labels or {0: "Normal", 1: "Attack"}
    rows = []
    for value, name in labels.items():
        mask = y_true == value
        rows.append({
            "class": name,
            "support": int(mask.sum()),
            "precision": _safe(lambda: precision_score(y_true == value, y_pred == value, zero_division=0)),
            "recall": _safe(lambda: recall_score(y_true == value, y_pred == value, zero_division=0)),
            "f1": _safe(lambda: f1_score(y_true == value, y_pred == value, zero_division=0)),
        })
    return pd.DataFrame(rows)


def per_category_detection(
    categories: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Detection rate for each attack category (recall within the category)."""
    rows = []
    categories = np.asarray(categories).astype(str)
    for cat in sorted(set(categories.tolist())):
        mask = categories == cat
        if not mask.any():
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        row = {
            "category": cat,
            "n_samples": int(mask.sum()),
            "attack_rate_in_group": float(yt.mean()),
            "detection_rate": float(yp[yt == 1].mean()) if (yt == 1).any() else float("nan"),
            "false_positive_rate": float(yp[yt == 0].mean()) if (yt == 0).any() else float("nan"),
        }
        if y_score is not None:
            try:
                row["roc_auc"] = float(roc_auc_score(yt, y_score[mask])) if len(np.unique(yt)) > 1 else float("nan")
            except Exception:
                row["roc_auc"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def summary_line(metrics: dict[str, Any], name: str = "") -> str:
    prefix = f"{name}: " if name else ""
    return (
        f"{prefix}acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
        f"rec={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
        f"auc={metrics['roc_auc']:.4f} fpr={metrics['fpr']:.4f} "
        f"fnr={metrics['fnr']:.4f}"
    )
