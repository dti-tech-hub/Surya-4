"""Publication-quality figures for the QE-IDS project."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import auc, precision_recall_curve, roc_curve

PALETTE = {
    "cnn_lstm": "#1f4e79",
    "qsvm": "#c0392b",
    "vqc": "#2e8b57",
    "pca": "#1f4e79",
    "mutual_info": "#d68910",
    "angle": "#1f4e79",
    "amplitude": "#c0392b",
}
MODEL_LABELS = {"cnn_lstm": "CNN+LSTM", "qsvm": "QSVM", "vqc": "VQC"}
MODEL_COLORS = {"cnn_lstm": "#1f4e79", "qsvm": "#c0392b", "vqc": "#2e8b57"}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
    })


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Data / features
# ---------------------------------------------------------------------------
def plot_class_distribution(train: pd.DataFrame, validation: pd.DataFrame,
                            test: pd.DataFrame, path: str | Path) -> Path:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    splits = [("Train", train), ("Validation", validation), ("Test", test)]
    binary = pd.DataFrame({
        name: [int((df["is_attack"] == 0).sum()), int((df["is_attack"] == 1).sum())]
        for name, df in splits
    }, index=["Normal", "Attack"]).T
    binary.plot(kind="bar", stacked=True, ax=axes[0],
                color=["#4c9f70", "#c0392b"], width=0.65)
    axes[0].set_title("Binary class distribution per split")
    axes[0].set_ylabel("Samples")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=0)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%d", fontsize=8, label_type="center")
    axes[0].legend(title="Class")

    cats = pd.concat([
        df.assign(split=name) for name, df in splits
    ], ignore_index=True)
    order = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    order = [c for c in order if c in set(cats["attack_category"])]
    counts = (
        cats.groupby(["split", "attack_category"]).size()
        .unstack(fill_value=0)
        .reindex(columns=order, fill_value=0)
        .reindex(["Train", "Validation", "Test"])
    )
    counts.plot(kind="bar", ax=axes[1], colormap="viridis", width=0.75)
    axes[1].set_title("Attack-category distribution per split")
    axes[1].set_ylabel("Samples")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="Category", ncol=2, fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_pca_explained_variance(explained: Sequence[float], cumulative: Sequence[float],
                                path: str | Path, n_features: int,
                                label: str = "PCA") -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    idx = np.arange(1, len(explained) + 1)
    ax.bar(idx, np.asarray(explained) * 100, color="#1f4e79", alpha=0.85,
           label="Individual")
    ax2 = ax.twinx()
    ax2.plot(idx, np.asarray(cumulative) * 100, "o-", color="#c0392b",
             linewidth=2, label="Cumulative")
    ax2.set_ylabel("Cumulative explained variance (%)", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax2.set_ylim(0, 105)
    ax2.grid(False)
    ax.set_xlabel(f"{label} component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title(
        f"{label} explained variance — {n_features} components retain "
        f"{np.asarray(cumulative)[-1] * 100:.1f}%"
    )
    ax.set_xticks(idx)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    fig.tight_layout()
    return _save(fig, path)


def plot_feature_importance(names: Sequence[str], values: Sequence[float],
                            path: str | Path, title: str = "Feature importance",
                            top_k: int = 20, xlabel: str = "Importance") -> Path:
    _style()
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(9, max(3.6, 0.32 * len(order) + 1.6)))
    colors = ["#c0392b" if v < 0 else "#1f4e79" for v in values[order]]
    ax.barh(range(len(order)), values[order][::-1], color=colors[::-1], alpha=0.9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([str(names[i]) for i in order][::-1], fontsize=8.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def plot_pca_component_heatmap(loadings: np.ndarray, feature_names: Sequence[str],
                               path: str | Path, top_k: int = 15) -> Path:
    _style()
    loadings = np.asarray(loadings)
    n_comp = loadings.shape[0]
    magnitude = np.abs(loadings).max(axis=0)
    top = np.argsort(magnitude)[::-1][:top_k]
    data = loadings[:, top]
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(top) + 3), max(3.5, 0.4 * n_comp + 2)))
    sns.heatmap(
        data, cmap="RdBu_r", center=0.0, annot=True, fmt=".2f",
        xticklabels=[str(feature_names[i]) for i in top],
        yticklabels=[f"PC{i + 1}" for i in range(n_comp)],
        cbar_kws={"label": "Loading"}, ax=ax, annot_kws={"size": 7},
    )
    ax.set_title("PCA component loadings on the strongest original features")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right", fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def plot_training_history(history: dict, path: str | Path) -> Path:
    _style()
    epochs = history.get("epochs", [])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    axes[0].plot(epochs, history.get("train_loss", []), label="Train", color="#1f4e79")
    axes[0].plot(epochs, history.get("val_loss", []), label="Validation", color="#c0392b")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title("CNN+LSTM loss curves")
    axes[0].legend()
    if history.get("best_epoch"):
        axes[0].axvline(history["best_epoch"], ls=":", color="grey", alpha=0.8,
                        label=f"best epoch {history['best_epoch']}")

    axes[1].plot(epochs, history.get("train_accuracy", []), label="Train accuracy", color="#1f4e79")
    axes[1].plot(epochs, history.get("val_accuracy", []), label="Val accuracy", color="#c0392b")
    axes[1].plot(epochs, history.get("val_f1", []), label="Val F1", color="#2e8b57", ls="--")
    if history.get("val_roc_auc"):
        axes[1].plot(epochs, history["val_roc_auc"], label="Val ROC-AUC",
                     color="#d68910", ls=":")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("CNN+LSTM validation metrics")
    axes[1].legend(fontsize=8.5)
    fig.tight_layout()
    return _save(fig, path)


def plot_vqc_convergence(history: list[dict], path: str | Path,
                         optimizer: str = "COBYLA") -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    if not history:
        ax.text(0.5, 0.5, "No convergence history recorded", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    evals = [h["evaluation"] for h in history]
    losses = [h["loss"] for h in history]
    ax.plot(evals, losses, color="#2e8b57", linewidth=1.6, label="Objective")
    best_so_far = np.minimum.accumulate(losses)
    ax.plot(evals, best_so_far, color="#c0392b", linewidth=2, ls="--", label="Best so far")
    ax.set_xlabel("Objective evaluation")
    ax.set_ylabel("Binary cross-entropy")
    ax.set_title(f"VQC convergence ({optimizer})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Evaluation curves
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, path: str | Path, title: str,
                          normalize: bool = False) -> Path:
    _style()
    raw = np.asarray(cm)
    if normalize:
        display = raw.astype(float)
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sums, out=np.zeros_like(display), where=row_sums > 0)
        fmt = ".3f"
    else:
        display = raw.astype(int)
        fmt = "d"
    fig, ax = plt.subplots(figsize=(4.9, 4.2))
    sns.heatmap(display, annot=True, fmt=fmt, cmap="Blues", cbar=False,
                xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"],
                ax=ax, annot_kws={"size": 12})
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, path)


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, path: str | Path,
                   label: str = "", color: str = "#1f4e79") -> Path:
    _style()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"{label} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    fig.tight_layout()
    return _save(fig, path)


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, path: str | Path,
                  label: str = "", color: str = "#c0392b") -> Path:
    _style()
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = auc(recall, precision)
    baseline = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(recall, precision, color=color, linewidth=2.2, label=f"{label} (AP = {ap:.4f})")
    ax.axhline(baseline, ls="--", color="grey", linewidth=1,
               label=f"Prevalence = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall curve")
    ax.legend(loc="lower left")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    fig.tight_layout()
    return _save(fig, path)


def plot_roc_comparison(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: str | Path,
                        title: str = "ROC comparison") -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    for name, (y_true, y_score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        color = MODEL_COLORS.get(name, None)
        ax.plot(fpr, tpr, linewidth=2.1, color=color,
                label=f"{MODEL_LABELS.get(name, name)} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    fig.tight_layout()
    return _save(fig, path)


def plot_pr_comparison(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: str | Path,
                       title: str = "Precision–recall comparison") -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    for name, (y_true, y_score) in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = auc(recall, precision)
        ax.plot(recall, precision, linewidth=2.1,
                color=MODEL_COLORS.get(name, None),
                label=f"{MODEL_LABELS.get(name, name)} (AP = {ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    fig.tight_layout()
    return _save(fig, path)


def plot_metric_comparison(df: pd.DataFrame, metrics: Sequence[str], path: str | Path,
                           model_col: str = "model") -> Path:
    _style()
    metrics = [m for m in metrics if m in df.columns]
    models = df[model_col].tolist()
    x = np.arange(len(models))
    width = 0.8 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(max(9, 1.4 * len(models) + 5), 4.8))
    cmap = plt.get_cmap("viridis")
    for i, metric in enumerate(metrics):
        values = df[metric].astype(float).fillna(0).values
        ax.bar(x + i * width - 0.4 + width / 2, values, width,
               label=metric.replace("_", " ").upper(), color=cmap(i / max(len(metrics) - 1, 1)))
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison on the common test set")
    ax.legend(ncol=min(len(metrics), 7), fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion_matrix_grid(confusions: dict[str, np.ndarray], path: str | Path) -> Path:
    _style()
    n = len(confusions)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.9))
    if n == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, confusions.items()):
        cm = np.asarray(cm).astype(int)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"],
                    ax=ax, annot_kws={"size": 11})
        ax.set_title(MODEL_LABELS.get(name, name))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle("Confusion matrices (common test set)", y=1.03, fontsize=13)
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Systems analysis
# ---------------------------------------------------------------------------
def plot_latency_comparison(df: pd.DataFrame, path: str | Path) -> Path:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    models = df["model"].tolist()
    labels = [MODEL_LABELS.get(m, m) for m in models]
    colors = [MODEL_COLORS.get(m, "#555") for m in models]

    if "inference_ms_per_sample" in df.columns:
        values = df["inference_ms_per_sample"].astype(float).values
        axes[0].bar(labels, values, color=colors, alpha=0.9)
        axes[0].set_ylabel("ms per sample (log scale)")
        axes[0].set_yscale("log")
        axes[0].set_title("Single-sample inference latency")
        for i, v in enumerate(values):
            axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    if "throughput_samples_per_s" in df.columns:
        values = df["throughput_samples_per_s"].astype(float).values
        axes[1].bar(labels, values, color=colors, alpha=0.9)
        axes[1].set_yscale("log")
        axes[1].set_ylabel("samples / second (log scale)")
        axes[1].set_title("Batch inference throughput")
        for i, v in enumerate(values):
            axes[1].text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_quantum_resources(df: pd.DataFrame, path: str | Path) -> Path:
    _style()
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No quantum results available", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    x = np.arange(len(df))
    names = df["model_label"].tolist() if "model_label" in df else df["model"].tolist()
    for ax, col, title, ylabel in [
        (axes[0], "qubits", "Qubit count", "Qubits"),
        (axes[1], "circuit_depth", "Circuit depth", "Depth"),
        (axes[2], "training_time_s", "Quantum training time", "Seconds"),
    ]:
        if col not in df.columns:
            ax.axis("off")
            continue
        values = pd.to_numeric(df[col], errors="coerce").fillna(0).values
        ax.bar(x, values, color=[MODEL_COLORS.get(m, "#555") for m in df.get("model", [])])
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if col == "training_time_s":
            ax.set_yscale("log")
    fig.suptitle("Quantum resource summary (simulation)", y=1.02)
    fig.tight_layout()
    return _save(fig, path)


def plot_encoding_comparison(df: pd.DataFrame, path: str | Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    if df.empty:
        ax.text(0.5, 0.5, "No encoding experiment results", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    df = df.copy()
    df["label"] = df["model"].str.upper() + "\n" + df["encoding"]
    x = np.arange(len(df))
    width = 0.38
    ax.bar(x - width / 2, df["f1"].astype(float), width, label="F1", color="#1f4e79")
    ax.bar(x + width / 2, df["roc_auc"].astype(float), width, label="ROC-AUC", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Angle vs amplitude encoding (same split, same evaluation)")
    ax.legend()
    for i, row in df.reset_index(drop=True).iterrows():
        ax.text(i - width / 2, float(row["f1"]) + 0.015, f"{float(row['f1']):.3f}",
                ha="center", fontsize=7.5)
        ax.text(i + width / 2, float(row["roc_auc"]) + 0.015, f"{float(row['roc_auc']):.3f}",
                ha="center", fontsize=7.5)
    fig.tight_layout()
    return _save(fig, path)


def plot_zero_day_results(df: pd.DataFrame, path: str | Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    if df.empty:
        ax.text(0.5, 0.5, "No unseen-attack results", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    x = np.arange(len(df))
    width = 0.27
    for i, (col, color, label) in enumerate([
        ("recall", "#c0392b", "Recall (unseen attack)"),
        ("precision", "#1f4e79", "Precision"),
        ("f1", "#2e8b57", "F1"),
    ]):
        if col in df.columns:
            ax.bar(x + (i - 1) * width, df[col].astype(float), width,
                   label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in df["model"]], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Unseen attack-category generalisation (R2L + U2R held out)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_drift_curve(df: pd.DataFrame, path: str | Path, x_col: str = "severity",
                     title: str = "Performance under controlled distribution shift") -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    if df.empty:
        ax.text(0.5, 0.5, "No drift results", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    for metric, color in [("f1", "#1f4e79"), ("roc_auc", "#c0392b"), ("recall", "#2e8b57")]:
        if metric in df.columns:
            ax.plot(df[x_col], df[metric], "o-", label=metric.upper(), color=color, linewidth=1.8)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_threshold_analysis(table: pd.DataFrame, path: str | Path,
                            selected: Optional[float] = None,
                            title: str = "Threshold trade-off") -> Path:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    axes[0].plot(table["threshold"], table["recall"], "o-", label="Recall", color="#c0392b")
    axes[0].plot(table["threshold"], table["precision"], "s-", label="Precision", color="#1f4e79")
    axes[0].plot(table["threshold"], table["f1"], "^-", label="F1", color="#2e8b57")
    axes[0].plot(table["threshold"], table["fpr"], "v--", label="FPR", color="#d68910")
    if selected is not None:
        axes[0].axvline(selected, ls=":", color="black", label=f"selected = {selected:.2f}")
    axes[0].set_xlabel("Decision threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title(title)
    axes[0].legend(fontsize=8)
    axes[1].plot(table["threshold"], table["fnr"], label="FNR", color="#c0392b")
    axes[1].plot(table["threshold"], table["fpr"], label="FPR", color="#1f4e79")
    axes[1].plot(table["threshold"], table["cost_per_sample"], "--",
                 label="Cost / sample", color="#6c3483")
    if selected is not None:
        axes[1].axvline(selected, ls=":", color="black")
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_title("Security trade-off (FN cost = 10x FP cost)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_cv_boxplot(cv_df: pd.DataFrame, path: str | Path,
                    metric: str = "f1") -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    if cv_df.empty or metric not in cv_df.columns:
        ax.text(0.5, 0.5, "No cross-validation results", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    data = [group[metric].dropna().values for _, group in cv_df.groupby("model")]
    labels = [MODEL_LABELS.get(k, k) for k, _ in cv_df.groupby("model")]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, key in zip(bp["boxes"], cv_df.groupby("model").groups.keys()):
        patch.set_facecolor(MODEL_COLORS.get(key, "#888"))
        patch.set_alpha(0.75)
    for i, values in enumerate(data, start=1):
        ax.scatter(np.random.default_rng(i).normal(i, 0.05, len(values)), values,
                   color="black", s=16, zorder=3, alpha=0.7)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} across cross-validation folds")
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Architecture diagram
# ---------------------------------------------------------------------------
def make_architecture_diagram(path: str | Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(13.5, 15.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22)
    ax.axis("off")

    def box(y, text, color="#dbe7f3", height=0.85, width=6.2, x=1.9, fontsize=9.5):
        rect = FancyBboxPatch(
            (x, y - height / 2), width, height,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.2, edgecolor="#24425f", facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + width / 2, y, text, ha="center", va="center",
                fontsize=fontsize, wrap=True)
        return y

    def arrow(y_from, y_to, x=5.0):
        ax.add_patch(FancyArrowPatch(
            (x, y_from), (x, y_to),
            arrowstyle="-|>", mutation_scale=14, linewidth=1.3, color="#24425f",
        ))

    ax.text(5, 21.4, "Quantum-Enhanced Intrusion Detection System",
            ha="center", fontsize=15, fontweight="bold")
    ax.text(5, 20.8, "End-to-end experiment architecture (simulation-based)",
            ha="center", fontsize=9.5, color="#555")

    y = 19.9
    box(y, "Network traffic dataset — NSL-KDD\n(KDDTrain+ pool · KDDTest+ external hold-out)", "#e8f1f8")
    arrow(y - 0.45, 18.65); y = 19.0
    box(y, "Data cleaning & preprocessing\nmissing values · inf detection · constant columns\nfit on TRAIN only")
    arrow(y - 0.45, 17.75); y = 18.1
    box(y, "Categorical encoding + numeric scaling\n(OneHotEncoder + StandardScaler)")
    arrow(y - 0.45, 16.85); y = 17.2
    box(y, "Reproducible stratified split\nTrain 70% · Validation 15% · Test 15%")
    arrow(y - 0.45, 15.95); y = 16.3
    box(y, "Feature reduction (fit on train)\nPCA or Mutual Information · 8 / 12 / 16 features", "#fdebd0")
    arrow(y - 0.45, 15.05); y = 15.4
    box(y, "Reduced feature space (default 8-D)\nshared identical representation for every model", "#fdebd0")

    # Three model branches
    y_top = 14.3
    bw, bh = 2.55, 2.5
    xs = [0.45, 3.72, 7.0]
    titles = [
        ("CNN + LSTM", "Classical baseline\nConv1D → ReLU → MaxPool\n→ LSTM → Dense → logit", "#d6e6f5"),
        ("QSVM", "Quantum kernel\nZZ / angle feature map\n→ fidelity kernel → SVM", "#f9d9d4"),
        ("VQC", "Variational circuit\nfeature map → RealAmplitudes\n→ P(qubit0 = 1)", "#d5efdf"),
    ]
    for x0, (title, body, color) in zip(xs, titles):
        rect = FancyBboxPatch(
            (x0, y_top - bh / 2), bw, bh,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.4, edgecolor="#24425f", facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x0 + bw / 2, y_top + 0.82, title, ha="center", fontsize=10.5, fontweight="bold")
        ax.text(x0 + bw / 2, y_top - 0.15, body, ha="center", va="center", fontsize=8.2)

    ax.plot([5, 5], [14.9, 14.3], color="#24425f", linewidth=1.3)
    ax.plot([1.72, 8.27], [14.3, 14.3], color="#24425f", linewidth=1.3)
    for x0 in xs:
        ax.add_patch(FancyArrowPatch((x0 + bw / 2, 14.3), (x0 + bw / 2, y_top - bh / 2),
                                     arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.3, color="#24425f"))

    y = 12.4
    for x0 in xs:
        ax.add_patch(FancyArrowPatch((x0 + bw / 2, y_top - bh / 2), (5.0, 11.9),
                                     arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.1, color="#24425f", alpha=0.8))
    box(11.5, "Fair comparison on the SAME test set\nAccuracy · Precision · Recall · F1 · ROC-AUC · FPR · FNR", "#eae6f7")
    arrow(11.07, 10.6); y = 10.95
    box(y, "Unseen attack-category evaluation (R2L + U2R held out)\n+ external novel attack types from KDDTest+", "#fdebd0")
    arrow(y - 0.45, 9.75); y = 10.1
    box(y, "Latency · throughput · false-positive cost · quantum resources", "#e8f1f8")
    arrow(y - 0.45, 8.85); y = 9.2
    box(y, "Statistical significance\nMcNemar · paired Wilcoxon · bootstrap CIs", "#e8f1f8")
    arrow(y - 0.45, 7.95); y = 8.3
    box(y, "Explainability\npermutation importance · PCA loadings · circuit sensitivity", "#e8f1f8")
    arrow(y - 0.45, 7.05); y = 7.4
    box(y, "Distribution shift / concept-drift stress test\n(controlled, simulated — not production drift detection)", "#e8f1f8")
    arrow(y - 0.45, 6.15); y = 6.5
    box(y, "Tiered risk & alert engine\nNORMAL → MONITOR → ALERT → QUARANTINE → ESCALATE", "#f5cba7")
    arrow(y - 0.45, 5.25); y = 5.6
    box(y, "Analyst dashboard + demo mode\n(offline results, no live traffic)", "#f5cba7")
    arrow(y - 0.45, 4.3)

    ax.text(5, 3.9, "Research integrity boundary", ha="center", fontsize=10,
            fontweight="bold", color="#7b241c")
    ax.text(
        5, 3.1,
        "All quantum results are simulation-based (Qiskit Statevector).\n"
        "No quantum-hardware execution, no production deployment, and no\n"
        "quantum-advantage claim is made unless the measured evidence supports it.",
        ha="center", va="center", fontsize=8.6, color="#7b241c",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fdedec", edgecolor="#c0392b"),
    )
    fig.tight_layout()
    return _save(fig, path)
