"""Classical CNN+LSTM baseline implemented in PyTorch.

Input representation (documented explicitly, because tabular data is not
naturally temporal)
---------------------------------------------------------------------------
The reduced feature vector ``x in R^d`` (d = 8 by default) is reshaped into a
single-channel sequence of length ``d``: ``(1, d)``.  Each reduced component is
therefore one "time step".  This is a *pseudo-temporal* reshaping: the ordering
is inherited from the PCA component index (variance order) or the mutual
information ranking, not from real packet timing.  The 1-D convolution learns
local interactions between adjacent components and the LSTM learns
order-dependent combinations of the convolved features.  No claim of genuine
temporal modelling is made.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


class CNNLSTM(nn.Module):
    """Configurable 1-D CNN -> LSTM -> dense classifier."""

    def __init__(
        self,
        n_features: int,
        conv_filters: int = 32,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.3,
        lstm_units: int = 32,
        lstm_layers: int = 1,
        dense_units: int = 32,
        n_classes: int = 1,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)

        kernel_size = min(int(kernel_size), max(1, self.n_features))
        if kernel_size % 2 == 0:
            kernel_size = max(1, kernel_size - 1)
        self.kernel_size = kernel_size

        effective_pool = int(pool_size)
        if self.n_features // max(1, effective_pool) < 2:
            effective_pool = 1
        self.pool_size = effective_pool

        self.input_norm = nn.BatchNorm1d(1)
        self.conv = nn.Conv1d(1, conv_filters, kernel_size, padding=kernel_size // 2)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool1d(effective_pool) if effective_pool > 1 else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_units,
            num_layers=max(1, lstm_layers),
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        out_dim = 1 if n_classes <= 2 else n_classes
        self.head = nn.Sequential(
            nn.Linear(lstm_units, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() == 3 and x.shape[1] != 1:
            x = x.transpose(1, 2)
        x = self.input_norm(x)
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)          # (B, L', filters)
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]
        x = self.head(x)
        if self.n_classes <= 2:
            return x.squeeze(-1)
        return x


@dataclass
class TrainingHistory:
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    val_roc_auc: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_roc_auc: float = 0.0
    stopped_early: bool = False
    epochs_requested: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "val_f1": self.val_f1,
            "val_roc_auc": self.val_roc_auc,
            "learning_rate": self.learning_rate,
            "epoch_seconds": self.epoch_seconds,
            "best_epoch": self.best_epoch,
            "best_val_roc_auc": self.best_val_roc_auc,
            "stopped_early": self.stopped_early,
            "epochs_requested": self.epochs_requested,
        }


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


def train_cnn_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg,
    seed: int = 42,
    epochs: Optional[int] = None,
    verbose: bool = True,
    logger: Any = None,
) -> tuple[CNNLSTM, TrainingHistory, dict[str, Any]]:
    """Train with early stopping on validation ROC-AUC and checkpoint the best epoch."""
    arch = cfg.get("classical", {})
    n_features = int(X_train.shape[1])
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_threads = int(arch.get("num_threads", 0) or 0)
    if num_threads > 0:
        torch.set_num_threads(num_threads)

    device = torch.device(str(arch.get("device", "cpu")))
    model = CNNLSTM(
        n_features=n_features,
        conv_filters=int(arch.get("conv_filters", 32)),
        kernel_size=int(arch.get("kernel_size", 3)),
        pool_size=int(arch.get("pool_size", 2)),
        dropout=float(arch.get("dropout", 0.3)),
        lstm_units=int(arch.get("lstm_units", 32)),
        lstm_layers=int(arch.get("lstm_layers", 1)),
        dense_units=int(arch.get("dense_units", 32)),
    ).to(device)

    epochs = int(epochs if epochs is not None else arch.get("epochs", 25))
    batch_size = int(arch.get("batch_size", 256))
    lr = float(arch.get("learning_rate", 1e-3))
    weight_decay = float(arch.get("weight_decay", 1e-4))
    patience = int(arch.get("patience", 6))
    min_delta = float(arch.get("min_delta", 5e-4))
    grad_clip = float(arch.get("grad_clip", 1.0))

    Xtr = torch.as_tensor(np.asarray(X_train, dtype=np.float32))
    ytr = torch.as_tensor(np.asarray(y_train, dtype=np.float32))
    Xva = torch.as_tensor(np.asarray(X_val, dtype=np.float32))
    yva = torch.as_tensor(np.asarray(y_val, dtype=np.float32))

    train_ds = TensorDataset(Xtr, ytr)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=generator, drop_last=False
    )
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size, shuffle=False)

    # Class weighting (computed on TRAIN only).
    class_weight = str(arch.get("class_weight", "balanced")).lower()
    if class_weight == "balanced":
        n_pos = float(ytr.sum().item())
        n_neg = float(len(ytr) - n_pos)
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device)
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    history = TrainingHistory(epochs_requested=epochs)
    best_state: dict[str, torch.Tensor] | None = None
    best_auc = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        t_epoch = time.perf_counter()
        model.train()
        running_loss = 0.0
        n_seen = 0
        correct = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += float(loss.item()) * len(xb)
            n_seen += len(xb)
            correct += int(((torch.sigmoid(logits) >= 0.5).float() == yb).sum().item())

        train_loss = running_loss / max(n_seen, 1)
        train_acc = correct / max(n_seen, 1)

        model.eval()
        val_loss_sum = 0.0
        val_scores: list[float] = []
        val_targets: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                val_loss_sum += float(criterion(logits, yb).item()) * len(xb)
                val_scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                val_targets.extend(yb.cpu().numpy().tolist())

        val_loss = val_loss_sum / max(len(yva), 1)
        val_scores_arr = np.asarray(val_scores)
        val_targets_arr = np.asarray(val_targets)
        val_pred = (val_scores_arr >= 0.5).astype(int)
        val_acc = float(accuracy_score(val_targets_arr, val_pred))
        val_f1 = float(f1_score(val_targets_arr, val_pred, zero_division=0))
        val_auc = _safe_auc(val_targets_arr, val_scores_arr)

        elapsed = time.perf_counter() - t_epoch
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.epochs.append(epoch)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_accuracy.append(train_acc)
        history.val_accuracy.append(val_acc)
        history.val_f1.append(val_f1)
        history.val_roc_auc.append(val_auc)
        history.learning_rate.append(current_lr)
        history.epoch_seconds.append(elapsed)

        score = val_auc if np.isfinite(val_auc) else val_f1
        improved = score > best_auc + min_delta
        if improved:
            best_auc = score
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        scheduler.step(score)

        if verbose:
            msg = (
                f"epoch {epoch:03d}/{epochs} | loss {train_loss:.4f} | "
                f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | "
                f"val_f1 {val_f1:.4f} | val_auc {val_auc:.4f} | "
                f"lr {current_lr:.2e} | {elapsed:.2f}s"
            )
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)

        if epochs_without_improvement >= patience:
            history.stopped_early = True
            if logger is not None:
                logger.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
            break

    history.best_epoch = best_epoch
    history.best_val_roc_auc = float(best_auc) if np.isfinite(best_auc) else float("nan")

    if best_state is not None:
        model.load_state_dict(best_state)

    info = {
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "n_features": n_features,
        "sequence_length": n_features,
        "conv_filters": int(model.conv.out_channels),
        "kernel_size": int(model.kernel_size),
        "pool_size": int(model.pool_size),
        "lstm_units": int(arch.get("lstm_units", 32)),
        "lstm_layers": int(arch.get("lstm_layers", 1)),
        "dropout": float(arch.get("dropout", 0.3)),
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_run": len(history.epochs),
        "best_epoch": best_epoch,
        "training_time_s": time.perf_counter() - t_start,
        "pos_weight": float(pos_weight.item()) if pos_weight is not None else None,
        "device": str(device),
        "input_representation": (
            "reduced feature vector reshaped to a single-channel sequence of "
            "length d (pseudo-temporal; not real packet timing)"
        ),
        "architecture": "BatchNorm1d -> Conv1d -> ReLU -> MaxPool -> Dropout -> LSTM -> Dense -> logit",
    }
    return model, history, info


def predict_proba(model: CNNLSTM, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    X = np.asarray(X, dtype=np.float32)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[start:start + batch_size]).to(device)
            logits = model(batch)
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    if not outputs:
        return np.array([], dtype=np.float32)
    return np.concatenate(outputs).astype(np.float32)


def save_cnn_lstm(model: CNNLSTM, info: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "architecture": {
                "n_features": model.n_features,
                "conv_filters": model.conv.out_channels,
                "kernel_size": model.kernel_size,
                "pool_size": model.pool_size,
                "lstm_units": model.lstm.hidden_size,
                "lstm_layers": model.lstm.num_layers,
                "dropout": model.dropout.p,
                "dense_units": model.head[0].out_features,
                "n_classes": model.n_classes,
            },
            "info": info,
        },
        path,
    )
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump({"architecture": {
            "n_features": model.n_features,
            "conv_filters": model.conv.out_channels,
            "kernel_size": model.kernel_size,
            "pool_size": model.pool_size,
            "lstm_units": model.lstm.hidden_size,
            "lstm_layers": model.lstm.num_layers,
            "dropout": model.dropout.p,
            "dense_units": model.head[0].out_features,
            "n_classes": model.n_classes,
        }, **info}, fh, indent=2, default=str)
    return path


def load_cnn_lstm(path: str | Path) -> tuple[CNNLSTM, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    arch = payload["architecture"]
    model = CNNLSTM(
        n_features=arch["n_features"],
        conv_filters=arch["conv_filters"],
        kernel_size=arch["kernel_size"],
        pool_size=arch["pool_size"],
        dropout=arch["dropout"],
        lstm_units=arch["lstm_units"],
        lstm_layers=arch["lstm_layers"],
        dense_units=arch["dense_units"],
        n_classes=arch.get("n_classes", 1),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("info", {})
