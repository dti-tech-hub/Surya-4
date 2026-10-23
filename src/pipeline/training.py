"""Shared training helpers used by the pipeline, cross-validation and hold-out runs."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.classical.cnn_lstm import (
    CNNLSTM,
    predict_proba as cnn_predict_proba,
    save_cnn_lstm,
    train_cnn_lstm,
)
from src.data.preprocessing import Preprocessor
from src.evaluation.metrics import compute_metrics, select_operating_threshold
from src.features.reduction import apply_reduction
from src.features.scaling import FeatureScaler
from src.quantum.encoding import EncodingConfig, encoding_config_from
from src.quantum.qsvm import QuantumSVM
from src.quantum.vqc import VariationalQuantumClassifier
from src.utils.logging import get_logger

LOGGER = get_logger("qeids.training")

MODEL_KINDS = ("cnn_lstm", "qsvm", "vqc")


@dataclass
class PreprocessingArtifacts:
    preprocessor: Preprocessor
    reducer: Any
    scaler: FeatureScaler
    feature_names: list[str]
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    method: str
    n_features: int
    y_train: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    X_test_common: Optional[np.ndarray] = None
    y_common: Optional[np.ndarray] = None
    common_idx: Optional[np.ndarray] = None
    train_q_idx: Optional[np.ndarray] = None
    val_q_idx: Optional[np.ndarray] = None
    timings: dict[str, float] = field(default_factory=dict)

    def input_dimension(self) -> int:
        return int(self.X_train.shape[1])


@dataclass
class ModelBundle:
    name: str
    model: Any
    kind: str
    encoding: str
    feature_method: str
    n_features: int
    resources: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    threshold: float = 0.5

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "cnn_lstm":
            return cnn_predict_proba(self.model, X)
        return np.asarray(self.model.predict_proba(X), dtype=float)

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        t = self.threshold if threshold is None else threshold
        return (self.predict_proba(X) >= t).astype(int)


def fit_preprocessing(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg,
    n_features: Optional[int] = None,
    method: Optional[str] = None,
    tag: str = "main",
) -> PreprocessingArtifacts:
    """Fit the full leakage-safe preprocessing chain on the training split."""
    seed = int(cfg.get_path("project.seed", 42))
    n_features = int(n_features or cfg.get_path("feature_reduction.n_features", 8))
    method = str(method or cfg.get_path("feature_reduction.method", "pca"))

    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    preprocessor = Preprocessor(
        missing_strategy=str(cfg.get_path("preprocessing.handle_missing", "median")),
        scale_method=str(cfg.get_path("preprocessing.scale_numeric", "standard")),
        min_category_frequency=int(cfg.get_path("preprocessing.min_category_frequency", 2)),
        drop_constant=bool(cfg.get_path("preprocessing.drop_constant_columns", True)),
    ).fit(train_df)
    Xtr_raw = preprocessor.transform(train_df)
    Xva_raw = preprocessor.transform(val_df)
    Xte_raw = preprocessor.transform(test_df)
    timings["preprocessing_fit_transform_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    Xtr, Xva, Xte, reducer, scaler = apply_reduction(
        method=method,
        k=n_features,
        X_train=Xtr_raw,
        y_train=train_df["is_attack"].to_numpy(),
        X_val=Xva_raw,
        X_test=Xte_raw,
        feature_names=preprocessor.feature_names,
        seed=seed,
    )
    timings["feature_reduction_s"] = time.perf_counter() - t0

    return PreprocessingArtifacts(
        preprocessor=preprocessor,
        reducer=reducer,
        scaler=scaler,
        feature_names=preprocessor.feature_names,
        X_train=Xtr,
        X_val=Xva,
        X_test=Xte,
        method=method,
        n_features=n_features,
        y_train=train_df["is_attack"].to_numpy(dtype=np.int8),
        y_val=val_df["is_attack"].to_numpy(dtype=np.int8),
        y_test=test_df["is_attack"].to_numpy(dtype=np.int8),
        timings=timings,
    )


def fit_model(
    kind: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg,
    encoding_override: Optional[str] = None,
    maxiter_override: Optional[int] = None,
    epochs_override: Optional[int] = None,
    verbose: bool = True,
) -> ModelBundle:
    """Train one model kind on an already-reduced feature matrix."""
    kind = kind.lower()
    seed = int(cfg.get_path("project.seed", 42))

    if kind == "cnn_lstm":
        model, history, info = train_cnn_lstm(
            X_train, y_train, X_val, y_val, cfg,
            seed=seed, epochs=epochs_override, verbose=verbose,
        )
        return ModelBundle(
            name="cnn_lstm",
            model=model,
            kind="cnn_lstm",
            encoding="none",
            feature_method=str(cfg.get_path("feature_reduction.method", "pca")),
            n_features=int(X_train.shape[1]),
            resources={"num_qubits": 0, "depth": 0, "num_parameters": info["n_parameters"]},
            timings={"training_time_s": info["training_time_s"]},
            info={"history": history.to_dict(), **info},
        )

    if kind == "qsvm":
        section = cfg.get_path("quantum.qsvm", {}) or {}
        enc_cfg = encoding_config_from(cfg, "qsvm", X_train.shape[1])
        if encoding_override:
            enc_cfg = EncodingConfig(
                method=encoding_override,
                n_features=int(X_train.shape[1]),
                map_name=enc_cfg.map_name if encoding_override == "angle" else "amplitude",
                reps=enc_cfg.reps if encoding_override == "angle" else 0,
                entanglement=enc_cfg.entanglement if encoding_override == "angle" else "none",
            )
        model = QuantumSVM(
            encoding=enc_cfg,
            C=float(section.get("C", 1.0)),
            class_weight=section.get("class_weight", "balanced"),
            probability=bool(section.get("probability", True)),
            seed=seed,
            validate_kernel=bool(section.get("validate_against_qiskit", True)),
            validation_shots=int(section.get("validation_shots", 100000)),
        )
        model.fit(X_train, y_train)
        return ModelBundle(
            name="qsvm",
            model=model,
            kind="qsvm",
            encoding=enc_cfg.method,
            feature_method=str(cfg.get_path("feature_reduction.method", "pca")),
            n_features=int(X_train.shape[1]),
            resources=model.resources(),
            timings=dict(model.timings_),
            info={
                "kernel_diagnostics": model.kernel_diagnostics_,
                "kernel_validation": model.validation_,
                "encoding": enc_cfg.to_dict(),
            },
        )

    if kind == "vqc":
        section = cfg.get_path("quantum.vqc", {}) or {}
        enc_cfg = encoding_config_from(cfg, "vqc", X_train.shape[1])
        if encoding_override:
            enc_cfg = EncodingConfig(
                method=encoding_override,
                n_features=int(X_train.shape[1]),
                map_name=enc_cfg.map_name if encoding_override == "angle" else "amplitude",
                reps=enc_cfg.reps if encoding_override == "angle" else 0,
                entanglement=enc_cfg.entanglement if encoding_override == "angle" else "none",
            )
        model = VariationalQuantumClassifier(
            encoding=enc_cfg,
            ansatz_reps=int(section.get("ansatz_reps", 2)),
            ansatz=str(section.get("ansatz", "real_amplitudes")),
            entanglement=str(section.get("entanglement", "linear")),
            optimizer=str(section.get("optimizer", "COBYLA")),
            maxiter=int(maxiter_override or section.get("maxiter", 150)),
            learning_rate=float(section.get("learning_rate", 0.05)),
            perturbation=float(section.get("perturbation", 0.1)),
            initial_point_scale=float(section.get("initial_point_scale", 0.1)),
            l2=float(section.get("l2", 1e-4)),
            seed=seed,
            shots=int(cfg.get_path("quantum.shots", 1024)),
        )
        model.fit(X_train, y_train, X_val, y_val, verbose=verbose)
        return ModelBundle(
            name="vqc",
            model=model,
            kind="vqc",
            encoding=enc_cfg.method,
            feature_method=str(cfg.get_path("feature_reduction.method", "pca")),
            n_features=int(X_train.shape[1]),
            resources=model.resources(),
            timings=dict(model.timings_),
            info={
                "optimizer_result": model.optimizer_result_,
                "convergence": model.history_,
                "encoding": enc_cfg.to_dict(),
            },
        )

    raise ValueError(f"Unknown model kind {kind!r}. Supported: {MODEL_KINDS}")


def tune_threshold(
    bundle: ModelBundle,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg,
    apply: bool = True,
) -> dict[str, Any]:
    """Select an operating threshold on validation data only."""
    if not bool(cfg.get_path("evaluation.optimize_threshold", True)):
        bundle.threshold = float(cfg.get_path("evaluation.threshold", 0.5))
        return {"objective": "fixed", "threshold": bundle.threshold}

    scores = bundle.predict_proba(X_val)
    selection = select_operating_threshold(
        y_val,
        scores,
        objective=str(cfg.get_path("evaluation.threshold_objective", "f1")),
        thresholds=cfg.get_path("evaluation.thresholds", None),
        false_negative_cost=float(cfg.get_path("evaluation.false_negative_cost", 10.0)),
        false_positive_cost=float(cfg.get_path("evaluation.false_positive_cost", 1.0)),
    )
    table = selection.pop("table")
    if apply:
        bundle.threshold = float(selection["threshold"])
    bundle.info["threshold_selection"] = selection
    bundle.info["threshold_table"] = table.to_dict(orient="records")
    return selection


def evaluate_bundle(
    bundle: ModelBundle,
    X: np.ndarray,
    y: np.ndarray,
    threshold: Optional[float] = None,
    categories: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Evaluate on a test set and return metrics + timing."""
    t = bundle.threshold if threshold is None else threshold
    t0 = time.perf_counter()
    scores = bundle.predict_proba(X)
    inference_time = time.perf_counter() - t0
    metrics = compute_metrics(y, scores, threshold=t)
    metrics.update({
        "model": bundle.name,
        "encoding": bundle.encoding,
        "feature_method": bundle.feature_method,
        "n_features": bundle.n_features,
        "inference_time_s": inference_time,
        "inference_ms_per_sample": inference_time / max(len(X), 1) * 1000.0,
        "throughput_samples_per_s": len(X) / max(inference_time, 1e-9),
    })
    if categories is not None:
        from src.evaluation.metrics import per_category_detection

        metrics["per_category"] = per_category_detection(
            categories, y, (scores >= t).astype(int), scores
        ).to_dict(orient="records")
    return metrics


def save_bundle(bundle: ModelBundle, cfg, tag: Optional[str] = None) -> dict[str, str]:
    """Persist model, metadata and config.  Returns the created paths."""
    tag = tag or bundle.name
    model_dir = Path(cfg.get_path("paths.models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    if bundle.kind == "cnn_lstm":
        path = model_dir / f"{tag}.pt"
        save_cnn_lstm(bundle.model, bundle.info, path)
        paths["model"] = str(path)
    elif bundle.kind == "qsvm":
        path = model_dir / f"{tag}.joblib"
        bundle.model.save(path)
        paths["model"] = str(path)
    elif bundle.kind == "vqc":
        path = model_dir / f"{tag}.joblib"
        bundle.model.save(path)
        paths["model"] = str(path)
    else:
        raise ValueError(f"Cannot save unknown kind {bundle.kind!r}")

    meta = {
        "name": bundle.name,
        "kind": bundle.kind,
        "encoding": bundle.encoding,
        "feature_method": bundle.feature_method,
        "n_features": bundle.n_features,
        "threshold": bundle.threshold,
        "resources": bundle.resources,
        "timings": bundle.timings,
        "info": _strip_large(bundle.info),
    }
    meta_path = model_dir / f"{tag}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(meta), fh, indent=2, default=str)
    paths["metadata"] = str(meta_path)
    return paths


def _strip_large(info: dict) -> dict:
    out = {}
    for k, v in info.items():
        if k == "threshold_table" and isinstance(v, list) and len(v) > 50:
            out[k] = v[:50]
        else:
            out[k] = v
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_bundle(kind: str, path: str | Path) -> Any:
    kind = kind.lower()
    if kind == "cnn_lstm":
        from src.classical.cnn_lstm import load_cnn_lstm

        return load_cnn_lstm(path)[0]
    if kind == "qsvm":
        return QuantumSVM.load(path)
    if kind == "vqc":
        return VariationalQuantumClassifier.load(path)
    raise ValueError(f"Unknown kind {kind!r}")
