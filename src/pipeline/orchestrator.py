"""End-to-end pipeline orchestration.

Stages are resumable: every artifact is written to disk and the run manifest
records exactly which stages completed, so ``--mode evaluate`` or
``--mode dashboard`` can reuse a previous run without retraining.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from src.alerting.risk_engine import AlertPolicy, RiskEngine, summarize_alerts
from src.classical.cnn_lstm import load_cnn_lstm
from src.data.loader import build_dataset, dataset_fingerprint
from src.data.preprocessing import build_class_distribution_report, clean_dataframe
from src.data.splitting import (
    load_splits,
    make_common_test_subset,
    save_splits,
    split_dataframe,
    stratified_subsample_indices,
)
from src.data.validation import (
    leakage_audit,
    save_validation_report,
    split_integrity_report,
    validate_design_matrix,
    validate_processed_frame,
)
from src.evaluation import drift as drift_mod
from src.evaluation import plots
from src.evaluation.comparison import build_master_table, save_master_table
from src.evaluation.cross_validation import paired_fold_scores, run_cross_validation
from src.evaluation.latency import measure_latency
from src.evaluation.metrics import compute_metrics, threshold_analysis
from src.evaluation.statistical_tests import (
    bootstrap_metric_ci,
    mcnemar_test,
    paired_bootstrap_difference,
    summarize_cv,
    wilcoxon_test,
)
from src.evaluation.zero_day import (
    evaluate_external_novel_attacks,
    run_unseen_category_experiment,
)
from src.explainability.explain import (
    example_explanation,
    pca_attribution,
    permutation_importance,
    quantum_feature_sensitivity,
    shap_explain_cnn,
)
from src.features.reduction import (
    apply_reduction,
    build_reducer,
    fit_reducer,
    select_configuration,
)
from src.pipeline.training import (
    evaluate_bundle,
    fit_model,
    fit_preprocessing,
    save_bundle,
    tune_threshold,
)
from src.quantum.circuits import save_circuit_artifacts
from src.quantum.qsvm import QuantumSVM
from src.quantum.vqc import VariationalQuantumClassifier
from src.utils.config import profile
from src.utils.io import ensure_dirs, jsonable, load_json, save_json, human_duration
from src.utils.logging import ExperimentLogger, get_logger
from src.utils.reproducibility import save_environment, set_global_seed

LOGGER = get_logger("qeids.pipeline")

ALL_MODELS = ("cnn_lstm", "qsvm", "vqc")
STAGES = (
    "prepare", "features", "train", "evaluate", "zero_day", "encoding",
    "latency", "drift", "explain", "cv", "stats", "tables", "dashboard", "report",
)


@dataclass
class RunContext:
    cfg: Any
    train_df: Optional[pd.DataFrame] = None
    val_df: Optional[pd.DataFrame] = None
    test_df: Optional[pd.DataFrame] = None
    external_df: Optional[pd.DataFrame] = None
    common_test_df: Optional[pd.DataFrame] = None
    prep: Any = None
    bundles: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, dict] = field(default_factory=dict)
    resources: dict[str, dict] = field(default_factory=dict)
    timings: dict[str, dict] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


class Pipeline:
    def __init__(self, cfg, models: Sequence[str] = ALL_MODELS, verbose: bool = True) -> None:
        self.cfg = cfg
        self.models = [m for m in models if m in ALL_MODELS]
        self.verbose = verbose
        ensure_dirs(cfg)
        self.p = {k: cfg.get_path(f"paths.{k}") for k in (
            "data_raw", "data_processed", "data_splits", "data_metadata", "models",
            "results", "metrics", "figures", "tables", "predictions", "logs",
            "circuits", "docs", "presentation",
        )}
        self.manifest_path = Path(self.p["results"]) / "run_manifest.json"
        self.manifest = load_json(self.manifest_path, default={}) or {}
        self.mode = str(cfg.get_path("experiment.mode", "quick"))
        self.profile = profile(cfg)
        self.seed = int(cfg.get_path("project.seed", 42))
        self.n_features = int(cfg.get_path("feature_reduction.n_features", 8))
        self.method = str(cfg.get_path("feature_reduction.method", "pca"))
        self.log_path = Path(self.p["logs"]) / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = ExperimentLogger(
            self.log_path, experiment=f"qeids-{self.mode}", model="pipeline", echo=verbose
        )
        self.start_time = time.perf_counter()

    # ------------------------------------------------------------------ util
    def _record(self, stage: str, status: str = "success", **fields: Any) -> None:
        self.manifest[stage] = {
            "status": status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "mode": self.mode,
            **jsonable(fields),
        }
        save_json(self.manifest, self.manifest_path)
        self.logger.log(stage, status=status, **fields)

    def _stage_done(self, stage: str) -> bool:
        return self.manifest.get(stage, {}).get("status") == "success"

    # ------------------------------------------------------------------ paths
    def _processed(self, name: str) -> Path:
        return Path(self.p["data_processed"]) / name

    def _fig(self, name: str) -> Path:
        return Path(self.p["figures"]) / name

    def _metrics_path(self, name: str) -> Path:
        return Path(self.p["metrics"]) / name

    def _pred_path(self, name: str) -> Path:
        return Path(self.p["predictions"]) / name

    # ================================================================ PREPARE
    def prepare(self, force: bool = False) -> RunContext:
        ctx = RunContext(cfg=self.cfg)
        set_global_seed(self.seed)

        env = save_environment(Path(self.p["results"]) / "environment.json")
        LOGGER.info("Environment captured (%s)", env["platform"]["processor"])

        dataset = build_dataset(self.cfg)
        validation = dataset.validate()
        save_json(validation, Path(self.p["data_metadata"]) / "dataset_validation.json")
        if not validation["valid"]:
            raise RuntimeError(f"Dataset validation failed: {validation}")

        pool = dataset.load()
        external = dataset.load_external()
        LOGGER.info(
            "Raw dataset: pool=%d rows, external=%d rows, %d attack types in pool",
            len(pool), len(external), pool["label"].nunique(),
        )

        pool, cleaning_report = clean_dataframe(pool, self.cfg)
        external, external_cleaning = clean_dataframe(external, self.cfg)
        save_json(cleaning_report, Path(self.p["data_metadata"]) / "cleaning_report.json")
        save_json(external_cleaning, Path(self.p["data_metadata"]) / "cleaning_report_external.json")

        max_samples = self.profile.get("max_samples")
        if max_samples is not None and len(pool) > int(max_samples):
            idx = stratified_subsample_indices(
                pool["is_attack"].to_numpy(), int(max_samples), self.seed
            )
            pool = pool.iloc[idx].reset_index(drop=True)
            LOGGER.info("Subsampled pool to %d rows (mode=%s)", len(pool), self.mode)

        train_df, val_df, test_df, split_meta = split_dataframe(pool, self.cfg)
        integrity = split_integrity_report(split_meta, train_df, val_df, test_df)
        if not integrity["passed"]:
            raise RuntimeError(f"Split integrity check failed: {integrity}")

        save_splits(train_df, val_df, test_df, self.cfg, extra={
            "mode": self.mode,
            "seed": self.seed,
            "pool_rows": int(len(pool)),
            "split_meta": {k: v for k, v in split_meta.items() if k != "indices"},
            "integrity": integrity,
        })
        save_json(
            {"split": {k: v for k, v in split_meta.items() if k != "indices"},
             "integrity": integrity,
             "pool_fingerprint": dataset_fingerprint(pool),
             "external_fingerprint": dataset_fingerprint(external)},
            Path(self.p["data_metadata"]) / "split_report.json",
        )

        external.to_parquet(self._processed("external.parquet"), index=False)

        common_n = self.profile.get("common_test_size")
        common_df, common_idx = make_common_test_subset(
            test_df, int(common_n) if common_n is not None else None, self.seed + 7
        )
        common_df.to_parquet(self._processed("common_test.parquet"), index=False)
        np.save(self._processed("common_test_indices.npy"), common_idx)

        class_report = build_class_distribution_report(train_df, val_df, test_df)
        save_json(class_report, Path(self.p["data_metadata"]) / "class_distribution.json")
        plots.plot_class_distribution(
            train_df, val_df, test_df, self._fig("class_distribution.png")
        )

        for name, frame in (("train", train_df), ("validation", val_df), ("test", test_df)):
            report = validate_processed_frame(frame, name)
            if not report["passed"]:
                LOGGER.warning("Frame %s validation issues: %s", name, report["issues"])

        self._record(
            "prepare",
            pool_rows=len(pool), train=int(len(train_df)), validation=int(len(val_df)),
            test=int(len(test_df)), common_test=int(len(common_df)),
            attack_types_pool=int(pool["label"].nunique()),
            external_rows=int(len(external)),
            duration_s=time.perf_counter() - self.start_time,
        )
        ctx.train_df, ctx.val_df, ctx.test_df = train_df, val_df, test_df
        ctx.external_df, ctx.common_test_df = external, common_df
        return ctx

    def load_prepared(self) -> RunContext:
        ctx = RunContext(cfg=self.cfg)
        train_df, val_df, test_df = load_splits(self.cfg)
        ctx.train_df, ctx.val_df, ctx.test_df = train_df, val_df, test_df
        ctx.external_df = pd.read_parquet(self._processed("external.parquet"))
        ctx.common_test_df = pd.read_parquet(self._processed("common_test.parquet"))
        return ctx

    # =============================================================== FEATURES
    def prepare_features(self, ctx: RunContext) -> RunContext:
        from src.features.scaling import FeatureScaler

        train_df, val_df, test_df = ctx.train_df, ctx.val_df, ctx.test_df
        assert train_df is not None

        t0 = time.perf_counter()
        prep = fit_preprocessing(
            train_df, val_df, test_df, self.cfg,
            n_features=self.n_features, method=self.method, tag="main",
        )

        # --- Validation-only configuration selection -----------------------
        from src.data.preprocessing import Preprocessor

        pre = Preprocessor(
            missing_strategy=str(self.cfg.get_path("preprocessing.handle_missing", "median")),
            scale_method=str(self.cfg.get_path("preprocessing.scale_numeric", "standard")),
            min_category_frequency=int(self.cfg.get_path("preprocessing.min_category_frequency", 2)),
            drop_constant=bool(self.cfg.get_path("preprocessing.drop_constant_columns", True)),
        ).fit(train_df)
        Xtr_raw = pre.transform(train_df)
        Xva_raw = pre.transform(val_df)

        candidates_m = list(self.cfg.get_path("feature_reduction.candidate_methods", ["pca", "mutual_info"]))
        candidates_k = list(self.cfg.get_path("feature_reduction.candidate_features", [8, 12, 16]))
        best, ablation = select_configuration(
            Xtr_raw, train_df["is_attack"].to_numpy(),
            Xva_raw, val_df["is_attack"].to_numpy(),
            pre.feature_names, candidates_m, candidates_k,
            metric=str(self.cfg.get_path("feature_reduction.selection_metric", "f1")),
            seed=self.seed,
        )
        ablation_df = pd.DataFrame(ablation)
        ablation_df.to_csv(Path(self.p["tables"]) / "feature_reduction_ablation.csv", index=False)

        # The comparative study is constrained to <= max_quantum_qubits.
        max_q = int(self.cfg.get_path("feature_reduction.max_quantum_qubits", 8))
        feasible = [r for r in ablation if r["status"] == "ok"
                    and r["n_features"] <= max_q and np.isfinite(r.get("f1", np.nan))]
        feasible_best = max(feasible, key=lambda r: (r["f1"], r.get("roc_auc", 0.0))) if feasible else best

        # The pipeline-level configuration can be overridden by the user; if the
        # override is infeasible for the simulator we keep 8 features and say so.
        selected_method = str(self.cfg.get_path("feature_reduction.method", feasible_best["method"]))
        selected_k = int(self.cfg.get_path("feature_reduction.n_features", feasible_best["n_features"]))
        if selected_k > max_q:
            LOGGER.warning(
                "n_features=%d exceeds max_quantum_qubits=%d; using %d for the "
                "fair comparison. The %d-feature configuration remains in the "
                "ablation table.", selected_k, max_q, max_q, selected_k,
            )
            selected_k = max_q

        if (selected_method, selected_k) != (prep.method, prep.n_features):
            LOGGER.info(
                "Re-fitting reducer with selected configuration %s/%d",
                selected_method, selected_k,
            )
            prep = fit_preprocessing(
                train_df, val_df, test_df, self.cfg,
                n_features=selected_k, method=selected_method, tag="main",
            )
        self.method = selected_method
        self.n_features = selected_k

        # --- Persist artefacts ---------------------------------------------
        prep.preprocessor.save(self._processed("preprocessor.joblib"))
        prep.reducer.save(self._processed("reducer.joblib"))
        prep.scaler.save(self._processed("scaler.joblib"))

        common_idx = np.load(self._processed("common_test_indices.npy"))
        X_test_common = prep.X_test[common_idx]
        y_train = train_df["is_attack"].to_numpy()
        y_val = val_df["is_attack"].to_numpy()
        y_test = test_df["is_attack"].to_numpy()
        y_common = y_test[common_idx]

        # Quantum subsets (stratified, fixed seed).
        q_train_n = int(self.profile.get("quantum_train_size", 600))
        q_val_n = int(self.profile.get("quantum_val_size", 200))
        train_q_idx = stratified_subsample_indices(y_train, min(q_train_n, len(y_train)), self.seed + 1)
        val_q_idx = stratified_subsample_indices(y_val, min(q_val_n, len(y_val)), self.seed + 2)

        prep.X_test_common = X_test_common
        prep.y_common = y_common
        prep.common_idx = common_idx
        prep.train_q_idx = train_q_idx
        prep.val_q_idx = val_q_idx

        np.savez_compressed(
            self._processed("arrays.npz"),
            X_train=prep.X_train, X_val=prep.X_val, X_test=prep.X_test,
            X_test_common=X_test_common,
            y_train=y_train, y_val=y_val, y_test=y_test, y_common=y_common,
            common_idx=common_idx, train_q_idx=train_q_idx, val_q_idx=val_q_idx,
        )

        selection = {
            "selected_method": self.method,
            "selected_n_features": self.n_features,
            "validation_best_all": best,
            "validation_best_feasible": feasible_best,
            "ablation": ablation,
            "policy": (
                "Configuration chosen on validation F1 only, restricted to "
                "n_features <= max_quantum_qubits for simulator feasibility. "
                "The test split was not consulted."
            ),
        }
        save_json(selection, Path(self.p["data_metadata"]) / "feature_selection.json")

        # --- Figures --------------------------------------------------------
        if hasattr(prep.reducer, "explained_variance_ratio_"):
            plots.plot_pca_explained_variance(
                prep.reducer.explained_variance_ratio_,
                prep.reducer.cumulative_variance(),
                self._fig("pca_explained_variance.png"),
                self.n_features,
            )
            plots.plot_pca_component_heatmap(
                prep.reducer.loadings(), prep.reducer.feature_names_in_,
                self._fig("pca_component_loadings.png"),
            )
            save_json(
                {
                    "explained_variance_ratio": prep.reducer.explained_variance_ratio_.tolist(),
                    "cumulative": prep.reducer.cumulative_variance().tolist(),
                    "total": float(prep.reducer.cumulative_variance()[-1]),
                    "n_features": self.n_features,
                },
                Path(self.p["metrics"]) / "pca_explained_variance.json",
            )
        else:
            ranking = prep.reducer.ranking()
            pd.DataFrame(ranking).to_csv(
                Path(self.p["tables"]) / "mutual_information_ranking.csv", index=False
            )
            plots.plot_feature_importance(
                [r["feature"] for r in ranking][:20],
                [r["mutual_information"] for r in ranking][:20],
                self._fig("mutual_information_scores.png"),
                title="Mutual information ranking (top 20)",
                xlabel="Mutual information",
            )

        plots.plot_feature_importance(
            [f"PC{i + 1}" if self.method == "pca" else f"F{i}" for i in range(self.n_features)],
            np.arange(self.n_features, 0, -1, dtype=float),
            self._fig("reduced_feature_space.png"),
            title=f"Retained reduced features ({self.method}, {self.n_features})",
            xlabel="Variance order (illustrative)",
        )

        # --- Leakage audit --------------------------------------------------
        audit = leakage_audit(
            prep.preprocessor, prep.reducer, train_df, val_df, test_df,
            prep.X_train, prep.X_val, prep.X_test,
            pca_explained=(
                prep.reducer.explained_variance_ratio_.tolist()
                if hasattr(prep.reducer, "explained_variance_ratio_") else None
            ),
        )
        audit["design_matrices"] = [
            validate_design_matrix(prep.X_train, "X_train"),
            validate_design_matrix(prep.X_val, "X_val"),
            validate_design_matrix(prep.X_test, "X_test"),
        ]
        audit["feature_selection"] = selection
        audit["split_integrity"] = load_json(
            Path(self.p["data_metadata"]) / "split_report.json", default={}
        ).get("integrity", {})
        save_validation_report(audit, Path(self.p["data_metadata"]) / "leakage_audit.json")
        if not audit["passed"]:
            LOGGER.warning("Leakage audit has failing checks: %s",
                           [c for c in audit["checks"] if not c.get("passed")])

        ctx.prep = prep
        self._record(
            "features", method=self.method, n_features=self.n_features,
            n_input_features=int(prep.X_train.shape[1]),
            train=int(prep.X_train.shape[0]), validation=int(prep.X_val.shape[0]),
            test=int(prep.X_test.shape[0]), quantum_train=int(len(train_q_idx)),
            quantum_val=int(len(val_q_idx)), common_test=int(len(common_idx)),
            explained_variance=(
                float(prep.reducer.cumulative_variance()[-1])
                if hasattr(prep.reducer, "cumulative_variance") else None
            ),
            leakage_audit_passed=bool(audit["passed"]),
            duration_s=time.perf_counter() - t0,
        )
        return ctx

    # ------------------------------------------------------------------ load
    def load_features(self, ctx: RunContext) -> RunContext:
        from src.data.preprocessing import Preprocessor
        from src.features.pca import PCAReducer
        from src.features.scaling import FeatureScaler

        ctx.prep = type("Prep", (), {})()
        ctx.prep.preprocessor = Preprocessor.load(self._processed("preprocessor.joblib"))
        ctx.prep.reducer = joblib.load(self._processed("reducer.joblib"))
        ctx.prep.scaler = FeatureScaler.load(self._processed("scaler.joblib"))
        ctx.prep.feature_names = ctx.prep.preprocessor.feature_names
        arrays = np.load(self._processed("arrays.npz"))
        ctx.prep.X_train = arrays["X_train"]
        ctx.prep.X_val = arrays["X_val"]
        ctx.prep.X_test = arrays["X_test"]
        ctx.prep.X_test_common = arrays["X_test_common"]
        ctx.prep.y_train = arrays["y_train"]
        ctx.prep.y_val = arrays["y_val"]
        ctx.prep.y_test = arrays["y_test"]
        ctx.prep.y_common = arrays["y_common"]
        ctx.prep.common_idx = arrays["common_idx"]
        ctx.prep.train_q_idx = arrays["train_q_idx"]
        ctx.prep.val_q_idx = arrays["val_q_idx"]
        ctx.prep.method = str(load_json(
            Path(self.p["data_metadata"]) / "feature_selection.json", default={}
        ).get("selected_method", self.method))
        ctx.prep.n_features = int(ctx.prep.X_train.shape[1])
        return ctx

    # ================================================================= TRAIN
    def train(self, ctx: RunContext, models: Optional[Sequence[str]] = None,
              tag_prefix: str = "", save: bool = True) -> RunContext:
        models = list(models or self.models)
        prep = ctx.prep
        assert prep is not None
        Xtr_all, ytr_all = prep.X_train, prep.y_train
        Xva_all, yva_all = prep.X_val, prep.y_val
        Xte, yte = prep.X_test_common, prep.y_common

        qtr, qva = prep.train_q_idx, prep.val_q_idx
        Xtr_q, ytr_q = Xtr_all[qtr], ytr_all[qtr]
        Xva_q, yva_q = Xva_all[qva], yva_all[qva]

        # Draw quantum circuits once per encoding.
        self._save_circuit_artifacts(ctx)

        for kind in models:
            LOGGER.info("=" * 72)
            LOGGER.info("Training %s (%s, %d features)", kind, self.mode, prep.n_features)
            t0 = time.perf_counter()
            if kind == "cnn_lstm":
                Xtr, ytr, Xva, yva = Xtr_all, ytr_all, Xva_all, yva_all
                epochs = int(self.profile.get("epochs", 25))
            else:
                Xtr, ytr, Xva, yva = Xtr_q, ytr_q, Xva_q, yva_q
                epochs = None

            bundle = fit_model(
                kind, Xtr, ytr, Xva, yva, self.cfg,
                epochs_override=epochs, verbose=self.verbose,
            )
            bundle.info["train_size"] = int(len(Xtr))
            bundle.info["val_size"] = int(len(Xva))
            bundle.info["test_size"] = int(len(Xte))
            tune_threshold(bundle, Xva, yva, self.cfg)

            metrics = evaluate_bundle(
                bundle, Xte, yte,
                threshold=bundle.threshold,
                categories=ctx.common_test_df["attack_category"].to_numpy(),
            )
            metrics["train_size"] = int(len(Xtr))
            metrics["val_size"] = int(len(Xva))
            metrics["evaluation_set"] = "common_test"
            metrics["mode"] = self.mode

            # Supplementary: classical model on the full test split.
            if kind == "cnn_lstm":
                full = evaluate_bundle(bundle, prep.X_test, prep.y_test, threshold=bundle.threshold,
                                       categories=ctx.test_df["attack_category"].to_numpy())
                full["evaluation_set"] = "full_test"
                full["mode"] = self.mode
                save_json(full, self._metrics_path("cnn_lstm_full_test.json"))

            # Predictions.
            scores = bundle.predict_proba(Xte)
            pred_df = pd.DataFrame({
                "row_id": np.arange(len(yte)),
                "y_true": yte.astype(int),
                "y_score": scores,
                "y_pred": (scores >= bundle.threshold).astype(int),
                "attack_category": ctx.common_test_df["attack_category"].to_numpy(),
                "label": ctx.common_test_df["label"].to_numpy(),
                "threshold": bundle.threshold,
                "model": kind,
            })
            pred_df.to_csv(self._pred_path(f"{tag_prefix}{kind}_test_predictions.csv"), index=False)

            # Threshold table on validation (where the threshold was chosen).
            val_scores = bundle.predict_proba(Xva)
            ttable = threshold_analysis(
                yva, val_scores,
                thresholds=self.cfg.get_path("evaluation.thresholds", None),
                false_negative_cost=float(self.cfg.get_path("evaluation.false_negative_cost", 10.0)),
                false_positive_cost=float(self.cfg.get_path("evaluation.false_positive_cost", 1.0)),
            )
            ttable.to_csv(self._pred_path(f"{tag_prefix}{kind}_threshold_validation.csv"), index=False)
            plots.plot_threshold_analysis(
                ttable, self._fig(f"{kind}_threshold_analysis.png"),
                selected=bundle.threshold,
                title=f"{kind.upper()} threshold trade-off (validation)",
            )
            plots.plot_threshold_analysis(
                threshold_analysis(yte, scores,
                                   thresholds=self.cfg.get_path("evaluation.thresholds", None),
                                   false_negative_cost=float(self.cfg.get_path("evaluation.false_negative_cost", 10.0)),
                                   false_positive_cost=float(self.cfg.get_path("evaluation.false_positive_cost", 1.0))),
                self._fig(f"{kind}_threshold_analysis_test.png"),
                selected=bundle.threshold,
                title=f"{kind.upper()} threshold trade-off (test)",
            )

            # Figures.
            plots.plot_confusion_matrix(
                np.asarray(metrics["confusion_matrix"]), self._fig(f"{kind}_confusion_matrix.png"),
                title=f"{kind.upper()} — common test set",
            )
            plots.plot_roc_curve(yte, scores, self._fig(f"{kind}_roc_curve.png"),
                                 label=kind.upper(), color=plots.MODEL_COLORS.get(kind, "#333"))
            plots.plot_pr_curve(yte, scores, self._fig(f"{kind}_pr_curve.png"),
                                label=kind.upper(), color=plots.MODEL_COLORS.get(kind, "#333"))
            if kind == "cnn_lstm" and bundle.info.get("history"):
                plots.plot_training_history(
                    bundle.info["history"], self._fig("cnn_lstm_training_curves.png")
                )
            if kind == "vqc" and bundle.info.get("convergence"):
                plots.plot_vqc_convergence(
                    bundle.info["convergence"], self._fig("vqc_convergence.png"),
                    optimizer=bundle.resources.get("optimizer", "COBYLA"),
                )

            metrics["duration_s"] = time.perf_counter() - t0
            save_json(metrics, self._metrics_path(f"{tag_prefix}{kind}.json"))
            if save:
                paths = save_bundle(bundle, self.cfg, tag=f"{tag_prefix}{kind}")
                metrics["model_path"] = paths.get("model")
                save_json(metrics, self._metrics_path(f"{tag_prefix}{kind}.json"))

            ctx.bundles[kind] = bundle
            ctx.metrics[kind] = metrics
            ctx.resources[kind] = bundle.resources
            ctx.timings[kind] = bundle.timings
            self.logger.log(
                f"train_{kind}", model=kind, accuracy=metrics["accuracy"],
                f1=metrics["f1"], roc_auc=metrics["roc_auc"],
                threshold=bundle.threshold, duration_s=metrics["duration_s"],
            )
            LOGGER.info(
                "%s: acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f fpr=%.4f "
                "(threshold=%.2f, %.1fs)", kind, metrics["accuracy"], metrics["precision"],
                metrics["recall"], metrics["f1"], metrics["roc_auc"], metrics["fpr"],
                bundle.threshold, metrics["duration_s"],
            )

        self._record(
            "train", models=list(models),
            metrics={k: {"accuracy": v["accuracy"], "f1": v["f1"],
                         "roc_auc": v["roc_auc"]} for k, v in ctx.metrics.items()},
            duration_s=time.perf_counter() - self.start_time,
        )
        return ctx

    def _save_circuit_artifacts(self, ctx: RunContext) -> None:
        prep = ctx.prep
        out = Path(self.p["circuits"])
        out.mkdir(parents=True, exist_ok=True)
        sample = prep.X_train[prep.train_q_idx[0]]
        from src.quantum.encoding import EncodingConfig, build_encoder

        for model_key, section in (("qsvm", "quantum.qsvm"), ("vqc", "quantum.vqc")):
            enc_name = str(self.cfg.get_path(f"{section}.encoding", "angle"))
            map_name = str(self.cfg.get_path(f"{section}.feature_map", "zz"))
            reps = int(self.cfg.get_path(f"{section}.feature_map_reps", 2))
            ent = str(self.cfg.get_path(f"{section}.entanglement", "linear"))
            if enc_name == "amplitude":
                map_name, reps, ent = "amplitude", 0, "none"
            enc = EncodingConfig(
                method=enc_name, n_features=prep.n_features,
                map_name=map_name, reps=reps, entanglement=ent,
            )
            encoder = build_encoder(enc)
            if enc_name == "angle":
                resource_circuit = encoder.circuit()
            else:
                resource_circuit = encoder.circuit(sample)
            save_circuit_artifacts(resource_circuit, out, model_key, sample=None)

            if model_key == "vqc":
                from qiskit.circuit.library import real_amplitudes

                ansatz = real_amplitudes(
                    prep.n_features, reps=int(self.cfg.get_path("quantum.vqc.ansatz_reps", 2)),
                    entanglement=ent if enc_name == "angle" else "linear",
                    parameter_prefix="theta",
                )
                from qiskit import QuantumCircuit

                full = QuantumCircuit(encoder.num_qubits, name="VQC_full")
                if enc_name == "angle":
                    full.compose(encoder.circuit().assign_parameters(sample), inplace=True)
                else:
                    full.compose(encoder.circuit(sample), inplace=True)
                full.compose(ansatz, inplace=True)
                full.measure_all()
                save_circuit_artifacts(full, out, "vqc_full")

    # ============================================================== EVALUATE
    def evaluate(self, ctx: RunContext) -> RunContext:
        prep = ctx.prep
        Xte, yte = prep.X_test_common, prep.y_common

        metrics: dict[str, dict] = {}
        for kind in self.models:
            path = self._metrics_path(f"{kind}.json")
            data = load_json(path)
            if data:
                metrics[kind] = data
        ctx.metrics = metrics or ctx.metrics

        table = build_master_table(ctx.metrics, ctx.resources, ctx.timings)
        paths = save_master_table(table, self.p["tables"])
        plots.plot_metric_comparison(
            table, ["accuracy", "precision", "recall", "f1", "roc_auc", "fpr", "fnr"],
            self._fig("model_comparison.png"),
        )
        plots.plot_confusion_matrix_grid(
            {k: np.asarray(v["confusion_matrix"]) for k, v in ctx.metrics.items()},
            self._fig("confusion_matrices.png"),
        )
        curves = {}
        for kind in ctx.metrics:
            pred_path = self._pred_path(f"{kind}_test_predictions.csv")
            if pred_path.exists():
                pred = pd.read_csv(pred_path)
                curves[kind] = (pred["y_true"].to_numpy(), pred["y_score"].to_numpy())
        if curves:
            plots.plot_roc_comparison(curves, self._fig("roc_comparison.png"))
            plots.plot_pr_comparison(curves, self._fig("pr_comparison.png"))

        # Alert engine demo over the common test set.
        engine = RiskEngine(AlertPolicy.from_config(self.cfg))
        pred = pd.read_csv(self._pred_path(f"{self.models[0]}_test_predictions.csv"))
        alert_summary = summarize_alerts(
            engine, pred["y_score"].to_numpy(),
            pred["attack_category"].astype(str).tolist(),
            pred["y_pred"].to_numpy(),
        )
        save_json(alert_summary, Path(self.p["metrics"]) / "alert_summary.json")

        ctx.summary["master_table"] = table.to_dict(orient="records")
        self._record("evaluate", models=list(ctx.metrics), table=paths,
                     duration_s=time.perf_counter() - self.start_time)
        return ctx

    # ================================================================ ZERO DAY
    def run_zero_day(self, ctx: RunContext) -> pd.DataFrame:
        pool = pd.concat([ctx.train_df, ctx.val_df], ignore_index=True)
        models = [m for m in self.models]
        df, details, artifacts = run_unseen_category_experiment(
            pool, self.cfg, models=models,
            n_features=self.n_features,
            quantum_train_size=int(self.profile.get("quantum_train_size", 600)),
            verbose=self.verbose,
        )
        df.to_csv(Path(self.p["tables"]) / "unseen_category_results.csv", index=False)
        save_json(details, Path(self.p["metrics"]) / "unseen_category_details.json")
        plots.plot_zero_day_results(df, self._fig("unseen_attack_performance.png"))

        external_novel = pd.DataFrame()
        ext_details: dict[str, Any] = {}
        if bool(self.cfg.get_path("zero_day.also_evaluate_official_novel_attacks", True)):
            # E5 scores the already-trained main models at their original
            # thresholds.  The unseen-category bundles are a separate
            # experiment and must not be mixed with the main preprocessing
            # pipeline.
            main_bundles = self._load_bundles(ctx, self.models)
            external_novel, ext_details = evaluate_external_novel_attacks(
                main_bundles, ctx.prep, ctx.external_df, ctx.train_df, seed=self.seed,
                reference_types=list(
                    (load_json(
                        Path(self.p["data_metadata"]) / "dataset_validation.json", default={}
                    ).get("train_labels") or {}).keys()
                ) or None,
            )
            if not external_novel.empty:
                external_novel.to_csv(
                    Path(self.p["tables"]) / "external_novel_attack_results.csv", index=False
                )
                save_json(ext_details, Path(self.p["metrics"]) / "external_novel_details.json")

        self._record(
            "zero_day",
            holdout=details["holdout_categories"],
            results=df[["model", "recall", "precision", "f1", "roc_auc"]].to_dict(orient="records")
            if not df.empty else [],
            external_novel_types=ext_details.get("novel_types", []),
            duration_s=time.perf_counter() - self.start_time,
        )
        return df

    # ================================================================ ENCODING
    def run_encoding_experiment(self, ctx: RunContext) -> pd.DataFrame:
        prep = ctx.prep
        qtr, qva = prep.train_q_idx, prep.val_q_idx
        Xtr, ytr = prep.X_train[qtr], prep.y_train[qtr]
        Xva, yva = prep.X_val[qva], prep.y_val[qva]
        Xte, yte = prep.X_test_common, prep.y_common

        encodings = list(self.cfg.get_path("encoding_experiment.encodings", ["angle", "amplitude"]))
        model_kinds = list(self.cfg.get_path("encoding_experiment.models", ["qsvm", "vqc"]))
        maxiter = int(self.cfg.get_path("encoding_experiment.maxiter", 100))

        rows = []
        for kind in model_kinds:
            for enc in encodings:
                LOGGER.info("Encoding experiment: %s + %s", kind, enc)
                try:
                    bundle = fit_model(
                        kind, Xtr, ytr, Xva, yva, self.cfg,
                        encoding_override=enc,
                        maxiter_override=maxiter if kind == "vqc" else None,
                        verbose=False,
                    )
                    bundle.info["train_size"] = int(len(Xtr))
                    tune_threshold(bundle, Xva, yva, self.cfg)
                    metrics = evaluate_bundle(bundle, Xte, yte, threshold=bundle.threshold)
                    rows.append({
                        "model": kind,
                        "encoding": enc,
                        "n_qubits": bundle.resources.get("num_qubits"),
                        "circuit_depth": bundle.resources.get("depth"),
                        "circuit_size": bundle.resources.get("size"),
                        "num_parameters": bundle.resources.get(
                            "num_parameters", bundle.resources.get("num_ansatz_parameters")
                        ),
                        "accuracy": metrics["accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "roc_auc": metrics["roc_auc"],
                        "fpr": metrics["fpr"],
                        "fnr": metrics["fnr"],
                        "threshold": bundle.threshold,
                        "training_time_s": bundle.timings.get("training_time_s"),
                        "inference_ms_per_sample": metrics["inference_ms_per_sample"],
                        "status": "ok",
                    })
                    if enc == str(self.cfg.get_path(f"quantum.{kind}.encoding", "angle")):
                        # This is the main-experiment configuration; reuse its
                        # predictions for paired statistical tests.
                        scores = bundle.predict_proba(Xte)
                        pd.DataFrame({
                            "y_true": yte, "y_score": scores,
                            "y_pred": (scores >= bundle.threshold).astype(int),
                            "model": kind, "encoding": enc,
                        }).to_csv(
                            self._pred_path(f"encoding_{kind}_{enc}_predictions.csv"), index=False
                        )
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.error("Encoding experiment %s/%s failed: %s", kind, enc, exc)
                    rows.append({
                        "model": kind, "encoding": enc, "status": f"failed: {exc}",
                        "n_qubits": None, "circuit_depth": None,
                    })
        df = pd.DataFrame(rows)
        df.to_csv(Path(self.p["tables"]) / "encoding_comparison.csv", index=False)
        plots.plot_encoding_comparison(df, self._fig("encoding_comparison.png"))
        self._record("encoding", results=df.to_dict(orient="records"),
                     duration_s=time.perf_counter() - self.start_time)
        return df

    # ================================================================= LATENCY
    def run_latency(self, ctx: RunContext) -> pd.DataFrame:
        prep = ctx.prep
        # raw rows corresponding to the common test set, in the same order
        idx = prep.common_idx
        raw = ctx.test_df.iloc[idx].reset_index(drop=True)
        X = prep.X_test_common
        y = prep.y_common
        bundles = self._load_bundles(ctx, self.models)
        df = measure_latency(prep, list(bundles.values()), raw, X, y, self.cfg)
        df.to_csv(Path(self.p["tables"]) / "latency_comparison.csv", index=False)
        save_json(df.to_dict(orient="records"), Path(self.p["metrics"]) / "latency.json")
        plots.plot_latency_comparison(df, self._fig("latency_comparison.png"))

        # Quantum resource table.
        qrows = []
        for kind, bundle in bundles.items():
            if kind == "cnn_lstm":
                continue
            res = bundle.resources
            qrows.append({
                "model": kind,
                "model_label": kind.upper(),
                "encoding": bundle.encoding,
                "qubits": res.get("num_qubits"),
                "circuit_depth": res.get("depth"),
                "circuit_size": res.get("size"),
                "num_parameters": res.get("num_parameters", res.get("num_ansatz_parameters")),
                "num_two_qubit_gates": res.get("num_two_qubit_gates"),
                "optimizer": res.get("optimizer", "-"),
                "shots": res.get("shots"),
                "backend": res.get("backend"),
                "quantum_training_time_s": bundle.timings.get("training_time_s"),
                "quantum_inference_ms_per_sample": float(
                    df.loc[df["model"] == kind, "inference_ms_per_sample"].iloc[0]
                ) if (df["model"] == kind).any() else None,
            })
        qdf = pd.DataFrame(qrows)
        if not qdf.empty:
            qdf.to_csv(Path(self.p["tables"]) / "quantum_resources.csv", index=False)
            save_json(qdf.to_dict(orient="records"), Path(self.p["metrics"]) / "quantum_resources.json")
            plots.plot_quantum_resources(qdf, self._fig("quantum_resources.png"))

        self._record("latency", results=df.to_dict(orient="records"),
                     duration_s=time.perf_counter() - self.start_time)
        return df

    # =================================================================== DRIFT
    def run_drift(self, ctx: RunContext) -> pd.DataFrame:
        prep = ctx.prep
        bundles = self._load_bundles(ctx, ["cnn_lstm"])
        if not bundles:
            return pd.DataFrame()
        bundle = bundles["cnn_lstm"]
        X = prep.X_test_common
        y = prep.y_common

        frames = [
            drift_mod.noise_shift_experiment(
                bundle, X, y, prep.X_train,
                self.cfg.get_path("drift.noise_levels", [0.0, 0.1, 0.25, 0.5, 1.0]),
                seed=self.seed,
            ),
            drift_mod.permutation_shift_experiment(
                bundle, X, y, prep.X_train,
                self.cfg.get_path("drift.permutation_fractions", [0.0, 0.1, 0.25, 0.5]),
                seed=self.seed + 1,
            ),
        ]
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(Path(self.p["tables"]) / "drift_controlled.csv", index=False)

        for shift_type, group in df.groupby("shift_type"):
            plots.plot_drift_curve(
                group.sort_values("severity"), self._fig(f"drift_{shift_type}.png"),
                title=f"Performance under {shift_type.replace('_', ' ')}",
            )

        # Simulated ordered windows on the external KDDTest+ file.
        X_ext = prep.preprocessor.transform(ctx.external_df)
        X_ext = prep.scaler.transform(prep.reducer.transform(X_ext))
        y_ext = ctx.external_df["is_attack"].to_numpy()
        windows = drift_mod.ordered_window_experiment(
            bundle, X_ext, y_ext, windows=int(self.cfg.get_path("drift.windows", 5))
        )
        windows.to_csv(Path(self.p["tables"]) / "drift_windows_external.csv", index=False)
        if not windows.empty:
            plots.plot_drift_curve(
                windows, self._fig("drift_simulated_windows.png"),
                x_col="window",
                title="Simulated ordered windows on KDDTest+ (no real timestamps)",
            )

        summary = drift_mod.drift_summary(df)
        summary.to_csv(Path(self.p["tables"]) / "drift_summary.csv", index=False)
        save_json(
            {
                "controlled": df.to_dict(orient="records"),
                "summary": summary.to_dict(orient="records"),
                "windows": windows.to_dict(orient="records"),
                "scope": (
                    "Controlled covariate shift and a simulated file-order window "
                    "scan. NSL-KDD has no timestamps; this is not production "
                    "concept-drift detection."
                ),
            },
            Path(self.p["metrics"]) / "drift.json",
        )
        self._record("drift", summary=summary.to_dict(orient="records"),
                     duration_s=time.perf_counter() - self.start_time)
        return df

    # ================================================================= EXPLAIN
    def run_explainability(self, ctx: RunContext) -> dict[str, Any]:
        prep = ctx.prep
        bundles = self._load_bundles(ctx, self.models)
        out_dir = Path(self.p["results"]) / "explainability"
        out_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, Any] = {}

        names = [f"PC{i + 1}" for i in range(prep.n_features)] if self.method == "pca" \
            else [f"F{i}" for i in range(prep.n_features)]

        if "cnn_lstm" in bundles:
            bundle = bundles["cnn_lstm"]
            LOGGER.info("Permutation importance (CNN+LSTM)")
            imp = permutation_importance(
                bundle.predict_proba, prep.X_val, prep.y_val, names,
                repeats=int(self.cfg.get_path("explainability.permutation_repeats", 10)),
                seed=self.seed,
                max_samples=int(self.cfg.get_path("explainability.permutation_max_samples", 500)),
            )
            imp.to_csv(out_dir / "permutation_importance.csv", index=False)
            plots.plot_feature_importance(
                imp["feature"], imp["importance_mean"], self._fig("permutation_importance.png"),
                title="CNN+LSTM permutation importance (validation, ROC-AUC drop)",
                xlabel="Mean ROC-AUC drop",
            )
            results["permutation_importance"] = imp.to_dict(orient="records")

            if self.method == "pca" and hasattr(prep.reducer, "loadings"):
                attrib = pca_attribution(prep.reducer, imp["importance_mean"].to_numpy(), top_k=20)
                attrib.to_csv(out_dir / "pca_feature_attribution.csv", index=False)
                plots.plot_feature_importance(
                    attrib["original_feature"], attrib["attributed_importance"],
                    self._fig("pca_feature_attribution.png"),
                    title="PCA loading attribution of CNN+LSTM importance",
                    xlabel="Attributed importance",
                )
                results["pca_attribution"] = attrib.to_dict(orient="records")

            if bool(self.cfg.get_path("explainability.shap_enabled", True)):
                LOGGER.info("SHAP explanation (CNN+LSTM)")
                results["shap"] = shap_explain_cnn(
                    bundle.model, prep.X_val, prep.X_test_common, names, out_dir,
                    max_background=int(self.cfg.get_path("explainability.shap_background", 50)),
                    max_samples=int(self.cfg.get_path("explainability.shap_samples", 100)),
                    seed=self.seed,
                )

        for kind in ("qsvm", "vqc"):
            if kind not in bundles:
                continue
            bundle = bundles[kind]
            LOGGER.info("Quantum feature sensitivity (%s)", kind)
            sens = quantum_feature_sensitivity(
                bundle.predict_proba, prep.X_test_common, names,
                max_samples=int(self.cfg.get_path("explainability.quantum_sensitivity_samples", 100)),
                seed=self.seed,
            )
            sens.to_csv(out_dir / f"{kind}_feature_sensitivity.csv", index=False)
            plots.plot_feature_importance(
                sens["feature"], sens["mean_abs_probability_change"],
                self._fig(f"{kind}_feature_sensitivity.png"),
                title=f"{kind.upper()} input-feature sensitivity (|ΔP| for Δx = 0.25 rad)",
                xlabel="Mean |Δ p(attack)|",
            )
            results[f"{kind}_sensitivity"] = sens.to_dict(orient="records")

            example = example_explanation(
                bundle.predict_proba, prep.X_test_common[0], names
            )
            example.to_csv(out_dir / f"{kind}_example_prediction.csv", index=False)

        results["scope_note"] = (
            "Permutation importance and PCA attribution are post-hoc, "
            "model-agnostic explanations of the classical pipeline. Quantum "
            "results are empirical input-sensitivity analyses of the encoded "
            "circuit, not Shapley attributions or 'quantum explainability'."
        )
        save_json(results, out_dir / "explainability_summary.json")
        self._record("explain", keys=list(results.keys()),
                     duration_s=time.perf_counter() - self.start_time)
        return results

    # ====================================================================== CV
    def run_cv(self, ctx: RunContext) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        pool = pd.concat([ctx.train_df, ctx.val_df], ignore_index=True)
        folds = int(self.profile.get("cv_folds", 3))
        cv_df, summary, details = run_cross_validation(
            pool, self.cfg,
            n_features=self.n_features, method=self.method,
            models=self.models, folds=folds,
            max_samples=int(self.profile.get("cv_max_samples", 3000)),
            quantum_train_size=int(self.profile.get("cv_quantum_train_size", 300)),
            seed=self.seed, verbose=self.verbose,
        )
        cv_df.to_csv(Path(self.p["tables"]) / "cross_validation_raw.csv", index=False)
        pd.DataFrame(summary).to_csv(Path(self.p["tables"]) / "cross_validation_summary.csv", index=False)
        save_json(details, Path(self.p["metrics"]) / "cross_validation_details.json")
        plots.plot_cv_boxplot(cv_df, self._fig("cross_validation_f1.png"))
        self._record("cv", folds=details["folds_completed"],
                     summary=summary, duration_s=time.perf_counter() - self.start_time)
        return cv_df, summary

    # ==================================================================== STATS
    def run_statistics(self, ctx: RunContext) -> dict[str, Any]:
        preds = {}
        for kind in self.models:
            path = self._pred_path(f"{kind}_test_predictions.csv")
            if path.exists():
                preds[kind] = pd.read_csv(path)
        if len(preds) < 2:
            LOGGER.warning("Not enough prediction files for statistical tests")
            return {}

        from sklearn.metrics import f1_score, roc_auc_score

        y_true = None
        results: dict[str, Any] = {"mcnemar": [], "bootstrap": [], "paired_bootstrap": []}
        keys = [k for k in self.models if k in preds]
        for k in keys:
            if y_true is None:
                y_true = preds[k]["y_true"].to_numpy()
            assert np.array_equal(y_true, preds[k]["y_true"].to_numpy()), "test sets differ"

        # Each model keeps the threshold selected on its own validation split.
        thresholds = {}
        for k in keys:
            col = preds[k].get("threshold")
            thresholds[k] = (
                float(col.dropna().iloc[0]) if col is not None and col.notna().any() else 0.5
            )

        def _f1_at(threshold: float):
            return lambda yt, ys: f1_score(
                yt, (ys >= threshold).astype(int), zero_division=0
            )

        def _acc_at(threshold: float):
            return lambda yt, ys: float(np.mean((ys >= threshold).astype(int) == yt))

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                mcn = mcnemar_test(
                    y_true, preds[a]["y_pred"].to_numpy(), preds[b]["y_pred"].to_numpy(),
                    alpha=float(self.cfg.get_path("evaluation.alpha", 0.05)),
                    name_a=a, name_b=b,
                )
                results["mcnemar"].append(mcn)

                # F1 difference (threshold-dependent) and AUC difference.
                pb_f1 = paired_bootstrap_difference(
                    y_true, preds[a]["y_score"].to_numpy(), preds[b]["y_score"].to_numpy(),
                    _f1_at(thresholds[a]),
                    n_boot=int(self.cfg.get_path("evaluation.bootstrap_samples", 1000)),
                    alpha=float(self.cfg.get_path("evaluation.alpha", 0.05)),
                    seed=self.seed,
                    metric_fn_b=_f1_at(thresholds[b]),
                )
                pb_f1.update({
                    "comparison": f"{a} vs {b}",
                    "metric": "f1@validation_threshold",
                    "threshold_a": thresholds[a],
                    "threshold_b": thresholds[b],
                })
                results["paired_bootstrap"].append(pb_f1)

                pb_auc = paired_bootstrap_difference(
                    y_true, preds[a]["y_score"].to_numpy(), preds[b]["y_score"].to_numpy(),
                    lambda yt, ys: roc_auc_score(yt, ys),
                    n_boot=int(self.cfg.get_path("evaluation.bootstrap_samples", 1000)),
                    alpha=float(self.cfg.get_path("evaluation.alpha", 0.05)),
                    seed=self.seed,
                )
                pb_auc.update({"comparison": f"{a} vs {b}", "metric": "roc_auc"})
                results["paired_bootstrap"].append(pb_auc)

        for k in keys:
            p = preds[k]
            for metric_name, fn in (
                ("accuracy", _acc_at(thresholds[k])),
                ("f1", _f1_at(thresholds[k])),
                ("roc_auc", lambda yt, ys: roc_auc_score(yt, ys)),
            ):
                ci = bootstrap_metric_ci(
                    y_true, p["y_score"].to_numpy(), fn,
                    n_boot=int(self.cfg.get_path("evaluation.bootstrap_samples", 1000)),
                    alpha=float(self.cfg.get_path("evaluation.alpha", 0.05)),
                    seed=self.seed,
                )
                ci.update({
                    "model": k,
                    "metric": metric_name,
                    "threshold": thresholds[k] if metric_name != "roc_auc" else None,
                })
                results["bootstrap"].append(ci)

        # Paired Wilcoxon across folds, if CV results exist.
        cv_path = Path(self.p["tables"]) / "cross_validation_raw.csv"
        results["wilcoxon"] = []
        if cv_path.exists():
            cv_df = pd.read_csv(cv_path)
            pivot = paired_fold_scores(cv_df, "f1")
            ids = [k for k in keys if k in pivot.columns]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    results["wilcoxon"].append(
                        wilcoxon_test(
                            pivot[a].to_numpy(), pivot[b].to_numpy(),
                            alpha=float(self.cfg.get_path("evaluation.alpha", 0.05)),
                            name_a=a, name_b=b, metric="f1",
                        )
                    )

        save_json(results, Path(self.p["metrics"]) / "statistical_tests.json")
        self._record("stats",
                     mcnemar=[{k: v for k, v in r.items() if k in
                               ("comparison", "p_value", "significant", "effect_size_cohens_h")}
                              for r in results["mcnemar"]],
                     duration_s=time.perf_counter() - self.start_time)
        return results

    # =================================================================== TABLES
    def build_tables(self, ctx: RunContext) -> dict[str, Any]:
        ctx = self.load_features(ctx) if ctx.prep is None else ctx
        # Reload metrics if not in memory.
        if not ctx.metrics:
            for kind in self.models:
                data = load_json(self._metrics_path(f"{kind}.json"))
                if data:
                    ctx.metrics[kind] = data
        if not ctx.resources:
            for kind in self.models:
                meta = load_json(Path(self.p["models"]) / f"{kind}_meta.json")
                if meta:
                    ctx.resources[kind] = meta.get("resources", {})
                    ctx.timings[kind] = meta.get("timings", {})

        table = build_master_table(ctx.metrics, ctx.resources, ctx.timings)
        save_master_table(table, self.p["tables"])
        plots.plot_metric_comparison(
            table, ["accuracy", "precision", "recall", "f1", "roc_auc", "fpr", "fnr"],
            self._fig("model_comparison.png"),
        )
        ctx.summary["master_table"] = table.to_dict(orient="records")
        self._record("tables", rows=len(table), duration_s=time.perf_counter() - self.start_time)
        return ctx.summary

    # ================================================================== LOADERS
    def _load_bundles(self, ctx: RunContext, kinds: Iterable[str]) -> dict[str, Any]:
        bundles = {}
        for kind in kinds:
            if kind in ctx.bundles:
                bundles[kind] = ctx.bundles[kind]
                continue
            path = Path(self.p["models"]) / (
                f"{kind}.pt" if kind == "cnn_lstm" else f"{kind}.joblib"
            )
            if not path.exists():
                LOGGER.warning("Model artifact missing for %s: %s", kind, path)
                continue
            meta = load_json(Path(self.p["models"]) / f"{kind}_meta.json", default={}) or {}
            if kind == "cnn_lstm":
                model, info = load_cnn_lstm(path)
                bundle = type("Bundle", (), {})()
                bundle.name = kind
                bundle.kind = kind
                bundle.model = model
                bundle.encoding = "none"
                bundle.feature_method = meta.get("feature_method", self.method)
                bundle.n_features = int(meta.get("n_features", self.n_features))
                bundle.resources = meta.get("resources", {})
                bundle.timings = meta.get("timings", {})
                bundle.info = meta.get("info", info)
                bundle.threshold = float(meta.get("threshold", 0.5))
                bundle.predict_proba = lambda X, m=model: __import__(
                    "src.classical.cnn_lstm", fromlist=["predict_proba"]
                ).predict_proba(m, X)
            else:
                model = QuantumSVM.load(path) if kind == "qsvm" else VariationalQuantumClassifier.load(path)
                bundle = type("Bundle", (), {})()
                bundle.name = kind
                bundle.kind = kind
                bundle.model = model
                bundle.encoding = meta.get("encoding", getattr(model, "encoding", None) and model.encoding.method)
                bundle.feature_method = meta.get("feature_method", self.method)
                bundle.n_features = int(meta.get("n_features", self.n_features))
                bundle.resources = meta.get("resources", {})
                bundle.timings = meta.get("timings", model.timings_)
                bundle.info = meta.get("info", {})
                bundle.threshold = float(meta.get("threshold", 0.5))
                bundle.predict_proba = model.predict_proba
            bundles[kind] = bundle
        return bundles

    # ===================================================================== RUN
    def run(
        self,
        stages: Optional[Sequence[str]] = None,
        experiments: Optional[Sequence[str]] = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Execute the requested stages in dependency order."""
        stages = list(stages) if stages else [
            "prepare", "features", "train", "evaluate", "cv", "stats",
            "zero_day", "encoding", "latency", "drift", "explain",
            "tables", "dashboard", "report",
        ]
        unknown = [s for s in stages if s not in STAGES and s != "all"]
        if unknown:
            raise ValueError(f"Unknown stage(s) {unknown}. Valid: {STAGES}")
        if "all" in stages:
            stages = [s for s in STAGES if s not in ("dashboard", "report")]
        experiments = set(experiments or [])

        LOGGER.info("Run start | mode=%s | models=%s | stages=%s",
                    self.mode, self.models, stages)
        LOGGER.info("Config: %s", self.cfg.get_path("project.name"))
        LOGGER.info("Results dir: %s", self.p["results"])

        ctx: Optional[RunContext] = None
        outputs: dict[str, Any] = {}

        # ---- data ---------------------------------------------------------
        if "prepare" in stages:
            ctx = self.prepare(force=force)
        elif any(s in stages for s in ("features", "train", "evaluate", "cv", "stats",
                                       "zero_day", "encoding", "latency", "drift",
                                       "explain", "tables")):
            ctx = self.load_prepared()

        if "features" in stages and ctx is not None:
            ctx = self.prepare_features(ctx)
        elif ctx is not None and any(s in stages for s in (
            "train", "evaluate", "cv", "stats", "zero_day", "encoding",
            "latency", "drift", "explain", "tables",
        )):
            ctx = self.load_features(ctx)

        if "train" in stages and ctx is not None:
            ctx = self.train(ctx)

        if "evaluate" in stages and ctx is not None:
            ctx = self.evaluate(ctx)

        if "cv" in stages and ctx is not None:
            cv_df, cv_summary = self.run_cv(ctx)
            outputs["cross_validation"] = cv_summary

        if "stats" in stages and ctx is not None:
            outputs["statistics"] = self.run_statistics(ctx)

        if "zero_day" in stages and ctx is not None:
            outputs["zero_day"] = self.run_zero_day(ctx).to_dict(orient="records")

        if "encoding" in stages and ctx is not None:
            outputs["encoding"] = self.run_encoding_experiment(ctx).to_dict(orient="records")

        if "latency" in stages and ctx is not None:
            outputs["latency"] = self.run_latency(ctx).to_dict(orient="records")

        if "drift" in stages and ctx is not None:
            outputs["drift"] = self.run_drift(ctx).to_dict(orient="records")

        if "explain" in stages and ctx is not None:
            outputs["explainability"] = self.run_explainability(ctx)

        if "tables" in stages and ctx is not None:
            outputs["tables"] = self.build_tables(ctx)

        # ---- reporting (works from disk, no model objects required) -------
        if any(s in stages for s in ("tables", "dashboard", "report")):
            from src.pipeline.reporting import generate_reports

            if "dashboard" in stages or "report" in stages:
                reports = generate_reports(self.cfg)
                outputs["reports"] = reports

        elapsed = time.perf_counter() - self.start_time
        self._record("run_complete", stages=stages, duration_s=elapsed,
                     duration_human=human_duration(elapsed))
        LOGGER.info("Run complete in %s", human_duration(elapsed))
        outputs["elapsed_s"] = elapsed
        outputs["manifest"] = self.manifest
        return outputs
