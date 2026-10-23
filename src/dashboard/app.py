"""Interactive Streamlit dashboard for QE-IDS results.

Run with::

    streamlit run src/dashboard/app.py

The dashboard is read-only: it never retrains a model and never touches the
test set.  It loads the artefacts produced by ``run_pipeline.py`` so the same
numbers shown here are the numbers stored in ``results/``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alerting.risk_engine import AlertPolicy, RiskEngine  # noqa: E402
from src.utils.config import load_config  # noqa: E402

MODEL_LABELS = {"cnn_lstm": "CNN+LSTM", "qsvm": "QSVM", "vqc": "VQC"}
LEVEL_ORDER = ["NORMAL", "MONITOR", "ALERT", "QUARANTINE", "ESCALATE"]


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _json(path: str, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def _csv(path: str):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _img(path: Path, caption: str | None = None) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure not found: {path.name}. Run the pipeline first.")


def _metric_row(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    cols = [c for c in ("model", "encoding", metric) if c in df.columns]
    out = df[cols].copy()
    if "model" in out.columns:
        out["model"] = out["model"].map(lambda m: MODEL_LABELS.get(m, m))
    return out


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_overview(cfg, summary: dict) -> None:
    st.title("QE-IDS — Quantum-Enhanced Intrusion Detection")
    st.caption(
        "Classical CNN+LSTM vs QSVM vs VQC on leakage-controlled, PCA-reduced "
        "NSL-KDD. All quantum results are statevector simulations."
    )

    master = pd.DataFrame(summary.get("master_table", []))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mode", str(summary.get("mode", "?")).upper())
    c2.metric("Seed", summary.get("seed", "?"))
    c3.metric("Reduced features", summary.get("feature_selection", {}).get("n_features", "?"))
    audit = summary.get("leakage_audit_passed")
    c4.metric("Leakage audit", "PASS" if audit else "CHECK")

    findings = summary.get("findings", {}) or {}
    st.subheader("Main finding")
    st.info(findings.get("quantum_comparison", "Run the pipeline to generate findings."))
    for note in findings.get("notes", []):
        st.write(f"- {note}")

    st.subheader("Model comparison (common test set)")
    if not master.empty:
        show = master.copy()
        if "model" in show.columns:
            show["model"] = show["model"].map(lambda m: MODEL_LABELS.get(m, m))
        cols = [c for c in (
            "model", "encoding", "accuracy", "precision", "recall", "f1",
            "roc_auc", "fpr", "fnr", "training_time_s", "inference_ms_per_sample",
            "qubits", "circuit_depth",
        ) if c in show.columns]
        st.dataframe(
            show[cols].style.format({
                "accuracy": "{:.4f}", "precision": "{:.4f}", "recall": "{:.4f}",
                "f1": "{:.4f}", "roc_auc": "{:.4f}", "fpr": "{:.4f}",
                "fnr": "{:.4f}", "training_time_s": "{:.2f}",
                "inference_ms_per_sample": "{:.4f}",
            }),
            use_container_width=True,
        )
    else:
        st.warning("No master table found. Run `python run_pipeline.py --mode quick`.")

    _img(ROOT / "results/figures/model_comparison.png", "Metric comparison")
    _img(ROOT / "results/figures/roc_comparison.png", "ROC curves")


def page_models(cfg, summary: dict) -> None:
    st.header("Model diagnostics")
    master = pd.DataFrame(summary.get("master_table", []))
    if not master.empty:
        st.subheader("Confusion matrices")
        _img(ROOT / "results/figures/confusion_matrices.png")

        st.subheader("Per-model curves")
        model = st.selectbox(
            "Model", [m for m in ("cnn_lstm", "qsvm", "vqc")
                      if (ROOT / f"results/figures/{m}_roc_curve.png").exists()],
            format_func=lambda m: MODEL_LABELS.get(m, m),
        )
        if model:
            a, b = st.columns(2)
            with a:
                _img(ROOT / f"results/figures/{model}_roc_curve.png", "ROC")
                _img(ROOT / f"results/figures/{model}_confusion_matrix.png", "Confusion matrix")
            with b:
                _img(ROOT / f"results/figures/{model}_pr_curve.png", "Precision–recall")
                _img(ROOT / f"results/figures/{model}_threshold_analysis.png",
                     "Validation threshold sweep")
            if model == "cnn_lstm":
                _img(ROOT / "results/figures/cnn_lstm_training_curves.png", "Training curves")
            if model == "vqc":
                _img(ROOT / "results/figures/vqc_convergence.png", "VQC convergence")

        st.subheader("Cross-validation")
        cv = pd.DataFrame(summary.get("cross_validation", []))
        f1 = cv[cv["metric"] == "f1"] if "metric" in cv.columns else cv
        if not f1.empty:
            st.dataframe(f1, use_container_width=True)
            _img(ROOT / "results/figures/cross_validation_f1.png")

        st.subheader("Paired statistical tests")
        stats = summary.get("statistics", {}) or {}
        if stats.get("mcnemar"):
            st.dataframe(pd.DataFrame(stats["mcnemar"]), use_container_width=True)
        if stats.get("wilcoxon"):
            st.dataframe(pd.DataFrame(stats["wilcoxon"]), use_container_width=True)
        if stats.get("bootstrap"):
            st.dataframe(pd.DataFrame(stats["bootstrap"]), use_container_width=True)


def page_quantum(cfg, summary: dict) -> None:
    st.header("Quantum experiments")
    st.warning(
        "Simulation only: exact Qiskit Statevector evaluation, no quantum "
        "hardware, no shot noise, and no claim of quantum advantage."
    )

    resources = pd.DataFrame(summary.get("quantum_resources", []))
    if not resources.empty:
        st.subheader("Circuit resources")
        st.dataframe(resources, use_container_width=True)
        _img(ROOT / "results/figures/quantum_resources.png")

    st.subheader("Encoding comparison")
    enc = pd.DataFrame(summary.get("encoding_comparison", []))
    if not enc.empty:
        st.dataframe(enc, use_container_width=True)
        _img(ROOT / "results/figures/encoding_comparison.png")
    else:
        st.info("Encoding experiment not run.")

    st.subheader("Circuit diagram (VQC, example sample)")
    circuit_path = ROOT / "results/circuits/vqc_full_circuit.txt"
    if circuit_path.exists():
        st.code(circuit_path.read_text(encoding="utf-8")[:12000], language="text")
    else:
        st.info("Circuit text not found.")


def page_robustness(cfg, summary: dict) -> None:
    st.header("Robustness and systems analysis")

    st.subheader("Unseen attack-category generalisation")
    st.caption(
        "R2L and U2R are removed from training. This measures unseen-category "
        "generalisation, not zero-day detection. The hold-out test is balanced "
        "(equal normal and attack rows), so F1 rewards recall."
    )
    unseen = pd.DataFrame(summary.get("unseen_category", []))
    if not unseen.empty:
        st.dataframe(unseen, use_container_width=True)
    _img(ROOT / "results/figures/unseen_attack_performance.png")

    ext = pd.DataFrame(summary.get("external_novel", []))
    if not ext.empty:
        st.subheader("External novel attack types (KDDTest+)")
        st.dataframe(ext, use_container_width=True)

    st.subheader("Latency")
    latency = pd.DataFrame(summary.get("latency", []))
    if not latency.empty:
        st.dataframe(latency, use_container_width=True)
        _img(ROOT / "results/figures/latency_comparison.png")

    st.subheader("Controlled distribution shift (classical baseline)")
    drift = pd.DataFrame(summary.get("drift_summary", []))
    if not drift.empty:
        st.dataframe(drift, use_container_width=True)
        _img(ROOT / "results/figures/drift_gaussian_noise.png")
        _img(ROOT / "results/figures/drift_feature_permutation.png")
        _img(ROOT / "results/figures/drift_simulated_windows.png")


def page_alerts(cfg, summary: dict) -> None:
    st.header("Tiered alert triage")
    st.caption(
        "Assistive policy: only p(attack) ≥ the ESCALATE band maps to a blocking "
        "action. Thresholds come from config/config.yaml."
    )
    policy = AlertPolicy.from_config(cfg)
    engine = RiskEngine(policy)

    pred_path = ROOT / "results/predictions/cnn_lstm_test_predictions.csv"
    if pred_path.exists():
        preds = pd.read_csv(pred_path)
        probs = preds["y_score"].to_numpy()
        cats = preds["attack_category"].astype(str).to_numpy()
        levels = [engine.assess(p, c)["risk_level"] for p, c in zip(probs, cats)]
        dist = {lvl: levels.count(lvl) for lvl in LEVEL_ORDER}

        a, b = st.columns(2)
        with a:
            st.subheader("Tier distribution (CNN+LSTM, test set)")
            st.bar_chart(pd.Series(dist, name="samples"))
        with b:
            st.subheader("Policy bands")
            st.json(policy.to_dict())

        st.subheader("Highest-risk detections")
        preds = preds.copy()
        preds["risk_level"] = levels
        preds["risk_score"] = preds["y_score"] * 100.0
        top = preds.sort_values("y_score", ascending=False).head(25)
        st.dataframe(
            top[["y_true", "y_score", "y_pred", "attack_category", "risk_level"]],
            use_container_width=True,
        )
    else:
        st.info("Prediction file not found. Run the evaluation stage first.")

    st.subheader("Interactive triage")
    p = st.slider("p(attack)", 0.0, 1.0, 0.72, 0.01)
    category = st.selectbox("Attack category hint", ["", "DoS", "Probe", "R2L", "U2R"])
    st.json(engine.assess(p, category or None))


def page_explain(cfg, summary: dict) -> None:
    st.header("Explainability")
    st.caption(
        "CNN+LSTM importances are computed in the reduced PCA space; they are "
        "not raw-packet attributions. Quantum sensitivity is a finite-difference "
        "probe of the encoded circuit, not a causal explanation."
    )
    base = ROOT / "results/explainability"
    perm = base / "permutation_importance.csv"
    pca = base / "pca_feature_attribution.csv"
    shap_df = base / "shap_importance.csv"
    a, b = st.columns(2)
    with a:
        if perm.exists():
            st.subheader("Permutation importance")
            st.dataframe(pd.read_csv(perm), use_container_width=True)
        _img(ROOT / "results/figures/permutation_importance.png")
        if shap_df.exists():
            st.subheader("SHAP (reduced features)")
            st.dataframe(pd.read_csv(shap_df), use_container_width=True)
    with b:
        if pca.exists():
            st.subheader("PCA attribution")
            st.dataframe(pd.read_csv(pca), use_container_width=True)
        _img(ROOT / "results/figures/pca_feature_attribution.png")

    for name in ("qsvm", "vqc"):
        path = base / f"{name}_feature_sensitivity.csv"
        if path.exists():
            st.subheader(f"{MODEL_LABELS[name]} feature sensitivity")
            st.dataframe(pd.read_csv(path), use_container_width=True)
            _img(ROOT / f"results/figures/{name}_feature_sensitivity.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="QE-IDS Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    cfg = load_config(root=ROOT)
    summary_path = ROOT / "results/final_summary.json"
    summary = _json(str(summary_path), default={}) or {}

    st.sidebar.title("QE-IDS")
    st.sidebar.caption(f"Mode: `{summary.get('mode', 'not run')}`")
    page = st.sidebar.radio(
        "Section",
        ["Overview", "Models", "Quantum", "Robustness", "Alert triage", "Explainability"],
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        "**Scientific scope**\n\n"
        "- Statevector simulation only\n"
        "- No hardware execution\n"
        "- No production deployment\n"
        "- Unseen category ≠ zero-day"
    )
    if not summary:
        st.error(
            "No `results/final_summary.json` found. Run "
            "`python run_pipeline.py --mode quick` first."
        )
        st.stop()

    if page == "Overview":
        page_overview(cfg, summary)
    elif page == "Models":
        page_models(cfg, summary)
    elif page == "Quantum":
        page_quantum(cfg, summary)
    elif page == "Robustness":
        page_robustness(cfg, summary)
    elif page == "Alert triage":
        page_alerts(cfg, summary)
    else:
        page_explain(cfg, summary)


if __name__ == "__main__":
    main()
