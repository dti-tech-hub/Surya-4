"""Result aggregation, final summary, paper draft, dashboard and presentation assets."""
from __future__ import annotations

import base64
import html
import json
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.io import load_json, save_json
from src.utils.logging import get_logger

LOGGER = get_logger("qeids.report")

MODEL_LABELS = {"cnn_lstm": "CNN+LSTM", "qsvm": "QSVM", "vqc": "VQC"}


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------
def collect_results(cfg) -> dict[str, Any]:
    p = {k: Path(cfg.get_path(f"paths.{k}")) for k in (
        "results", "metrics", "tables", "figures", "predictions", "data_metadata",
        "data_splits", "models", "presentation",
    )}

    def j(name: str, default=None):
        return load_json(p["metrics"] / name, default=default)

    def csv(name: str):
        path = p["tables"] / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    manifest = load_json(p["results"] / "run_manifest.json", default={}) or {}
    env = load_json(p["results"] / "environment.json", default={}) or {}
    master = csv("master_comparison.csv")
    class_dist = load_json(Path(cfg.get_path("paths.data_metadata")) / "class_distribution.json", default={})
    selection = load_json(Path(cfg.get_path("paths.data_metadata")) / "feature_selection.json", default={})
    leakage = load_json(Path(cfg.get_path("paths.data_metadata")) / "leakage_audit.json", default={})

    summary = {
        "project": cfg.get_path("project.name", "Quantum-Enhanced Intrusion Detection System"),
        "mode": cfg.get_path("experiment.mode", "quick"),
        "seed": cfg.get_path("project.seed", 42),
        "dataset": cfg.get_path("dataset.name", "nsl-kdd"),
        "dataset_validation": load_json(
            Path(cfg.get_path("paths.data_metadata")) / "dataset_validation.json", default={}
        ),
        "class_distribution": class_dist,
        "feature_selection": {
            "method": selection.get("selected_method"),
            "n_features": selection.get("selected_n_features"),
            "policy": selection.get("policy"),
            "ablation": selection.get("ablation", []),
        },
        "leakage_audit_passed": leakage.get("passed"),
        "master_table": master.to_dict(orient="records") if not master.empty else [],
        "cross_validation": csv("cross_validation_summary.csv").to_dict(orient="records"),
        "unseen_category": csv("unseen_category_results.csv").to_dict(orient="records"),
        "external_novel": csv("external_novel_attack_results.csv").to_dict(orient="records"),
        "encoding_comparison": csv("encoding_comparison.csv").to_dict(orient="records"),
        "latency": csv("latency_comparison.csv").to_dict(orient="records"),
        "quantum_resources": csv("quantum_resources.csv").to_dict(orient="records"),
        "drift_summary": csv("drift_summary.csv").to_dict(orient="records"),
        "statistics": j("statistical_tests.json", default={}),
        "unseen_details": j("unseen_category_details.json", default={}),
        "external_novel_details": j("external_novel_details.json", default={}),
        "alert_summary": j("alert_summary.json", default={}),
        "environment": env,
        "manifest": manifest,
    }
    return summary


def _fmt(value: Any, digits: int = 4, pct: bool = False) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    if pct:
        return f"{v * 100:.2f}%"
    return f"{v:.{digits}f}"


def _find_best(summary: dict, model: str) -> Optional[dict]:
    for row in summary.get("master_table", []):
        if row.get("model") == model:
            return row
    return None


def _best_overall(summary: dict) -> Optional[dict]:
    rows = [r for r in summary.get("master_table", []) if r.get("f1") is not None]
    if not rows:
        return None
    return max(rows, key=lambda r: (r.get("f1") or 0, r.get("roc_auc") or 0))


def derive_findings(summary: dict) -> dict[str, Any]:
    """Data-driven narrative: no claimed advantage without evidence."""
    best = _best_overall(summary)
    cnn = _find_best(summary, "cnn_lstm")
    qsvm = _find_best(summary, "qsvm")
    vqc = _find_best(summary, "vqc")
    stats = summary.get("statistics", {}) or {}
    mcnemar = stats.get("mcnemar", []) or []

    def sig(pair: str) -> Optional[dict]:
        for r in mcnemar:
            if r.get("comparison") == pair:
                return r
        return None

    quantum_models = [r for r in (qsvm, vqc) if r]
    best_quantum = max(quantum_models, key=lambda r: r.get("f1") or 0) if quantum_models else None

    findings: dict[str, Any] = {
        "best_overall": best,
        "best_classical": cnn,
        "best_quantum": best_quantum,
        "quantum_advantage_claim": False,
        "quantum_comparison": "",
        "notes": [],
    }

    if not best or not cnn:
        findings["quantum_comparison"] = "Insufficient results to compare models."
        return findings

    if best_quantum:
        delta_f1 = (best_quantum.get("f1") or 0) - (cnn.get("f1") or 0)
        delta_auc = (best_quantum.get("roc_auc") or 0) - (cnn.get("roc_auc") or 0)
        pair = f"cnn_lstm vs {best_quantum['model']}"
        mc = sig(pair)
        p_txt = f"McNemar p = {mc['p_value']:.4g}" if mc else "McNemar not available"
        if delta_f1 > 0 and mc and mc.get("p_value", 1) < 0.05:
            findings["quantum_advantage_claim"] = True
            findings["quantum_comparison"] = (
                f"{MODEL_LABELS[best_quantum['model']]} exceeds the classical baseline "
                f"by {delta_f1 * 100:.2f} F1 points (ΔROC-AUC {delta_auc:+.4f}) "
                f"and the paired McNemar test is significant ({p_txt}). "
                "This is a statistically detectable difference on this benchmark, "
                "not proof of a general quantum advantage."
            )
        elif delta_f1 > 0:
            findings["quantum_comparison"] = (
                f"{MODEL_LABELS[best_quantum['model']]} is {delta_f1 * 100:+.2f} F1 "
                f"points above the classical baseline, but the difference is not "
                f"statistically significant ({p_txt}). No quantum-advantage claim "
                "is warranted."
            )
        else:
            auc_dir = "higher" if delta_auc > 0 else "lower"
            findings["quantum_comparison"] = (
                f"The classical CNN+LSTM outperforms the best quantum model "
                f"({MODEL_LABELS[best_quantum['model']]}) by "
                f"{abs(delta_f1) * 100:.2f} F1 points. The quantum model's "
                f"ROC-AUC is {abs(delta_auc):.4f} {auc_dir} "
                f"({best_quantum.get('roc_auc', float('nan')):.4f} vs "
                f"{cnn.get('roc_auc', float('nan')):.4f}). {p_txt}. The quantum "
                "models did not provide an advantage on the primary common test "
                "under this experimental design."
            )

    # Latency
    latency = {r["model"]: r for r in summary.get("latency", []) if "model" in r}
    if "cnn_lstm" in latency and "qsvm" in latency:
        slow = latency["qsvm"].get("slowdown_vs_classical")
        if slow is not None and np.isfinite(slow):
            findings["notes"].append(
                f"Single-sample QSVM inference is {slow:.1f}x slower than the "
                "CNN+LSTM in simulation."
            )

    # Unseen attack
    unseen = summary.get("unseen_category", [])
    if unseen:
        rows = sorted(unseen, key=lambda r: r.get("recall") or 0, reverse=True)
        findings["notes"].append(
            "Unseen-category recall on a balanced Normal/attack hold-out "
            "(R2L+U2R withheld; per-model validation-selected threshold): "
            + ", ".join(
                f"{MODEL_LABELS.get(r['model'], r['model'])} {_fmt(r.get('recall'))}"
                for r in rows
            )
        )

    # External novel types
    external = summary.get("external_novel", [])
    if external:
        ext_rows = [r for r in external if r.get("model")]
        if ext_rows:
            best_ext = max(ext_rows, key=lambda r: r.get("f1") or 0)
            cnn_ext = next((r for r in ext_rows if r.get("model") == "cnn_lstm"), None)
            note = (
                "External novel-type F1: "
                + ", ".join(
                    f"{MODEL_LABELS.get(r['model'], r['model'])} {_fmt(r.get('f1'))}"
                    for r in sorted(ext_rows, key=lambda r: r.get("f1") or 0, reverse=True)
                )
                + f"; best is {MODEL_LABELS.get(best_ext['model'], best_ext['model'])}."
            )
            if cnn_ext is not None and best_ext.get("model") != "cnn_lstm":
                note += (
                    f" This is a scope-specific external-subset result, not a general "
                    f"advantage over the primary common test."
                )
            findings["notes"].append(note)

    # CV
    cv = summary.get("cross_validation", [])
    f1_rows = [r for r in cv if r.get("metric") == "f1"]
    if f1_rows:
        n_folds = f1_rows[0].get("n_folds", "?")
        findings["notes"].append(
            f"{n_folds}-fold CV F1: "
            + ", ".join(
                f"{MODEL_LABELS.get(r['model'], r['model'])} "
                f"{_fmt(r['mean'])} ± {_fmt(r['std'])}"
                for r in f1_rows
            )
        )

    if not findings.get("quantum_advantage_claim"):
        findings["notes"].append(
            "All quantum results are simulation-based (Qiskit Statevector) and "
            "shot-noise free; they do not represent quantum-hardware performance."
        )
    return findings


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def render_summary_md(summary: dict, findings: dict) -> str:
    best = findings.get("best_overall") or {}
    ds = summary.get("dataset_validation", {}) or {}
    fs = summary.get("feature_selection", {}) or {}
    lines: list[str] = []
    add = lines.append

    add(f"# {summary['project']} — Final Summary\n")
    add(f"**Mode:** `{summary['mode']}` · **Seed:** {summary['seed']} · "
        f"**Dataset:** {summary['dataset']}\n")
    add("## 1. Dataset and preprocessing\n")
    add(f"| Item | Value |\n|---|---|")
    add(f"| Raw training pool | {ds.get('train_rows', 'n/a'):,} rows |")
    add(f"| External hold-out (KDDTest+) | {ds.get('test_rows', 'n/a'):,} rows |")
    add(f"| Input features | {ds.get('n_features', 'n/a')} |")
    add(f"| Reduction | {fs.get('method', 'n/a').upper()} → {fs.get('n_features', 'n/a')} components |")
    add(f"| Leakage audit | {'PASSED' if summary.get('leakage_audit_passed') else 'FAILED / not run'} |")
    cd = summary.get("class_distribution", {}) or {}
    for split in ("train", "validation", "test"):
        info = cd.get(split, {})
        if info:
            add(f"| {split.title()} size | {info.get('rows', 0):,} (attack rate "
                f"{info.get('attack_rate', 0) * 100:.2f}%) |")
    add("")

    add("## 2. Model comparison (common test set)\n")
    rows = summary.get("master_table", [])
    if rows:
        add("| Model | Encoding | Acc | Prec | Rec | F1 | ROC-AUC | FPR | FNR | Train (s) | Batch infer (ms/sample) | Qubits | Depth |")
        add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            add(
                f"| {MODEL_LABELS.get(r['model'], r['model'])} | {r.get('encoding', '-')} | "
                f"{_fmt(r.get('accuracy'))} | {_fmt(r.get('precision'))} | "
                f"{_fmt(r.get('recall'))} | {_fmt(r.get('f1'))} | {_fmt(r.get('roc_auc'))} | "
                f"{_fmt(r.get('fpr'))} | {_fmt(r.get('fnr'))} | "
                f"{_fmt(r.get('training_time_s'), 2)} | {_fmt(r.get('inference_ms_per_sample'), 4)} | "
                f"{int(r['qubits']) if r.get('qubits') is not None else '-'} | "
                f"{int(r['circuit_depth']) if r.get('circuit_depth') is not None else '-'} |"
            )
    else:
        add("_No results available._")
    add("")

    add("## 3. Main finding\n")
    add(findings.get("quantum_comparison", "n/a") + "\n")
    if findings.get("notes"):
        add("Additional observations:\n")
        for note in findings["notes"]:
            add(f"- {note}")
        add("")

    add("## 4. Cross-validation\n")
    cv = [r for r in summary.get("cross_validation", []) if r.get("metric") == "f1"]
    if cv:
        add("| Model | Folds | Mean F1 | Std | 95% CI |")
        add("|---|---|---|---|---|")
        for r in cv:
            add(f"| {MODEL_LABELS.get(r['model'], r['model'])} | {r['n_folds']} | "
                f"{_fmt(r['mean'])} | {_fmt(r['std'])} | "
                f"[{_fmt(r.get('ci_low'))}, {_fmt(r.get('ci_high'))}] |")
    else:
        add("_Cross-validation not run._")
    add("")

    add("## 5. Statistical significance\n")
    stats = summary.get("statistics", {}) or {}
    if stats.get("mcnemar"):
        add("| Comparison | Discordant | McNemar p | Significant | Cohen's h | Effect |")
        add("|---|---|---|---|---|---|")
        for r in stats["mcnemar"]:
            add(f"| {r['comparison']} | {r.get('n_discordant', '-')} | "
                f"{_fmt(r.get('p_value'), 6)} | {r.get('significant')} | "
                f"{_fmt(r.get('effect_size_cohens_h'), 3)} | {r.get('effect_interpretation', '-')} |")
    else:
        add("_No paired tests available._")
    add("")

    pb = stats.get("paired_bootstrap", [])
    if pb:
        add("Paired bootstrap (1,000 resamples; F1 at each model's validation-selected threshold):\n")
        add("| Comparison | Metric | Difference | 95% CI | Crosses zero |")
        add("|---|---|---|---|---|")
        for r in pb:
            add(f"| {r.get('comparison', '-')} | {r.get('metric', '-')} | "
                f"{_fmt(r.get('difference'), 4)} | "
                f"[{_fmt(r.get('ci_low'), 4)}, {_fmt(r.get('ci_high'), 4)}] | "
                f"{r.get('crosses_zero', '-')} |")
        add("")

    wil = stats.get("wilcoxon", [])
    if wil:
        folds = wil[0].get("n_folds", "-")
        add(f"Wilcoxon signed-rank across {folds} CV folds (supporting evidence only; "
            "the attainable two-sided p-value is bounded by the fold count):\n")
        add("| Comparison | Mean F1 (A) | Mean F1 (B) | p-value |")
        add("|---|---|---|---|")
        for r in wil:
            add(f"| {r.get('comparison', '-')} | {_fmt(r.get('mean_a'))} | "
                f"{_fmt(r.get('mean_b'))} | {_fmt(r.get('p_value'), 3)} |")
        add("")

    add("## 6. Unseen attack-category generalisation\n")
    unseen = summary.get("unseen_category", [])
    if unseen:
        add("| Model | Encoding | Recall | Precision | F1 | ROC-AUC | FPR | Unseen recall |")
        add("|---|---|---|---|---|---|---|---|")
        for r in unseen:
            add(f"| {MODEL_LABELS.get(r['model'], r['model'])} | {r.get('encoding', '-')} | "
                f"{_fmt(r.get('recall'))} | {_fmt(r.get('precision'))} | {_fmt(r.get('f1'))} | "
                f"{_fmt(r.get('roc_auc'))} | {_fmt(r.get('fpr'))} | "
                f"{_fmt(r.get('unseen_attack_recall'))} |")
        ud = summary.get("unseen_details", {}) or {}
        add(f"\nHeld-out categories: **{', '.join(ud.get('holdout_categories', []))}**. "
            f"Training categories: {', '.join(ud.get('train_categories', []))}. "
            f"Test categories: {', '.join(ud.get('test_categories', []))}.\n")
    else:
        add("_Experiment not run._")
    add("")

    ext = summary.get("external_novel", [])
    if ext:
        add("### External novel attack types (KDDTest+)\n")
        add("| Model | Novel types | Novel / normal samples | Recall | Precision | F1 | ROC-AUC |")
        add("|---|---|---|---|---|---|---|")
        for r in ext:
            add(f"| {MODEL_LABELS.get(r['model'], r['model'])} | {r.get('novel_attack_types', '-')} | "
                f"{r.get('n_novel_attack_samples', '-')} / {r.get('n_negative', '-')} | "
                f"{_fmt(r.get('recall'))} | "
                f"{_fmt(r.get('precision'))} | {_fmt(r.get('f1'))} | {_fmt(r.get('roc_auc'))} |")
        ed = summary.get("external_novel_details", {}) or {}
        add(f"\nNovel attack types: {', '.join(ed.get('novel_types', []))}.\n")

    add("## 7. Encoding comparison\n")
    enc = summary.get("encoding_comparison", [])
    if enc:
        add("| Model | Encoding | Qubits | Depth | F1 | ROC-AUC | Train (s) |")
        add("|---|---|---|---|---|---|---|")
        for r in enc:
            add(f"| {r['model'].upper()} | {r['encoding']} | {r.get('n_qubits', '-')} | "
                f"{r.get('circuit_depth', '-')} | {_fmt(r.get('f1'))} | "
                f"{_fmt(r.get('roc_auc'))} | {_fmt(r.get('training_time_s'), 2)} |")
    else:
        add("_Experiment not run._")
    add("")

    add("## 8. Latency\n")
    lat = summary.get("latency", [])
    if lat:
        add("| Model | Preprocess (ms) | Reduction (ms) | Single-sample inference (ms) | Throughput (samples/s) | Slowdown vs classical |")
        add("|---|---|---|---|---|---|")
        for r in lat:
            add(f"| {MODEL_LABELS.get(r['model'], r['model'])} | "
                f"{_fmt(r.get('preprocessing_ms_per_sample'), 5)} | "
                f"{_fmt(r.get('feature_reduction_ms_per_sample'), 5)} | "
                f"{_fmt(r.get('inference_ms_per_sample'), 4)} | "
                f"{_fmt(r.get('throughput_samples_per_s'), 1)} | "
                f"{_fmt(r.get('slowdown_vs_classical'), 2)}x |")
        repeats = int(lat[0].get("repeats", 1) or 1)
        add(f"\nSingle-sample end-to-end latency, median of {repeats} repeats on CPU. "
            "Throughput uses batch size "
            f"{int(lat[0].get('batch_size', 0) or 0)}. Timings are hardware-dependent; "
            "batch inference in Section 2 is not directly comparable.\n")
    else:
        add("_Experiment not run._")
    add("")

    add("## 9. Quantum resources (simulation)\n")
    qr = summary.get("quantum_resources", [])
    if qr:
        add("| Model | Encoding | Qubits | Depth | Size | Parameters | 2Q gates | Backend |")
        add("|---|---|---|---|---|---|---|---|")
        for r in qr:
            add(f"| {r['model'].upper()} | {r.get('encoding')} | {r.get('qubits')} | "
                f"{r.get('circuit_depth')} | {r.get('circuit_size')} | "
                f"{r.get('num_parameters')} | {r.get('num_two_qubit_gates')} | "
                f"{r.get('backend')} |")
    else:
        add("_Experiment not run._")
    add("")

    add("## 10. Distribution shift\n")
    drift = summary.get("drift_summary", [])
    if drift:
        add("| Shift | Model | Baseline F1 | Worst F1 | ΔF1 | Worst KS |")
        add("|---|---|---|---|---|---|")
        for r in drift:
            add(f"| {r['shift_type']} | {r['model']} | {_fmt(r['baseline_f1'])} | "
                f"{_fmt(r['worst_f1'])} | {_fmt(r['f1_degradation'])} | {_fmt(r['worst_ks_mean'], 3)} |")
    else:
        add("_Experiment not run._")
    add("")

    add("## 11. Limitations\n")
    for text in LIMITATIONS:
        add(f"- {text}")
    add("")

    add("## 12. Conclusion\n")
    add(
        "This study compared a properly tuned classical CNN+LSTM with two "
        "quantum classifiers on an identical, leakage-controlled, "
        "dimensionality-reduced NSL-KDD benchmark. "
        + findings.get("quantum_comparison", "")
        + " All quantum results were produced with exact statevector simulation; "
        "no quantum hardware was used and no production deployment was attempted."
    )
    add("")
    add("## 13. Reproducibility\n")
    env = summary.get("environment", {}) or {}
    add("```")
    add(f"Python: {'.'.join(map(str, env.get('python', {}).get('version_info', [])))}")
    add(f"Platform: {env.get('platform', {}).get('system')} {env.get('platform', {}).get('release')}")
    add(f"CPU: {env.get('platform', {}).get('processor')}")
    add(f"RAM: {env.get('ram_gb')} GB")
    pkgs = env.get("packages", {})
    add("Packages: " + ", ".join(f"{k}=={v}" for k, v in pkgs.items() if v))
    add("```")
    add("")
    return "\n".join(lines)


LIMITATIONS = [
    "All quantum experiments use Qiskit's exact statevector simulator. They are simulation-based and do not reflect quantum-hardware noise, decoherence or gate errors.",
    "The quantum feature dimension is capped at 8 qubits for tractability. Larger circuits (12–16 qubits) are reported only for the classical probe ablation.",
    "The quantum models are trained on a smaller stratified subset than the classical model because circuit evaluation cost grows with training-set size. The test set is identical for every model.",
    "NSL-KDD is a 2009 benchmark with known limitations (age, synthetic attack mix, no timestamps). Results do not transfer automatically to modern enterprise traffic.",
    "Unseen attack categories are measured by holding out R2L and U2R. This is unseen-category generalisation, not a zero-day claim.",
    "Concept drift is studied with controlled covariate-shift injections and a simulated file-order scan. NSL-KDD has no capture timestamps, so no real temporal drift detection is claimed.",
    "The external novel-type subset is one normal-heavy KDDTest+ file with strong covariate shift; its metrics are point estimates without external-set confidence intervals.",
    "Amplitude encoding reduces to a squared-cosine fidelity kernel, which is a classical polynomial kernel. The encoding comparison is therefore an empirical check, not evidence of quantum advantage.",
    "McNemar and Wilcoxon tests establish whether differences are unlikely under a null of equal performance on this sample; they do not establish a general quantum advantage.",
]


# ---------------------------------------------------------------------------
# Paper draft
# ---------------------------------------------------------------------------
def render_paper_md(summary: dict, findings: dict) -> str:
    ds = summary.get("dataset_validation", {}) or {}
    fs = summary.get("feature_selection", {}) or {}
    rows = summary.get("master_table", [])
    cnn = _find_best(summary, "cnn_lstm")
    qsvm = _find_best(summary, "qsvm")
    vqc = _find_best(summary, "vqc")
    unseen = summary.get("unseen_category", [])
    ext = summary.get("external_novel", [])
    enc = summary.get("encoding_comparison", [])
    lat = structure_by_model(summary.get("latency", []))
    stats = summary.get("statistics", {}) or {}
    cv = [r for r in summary.get("cross_validation", []) if r.get("metric") == "f1"]
    cd = summary.get("class_distribution", {}) or {}
    train_info = cd.get("train", {}) or {}
    test_info = cd.get("test", {}) or {}
    mode = summary.get("mode", "quick")

    def row(model: dict | None, key: str, digits: int = 4) -> str:
        return _fmt((model or {}).get(key), digits)

    lines: list[str] = []
    add = lines.append

    add("# Quantum-Enhanced Intrusion Detection: A Fair Comparison of a Classical CNN+LSTM Baseline with Quantum SVM and Variational Quantum Classifier on NSL-KDD\n")
    add("> **Generated from executed experiments.** Every numeric value in this draft "
        "was read from `results/` after the pipeline completed. Values are not "
        "placeholders. Re-running `python run_all.py --mode full` will regenerate "
        "this document with the full-mode numbers.\n")

    add("## Abstract\n")
    add(
        "Network intrusion detection systems must separate malicious traffic from "
        "benign traffic under severe class imbalance and evolving attack behaviour. "
        "Quantum machine learning is frequently proposed for this task, but "
        "comparisons are rarely conducted on identical reduced features, identical "
        "splits and identical evaluation protocols. We present a controlled "
        "benchmark on NSL-KDD comparing three classifiers: a classical CNN+LSTM, a "
        "Quantum Support Vector Machine (QSVM) using a fidelity quantum kernel, and "
        "a Variational Quantum Classifier (VQC). All models consume the same "
        f"{fs.get('n_features', 8)}-component {str(fs.get('method', 'pca')).upper()} "
        "representation, are trained with leakage-safe pipelines fitted only on the "
        "training split, and are scored on the same held-out test set. "
        f"The classical CNN+LSTM reached F1 = {row(cnn, 'f1')} and ROC-AUC = "
        f"{row(cnn, 'roc_auc')}; the best quantum model reached F1 = "
        f"{row(findings.get('best_quantum'), 'f1')} and ROC-AUC = "
        f"{row(findings.get('best_quantum'), 'roc_auc')}. "
        + (
            "The quantum models did not outperform the classical baseline."
            if (findings.get("best_quantum") or {}).get("f1", 0) <= (cnn or {}).get("f1", 0)
            else "A numerical improvement was observed for a quantum model; its statistical significance is reported below."
        )
        + " We report unseen attack-category generalisation, latency, quantum "
        "resource usage and threshold-dependent operating points, and we explicitly "
        "state the limits of simulation-based evidence.\n"
    )
    add("**Keywords:** intrusion detection, NSL-KDD, quantum machine learning, "
        "quantum kernel, variational quantum classifier, CNN-LSTM, reproducibility\n")

    add("## 1. Introduction\n")
    add(
        "Intrusion detection remains a core defensive control, yet signature-based "
        "systems struggle with novel behaviour and classical deep models can be "
        "expensive to tune and explain. Quantum classifiers promise richer feature "
        "spaces through high-dimensional state representations. That promise is "
        "frequently evaluated with inconsistent baselines, different feature "
        "subsets, or test-set tuning. This work asks a narrower and testable "
        "question: *given identical reduced inputs and an identical evaluation "
        "protocol, does a quantum classifier improve intrusion detection over a "
        "properly tuned classical CNN+LSTM?* The answer reported here is derived "
        "only from measured results.\n")
    add("### Contributions\n")
    add("1. A leakage-audited pipeline in which imputation, one-hot encoding, "
        "scaling, PCA/mutual-information selection and quantum angle rescaling are "
        "fitted exclusively on training data.\n"
        "2. A fair three-way comparison on the same reduced representation and the "
        "same test samples, including validation-only threshold selection.\n"
        "3. An unseen attack-category experiment in which R2L and U2R are removed "
        "from training entirely, plus an external novel-type test on KDDTest+.\n"
        "4. A systems analysis covering latency, throughput, quantum resources, "
        "controlled distribution shift and explainability.\n"
        "5. Explicit reporting of negative or inconclusive results.\n")

    add("## 2. Related work\n")
    add(
        "Quantum kernel methods estimate similarity in a state space induced by a "
        "feature map and are attractive because they can be analysed with the "
        "machinery of kernel methods. Variational quantum classifiers instead train "
        "a parameterised circuit with a classical optimiser. Prior IDS studies "
        "report strong quantum results, but often on small subsamples, with "
        "classical baselines that are not tuned to convergence, or without paired "
        "statistical testing. Our protocol addresses those three weaknesses "
        "directly. NSL-KDD remains a standard benchmark despite its age, which "
        "makes it useful for controlled comparison rather than for claims about "
        "modern networks.\n")

    add("## 3. Dataset\n")
    add(f"NSL-KDD provides {ds.get('train_rows', 0):,} labelled records in "
        f"KDDTrain+ and {ds.get('test_rows', 0):,} in KDDTest+. Each record has "
        f"{ds.get('n_features', 41)} features (3 categorical: protocol_type, "
        "service, flag; 38 numeric) plus a label and a difficulty score. The "
        f"training pool contains {ds.get('train_attack_types', 'n/a')} distinct "
        f"labels; KDDTest+ contains {ds.get('test_attack_types', 'n/a')}, of which "
        f"{len(ds.get('test_only_attack_types', []))} attack types never appear in "
        "KDDTrain+.\n")
    add("| Split | Rows | Attack rate |\n|---|---|---|")
    add(f"| Train (70%) | {train_info.get('rows', 'n/a'):,} | "
        f"{train_info.get('attack_rate', 0) * 100:.2f}% |")
    for split in ("validation", "test"):
        info = cd.get(split, {})
        add(f"| {split.title()} | {info.get('rows', 'n/a'):,} | "
            f"{info.get('attack_rate', 0) * 100:.2f}% |")
    add("")
    add(f"Attack categories follow the NSL-KDD taxonomy (Normal, DoS, Probe, R2L, "
        f"U2R). The official KDDTest+ file is used only as an external, "
        f"never-trained-on distribution-shift set.\n")

    add("## 4. Methodology\n")
    add("### 4.1 Preprocessing\n")
    add(
        "Constant columns are detected on the training split and dropped. Numeric "
        "features are median-imputed (train medians) and standardised; categorical "
        "features are one-hot encoded with an unknown-level-safe vocabulary learned "
        "on the training split. Infinite values are replaced before imputation. "
        "No transformation is fitted on validation or test data.\n")
    add("### 4.2 Dimensionality reduction\n")
    add(
        f"Two families were evaluated on validation data only: PCA and mutual-"
        f"information ranking at 8, 12 and 16 dimensions. The comparative study uses "
        f"**{str(fs.get('method', 'pca')).upper()} with {fs.get('n_features', 8)} "
        f"components**, selected by validation F1 subject to the simulator budget of "
        f"8 qubits. The full 8/12/16 ablation is reported in "
        f"`results/tables/feature_reduction_ablation.csv`. The selected reduced "
        f"features are then min–max rescaled to [0, π] using training statistics, "
        f"and this identical matrix is supplied to all three models.\n")
    add("### 4.3 Classical baseline\n")
    add(
        "The CNN+LSTM receives the reduced vector as a single-channel sequence of "
        "length equal to the number of components. This is a *pseudo-temporal* "
        "reshaping, not real packet timing: the order is the component index. The "
        "network is BatchNorm1d → Conv1d → ReLU → MaxPool → Dropout → LSTM → Dense "
        "→ logit, trained with AdamW, class-weighted BCE, early stopping on "
        "validation ROC-AUC and best-epoch checkpointing.\n")
    add("### 4.4 QSVM\n")
    add(
        "Reduced features are encoded with a ZZ feature map (angle encoding). The "
        "fidelity kernel K(x, y) = |⟨φ(x)|φ(y)⟩|² is computed exactly from "
        "statevectors produced by Qiskit. A soft-margin SVM with Platt scaling is "
        "trained on the precomputed kernel. The exact kernel is cross-checked "
        "against Qiskit's circuit-based FidelityQuantumKernel/ComputeUncompute "
        "estimator; the comparison tolerance is derived from the finite-shot "
        "binomial noise of that reference.\n")
    add("### 4.5 VQC\n")
    add(
        "The same feature map is followed by a RealAmplitudes ansatz. The decision "
        "is P(qubit 0 = |1⟩), optimised with COBYLA (SPSA is supported) on binary "
        "cross-entropy. Training uses the identity "
        "|ψ(x;θ)⟩ = U_ansatz(θ)|φ(x)⟩ to apply one ansatz unitary to a precomputed "
        "batch of feature states; this is mathematically identical to full-circuit "
        "simulation and is verified in the test suite against direct simulation.\n")
    add("### 4.6 Evaluation protocol\n")
    add(
        "All models are scored on the identical common test subset. The decision "
        "threshold is selected on validation data only, using either F1 or a "
        "cost-sensitive objective (a missed attack costs 10× a false alarm). "
        "Reported metrics are accuracy, precision, recall, F1, ROC-AUC, FPR, FNR, "
        "specificity, MCC and Brier score. Pairwise differences are tested with "
        "McNemar's exact/chi-square test on paired predictions, paired bootstrap "
        "confidence intervals and, across folds, the Wilcoxon signed-rank test.\n")

    add("## 5. Results\n")
    add("### 5.1 Main comparison\n")
    add("| Model | Encoding | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR | Train (s) | Inference (ms/sample) |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        add(f"| {MODEL_LABELS.get(r['model'], r['model'])} | {r.get('encoding', '-')} | "
            f"{_fmt(r.get('accuracy'))} | {_fmt(r.get('precision'))} | {_fmt(r.get('recall'))} | "
            f"{_fmt(r.get('f1'))} | {_fmt(r.get('roc_auc'))} | {_fmt(r.get('fpr'))} | "
            f"{_fmt(r.get('fnr'))} | {_fmt(r.get('training_time_s'), 2)} | "
            f"{_fmt(r.get('inference_ms_per_sample'), 4)} |")
    add("")
    add("### 5.2 Cross-validation\n")
    if cv:
        add("| Model | Folds | Mean F1 | Std | 95% CI |")
        add("|---|---|---|---|---|")
        for r in cv:
            add(f"| {MODEL_LABELS.get(r['model'], r['model'])} | {r['n_folds']} | "
                f"{_fmt(r['mean'])} | {_fmt(r['std'])} | "
                f"[{_fmt(r.get('ci_low'))}, {_fmt(r.get('ci_high'))}] |")
    else:
        add("_Not run in this mode._")
    add("")
    add("### 5.3 Statistical tests\n")
    if stats.get("mcnemar"):
        add("| Comparison | Discordant pairs | McNemar p | Significant | Cohen's h | Effect |")
        add("|---|---|---|---|---|---|")
        for r in stats["mcnemar"]:
            add(f"| {r['comparison']} | {r.get('n_discordant', '-')} | "
                f"{_fmt(r.get('p_value'), 6)} | {r.get('significant')} | "
                f"{_fmt(r.get('effect_size_cohens_h'), 3)} | {r.get('effect_interpretation', '-')} |")
    else:
        add("_No paired tests available._")
    add("")
    add("### 5.4 Unseen attack-category generalisation\n")
    if unseen:
        add("| Model | Recall | Precision | F1 | ROC-AUC | Unseen-attack recall |")
        add("|---|---|---|---|---|---|")
        for r in unseen:
            add(f"| {MODEL_LABELS.get(r['model'], r['model'])} | {_fmt(r.get('recall'))} | "
                f"{_fmt(r.get('precision'))} | {_fmt(r.get('f1'))} | {_fmt(r.get('roc_auc'))} | "
                f"{_fmt(r.get('unseen_attack_recall'))} |")
    else:
        add("_Not run._")
    add("")
    if ext:
        add("### 5.5 External novel attack types\n")
        add("| Model | Novel types | Recall | Precision | F1 | ROC-AUC |")
        add("|---|---|---|---|---|---|")
        for r in ext:
            add(f"| {MODEL_LABELS.get(r['model'], r['model'])} | {r.get('novel_attack_types')} | "
                f"{_fmt(r.get('recall'))} | {_fmt(r.get('precision'))} | {_fmt(r.get('f1'))} | "
                f"{_fmt(r.get('roc_auc'))} |")
        add("")
    add("### 5.6 Encoding comparison\n")
    if enc:
        add("| Model | Encoding | Qubits | Depth | F1 | ROC-AUC |")
        add("|---|---|---|---|---|---|")
        for r in enc:
            add(f"| {r['model'].upper()} | {r['encoding']} | {r.get('n_qubits', '-')} | "
                f"{r.get('circuit_depth', '-')} | {_fmt(r.get('f1'))} | {_fmt(r.get('roc_auc'))} |")
    else:
        add("_Not run._")
    add("")
    add("### 5.7 Latency and quantum resources\n")
    if lat:
        add("| Model | Inference (ms/sample) | Throughput (samples/s) | Slowdown vs classical |")
        add("|---|---|---|---|")
        for r in lat.values():
            add(f"| {MODEL_LABELS.get(r['model'], r['model'])} | "
                f"{_fmt(r.get('inference_ms_per_sample'), 4)} | "
                f"{_fmt(r.get('throughput_samples_per_s'), 1)} | "
                f"{_fmt(r.get('slowdown_vs_classical'), 2)}× |")
    else:
        add("_Not run._")
    add("")

    add("## 6. Discussion\n")
    add(findings.get("quantum_comparison", "") + "\n")
    add(
        "The classical baseline benefits from being trained on the full training "
        "split, whereas the quantum models are trained on a stratified subset "
        "whose size is bounded by simulation cost. This is a property of the "
        "experiment, not a hidden advantage: every model is scored on the same "
        "test samples, and the asymmetry is stated explicitly. It also means the "
        "comparison answers a practical question — what a student-scale quantum "
        "simulation can deliver — rather than an idealised hardware question.\n")
    ext = [r for r in summary.get("external_novel", []) if r.get("model")]
    if ext:
        best_ext = max(ext, key=lambda r: r.get("f1") or 0)
        cnn_ext = next((r for r in ext if r.get("model") == "cnn_lstm"), None)
        if cnn_ext is not None and best_ext.get("model") != "cnn_lstm":
            add(
                f"On the secondary external novel-type subset, "
                f"{MODEL_LABELS.get(best_ext['model'], best_ext['model'])} leads by "
                f"F1 {_fmt(best_ext.get('f1'))} versus {_fmt(cnn_ext.get('f1'))} for "
                f"the CNN+LSTM, with ROC-AUC {_fmt(best_ext.get('roc_auc'))} versus "
                f"{_fmt(cnn_ext.get('roc_auc'))}. This is a scope-specific result on "
                f"one normal-heavy external file, not a general quantum advantage.\n")
    add(
        "Latency is dominated by quantum state preparation and kernel evaluation. "
        "Because every quantum prediction requires fresh circuit simulation, the "
        "quantum models are not competitive for real-time inline detection in this "
        "form. A hardware deployment would face additional sampling, queueing and "
        "error-correction costs.\n")

    add("## 7. Limitations\n")
    for text in LIMITATIONS:
        add(f"- {text}")
    add("")

    add("## 8. Conclusion\n")
    add(
        "We built a reproducible, leakage-audited benchmark for quantum versus "
        "classical intrusion detection and executed it end to end. On NSL-KDD, "
        "with 8 reduced features, identical splits and identical test samples, "
        + (
            "the classical CNN+LSTM remained the strongest model on the primary "
            "common test. "
            if (findings.get("best_quantum") or {}).get("f1", 0) <= (cnn or {}).get("f1", 0)
            else "a quantum model achieved the highest F1 on the primary common test, "
                 "with the caveat that the difference must be interpreted through the "
                 "paired tests above. "
        )
        + "The contribution is methodological as much as empirical: an honest, "
        "fully executable protocol that others can extend to larger circuits or "
        "real quantum hardware without changing the evaluation contract.\n")

    add("## References\n")
    add("1. Tavallaee, M., Bagheri, E., Lu, W., Ghorbani, A. A. (2009). A detailed "
        "analysis of the KDD CUP 99 data set. *IEEE CISDA*.\n"
        "2. Havlíček, V. et al. (2019). Supervised learning with quantum-enhanced "
        "feature spaces. *Nature* 567, 209–212.\n"
        "3. Schuld, M., Killoran, N. (2019). Quantum machine learning in feature "
        "Hilbert spaces. *Physical Review Letters* 122, 040504.\n"
        "4. Cerezo, M. et al. (2021). Variational quantum algorithms. *Nature "
        "Reviews Physics* 3, 625–644.\n"
        "5. Qiskit contributors. Qiskit: An open-source framework for quantum "
        "computing (2024–2026).\n"
        "6. Dietterich, T. G. (1998). Approximate statistical tests for comparing "
        "supervised classification learning algorithms. *Neural Computation* "
        "10(7), 1895–1923.\n")
    return "\n".join(lines)


def structure_by_model(rows: list[dict]) -> dict[str, dict]:
    return {r["model"]: r for r in rows if "model" in r}


# ---------------------------------------------------------------------------
# Static dashboard
# ---------------------------------------------------------------------------
def _b64_image(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _table_html(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "<p class='muted'>No data.</p>"
    return df.to_html(index=False, border=0, classes="data-table", float_format=lambda v: f"{v:.{digits}f}")


def build_static_dashboard(cfg, summary: dict, findings: dict) -> Path:
    fig_dir = Path(cfg.get_path("paths.figures"))
    metrics_dir = Path(cfg.get_path("paths.metrics"))
    out = Path(cfg.get_path("paths.results")) / "dashboard.html"

    def img(name: str, caption: str) -> str:
        b64 = _b64_image(fig_dir / name)
        if not b64:
            return f"<figure><div class='missing'>Figure not available: {html.escape(name)}</div></figure>"
        return (f"<figure><img src='{b64}' alt='{html.escape(caption)}'/>"
                f"<figcaption>{html.escape(caption)}</figcaption></figure>")

    master = pd.DataFrame(summary.get("master_table", []))
    if not master.empty:
        show_cols = ["model_label", "encoding", "accuracy", "precision", "recall", "f1",
                     "roc_auc", "fpr", "fnr", "training_time_s", "qubits", "circuit_depth"]
        master_show = master[[c for c in show_cols if c in master.columns]]
    else:
        master_show = master

    cv = pd.DataFrame([r for r in summary.get("cross_validation", []) if r.get("metric") == "f1"])
    unseen = pd.DataFrame(summary.get("unseen_category", []))
    enc = pd.DataFrame(summary.get("encoding_comparison", []))
    lat = pd.DataFrame(summary.get("latency", []))
    qr = pd.DataFrame(summary.get("quantum_resources", []))
    stats = summary.get("statistics", {}) or {}
    mcnemar = pd.DataFrame(stats.get("mcnemar", []))
    alerts = summary.get("alert_summary", {}) or {}

    best = findings.get("best_overall") or {}
    ds = summary.get("dataset_validation", {}) or {}
    fs = summary.get("feature_selection", {}) or {}
    cd = summary.get("class_distribution", {}) or {}
    train_rows = cd.get("train", {}).get("rows", "-")
    test_rows = cd.get("test", {}).get("rows", "-")

    # Alert demo samples: take a handful of scored test rows.
    demo_rows = []
    pred_path = Path(cfg.get_path("paths.predictions")) / "cnn_lstm_test_predictions.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        sample = pred.sample(n=min(25, len(pred)), random_state=int(cfg.get_path("project.seed", 42)))
        for _, r in sample.iterrows():
            demo_rows.append({
                "true": "ATTACK" if int(r["y_true"]) == 1 else "NORMAL",
                "score": round(float(r["y_score"]), 4),
                "category": str(r.get("attack_category", "")),
            })

    policy = alerts.get("policy", {}) or {}
    total = max(alerts.get("level_counts", {}).get("NORMAL", 0)
                + alerts.get("level_counts", {}).get("MONITOR", 0)
                + alerts.get("level_counts", {}).get("ALERT", 0)
                + alerts.get("level_counts", {}).get("QUARANTINE", 0)
                + alerts.get("level_counts", {}).get("ESCALATE", 0), 1)
    level_counts = alerts.get("level_counts", {})

    alert_html = "".join(
        f"<div class='alertbar'><span class='lvl {lvl.lower()}'>{lvl}</span>"
        f"<div class='bar'><div class='fill {lvl.lower()}' style='width:{level_counts.get(lvl, 0) / total * 100:.2f}%'></div></div>"
        f"<span class='count'>{level_counts.get(lvl, 0)}</span></div>"
        for lvl in ["NORMAL", "MONITOR", "ALERT", "QUARANTINE", "ESCALATE"]
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(summary['project'])} — Dashboard</title>
<style>
  :root {{
    --navy:#1f4e79; --red:#c0392b; --green:#2e8b57; --ink:#1b2631; --muted:#6b7a86;
    --bg:#f7f9fb; --card:#ffffff; --line:#e1e8ee;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         background:var(--bg); color:var(--ink); line-height:1.5; }}
  header {{ background:linear-gradient(135deg,#12324f,#1f4e79 55%,#2e8b57);
            color:white; padding:34px 40px 30px; }}
  header h1 {{ margin:0 0 6px; font-size:26px; letter-spacing:.2px; }}
  header p {{ margin:2px 0; opacity:.9; font-size:13.5px; }}
  .wrap {{ max-width:1240px; margin:0 auto; padding:26px 22px 70px; }}
  .grid {{ display:grid; gap:16px; }}
  .kpis {{ grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); margin-top:-42px; }}
  .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
          padding:16px 18px; box-shadow:0 6px 18px rgba(20,50,80,.06); }}
  .kpi .label {{ font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }}
  .kpi .value {{ font-size:22px; font-weight:700; color:var(--navy); margin-top:4px; }}
  section {{ margin-top:28px; }}
  h2 {{ font-size:18px; margin:0 0 12px; padding-bottom:7px; border-bottom:2px solid var(--line); }}
  h3 {{ font-size:14px; margin:18px 0 8px; color:var(--navy); }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
           padding:18px; box-shadow:0 4px 14px rgba(20,50,80,.045); }}
  .cols {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  @media (max-width:900px) {{ .cols {{ grid-template-columns:1fr; }} }}
  table.data-table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
  table.data-table th {{ background:#eef3f8; text-align:left; padding:8px 9px; font-weight:600;
                         border-bottom:1px solid var(--line); white-space:nowrap; }}
  table.data-table td {{ padding:7px 9px; border-bottom:1px solid #f0f4f7; }}
  table.data-table tr:hover td {{ background:#f7fbfd; }}
  figure {{ margin:0; }}
  figure img {{ width:100%; border:1px solid var(--line); border-radius:10px; background:white; }}
  figcaption {{ font-size:11.5px; color:var(--muted); margin-top:6px; text-align:center; }}
  .missing {{ padding:30px; text-align:center; color:var(--muted); border:1px dashed var(--line);
              border-radius:10px; font-size:12px; }}
  .alertbar {{ display:grid; grid-template-columns:110px 1fr 48px; gap:10px; align-items:center;
               margin:7px 0; font-size:12.5px; }}
  .bar {{ background:#eef2f5; border-radius:99px; height:14px; overflow:hidden; }}
  .fill {{ height:100%; border-radius:99px; }}
  .lvl {{ font-weight:700; font-size:11px; letter-spacing:.05em; }}
  .normal {{ color:#2e8b57; }} .fill.normal {{ background:#2e8b57; }}
  .monitor {{ color:#1f4e79; }} .fill.monitor {{ background:#1f4e79; }}
  .alert {{ color:#b9770e; }} .fill.alert {{ background:#d68910; }}
  .quarantine {{ color:#a04000; }} .fill.quarantine {{ background:#e67e22; }}
  .escalate {{ color:#c0392b; }} .fill.escalate {{ background:#c0392b; }}
  .banner {{ border-left:4px solid var(--red); background:#fdedec; padding:14px 16px;
             border-radius:8px; font-size:13px; }}
  .ok {{ border-left-color:var(--green); background:#eafaf1; }}
  .muted {{ color:var(--muted); font-size:12.5px; }}
  .pill {{ display:inline-block; padding:2px 9px; border-radius:99px; font-size:11px;
           background:#eef3f8; color:var(--navy); font-weight:600; margin-right:6px; }}
  select, button {{ font:inherit; padding:7px 10px; border-radius:8px; border:1px solid var(--line);
                    background:white; }}
  button {{ background:var(--navy); color:white; border:none; cursor:pointer; }}
  .demo-out {{ margin-top:12px; padding:14px; border-radius:10px; background:#f4f8fb;
               border:1px solid var(--line); font-size:13px; }}
  .risk {{ font-size:20px; font-weight:800; }}
  footer {{ text-align:center; color:var(--muted); font-size:11.5px; margin-top:40px; }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(summary['project'])}</h1>
  <p>Offline experimental dashboard · mode <strong>{html.escape(str(summary['mode']))}</strong> · seed {summary['seed']}</p>
  <p>Simulation-based evaluation. Not connected to live network traffic.</p>
</header>
<div class="wrap">

  <div class="grid kpis">
    <div class="kpi"><div class="label">Dataset</div><div class="value">{html.escape(str(summary['dataset']))}</div></div>
    <div class="kpi"><div class="label">Raw training pool</div><div class="value">{ds.get('train_rows', 0):,}</div></div>
    <div class="kpi"><div class="label">Original features</div><div class="value">{ds.get('n_features', 0)}</div></div>
    <div class="kpi"><div class="label">Reduced features</div><div class="value">{fs.get('n_features', '—')}</div></div>
    <div class="kpi"><div class="label">Models evaluated</div><div class="value">{len(master)}</div></div>
    <div class="kpi"><div class="label">Best F1</div><div class="value">{_fmt(best.get('f1'))}</div></div>
  </div>

  <section>
    <h2>Overview</h2>
    <div class="card">
      <p><span class="pill">Reduction: {str(fs.get('method', 'n/a')).upper()} → {fs.get('n_features', 'n/a')}</span>
         <span class="pill">Leakage audit: {'PASSED' if summary.get('leakage_audit_passed') else 'FAILED / not run'}</span>
         <span class="pill">Test rows: {test_rows}</span>
         <span class="pill">Best model: {MODEL_LABELS.get(best.get('model', ''), '—')}</span></p>
      <div class="banner {'ok' if summary.get('leakage_audit_passed') else ''}">
        <strong>Finding.</strong> {html.escape(findings.get('quantum_comparison', 'n/a'))}
      </div>
    </div>
  </section>

  <section>
    <h2>Model comparison — identical test set</h2>
    <div class="card">{_table_html(master_show)}</div>
    <div class="cols" style="margin-top:16px">
      <div class="card">{img('model_comparison.png', 'Metric comparison across models')}</div>
      <div class="card">{img('confusion_matrices.png', 'Confusion matrices on the common test set')}</div>
    </div>
    <div class="cols" style="margin-top:16px">
      <div class="card">{img('roc_comparison.png', 'ROC curves')}</div>
      <div class="card">{img('pr_comparison.png', 'Precision–recall curves')}</div>
    </div>
  </section>

  <section>
    <h2>Statistical comparison</h2>
    <div class="card">{_table_html(mcnemar[['comparison','n_discordant','p_value','significant','effect_interpretation']] if not mcnemar.empty and 'p_value' in mcnemar.columns else mcnemar)}
      <p class="muted">McNemar tests whether the pattern of errors differs between model pairs. A non-significant result does not prove equivalence.</p>
    </div>
  </section>

  <section>
    <h2>Cross-validation</h2>
    <div class="card">{_table_html(cv)}</div>
    <div class="card" style="margin-top:16px">{img('cross_validation_f1.png', 'F1 across cross-validation folds')}</div>
  </section>

  <section>
    <h2>Unseen attack-category generalisation</h2>
    <div class="card">{_table_html(unseen[['model','encoding','recall','precision','f1','roc_auc','unseen_attack_recall']] if not unseen.empty and 'unseen_attack_recall' in unseen.columns else unseen)}
      <p class="muted">R2L and U2R categories were removed from training entirely. This is unseen-category generalisation, not a zero-day claim.</p>
    </div>
    <div class="card" style="margin-top:16px">{img('unseen_attack_performance.png', 'Unseen attack-category performance')}</div>
  </section>

  <section>
    <h2>Quantum information</h2>
    <div class="card">{_table_html(qr)}</div>
    <div class="cols" style="margin-top:16px">
      <div class="card">{img('quantum_resources.png', 'Quantum resource comparison')}</div>
      <div class="card">{img('encoding_comparison.png', 'Angle vs amplitude encoding')}</div>
    </div>
    <div class="cols" style="margin-top:16px">
      <div class="card">{img('qsvm_circuit.png', 'QSVM feature-map circuit (sample)')}</div>
      <div class="card">{img('vqc_full_circuit.png', 'VQC full circuit (sample)')}</div>
    </div>
    <div class="card" style="margin-top:16px">{_table_html(enc)}</div>
  </section>

  <section>
    <h2>Latency and throughput</h2>
    <div class="card">{_table_html(lat)}</div>
    <div class="card" style="margin-top:16px">{img('latency_comparison.png', 'Inference latency and throughput')}</div>
  </section>

  <section>
    <h2>Feature reduction</h2>
    <div class="cols">
      <div class="card">{img('pca_explained_variance.png', 'PCA explained variance')}</div>
      <div class="card">{img('pca_component_loadings.png', 'PCA component loadings')}</div>
    </div>
    <div class="card" style="margin-top:16px">{img('permutation_importance.png', 'Permutation importance (CNN+LSTM)')}</div>
  </section>

  <section>
    <h2>Alert triage</h2>
    <div class="card">
      <div class="alertbar"><span class="lvl normal">NORMAL</span><div class="bar"><div class="fill normal" style="width:{level_counts.get('NORMAL', 0) / total * 100:.2f}%"></div></div><span class="count">{level_counts.get('NORMAL', 0)}</span></div>
      <div class="alertbar"><span class="lvl monitor">MONITOR</span><div class="bar"><div class="fill monitor" style="width:{level_counts.get('MONITOR', 0) / total * 100:.2f}%"></div></div><span class="count">{level_counts.get('MONITOR', 0)}</span></div>
      <div class="alertbar"><span class="lvl alert">ALERT</span><div class="bar"><div class="fill alert" style="width:{level_counts.get('ALERT', 0) / total * 100:.2f}%"></div></div><span class="count">{level_counts.get('ALERT', 0)}</span></div>
      <div class="alertbar"><span class="lvl quarantine">QUARANTINE</span><div class="bar"><div class="fill quarantine" style="width:{level_counts.get('QUARANTINE', 0) / total * 100:.2f}%"></div></div><span class="count">{level_counts.get('QUARANTINE', 0)}</span></div>
      <div class="alertbar"><span class="lvl escalate">ESCALATE</span><div class="bar"><div class="fill escalate" style="width:{level_counts.get('ESCALATE', 0) / total * 100:.2f}%"></div></div><span class="count">{level_counts.get('ESCALATE', 0)}</span></div>
      <p class="muted">Distribution over the common test set using the primary model. Policy: p&lt;{policy.get('normal_max', 0.20)} NORMAL, &lt;{policy.get('monitor_max', 0.45)} MONITOR, &lt;{policy.get('alert_max', 0.70)} ALERT, &lt;{policy.get('quarantine_max', 0.90)} QUARANTINE, otherwise ESCALATE. U2R/R2L predictions receive a one-tier severity bump.</p>
    </div>
    <div class="card" style="margin-top:16px">
      <h3>Alert demo — select a captured test sample</h3>
      <select id="sampleSelect"></select>
      <button onclick="assess()">Assess sample</button>
      <div class="demo-out" id="demoOut">Select a sample and press <em>Assess sample</em>.</div>
      <p class="muted">These are pre-scored offline test samples, not live traffic.</p>
    </div>
  </section>

  <section>
    <h2>Distribution shift</h2>
    <div class="cols">
      <div class="card">{img('drift_gaussian_noise.png', 'Performance under Gaussian covariate shift')}</div>
      <div class="card">{img('drift_feature_permutation.png', 'Performance under feature permutation')}</div>
    </div>
    <p class="muted">Controlled simulation; NSL-KDD has no timestamps so no real temporal drift is claimed.</p>
  </section>

  <footer>
    Generated from executed artifacts · {html.escape(str(summary.get('environment', {}).get('timestamp_utc', '')))} ·
    Quantum results: Qiskit statevector simulation only.
  </footer>
</div>

<script>
const SAMPLES = {json.dumps(demo_rows)};
const POLICY = {json.dumps(policy)};
const sel = document.getElementById('sampleSelect');
SAMPLES.forEach((s, i) => {{
  const o = document.createElement('option');
  o.value = i;
  o.textContent = `#${{i+1}} · true=${{s.true}} · p(attack)=${{s.score.toFixed(3)}} · ${{s.category}}`;
  sel.appendChild(o);
}});
function tier(p, cat) {{
  let base;
  if (p < POLICY.normal_max) base = 0;
  else if (p < POLICY.monitor_max) base = 1;
  else if (p < POLICY.alert_max) base = 2;
  else if (p < POLICY.quarantine_max) base = 3;
  else base = 4;
  const bump = (POLICY.severity_bump_categories || []).includes(cat) && p >= POLICY.monitor_max;
  if (bump) base = Math.min(4, base + (POLICY.severity_bump || 1));
  return base;
}}
const LEVELS = ['NORMAL','MONITOR','ALERT','QUARANTINE','ESCALATE'];
const ACTIONS = POLICY.actions || {{}};
function assess() {{
  const i = parseInt(sel.value || '0', 10);
  const s = SAMPLES[i];
  const lvl = tier(s.score, s.category);
  const name = LEVELS[lvl];
  const color = ['#2e8b57','#1f4e79','#d68910','#e67e22','#c0392b'][lvl];
  document.getElementById('demoOut').innerHTML = `
    <div>Predicted class: <strong>${{s.score >= 0.5 ? 'ATTACK' : 'NORMAL'}}</strong>
    &nbsp;·&nbsp; confidence: <strong>${{Math.max(s.score, 1-s.score).toFixed(4)}}</strong>
    &nbsp;·&nbsp; p(attack): <strong>${{s.score.toFixed(4)}}</strong></div>
    <div style="margin-top:8px">Risk level: <span class="risk" style="color:${{color}}">${{name}}</span></div>
    <div>Recommended action: <strong>${{ACTIONS[name] || 'NOTIFY_ANALYST'}}</strong></div>
    <div class="muted" style="margin-top:6px">Ground-truth label for this offline sample: ${{s.true}} · category: ${{s.category || 'n/a'}}</div>`;
}}
</script>
</body>
</html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Presentation assets
# ---------------------------------------------------------------------------
PRESENTATION_MAP = {
    "architecture.png": "architecture.png",
    "comparison.png": "model_comparison.png",
    "confusion_matrices.png": "confusion_matrices.png",
    "roc_comparison.png": "roc_comparison.png",
    "latency_comparison.png": "latency_comparison.png",
    "quantum_resources.png": "quantum_resources.png",
    "zero_day_results.png": "unseen_attack_performance.png",
    "feature_reduction.png": "pca_explained_variance.png",
    "encoding_comparison.png": "encoding_comparison.png",
    "cross_validation.png": "cross_validation_f1.png",
    "threshold_analysis.png": "cnn_lstm_threshold_analysis_test.png",
    "training_curves.png": "cnn_lstm_training_curves.png",
}


def copy_presentation_assets(cfg) -> list[str]:
    fig_dir = Path(cfg.get_path("paths.figures"))
    out_dir = Path(cfg.get_path("paths.presentation"))
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for dest, src in PRESENTATION_MAP.items():
        source = fig_dir / src
        if source.exists():
            shutil.copy2(source, out_dir / dest)
            copied.append(str(out_dir / dest))
    arch = Path(cfg.get_path("paths.docs")) / "architecture.png"
    if arch.exists():
        shutil.copy2(arch, out_dir / "architecture.png")
        copied.append(str(out_dir / "architecture.png"))
    return copied


# ---------------------------------------------------------------------------
# Top-level report generation
# ---------------------------------------------------------------------------
def generate_reports(cfg) -> dict[str, Any]:
    summary = collect_results(cfg)
    findings = derive_findings(summary)
    summary["findings"] = findings
    summary["limitations"] = LIMITATIONS

    results_dir = Path(cfg.get_path("paths.results"))
    save_json(summary, results_dir / "final_summary.json")

    md = render_summary_md(summary, findings)
    (results_dir / "final_summary.md").write_text(md, encoding="utf-8")

    docs = Path(cfg.get_path("paths.docs"))
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "results.md").write_text(md, encoding="utf-8")
    (docs / "paper_draft.md").write_text(render_paper_md(summary, findings), encoding="utf-8")

    dashboard = build_static_dashboard(cfg, summary, findings)
    copied = copy_presentation_assets(cfg)

    from src.evaluation.plots import make_architecture_diagram

    if not (docs / "architecture.png").exists():
        make_architecture_diagram(docs / "architecture.png")
        copied += copy_presentation_assets(cfg)

    LOGGER.info("Reports written: %s", results_dir / "final_summary.md")
    return {
        "summary_json": str(results_dir / "final_summary.json"),
        "summary_md": str(results_dir / "final_summary.md"),
        "paper_draft": str(docs / "paper_draft.md"),
        "dashboard": str(dashboard),
        "presentation_assets": copied,
        "findings": findings,
    }
