#!/usr/bin/env python
"""Interactive / scripted demo of a trained QE-IDS model.

Examples
--------
python demo.py
python demo.py --model qsvm --n-samples 20
python demo.py --all-models --sample-index 7
python demo.py --json
python demo.py --dashboard

The demo reads artefacts produced by ``run_pipeline.py``.  It performs no
training and does not modify any result file.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alerting.risk_engine import AlertPolicy, RiskEngine, summarize_alerts
from src.classical.cnn_lstm import load_cnn_lstm, predict_proba as cnn_predict_proba
from src.quantum.qsvm import QuantumSVM
from src.quantum.vqc import VariationalQuantumClassifier
from src.utils.config import load_config

MODEL_LABELS = {"cnn_lstm": "CNN+LSTM", "qsvm": "QSVM", "vqc": "VQC"}
MODEL_FILES = {"cnn_lstm": "cnn_lstm.pt", "qsvm": "qsvm.joblib", "vqc": "vqc.joblib"}


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def _load_model(cfg, kind: str):
    model_dir = Path(cfg.get_path("paths.models"))
    path = model_dir / MODEL_FILES[kind]
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\nRun: python run_pipeline.py --mode quick"
        )
    meta_path = model_dir / f"{kind}_meta.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

    if kind == "cnn_lstm":
        model, _ = load_cnn_lstm(path)

        def score(X: np.ndarray) -> np.ndarray:
            return cnn_predict_proba(model, X)
    elif kind == "qsvm":
        model = QuantumSVM.load(path)

        def score(X: np.ndarray) -> np.ndarray:
            return np.asarray(model.predict_proba(X), dtype=float)
    elif kind == "vqc":
        model = VariationalQuantumClassifier.load(path)

        def score(X: np.ndarray) -> np.ndarray:
            return np.asarray(model.predict_proba(X), dtype=float)
    else:
        raise ValueError(f"Unknown model {kind!r}")
    return score, float(meta.get("threshold", 0.5)), meta


def _fmt_table(rows: list[dict[str, Any]]) -> str:
    header = (
        f"{'idx':>4}  {'true':>6}  {'p(attack)':>9}  {'pred':>6}  "
        f"{'category':>10}  {'tier':>12}  action"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['index']:>4}  {r['true_label']:>6}  {r['probability']:>9.4f}  "
            f"{r['predicted_label']:>6}  {r['true_category']:>10}  "
            f"{r['risk_level']:>12}  {r['action']}"
        )
    return "\n".join(lines)


def run_demo(args) -> int:
    cfg = load_config(root=ROOT)
    arrays_path = Path(cfg.get_path("paths.data_processed")) / "arrays.npz"
    common_path = Path(cfg.get_path("paths.data_processed")) / "common_test.parquet"
    if not arrays_path.exists() or not common_path.exists():
        print(
            "[ERROR] Processed artefacts missing. Run:\n"
            "  python run_pipeline.py --mode quick",
            file=sys.stderr,
        )
        return 1

    arrays = np.load(arrays_path)
    X = arrays["X_test_common"]
    y = arrays["y_common"]
    common = pd.read_parquet(common_path)
    if len(common) != len(y):
        print("[ERROR] common_test.parquet does not align with arrays.npz", file=sys.stderr)
        return 1

    kinds = list(MODEL_LABELS) if args.all_models else [args.model]
    engine = RiskEngine(AlertPolicy.from_config(cfg))

    if args.sample_index is not None:
        if not (0 <= args.sample_index < len(X)):
            print(f"[ERROR] sample-index must be in [0, {len(X) - 1}]", file=sys.stderr)
            return 1
        indices = np.array([args.sample_index])
    else:
        n = max(1, min(int(args.n_samples), len(X)))
        # Deterministic, class-balanced spread across the test set.
        rng = np.random.default_rng(int(cfg.get_path("project.seed", 42)))
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        take_pos = min(len(pos), n // 2)
        take_neg = min(len(neg), n - take_pos)
        sel = np.concatenate([
            rng.choice(pos, size=take_pos, replace=False),
            rng.choice(neg, size=take_neg, replace=False),
        ])
        indices = np.sort(sel)

    payload: dict[str, Any] = {
        "mode": "demo",
        "n_samples": int(len(indices)),
        "threshold_policy": "per-model validation-selected threshold",
        "caveat": (
            "Simulation/demo only. Not a production IDS. Quantum scores come "
            "from exact statevector simulation."
        ),
        "models": {},
    }

    for kind in kinds:
        try:
            score_fn, threshold, meta = _load_model(cfg, kind)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}", file=sys.stderr)
            continue
        Xs = X[indices]
        scores = score_fn(Xs)
        t = float(args.threshold) if args.threshold is not None else threshold
        rows = []
        for i, (idx, p, truth) in enumerate(zip(indices, scores, y[indices])):
            raw = common.iloc[int(idx)]
            category = str(raw.get("attack_category", "Unknown"))
            assessment = engine.assess(float(p), category, int(float(p) >= t))
            rows.append({
                "index": int(idx),
                "true_label": "ATTACK" if int(truth) else "NORMAL",
                "true_category": category,
                "probability": float(p),
                "predicted_label": assessment["predicted_class"],
                "risk_level": assessment["risk_level"],
                "risk_score": assessment["risk_score"],
                "action": assessment["action"],
            })

        yt = y[indices].astype(int)
        yp = (scores >= t).astype(int)
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        summary = summarize_alerts(
            engine, scores, common.iloc[indices]["attack_category"].astype(str).tolist(), yp
        )
        payload["models"][kind] = {
            "label": MODEL_LABELS[kind],
            "threshold": t,
            "encoding": meta.get("encoding"),
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "alert_summary": summary,
            "samples": rows,
        }

        if not args.json:
            print("=" * 84)
            print(f"{MODEL_LABELS[kind]}  |  threshold={t:.2f}  |  "
                  f"encoding={meta.get('encoding', '-')}")
            print("=" * 84)
            print(_fmt_table(rows))
            print(
                f"\nSample confusion: TP={tp} FP={fp} TN={tn} FN={fn}  "
                f"| tiers: " + ", ".join(
                    f"{k}={v}" for k, v in summary["level_counts"].items()
                )
            )
            print()

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(
            "Scope: this demo is a research prototype, not a deployed IDS. "
            "Only ESCALATE maps to a blocking action; all other tiers are "
            "assistive triage."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    parser = argparse.ArgumentParser(description="QE-IDS demo")
    parser.add_argument("--model", choices=tuple(MODEL_LABELS), default="cnn_lstm")
    parser.add_argument("--all-models", action="store_true",
                        help="Show every trained model side by side")
    parser.add_argument("--n-samples", type=int, default=12,
                        help="Number of class-balanced test samples (default: 12)")
    parser.add_argument("--sample-index", type=int, default=None,
                        help="Score one specific row of the common test set")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the validation-selected threshold (diagnostic only)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch the Streamlit dashboard instead")
    args = parser.parse_args(argv)

    if args.dashboard:
        app = ROOT / "src/dashboard/app.py"
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])
    return run_demo(args)


if __name__ == "__main__":
    sys.exit(main())
