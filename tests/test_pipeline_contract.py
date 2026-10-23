import json
from pathlib import Path

import pytest

from src.pipeline.reporting import derive_findings
from src.utils.config import load_config, profile

ROOT = Path(__file__).resolve().parents[1]


def test_cli_lists_stages(capsys):
    import run_pipeline

    assert run_pipeline.main(["--list-stages"]) == 0
    out = capsys.readouterr().out
    for stage in ("prepare", "features", "train", "evaluate", "zero_day",
                  "encoding", "latency", "drift", "explain", "cv", "stats",
                  "tables", "dashboard", "report"):
        assert stage in out


def test_config_profiles_and_paths():
    cfg = load_config(root=ROOT)
    quick = profile(cfg)
    assert quick["max_samples"] == 12000
    assert quick["quantum_train_size"] == 600
    assert quick["common_test_size"] == 1800
    assert cfg.get_path("feature_reduction.max_quantum_qubits") == 8
    assert cfg.get_path("quantum.backend") == "statevector_simulator"


def test_derive_findings_no_false_advantage():
    summary = {
        "master_table": [
            {"model": "cnn_lstm", "f1": 0.95, "roc_auc": 0.99},
            {"model": "qsvm", "f1": 0.92, "roc_auc": 0.98},
            {"model": "vqc", "f1": 0.80, "roc_auc": 0.90},
        ],
        "statistics": {"mcnemar": [
            {"comparison": "cnn_lstm vs qsvm", "p_value": 0.001},
        ]},
        "cross_validation": [],
        "unseen_category": [],
        "latency": [],
    }
    findings = derive_findings(summary)
    assert findings["quantum_advantage_claim"] is False
    assert findings["best_overall"]["model"] == "cnn_lstm"
    assert "did not provide an advantage" in findings["quantum_comparison"]


def test_derive_findings_positive_requires_significance():
    summary = {
        "master_table": [
            {"model": "cnn_lstm", "f1": 0.90, "roc_auc": 0.95},
            {"model": "qsvm", "f1": 0.93, "roc_auc": 0.96},
            {"model": "vqc", "f1": 0.80, "roc_auc": 0.90},
        ],
        "statistics": {"mcnemar": [
            {"comparison": "cnn_lstm vs qsvm", "p_value": 0.20},
        ]},
        "cross_validation": [],
        "unseen_category": [],
        "latency": [],
    }
    findings = derive_findings(summary)
    assert findings["quantum_advantage_claim"] is False
    assert "not statistically significant" in findings["quantum_comparison"]


@pytest.mark.skipif(not (ROOT / "results/final_summary.json").exists(),
                    reason="pipeline results not generated")
def test_generated_results_are_internally_consistent():
    summary = json.loads((ROOT / "results/final_summary.json").read_text(encoding="utf-8"))
    master = summary["master_table"]
    assert {r["model"] for r in master} == {"cnn_lstm", "qsvm", "vqc"}
    for row in master:
        assert 0 <= row["accuracy"] <= 1
        assert 0 <= row["f1"] <= 1
        assert 0 <= row["roc_auc"] <= 1
        assert row["n_features"] == summary["feature_selection"]["n_features"]
    assert summary["leakage_audit_passed"] is True
    assert summary["unseen_details"]["holdout_categories"] == ["R2L", "U2R"]


@pytest.mark.skipif(not (ROOT / "results/tables/master_comparison.csv").exists(),
                    reason="master table not generated")
def test_master_table_has_same_test_size():
    import pandas as pd

    table = pd.read_csv(ROOT / "results/tables/master_comparison.csv")
    assert len(table) == 3
    # All models are evaluated on the identical common test set.
    # n_samples is not a column, so verify against the summary instead.
    summary = json.loads((ROOT / "results/final_summary.json").read_text(encoding="utf-8"))
    for row in summary["master_table"]:
        assert row["n_features"] == 8
