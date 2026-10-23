import numpy as np
import pandas as pd

from src.alerting.risk_engine import AlertPolicy, RiskEngine
from src.evaluation.metrics import (
    compute_metrics,
    select_operating_threshold,
    threshold_analysis,
)
from src.evaluation.statistical_tests import (
    bootstrap_metric_ci,
    mcnemar_test,
    paired_bootstrap_difference,
    summarize_cv,
    wilcoxon_test,
)


def test_compute_metrics_known_confusion():
    y = np.array([0, 0, 1, 1, 1, 0])
    scores = np.array([0.1, 0.2, 0.9, 0.6, 0.4, 0.8])
    m = compute_metrics(y, scores, threshold=0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 1, 1, 2)
    assert m["accuracy"] == 4 / 6
    assert m["recall"] == 2 / 3
    assert m["precision"] == 2 / 3
    assert m["confusion_matrix"] == [[2, 1], [1, 2]]
    assert 0 <= m["brier_score"] <= 1


def test_threshold_analysis_and_selection():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    scores = np.clip(0.7 * y + rng.normal(0, 0.25, 200), 0, 1)
    table = threshold_analysis(y, scores, thresholds=[0.2, 0.5, 0.8])
    assert len(table) == 3
    assert set(["fpr", "fnr", "expected_cost", "tp", "fp"]).issubset(table.columns)
    chosen = select_operating_threshold(y, scores, objective="f1", thresholds=np.linspace(0.05, 0.95, 19))
    assert 0.05 <= chosen["threshold"] <= 0.95
    assert "table" in chosen


def test_mcnemar_identical_predictions():
    y = np.array([0, 1, 0, 1, 1, 0])
    result = mcnemar_test(y, y.copy(), y.copy(), name_a="a", name_b="b")
    assert result["n_discordant"] == 0
    assert result["p_value"] == 1.0
    assert not result["significant"]


def test_mcnemar_detects_difference():
    y = np.array([1] * 40 + [0] * 40)
    perfect = y.copy()
    bad = np.array([1] * 30 + [0] * 10 + [0] * 30 + [1] * 10)
    result = mcnemar_test(y, perfect, bad, name_a="perfect", name_b="bad")
    assert result["n_discordant"] == 20
    assert result["p_value"] < 0.05
    assert result["significant"]


def test_wilcoxon_and_bootstrap():
    from sklearn.metrics import f1_score

    def f1_fn(yt, ys):
        return f1_score(yt, (ys >= 0.5).astype(int), zero_division=0)

    a = np.array([0.90, 0.91, 0.93, 0.92, 0.94])
    b = np.array([0.80, 0.82, 0.81, 0.83, 0.79])
    w = wilcoxon_test(a, b, name_a="a", name_b="b")
    assert "p_value" in w
    assert not np.isnan(w["p_value"])
    ci = bootstrap_metric_ci(
        np.array([0, 1, 1, 0, 1, 0]),
        np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.4]),
        metric_fn=f1_fn, n_boot=100, seed=42,
    )
    assert 0 <= ci["ci_low"] <= ci["ci_high"] <= 1
    diff = paired_bootstrap_difference(
        np.array([0, 1, 1, 0, 1, 0]),
        np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.4]),
        np.array([0.6, 0.4, 0.3, 0.7, 0.2, 0.8]),
        metric_fn=f1_fn, n_boot=100, seed=42,
    )
    assert diff["difference"] > 0
    assert diff["ci_low"] <= diff["difference"] <= diff["ci_high"]


def test_summarize_cv():
    rows = []
    for model in ("a", "b"):
        for fold, value in enumerate([0.8, 0.82, 0.84], start=1):
            rows.append({"model": model, "fold": fold, "f1": value})
    summary = summarize_cv(pd.DataFrame(rows), metric="f1")
    assert len(summary) == 2
    assert all(r["n_folds"] == 3 for r in summary)
    assert all("ci_low" in r and "ci_high" in r for r in summary)


def test_alert_engine_tiers():
    engine = RiskEngine(AlertPolicy())
    assert engine.assess(0.01)["risk_level"] == "NORMAL"
    assert engine.assess(0.30)["risk_level"] == "MONITOR"
    assert engine.assess(0.50)["risk_level"] == "ALERT"
    assert engine.assess(0.80)["risk_level"] == "QUARANTINE"
    assert engine.assess(0.99)["risk_level"] == "ESCALATE"
    bumped = engine.assess(0.50, "R2L")
    assert bumped["risk_level"] == "QUARANTINE"
    assert bumped["severity_bumped"]
    # A low probability is not escalated merely because of the category hint.
    assert engine.assess(0.30, "R2L")["risk_level"] == "MONITOR"
