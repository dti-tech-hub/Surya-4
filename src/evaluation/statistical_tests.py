"""Statistical comparison utilities.

What each test establishes
--------------------------
* McNemar's test: whether two classifiers make *different* errors on the same
  paired test samples.  It does not measure the size of the difference.
* Paired Wilcoxon signed-rank: whether per-fold scores differ systematically.
  It is underpowered with 3-5 folds and is reported as supporting evidence only.
* Bootstrap CI: uncertainty of a metric on a finite test set.
* Cohen's h: practical magnitude of a difference between two proportions.

No test here can establish "quantum advantage"; they only quantify whether the
observed differences are unlikely under the null hypothesis of equal
performance on this particular sample.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from scipy import stats


def _interpret_p(p: float, alpha: float = 0.05) -> str:
    if not np.isfinite(p):
        return "not interpretable (test undefined)"
    if p < alpha:
        return f"statistically significant at alpha={alpha}"
    return f"not significant at alpha={alpha}"


def cohens_h(p1: float, p2: float) -> float:
    p1 = float(np.clip(p1, 1e-6, 1 - 1e-6))
    p2 = float(np.clip(p2, 1e-6, 1 - 1e-6))
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def _effect_label(h: float) -> str:
    a = abs(h)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    alpha: float = 0.05,
    name_a: str = "model_a",
    name_b: str = "model_b",
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    a_correct = (np.asarray(y_pred_a).astype(int) == y_true)
    b_correct = (np.asarray(y_pred_b).astype(int) == y_true)

    n10 = int(np.sum(a_correct & ~b_correct))   # A right, B wrong
    n01 = int(np.sum(~a_correct & b_correct))   # A wrong, B right
    n00 = int(np.sum(~a_correct & ~b_correct))
    n11 = int(np.sum(a_correct & b_correct))
    discordant = n10 + n01

    if discordant == 0:
        return {
            "test": "McNemar",
            "comparison": f"{name_a} vs {name_b}",
            "n_discordant": 0,
            "p_value": 1.0,
            "statistic": 0.0,
            "method": "undefined (no discordant pairs)",
            "interpretation": "The two models make identical correct/incorrect decisions.",
            "alpha": alpha,
            "significant": False,
            "n10_a_only_correct": n10,
            "n01_b_only_correct": n01,
            "n00_both_wrong": n00,
            "n11_both_correct": n11,
            "odds_ratio": None,
            "effect_size_cohens_h": 0.0,
            "effect_interpretation": "none",
        }

    if discordant < 25:
        result = stats.binomtest(min(n10, n01), discordant, 0.5, alternative="two-sided")
        p_value = float(result.pvalue)
        statistic = float(min(n10, n01))
        method = "exact binomial"
    else:
        statistic = (abs(n10 - n01) - 1) ** 2 / discordant
        p_value = float(stats.chi2.sf(statistic, df=1))
        method = "chi-square with continuity correction"

    acc_a = float(a_correct.mean())
    acc_b = float(b_correct.mean())
    h = cohens_h(acc_a, acc_b)
    return {
        "test": "McNemar",
        "comparison": f"{name_a} vs {name_b}",
        "statistic": statistic,
        "p_value": p_value,
        "method": method,
        "alpha": alpha,
        "significant": bool(p_value < alpha),
        "interpretation": _interpret_p(p_value, alpha),
        "n10_a_only_correct": n10,
        "n01_b_only_correct": n01,
        "n00_both_wrong": n00,
        "n11_both_correct": n11,
        "n_discordant": discordant,
        "accuracy_a": acc_a,
        "accuracy_b": acc_b,
        "accuracy_difference_a_minus_b": acc_a - acc_b,
        "odds_ratio": (n10 / n01) if n01 else None,
        "effect_size_cohens_h": h,
        "effect_interpretation": _effect_label(h),
        "note": (
            "McNemar tests whether the pattern of errors differs; a "
            "non-significant result does not prove equivalence."
        ),
    }


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
    name_a: str = "model_a",
    name_b: str = "model_b",
    metric: str = "f1",
) -> dict[str, Any]:
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    diff = a - b
    if len(a) < 3:
        return {
            "test": "Wilcoxon signed-rank",
            "comparison": f"{name_a} vs {name_b}",
            "metric": metric,
            "n_folds": int(len(a)),
            "statistic": None,
            "p_value": None,
            "significant": False,
            "interpretation": "insufficient folds for a meaningful test",
        }
    if np.allclose(diff, 0):
        return {
            "test": "Wilcoxon signed-rank",
            "comparison": f"{name_a} vs {name_b}",
            "metric": metric,
            "n_folds": int(len(a)),
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "interpretation": "identical fold scores",
            "mean_difference": 0.0,
        }
    try:
        res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        statistic = float(res.statistic)
        p_value = float(res.pvalue)
    except ValueError:
        statistic, p_value = float("nan"), float("nan")
    return {
        "test": "Wilcoxon signed-rank",
        "comparison": f"{name_a} vs {name_b}",
        "metric": metric,
        "n_folds": int(len(a)),
        "statistic": statistic,
        "p_value": p_value,
        "alpha": alpha,
        "significant": bool(np.isfinite(p_value) and p_value < alpha),
        "interpretation": _interpret_p(p_value, alpha),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_difference": float(np.mean(diff)),
        "std_difference": float(np.std(diff, ddof=1)) if len(diff) > 1 else None,
        "note": (
            "With only 3-5 folds the minimum attainable two-sided p-value is "
            "0.25 (3 folds) or 0.0625 (5 folds); treat as supporting evidence."
        ),
    }


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Percentile bootstrap CI for a metric (resampling samples)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            values.append(float(metric_fn(yt, y_score[idx])))
        except Exception:
            continue
    if not values:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_bootstrap": 0}
    point = float(metric_fn(y_true, y_score))
    lo = float(np.percentile(values, 100 * alpha / 2))
    hi = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return {
        "point": point,
        "ci_low": lo,
        "ci_high": hi,
        "ci_level": 1 - alpha,
        "n_bootstrap": len(values),
        "bootstrap_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
    }


def paired_bootstrap_difference(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    metric_fn_b: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> dict[str, float]:
    """Paired bootstrap CI of metric(A) - metric(B) on identical samples.

    ``metric_fn_b`` allows the two models to use different decision rules
    (e.g. separately validation-tuned thresholds).  It defaults to
    ``metric_fn``.
    """
    y_true = np.asarray(y_true).astype(int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    metric_fn_b = metric_fn_b or metric_fn
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            diffs.append(float(metric_fn(yt, a[idx]) - metric_fn_b(yt, b[idx])))
        except Exception:
            continue
    if not diffs:
        return {"difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    point = float(metric_fn(y_true, a) - metric_fn_b(y_true, b))
    return {
        "difference": point,
        "ci_low": float(np.percentile(diffs, 100 * alpha / 2)),
        "ci_high": float(np.percentile(diffs, 100 * (1 - alpha / 2))),
        "ci_level": 1 - alpha,
        "n_bootstrap": len(diffs),
        "crosses_zero": bool(
            np.percentile(diffs, 100 * alpha / 2) <= 0 <= np.percentile(diffs, 100 * (1 - alpha / 2))
        ),
    }


def summarize_cv(
    cv_df,
    metric: str = "f1",
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Mean / std / 95% CI of a metric per model across folds."""
    from scipy import stats as st

    rows = []
    for model, group in cv_df.groupby("model"):
        values = group[metric].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        if len(values) > 1:
            sem = std / np.sqrt(len(values))
            tcrit = float(st.t.ppf(1 - alpha / 2, df=len(values) - 1))
            ci = (mean - tcrit * sem, mean + tcrit * sem)
        else:
            ci = (float("nan"), float("nan"))
        rows.append({
            "model": model,
            "metric": metric,
            "n_folds": int(len(values)),
            "mean": mean,
            "std": std,
            "ci_low": ci[0],
            "ci_high": ci[1],
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "fold_values": values.tolist(),
        })
    return rows
