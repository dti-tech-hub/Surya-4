"""Tiered risk and alert engine.

The engine is deliberately *assistive*: only the highest-confidence detections
are mapped to blocking actions.  Everything else is routed to an analyst.  The
thresholds are configuration, not hard-coded policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

LEVELS = ["NORMAL", "MONITOR", "ALERT", "QUARANTINE", "ESCALATE"]
_SEVERITY_INDEX = {name: i for i, name in enumerate(LEVELS)}


@dataclass
class AlertPolicy:
    normal_max: float = 0.20
    monitor_max: float = 0.45
    alert_max: float = 0.70
    quarantine_max: float = 0.90
    severity_bump_categories: tuple[str, ...] = ("U2R", "R2L")
    severity_bump: int = 1
    actions: dict[str, str] = field(default_factory=lambda: {
        "NORMAL": "ALLOW",
        "MONITOR": "LOG_AND_MONITOR",
        "ALERT": "NOTIFY_ANALYST",
        "QUARANTINE": "QUARANTINE_HOST",
        "ESCALATE": "BLOCK_AND_ESCALATE",
    })

    @classmethod
    def from_config(cls, cfg) -> "AlertPolicy":
        section = cfg.get("alerting", {}) or {}
        actions = section.get("actions", {}) or {}
        default_actions = cls().actions
        return cls(
            normal_max=float(section.get("normal_max", 0.20)),
            monitor_max=float(section.get("monitor_max", 0.45)),
            alert_max=float(section.get("alert_max", 0.70)),
            quarantine_max=float(section.get("quarantine_max", 0.90)),
            severity_bump_categories=tuple(section.get("severity_bump_categories", ("U2R", "R2L"))),
            severity_bump=int(section.get("severity_bump", 1)),
            actions={**default_actions, **{str(k): str(v) for k, v in actions.items()}},
        )

    def validate(self) -> None:
        values = [self.normal_max, self.monitor_max, self.alert_max, self.quarantine_max]
        if values != sorted(values):
            raise ValueError(f"Alert thresholds must be non-decreasing, got {values}")
        if not (0 <= values[0] and values[-1] <= 1):
            raise ValueError("Alert thresholds must lie in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "normal_max": self.normal_max,
            "monitor_max": self.monitor_max,
            "alert_max": self.alert_max,
            "quarantine_max": self.quarantine_max,
            "severity_bump_categories": list(self.severity_bump_categories),
            "severity_bump": self.severity_bump,
            "actions": dict(self.actions),
        }


class RiskEngine:
    """Map ``p(attack)`` (+ optional category) to a risk tier and an action."""

    def __init__(self, policy: Optional[AlertPolicy] = None) -> None:
        self.policy = policy or AlertPolicy()
        self.policy.validate()

    # ------------------------------------------------------------------
    def _base_level(self, probability: float) -> str:
        p = float(probability)
        if p < self.policy.normal_max:
            return "NORMAL"
        if p < self.policy.monitor_max:
            return "MONITOR"
        if p < self.policy.alert_max:
            return "ALERT"
        if p < self.policy.quarantine_max:
            return "QUARANTINE"
        return "ESCALATE"

    def assess(
        self,
        probability: float,
        attack_category: Optional[str] = None,
        predicted_label: Optional[int] = None,
    ) -> dict[str, Any]:
        probability = float(min(max(probability, 0.0), 1.0))
        base = self._base_level(probability)
        level = base

        category = (attack_category or "").strip()
        bumped = False
        if (
            category in self.policy.severity_bump_categories
            and probability >= self.policy.monitor_max
            and _SEVERITY_INDEX[base] < len(LEVELS) - 1
        ):
            level = LEVELS[_SEVERITY_INDEX[base] + max(0, self.policy.severity_bump)]
            bumped = True

        # Risk score in [0, 100]: probability drives the score, category bumps
        # it by a fixed amount so a high-confidence U2R is never under-rated.
        risk_score = probability * 100.0
        if bumped:
            risk_score = min(100.0, risk_score + 10.0 * self.policy.severity_bump)

        action = self.policy.actions.get(level, "NOTIFY_ANALYST")
        predicted = (
            int(predicted_label) if predicted_label is not None
            else int(probability >= 0.5)
        )

        reasons = [
            f"p(attack)={probability:.4f}",
            f"base_tier={base}",
        ]
        if bumped:
            reasons.append(f"severity bump for category '{category}'")
        reasons.append("assistive triage only — no automatic blocking below ESCALATE")

        return {
            "predicted_label": predicted,
            "predicted_class": "ATTACK" if predicted == 1 else "NORMAL",
            "probability": probability,
            "confidence": max(probability, 1 - probability),
            "attack_category": category or None,
            "risk_score": round(float(risk_score), 2),
            "risk_level": level,
            "action": action,
            "severity_bumped": bumped,
            "rationale": "; ".join(reasons),
        }

    # ------------------------------------------------------------------
    def assess_batch(
        self,
        probabilities: Iterable[float],
        attack_categories: Optional[Iterable[Optional[str]]] = None,
        predicted_labels: Optional[Iterable[int]] = None,
    ) -> list[dict[str, Any]]:
        probs = list(probabilities)
        cats = list(attack_categories) if attack_categories is not None else [None] * len(probs)
        labels = list(predicted_labels) if predicted_labels is not None else [None] * len(probs)
        return [
            self.assess(p, c, l) for p, c, l in zip(probs, cats, labels)
        ]

    def distribution(self, probabilities: Iterable[float],
                     attack_categories: Optional[Iterable[Optional[str]]] = None) -> dict[str, int]:
        results = self.assess_batch(probabilities, attack_categories)
        counts = {level: 0 for level in LEVELS}
        for item in results:
            counts[item["risk_level"]] += 1
        return counts


def summarize_alerts(engine: RiskEngine, probabilities, attack_categories=None,
                     labels=None) -> dict[str, Any]:
    results = engine.assess_batch(probabilities, attack_categories, labels)
    counts: dict[str, int] = {level: 0 for level in LEVELS}
    actions: dict[str, int] = {}
    for item in results:
        counts[item["risk_level"]] += 1
        actions[item["action"]] = actions.get(item["action"], 0) + 1
    total = max(len(results), 1)
    return {
        "n_samples": len(results),
        "level_counts": counts,
        "level_fractions": {k: v / total for k, v in counts.items()},
        "action_counts": actions,
        "policy": engine.policy.to_dict(),
        "examples": results[:10],
    }
