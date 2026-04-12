from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


SEVERITY_WEIGHTS = {
    "low": 10.0,
    "medium": 25.0,
    "high": 40.0,
    "critical": 60.0,
}


@dataclass(frozen=True)
class FindingScoreInput:
    severity: str
    confirmed: bool
    evidence_completeness: float
    remediation_effect_size: float


def score_branch(
    findings: Sequence[FindingScoreInput],
    *,
    negative_control_penalty: float = 0.0,
) -> dict[str, float | str]:
    score = 0.0
    for finding in findings:
        if not finding.confirmed:
            continue
        severity_weight = SEVERITY_WEIGHTS[finding.severity]
        evidence_completeness = min(max(finding.evidence_completeness, 0.0), 1.0)
        remediation_effect = min(max(abs(finding.remediation_effect_size), 0.0), 1.0)
        score += severity_weight * (0.5 + 0.5 * evidence_completeness)
        score += remediation_effect * 20.0

    if not findings:
        score = 15.0

    penalty = max(0.0, negative_control_penalty) * 20.0
    value = max(0.0, min(100.0, score - penalty))
    if value >= 70.0:
        rating = "critical"
    elif value >= 35.0:
        rating = "warning"
    else:
        rating = "clean"

    return {
        "value": round(value, 2),
        "max_value": 100.0,
        "rating": rating,
    }
