from ai_scientist.audits.scoring import FindingScoreInput, score_branch


def test_score_branch_orders_stronger_findings_higher():
    weak = score_branch(
        [
            FindingScoreInput(
                severity="medium",
                confirmed=True,
                evidence_completeness=0.5,
                remediation_effect_size=0.1,
            )
        ]
    )
    strong = score_branch(
        [
            FindingScoreInput(
                severity="high",
                confirmed=True,
                evidence_completeness=1.0,
                remediation_effect_size=0.6,
            )
        ]
    )

    assert strong["value"] > weak["value"]


def test_score_branch_applies_negative_control_penalty():
    unpenalized = score_branch(
        [
            FindingScoreInput(
                severity="high",
                confirmed=True,
                evidence_completeness=1.0,
                remediation_effect_size=0.4,
            )
        ],
        negative_control_penalty=0.0,
    )
    penalized = score_branch(
        [
            FindingScoreInput(
                severity="high",
                confirmed=True,
                evidence_completeness=1.0,
                remediation_effect_size=0.4,
            )
        ],
        negative_control_penalty=0.8,
    )

    assert penalized["value"] < unpenalized["value"]


def test_score_branch_assigns_clean_rating_to_clean_audit():
    clean = score_branch([], negative_control_penalty=0.0)
    assert clean["rating"] == "clean"
