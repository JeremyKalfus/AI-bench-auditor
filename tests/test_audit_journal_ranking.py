import json
from pathlib import Path

from ai_scientist.audits.schema import (
    build_example_audit_results,
    build_example_metrics_before_after,
)
from ai_scientist.treesearch import journal as journal_module
from ai_scientist.treesearch.journal import Journal, Node
from ai_scientist.treesearch.utils.metric import MetricValue


def write_audit_artifacts(
    artifact_dir: Path,
    *,
    audit_score: float = 82.5,
    evidence_coverage: float = 0.8,
    findings_total: int = 3,
    include_metrics_before_after: bool = False,
    reproducibility_signal: float | None = None,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    audit_results = build_example_audit_results()
    audit_results["audit_score"]["value"] = audit_score
    audit_results["confidence"]["evidence_coverage"] = evidence_coverage
    audit_results["findings_summary"]["total_findings"] = findings_total
    audit_results["findings_summary"]["open_findings"] = findings_total
    if findings_total == 0:
        audit_results["findings_summary"]["by_severity"] = {}
        audit_results["findings_summary"]["by_detector"] = {}

    (artifact_dir / "audit_results.json").write_text(
        json.dumps(audit_results, indent=2)
    )

    if include_metrics_before_after:
        metrics_before_after = build_example_metrics_before_after()
        metrics_before_after["provenance"]["dataset_fingerprint"] = audit_results[
            "provenance"
        ]["dataset_fingerprint"]
        metrics_before_after["split_information"][
            "split_manifest_path"
        ] = "split_manifest.json"
        (artifact_dir / "metrics_before_after.json").write_text(
            json.dumps(metrics_before_after, indent=2)
        )

    if reproducibility_signal is not None:
        (artifact_dir / "reproducibility_summary.json").write_text(
            json.dumps(
                {
                    "reproducibility_score": reproducibility_signal,
                },
                indent=2,
            )
        )


def make_audit_node(node_id: str, artifact_dir: Path) -> Node:
    return Node(
        id=node_id,
        code="print('audit')",
        is_buggy=False,
        is_buggy_plots=False,
        metric=MetricValue(0.0, maximize=True, name="audit_score"),
        exp_results_dir=str(artifact_dir),
    )


def make_non_audit_node(node_id: str, metric_value: float) -> Node:
    return Node(
        id=node_id,
        code="print('paper')",
        is_buggy=False,
        is_buggy_plots=False,
        metric=MetricValue(metric_value, maximize=True, name="accuracy"),
    )


def test_audit_journal_ranking_bypasses_llm_and_prefers_higher_audit_score(
    monkeypatch, tmp_path
):
    lower_dir = tmp_path / "lower"
    higher_dir = tmp_path / "higher"
    write_audit_artifacts(lower_dir, audit_score=71.0)
    write_audit_artifacts(higher_dir, audit_score=88.0)

    def fail_query(*args, **kwargs):
        raise AssertionError("query should not be called for audit ranking")

    monkeypatch.setattr(journal_module, "query", fail_query)

    journal = Journal(
        nodes=[
            make_audit_node("node-low", lower_dir),
            make_audit_node("node-high", higher_dir),
        ]
    )

    best_node = journal.get_best_node(cfg=None)

    assert best_node is not None
    assert best_node.id == "node-high"


def test_audit_journal_ranking_uses_evidence_coverage_as_second_tiebreaker(tmp_path):
    lower_dir = tmp_path / "lower_coverage"
    higher_dir = tmp_path / "higher_coverage"
    write_audit_artifacts(lower_dir, audit_score=82.5, evidence_coverage=0.41)
    write_audit_artifacts(higher_dir, audit_score=82.5, evidence_coverage=0.97)

    journal = Journal(
        nodes=[
            make_audit_node("node-low-coverage", lower_dir),
            make_audit_node("node-high-coverage", higher_dir),
        ]
    )

    best_node = journal.get_best_node(cfg=None)

    assert best_node is not None
    assert best_node.id == "node-high-coverage"


def test_audit_journal_ranking_prefers_remediation_confirmation(tmp_path):
    no_remediation_dir = tmp_path / "no_remediation"
    remediation_dir = tmp_path / "with_remediation"
    write_audit_artifacts(
        no_remediation_dir,
        audit_score=82.5,
        evidence_coverage=0.9,
        findings_total=2,
        include_metrics_before_after=False,
    )
    write_audit_artifacts(
        remediation_dir,
        audit_score=82.5,
        evidence_coverage=0.9,
        findings_total=2,
        include_metrics_before_after=True,
    )

    journal = Journal(
        nodes=[
            make_audit_node("node-unconfirmed", no_remediation_dir),
            make_audit_node("node-confirmed", remediation_dir),
        ]
    )

    best_node = journal.get_best_node(cfg=None)

    assert best_node is not None
    assert best_node.id == "node-confirmed"


def test_audit_journal_ranking_prefers_reproducibility_signal_when_present(tmp_path):
    lower_signal_dir = tmp_path / "lower_signal"
    higher_signal_dir = tmp_path / "higher_signal"
    write_audit_artifacts(
        lower_signal_dir,
        audit_score=82.5,
        evidence_coverage=0.9,
        findings_total=0,
        reproducibility_signal=0.2,
    )
    write_audit_artifacts(
        higher_signal_dir,
        audit_score=82.5,
        evidence_coverage=0.9,
        findings_total=0,
        reproducibility_signal=0.8,
    )

    journal = Journal(
        nodes=[
            make_audit_node("node-lower-signal", lower_signal_dir),
            make_audit_node("node-higher-signal", higher_signal_dir),
        ]
    )

    best_node = journal.get_best_node(cfg=None)

    assert best_node is not None
    assert best_node.id == "node-higher-signal"


def test_audit_journal_ranking_breaks_full_ties_by_node_id(tmp_path):
    b_dir = tmp_path / "b"
    a_dir = tmp_path / "a"
    write_audit_artifacts(a_dir, audit_score=82.5, evidence_coverage=0.9, findings_total=0)
    write_audit_artifacts(b_dir, audit_score=82.5, evidence_coverage=0.9, findings_total=0)

    journal = Journal(
        nodes=[
            make_audit_node("node-b", b_dir),
            make_audit_node("node-a", a_dir),
        ]
    )

    best_node = journal.get_best_node(cfg=None)

    assert best_node is not None
    assert best_node.id == "node-a"


def test_audit_journal_get_ranked_nodes_uses_audit_precedence(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    third_dir = tmp_path / "third"
    write_audit_artifacts(
        first_dir,
        audit_score=82.5,
        evidence_coverage=0.95,
        findings_total=2,
        include_metrics_before_after=True,
    )
    write_audit_artifacts(
        second_dir,
        audit_score=82.5,
        evidence_coverage=0.95,
        findings_total=2,
        include_metrics_before_after=False,
    )
    write_audit_artifacts(
        third_dir,
        audit_score=79.0,
        evidence_coverage=1.0,
        findings_total=0,
    )

    journal = Journal(
        nodes=[
            make_audit_node("node-third", third_dir),
            make_audit_node("node-second", second_dir),
            make_audit_node("node-first", first_dir),
        ]
    )

    ranked_ids = [node.id for node in journal.get_ranked_nodes()]

    assert ranked_ids == ["node-first", "node-second", "node-third"]


def test_non_audit_journal_selection_still_uses_query(monkeypatch):
    calls = []

    def fake_query(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return {"selected_id": "paper-b", "reasoning": "better overall"}

    monkeypatch.setattr(journal_module, "query", fake_query)

    journal = Journal(
        nodes=[
            make_non_audit_node("paper-a", 0.2),
            make_non_audit_node("paper-b", 0.3),
        ]
    )

    best_node = journal.get_best_node(cfg=None)

    assert best_node is not None
    assert best_node.id == "paper-b"
    assert len(calls) == 1
