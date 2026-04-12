import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ai_scientist.audits.schema import (
    build_example_audit_results,
    build_example_findings_columns,
    build_example_metrics_before_after,
    build_example_split_manifest,
)
from ai_scientist.treesearch.agent_manager import (
    AUDIT_STAGE_TITLES,
    AgentManager,
    Journal,
    Node,
    Stage,
)
from ai_scientist.treesearch.utils.metric import MetricValue


class DummyAgent:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def step(self, exec_callback):
        exec_callback()

    def _run_multi_seed_evaluation(self, best_node):
        return []

    def _run_plot_aggregation(self, best_node, seed_nodes):
        return None


def make_cfg():
    return SimpleNamespace(
        agent=SimpleNamespace(
            search=SimpleNamespace(num_drafts=1),
            stages=SimpleNamespace(
                stage1_max_iters=2,
                stage2_max_iters=2,
                stage3_max_iters=2,
                stage4_max_iters=2,
            ),
            steps=2,
        ),
        exec=SimpleNamespace(timeout=3600),
    )


def make_audit_task_desc() -> str:
    return json.dumps(
        {
            "Title": "Benchmark Leakage Audit",
            "Abstract": "Audit a benchmark for leakage using deterministic artifacts.",
            "Short Hypothesis": "Deterministic artifacts should drive stage progression.",
            "Experiments": ["Reproduce the benchmark and audit the split integrity."],
            "Risk Factors and Limitations": "Tiny synthetic fixtures only.",
            "Audit Targets": ["exact_duplicate", "group_overlap"],
            "Leakage Taxonomy": ["duplicate leakage", "group leakage"],
            "Acceptance Criteria": [
                "split_manifest.json validates",
                "audit_results.json validates",
            ],
            "Benchmark Metadata": {
                "dataset_name": "demo-dataset",
            },
            "Dataset Context": "Dataset card and split manifest are available.",
        }
    )


def make_manager(tmp_path: Path) -> AgentManager:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    return AgentManager(
        task_desc=make_audit_task_desc(),
        cfg=make_cfg(),
        workspace_dir=workspace_dir,
    )


def make_good_node(exp_results_dir: Path | None = None) -> Node:
    return Node(
        id="node-001",
        plan="audit",
        code="print('audit')",
        is_buggy=False,
        is_buggy_plots=False,
        metric=MetricValue(0.91, maximize=True, name="audit_score"),
        exp_results_dir=str(exp_results_dir) if exp_results_dir is not None else None,
        exec_time=120.0,
    )


def write_dataset_card(artifact_dir: Path) -> None:
    (artifact_dir / "dataset_card.md").write_text(
        "# Dataset Card\n\nSplit names: train, test\n"
    )


def write_findings_csv(artifact_dir: Path, provenance: dict, include_row: bool) -> None:
    columns = list(build_example_findings_columns())
    rows = []
    if include_row:
        rows.append(
            {
                "finding_id": "finding-001",
                "detector_name": "exact_duplicate",
                "severity": "high",
                "confidence": 0.99,
                "evidence_pointer": "evidence/exact_duplicate_pairs.parquet",
                "remediation_status": "open",
                "provenance_schema_version": provenance["schema_version"],
                "provenance_git_sha": provenance["git_sha"],
                "provenance_dataset_fingerprint": provenance["dataset_fingerprint"],
                "provenance_seed": provenance["seed"],
                "provenance_run_id": provenance["run_id"],
                "provenance_detector_versions_json": json.dumps(
                    provenance["detector_versions"], sort_keys=True
                ),
                "provenance_created_at": provenance["created_at"],
                "provenance_updated_at": provenance["updated_at"],
            }
        )
    pd.DataFrame(rows, columns=columns).to_csv(artifact_dir / "findings.csv", index=False)


def write_audit_artifacts(
    artifact_dir: Path,
    *,
    include_findings: bool,
    findings_total: int,
    include_metrics_before_after: bool,
    high_confidence_clean: bool = False,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_dataset_card(artifact_dir)

    split_manifest = build_example_split_manifest()
    audit_results = build_example_audit_results()
    audit_results["benchmark_summary"]["split_names"] = [
        split["name"] for split in split_manifest["splits"]
    ]
    audit_results["benchmark_summary"]["dataset_name"] = split_manifest["dataset_name"]

    if high_confidence_clean:
        audit_results["findings_summary"] = {
            "total_findings": 0,
            "open_findings": 0,
            "by_severity": {},
            "by_detector": {},
        }
        audit_results["confidence"]["overall"] = 0.96
        audit_results["confidence"]["evidence_coverage"] = 0.97
        audit_results["evidence_references"] = []
    else:
        audit_results["findings_summary"]["total_findings"] = findings_total
        audit_results["findings_summary"]["open_findings"] = findings_total

    (artifact_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2))
    (artifact_dir / "audit_results.json").write_text(json.dumps(audit_results, indent=2))

    if include_findings:
        write_findings_csv(
            artifact_dir,
            audit_results["provenance"],
            include_row=findings_total > 0 and not high_confidence_clean,
        )

    if include_metrics_before_after:
        metrics_before_after = build_example_metrics_before_after()
        metrics_before_after["provenance"]["dataset_fingerprint"] = audit_results[
            "provenance"
        ]["dataset_fingerprint"]
        (artifact_dir / "metrics_before_after.json").write_text(
            json.dumps(metrics_before_after, indent=2)
        )


def test_audit_stage_task_descriptions_use_audit_titles_and_goals(tmp_path):
    manager = make_manager(tmp_path)

    stage1 = manager.current_stage
    assert stage1 is not None
    stage2 = manager._create_next_main_stage(stage1, Journal())
    assert stage2 is not None
    stage3 = manager._create_next_main_stage(stage2, Journal())
    assert stage3 is not None
    stage4 = manager._create_next_main_stage(stage3, Journal())
    assert stage4 is not None

    rendered_stage1 = manager._build_task_desc_for_stage(stage1)
    rendered_stage2 = manager._build_task_desc_for_stage(stage2)
    rendered_stage3 = manager._build_task_desc_for_stage(stage3)
    rendered_stage4 = manager._build_task_desc_for_stage(stage4)

    assert f"Current Main Stage: {AUDIT_STAGE_TITLES[1]}" in rendered_stage1
    assert "Reproduce the benchmark protocol on the provided dataset splits." in rendered_stage1
    assert f"Current Main Stage: {AUDIT_STAGE_TITLES[2]}" in rendered_stage2
    assert "Run leakage detectors on the declared benchmark splits." in rendered_stage2
    assert f"Current Main Stage: {AUDIT_STAGE_TITLES[3]}" in rendered_stage3
    assert "Confirm candidate findings with remediation or falsification attempts." in rendered_stage3
    assert f"Current Main Stage: {AUDIT_STAGE_TITLES[4]}" in rendered_stage4
    assert "Run robustness checks on confirmed or disputed findings." in rendered_stage4


def test_run_logs_all_four_audit_main_stages(tmp_path, monkeypatch, capsys):
    manager = make_manager(tmp_path)
    dummy_node = make_good_node()

    monkeypatch.setattr(manager, "_create_agent_for_stage", lambda stage: DummyAgent())
    monkeypatch.setattr(manager, "_save_checkpoint", lambda: None)
    monkeypatch.setattr(manager, "_get_best_implementation", lambda stage_name: dummy_node)
    monkeypatch.setattr(manager, "_check_stage_completion", lambda stage: (True, "complete"))

    manager.run(exec_callback=lambda: None)
    captured = capsys.readouterr().out

    for title in AUDIT_STAGE_TITLES.values():
        assert title in captured


def test_stage_one_completion_requires_deterministic_baseline_artifacts(tmp_path):
    manager = make_manager(tmp_path)
    stage1 = manager.current_stage
    assert stage1 is not None

    artifact_dir = tmp_path / "stage1_artifacts"
    write_audit_artifacts(
        artifact_dir,
        include_findings=False,
        findings_total=1,
        include_metrics_before_after=False,
    )

    journal = Journal()
    journal.append(make_good_node(artifact_dir))
    manager.journals[stage1.name] = journal

    is_complete, reason = manager._check_stage_completion(stage1)
    assert is_complete is True
    assert "Baseline protocol validated" in reason


def test_stage_two_followup_substage_uses_stage_two_gate(tmp_path):
    manager = make_manager(tmp_path)
    stage2 = Stage(
        name="2_baseline_tuning_2_followup",
        description="followup",
        goals=manager.main_stage_goals[2],
        max_iterations=2,
        num_drafts=0,
        stage_number=99,
    )

    artifact_dir = tmp_path / "stage2_artifacts"
    write_audit_artifacts(
        artifact_dir,
        include_findings=True,
        findings_total=2,
        include_metrics_before_after=False,
    )

    journal = Journal()
    journal.append(make_good_node(artifact_dir))
    manager.journals[stage2.name] = journal

    is_complete, reason = manager._check_stage_completion(stage2)
    assert is_complete is True
    assert "confirmed finding" in reason.lower()


def test_stage_three_followup_requires_metrics_before_after_for_findings(tmp_path):
    manager = make_manager(tmp_path)
    stage3 = Stage(
        name="3_creative_research_2_followup",
        description="followup",
        goals=manager.main_stage_goals[3],
        max_iterations=2,
        num_drafts=0,
        stage_number=88,
    )

    artifact_dir = tmp_path / "stage3_artifacts"
    write_audit_artifacts(
        artifact_dir,
        include_findings=True,
        findings_total=2,
        include_metrics_before_after=False,
    )

    journal = Journal()
    journal.append(make_good_node(artifact_dir))
    manager.journals[stage3.name] = journal

    is_complete, reason = manager._check_stage_completion(stage3)
    assert is_complete is False
    assert "remediation or falsification" in reason.lower()


def test_stage_four_followup_allows_high_confidence_clean_audit(tmp_path):
    manager = make_manager(tmp_path)
    stage4 = Stage(
        name="4_ablation_studies_3_followup",
        description="followup",
        goals=manager.main_stage_goals[4],
        max_iterations=2,
        num_drafts=0,
        stage_number=77,
    )

    artifact_dir = tmp_path / "stage4_artifacts"
    write_audit_artifacts(
        artifact_dir,
        include_findings=True,
        findings_total=0,
        include_metrics_before_after=False,
        high_confidence_clean=True,
    )

    journal = Journal()
    journal.append(make_good_node(artifact_dir))
    manager.journals[stage4.name] = journal

    is_complete, reason = manager._check_stage_completion(stage4)
    assert is_complete is True
    assert "high-confidence clean audit" in reason.lower()
