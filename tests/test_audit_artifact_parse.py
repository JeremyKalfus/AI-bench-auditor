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
from ai_scientist.treesearch import parallel_agent as parallel_agent_module
from ai_scientist.treesearch.interpreter import ExecutionResult
from ai_scientist.treesearch.journal import Node
from ai_scientist.treesearch.parallel_agent import MinimalAgent


def make_cfg():
    return SimpleNamespace(
        exec=SimpleNamespace(timeout=3600),
        experiment=SimpleNamespace(num_syn_datasets=1),
        agent=SimpleNamespace(
            data_preview=False,
            k_fold_validation=1,
            code=SimpleNamespace(model="test-model", temp=0.0),
            feedback=SimpleNamespace(model="test-model", temp=0.0),
        ),
    )


def make_audit_task_desc() -> str:
    return "\n".join(
        [
            "Title:",
            "Benchmark Leakage Audit",
            "Audit Targets:",
            "- exact_duplicate",
            "Leakage Taxonomy:",
            "- duplicate leakage",
            "Acceptance Criteria:",
            "- emit audit artifacts",
            "Benchmark Metadata:",
            '{"dataset_name": "demo-dataset"}',
            "Dataset Context:",
            "Split names: train, test",
        ]
    )


def make_agent() -> MinimalAgent:
    return MinimalAgent(
        task_desc=make_audit_task_desc(),
        cfg=make_cfg(),
        memory_summary=None,
        evaluation_metrics="overall audit confidence",
        stage_name="1_initial_implementation_1_preliminary",
    )


def make_exec_result() -> ExecutionResult:
    return ExecutionResult(
        term_out=["audit run complete\n"],
        exec_time=1.25,
        exc_type=None,
        exc_info=None,
        exc_stack=None,
    )


def write_findings_csv(path: Path) -> None:
    columns = list(build_example_findings_columns())
    pd.DataFrame(
        [
            {
                "finding_id": "finding-001",
                "detector_name": "exact_duplicate",
                "severity": "high",
                "confidence": 0.99,
                "evidence_pointer": "evidence/exact_duplicate_pairs.parquet",
                "remediation_status": "open",
                "provenance_schema_version": "0.1.0",
                "provenance_git_sha": "96bd51617cfdbb494a9fc283af00fe090edfae48",
                "provenance_dataset_fingerprint": "sha256:demo-dataset-fingerprint",
                "provenance_seed": 7,
                "provenance_run_id": "audit-run-0001",
                "provenance_detector_versions_json": '{"exact_duplicate": "1.0.0"}',
                "provenance_created_at": "2026-04-11T15:30:00Z",
                "provenance_updated_at": "2026-04-11T15:30:00Z",
            }
        ],
        columns=columns,
    ).to_csv(path, index=False)


def write_valid_audit_bundle(working_dir: Path) -> None:
    working_dir.mkdir(parents=True, exist_ok=True)
    (working_dir / "dataset_card.md").write_text("# Dataset Card\n")
    (working_dir / "audit_results.json").write_text(
        json.dumps(build_example_audit_results(), indent=2)
    )
    (working_dir / "split_manifest.json").write_text(
        json.dumps(build_example_split_manifest(), indent=2)
    )
    (working_dir / "metrics_before_after.json").write_text(
        json.dumps(build_example_metrics_before_after(), indent=2)
    )
    write_findings_csv(working_dir / "findings.csv")


def test_parse_exec_result_uses_structured_audit_artifacts_without_llm(monkeypatch, tmp_path):
    agent = make_agent()
    node = Node(code="print('audit')", is_buggy=False)
    exec_result = make_exec_result()
    write_valid_audit_bundle(tmp_path)

    def fail_query(*args, **kwargs):
        raise AssertionError("query should not be called for valid structured audit artifacts")

    monkeypatch.setattr(parallel_agent_module, "query", fail_query)

    finalized = agent.parse_exec_result(node=node, exec_result=exec_result, workspace=tmp_path)

    assert finalized is True
    assert node.is_buggy is False
    assert node.metric.value == 82.5
    assert "Structured audit summary" in node.analysis
    assert "3 total findings" in node.analysis


def test_parse_exec_result_missing_audit_results_marks_node_invalid_and_uses_fallback_review(
    monkeypatch, tmp_path
):
    agent = make_agent()
    node = Node(code="print('audit')", is_buggy=False)
    exec_result = make_exec_result()

    def fallback_query(*args, **kwargs):
        return {"is_bug": False, "summary": "stdout fallback review"}

    monkeypatch.setattr(parallel_agent_module, "query", fallback_query)

    finalized = agent.parse_exec_result(node=node, exec_result=exec_result, workspace=tmp_path)

    assert finalized is True
    assert node.is_buggy is True
    assert node.metric.value is None
    assert "Missing required audit artifact: audit_results.json" in node.analysis
    assert "stdout fallback review" in node.analysis


def test_parse_exec_result_invalid_audit_results_marks_node_invalid_without_fallback(
    monkeypatch, tmp_path
):
    agent = make_agent()
    node = Node(code="print('audit')", is_buggy=False)
    exec_result = make_exec_result()
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "audit_results.json").write_text(json.dumps({"run_metadata": {}}))

    def fail_query(*args, **kwargs):
        raise AssertionError("query should not be called for malformed audit artifacts")

    monkeypatch.setattr(parallel_agent_module, "query", fail_query)

    finalized = agent.parse_exec_result(node=node, exec_result=exec_result, workspace=tmp_path)

    assert finalized is True
    assert node.is_buggy is True
    assert node.metric.value is None
    assert "Invalid audit_results.json" in node.analysis


def test_copy_audit_artifacts_copies_primary_files(tmp_path):
    agent = make_agent()
    working_dir = tmp_path / "working"
    exp_results_dir = tmp_path / "experiment_results"
    write_valid_audit_bundle(working_dir)
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    agent._copy_audit_artifacts(working_dir, exp_results_dir)

    assert (exp_results_dir / "audit_results.json").exists()
    assert (exp_results_dir / "split_manifest.json").exists()
    assert (exp_results_dir / "metrics_before_after.json").exists()
    assert (exp_results_dir / "findings.csv").exists()
    assert (exp_results_dir / "dataset_card.md").exists()
