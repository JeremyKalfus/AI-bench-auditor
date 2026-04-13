import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ai_scientist.audits import empty_findings_dataframe
from ai_scientist.audits.artifacts import load_validated_audit_bundle
from ai_scientist.audits.schema import (
    build_example_audit_results,
    build_example_findings_columns,
    build_example_metrics_before_after,
    build_provenance_block,
    build_example_split_manifest,
)
from ai_scientist.treesearch import parallel_agent as parallel_agent_module
from ai_scientist.treesearch.interpreter import ExecutionResult
from ai_scientist.treesearch.journal import Journal, Node
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


def test_parse_exec_result_accepts_valid_audit_bundle_even_if_exec_reports_exception(
    monkeypatch, tmp_path
):
    agent = make_agent()
    node = Node(code="print('audit')", is_buggy=False)
    exec_result = ExecutionResult(
        term_out=["late assertion\n"],
        exec_time=1.25,
        exc_type="AssertionError",
        exc_info={"args": ["late assertion"]},
        exc_stack=None,
    )
    write_valid_audit_bundle(tmp_path)

    def fail_query(*args, **kwargs):
        raise AssertionError("query should not be called for valid structured audit artifacts")

    monkeypatch.setattr(parallel_agent_module, "query", fail_query)

    finalized = agent.parse_exec_result(node=node, exec_result=exec_result, workspace=tmp_path)

    assert finalized is True
    assert node.is_buggy is False
    assert node.metric.value == 82.5
    assert "Execution ended with `AssertionError`" in node.analysis


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


def test_parse_exec_result_normalizes_near_miss_audit_bundle(tmp_path):
    agent = make_agent()
    node = Node(code="print('audit')", is_buggy=False)
    exec_result = make_exec_result()

    workspace_root = tmp_path / "workspace"
    working_dir = workspace_root / "working"
    workspace_root.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)

    idea_payload = {
        "Title": "Benchmark Leakage Audit",
        "Abstract": "Synthetic normalization test.",
        "Short Hypothesis": "Normalization should recover a valid audit bundle.",
        "Experiments": ["Audit the staged benchmark splits."],
        "Risk Factors and Limitations": "Synthetic test fixture only.",
        "Audit Targets": ["group_overlap", "near_duplicate"],
        "Benchmark Metadata": {
            "benchmark_name": "demo-benchmark",
            "dataset_name": "demo-dataset",
            "expected_issue_detectors": ["group_overlap", "near_duplicate"],
            "baseline_metrics": [
                {
                    "metric_name": "accuracy",
                    "split": "test",
                    "value": 0.93,
                    "higher_is_better": True,
                }
            ],
            "remediated_metrics": [
                {
                    "metric_name": "accuracy",
                    "split": "test",
                    "value": 0.84,
                    "higher_is_better": True,
                }
            ],
            "metrics_notes": "Synthetic remediation delta.",
        },
    }
    (workspace_root / "idea.json").write_text(json.dumps(idea_payload, indent=2))
    (workspace_root / "split_manifest.json").write_text(
        json.dumps(build_example_split_manifest(), indent=2)
    )
    (working_dir / "audit_results.json").write_text(
        json.dumps(
            {
                "benchmark_id": "demo-benchmark",
                "group_overlap": 1,
                "near_duplicates": 1,
            },
            indent=2,
        )
    )
    (working_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "splits": [
                    {"split_name": "train", "file": "data/train.csv", "num_rows": 3},
                    {"split_name": "test", "file": "data/test.csv", "num_rows": 3},
                ],
            },
            indent=2,
        )
    )
    (working_dir / "metrics_before_after.json").write_text(
        json.dumps(
            {
                "baseline_metrics": {"accuracy": 0.93},
                "remediated_metrics": {"accuracy": 0.84},
            },
            indent=2,
        )
    )
    pd.DataFrame(
        [
            {
                "customer_id": "cust_002",
                "issue_type": "group_overlap",
                "train_text": None,
                "test_text": None,
                "similarity": None,
            },
            {
                "customer_id": None,
                "issue_type": None,
                "train_text": "Reset password request for billing portal access",
                "test_text": "Reset password request for billing portal access today",
                "similarity": 94.1,
            },
        ]
    ).to_csv(working_dir / "findings.csv", index=False)

    finalized = agent.parse_exec_result(
        node=node,
        exec_result=exec_result,
        workspace=working_dir,
    )

    assert finalized is True
    assert node.is_buggy is False
    assert node.metric.value == 100.0

    normalized_audit_results = json.loads((working_dir / "audit_results.json").read_text())
    assert normalized_audit_results["findings_summary"]["total_findings"] == 2
    assert normalized_audit_results["detectors_run"][0]["status"] == "completed"

    normalized_metrics = json.loads((working_dir / "metrics_before_after.json").read_text())
    assert normalized_metrics["deltas"][0]["delta"] == -0.09
    assert (working_dir / "evidence" / "group_overlap_001.json").exists()


def test_parse_exec_result_normalizes_raw_audit_results_findings_list(tmp_path):
    agent = make_agent()
    node = Node(code="print('audit')", is_buggy=False)
    exec_result = make_exec_result()

    workspace_root = tmp_path / "workspace"
    working_dir = workspace_root / "working"
    workspace_root.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)

    idea_payload = {
        "Title": "Benchmark Leakage Audit",
        "Abstract": "Synthetic normalization test.",
        "Short Hypothesis": "Normalization should recover live findings from audit_results.json.",
        "Experiments": ["Audit the staged benchmark splits."],
        "Risk Factors and Limitations": "Synthetic test fixture only.",
        "Audit Targets": ["group_overlap", "near_duplicate"],
        "Benchmark Metadata": {
            "benchmark_name": "demo-benchmark",
            "dataset_name": "demo-dataset",
            "expected_issue_detectors": ["group_overlap", "near_duplicate"],
            "baseline_metrics": [
                {
                    "metric_name": "accuracy",
                    "split": "test",
                    "value": 0.93,
                    "higher_is_better": True,
                }
            ],
            "remediated_metrics": [
                {
                    "metric_name": "accuracy",
                    "split": "test",
                    "value": 0.84,
                    "higher_is_better": True,
                }
            ],
            "metrics_notes": "Synthetic remediation delta.",
        },
    }
    (workspace_root / "idea.json").write_text(json.dumps(idea_payload, indent=2))
    (workspace_root / "split_manifest.json").write_text(
        json.dumps(build_example_split_manifest(), indent=2)
    )
    (working_dir / "audit_results.json").write_text(
        json.dumps(
            {
                "benchmark_id": "demo-benchmark",
                "findings": [
                    {
                        "detector": "group_overlap",
                        "description": "1 overlapping customer detected across train and test.",
                        "evidence": ["cust_002"],
                    },
                    {
                        "detector": "near_duplicate",
                        "description": "1 semantically repeated ticket pair detected.",
                        "evidence": [
                            {
                                "train_index": 0,
                                "test_index": 1,
                                "similarity": 94.1,
                            }
                        ],
                    },
                ],
            },
            indent=2,
        )
    )
    (working_dir / "metrics_before_after.json").write_text(
        json.dumps(
            {
                "baseline_metrics": {"accuracy": 0.93},
                "remediated_metrics": {"accuracy": 0.84},
            },
            indent=2,
        )
    )

    finalized = agent.parse_exec_result(
        node=node,
        exec_result=exec_result,
        workspace=working_dir,
    )

    assert finalized is True
    assert node.is_buggy is False
    assert node.metric.value == 100.0

    normalized_audit_results = json.loads((working_dir / "audit_results.json").read_text())
    assert normalized_audit_results["findings_summary"]["total_findings"] == 2
    assert normalized_audit_results["findings_summary"]["by_detector"] == {
        "group_overlap": 1,
        "near_duplicate": 1,
    }

    findings = pd.read_csv(working_dir / "findings.csv")
    assert findings["detector_name"].tolist() == ["group_overlap", "near_duplicate"]
    assert (working_dir / "evidence" / "group_overlap_001.json").exists()
    assert (working_dir / "evidence" / "near_duplicate_002.json").exists()


def test_parse_exec_result_recovers_detector_names_from_malformed_contract_findings(
    tmp_path,
):
    agent = make_agent()
    node = Node(code="print('audit')", is_buggy=False)
    exec_result = make_exec_result()

    workspace_root = tmp_path / "workspace"
    working_dir = workspace_root / "working"
    workspace_root.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)

    idea_payload = {
        "Title": "Benchmark Leakage Audit",
        "Abstract": "Synthetic normalization test.",
        "Short Hypothesis": "Normalization should recover detector names from malformed findings.",
        "Experiments": ["Audit the staged benchmark splits."],
        "Risk Factors and Limitations": "Synthetic test fixture only.",
        "Audit Targets": ["group_overlap", "near_duplicate"],
        "Benchmark Metadata": {
            "benchmark_name": "demo-benchmark",
            "dataset_name": "demo-dataset",
            "expected_issue_detectors": ["group_overlap", "near_duplicate"],
            "baseline_metrics": [
                {
                    "metric_name": "accuracy",
                    "split": "test",
                    "value": 0.93,
                    "higher_is_better": True,
                }
            ],
            "remediated_metrics": [
                {
                    "metric_name": "accuracy",
                    "split": "test",
                    "value": 0.84,
                    "higher_is_better": True,
                }
            ],
            "metrics_notes": "Synthetic remediation delta.",
        },
    }
    (workspace_root / "idea.json").write_text(json.dumps(idea_payload, indent=2))
    (workspace_root / "split_manifest.json").write_text(
        json.dumps(build_example_split_manifest(), indent=2)
    )
    (working_dir / "audit_results.json").write_text(
        json.dumps({"benchmark_id": "demo-benchmark"}, indent=2)
    )
    (working_dir / "metrics_before_after.json").write_text(
        json.dumps(
            {
                "baseline_metrics": {"accuracy": 0.93},
                "remediated_metrics": {"accuracy": 0.84},
            },
            indent=2,
        )
    )
    malformed_findings = pd.concat(
        [
            empty_findings_dataframe(),
            pd.DataFrame(
                [
                    {
                        "detector": "group_overlap",
                        "detail": "Customer ID cust_002 in both splits",
                    },
                    {
                        "detector": "near_duplicate",
                        "detail": "Text near-duplicate between train row 1 and test row 1",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    malformed_findings.to_csv(working_dir / "findings.csv", index=False)

    finalized = agent.parse_exec_result(
        node=node,
        exec_result=exec_result,
        workspace=working_dir,
    )

    assert finalized is True
    assert node.is_buggy is False
    assert node.metric.value == 100.0

    normalized_audit_results = json.loads((working_dir / "audit_results.json").read_text())
    assert normalized_audit_results["findings_summary"]["by_detector"] == {
        "group_overlap": 1,
        "near_duplicate": 1,
    }

    findings = pd.read_csv(working_dir / "findings.csv")
    assert findings["detector_name"].tolist() == ["group_overlap", "near_duplicate"]


def test_parse_exec_result_runs_deterministic_detector_fallback(tmp_path):
    agent = make_agent()
    node = Node(code="print('audit')", is_buggy=False)
    exec_result = make_exec_result()

    workspace_root = tmp_path / "workspace"
    working_dir = workspace_root / "working"
    data_dir = workspace_root / "data"
    workspace_root.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(
        [
            {
                "customer_id": "cust_001",
                "text": "Reset password request for billing portal access",
                "label": "password",
                "event_time": "2026-01-01T00:00:00Z",
            },
            {
                "customer_id": "cust_002",
                "text": "Unable to update payment method on file",
                "label": "billing",
                "event_time": "2026-01-02T00:00:00Z",
            },
        ]
    )
    test = pd.DataFrame(
        [
            {
                "customer_id": "cust_002",
                "text": "Unable to update payment method on file today",
                "label": "billing",
                "event_time": "2026-01-03T00:00:00Z",
            },
            {
                "customer_id": "cust_003",
                "text": "New account setup question",
                "label": "account",
                "event_time": "2026-01-04T00:00:00Z",
            },
        ]
    )
    train.to_csv(data_dir / "00_train_train.csv", index=False)
    test.to_csv(data_dir / "01_test_test.csv", index=False)

    idea_payload = {
        "Title": "Benchmark Leakage Audit",
        "Abstract": "Synthetic deterministic detector fallback test.",
        "Short Hypothesis": "Fallback detectors should recover the expected issues.",
        "Experiments": ["Audit the staged benchmark splits."],
        "Risk Factors and Limitations": "Synthetic test fixture only.",
        "Audit Targets": ["group_overlap", "near_duplicate"],
        "Benchmark Metadata": {
            "benchmark_id": "demo-benchmark",
            "benchmark_name": "demo-benchmark",
            "dataset_name": "demo-dataset",
            "expected_issue_detectors": ["group_overlap", "near_duplicate"],
            "candidate_key_columns": ["customer_id"],
            "text_columns": ["text"],
            "near_duplicate_similarity_threshold": 90,
            "baseline_metrics": [
                {
                    "metric_name": "accuracy",
                    "split": "test",
                    "value": 0.93,
                    "higher_is_better": True,
                }
            ],
            "remediated_metrics": [
                {
                    "metric_name": "accuracy",
                    "split": "test",
                    "value": 0.84,
                    "higher_is_better": True,
                }
            ],
            "metrics_notes": "Synthetic remediation delta.",
        },
    }
    (workspace_root / "idea.json").write_text(json.dumps(idea_payload, indent=2))

    provenance = build_provenance_block(
        git_sha="96bd51617cfdbb494a9fc283af00fe090edfae48",
        dataset_fingerprint="sha256:demo-dataset-fingerprint",
        seed=7,
        run_id="audit-run-0001",
        detector_versions={
            "group_overlap": "0.1.0",
            "near_duplicate": "0.1.0",
        },
    )
    (workspace_root / "split_manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "demo-dataset",
                "file_paths_used": [
                    "data/00_train_train.csv",
                    "data/01_test_test.csv",
                ],
                "splits": [
                    {
                        "name": "train",
                        "record_count": 2,
                        "file_paths": ["data/00_train_train.csv"],
                    },
                    {
                        "name": "test",
                        "record_count": 2,
                        "file_paths": ["data/01_test_test.csv"],
                    },
                ],
                "provenance": provenance,
            },
            indent=2,
        )
    )
    (working_dir / "audit_results.json").write_text(
        json.dumps({"benchmark_id": "demo-benchmark"}, indent=2)
    )

    finalized = agent.parse_exec_result(
        node=node,
        exec_result=exec_result,
        workspace=working_dir,
    )

    assert finalized is True
    assert node.is_buggy is False
    assert node.metric.value == 100.0

    normalized_audit_results = json.loads((working_dir / "audit_results.json").read_text())
    assert normalized_audit_results["findings_summary"]["by_detector"] == {
        "group_overlap": 1,
        "near_duplicate": 1,
    }
    assert normalized_audit_results["audit_score"]["rating"] == "warning"
    assert (working_dir / "evidence" / "group_overlap_001.json").exists()
    assert (working_dir / "evidence" / "near_duplicate_002.json").exists()


def test_copy_audit_artifacts_copies_primary_files(tmp_path):
    agent = make_agent()
    working_dir = tmp_path / "working"
    exp_results_dir = tmp_path / "experiment_results"
    write_valid_audit_bundle(working_dir)
    (working_dir / "evidence").mkdir(parents=True, exist_ok=True)
    (working_dir / "evidence" / "exact_duplicate_pairs.json").write_text("{}")
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    agent._copy_audit_artifacts(working_dir, exp_results_dir)

    assert (exp_results_dir / "audit_results.json").exists()
    assert (exp_results_dir / "split_manifest.json").exists()
    assert (exp_results_dir / "metrics_before_after.json").exists()
    assert (exp_results_dir / "findings.csv").exists()
    assert (exp_results_dir / "dataset_card.md").exists()
    assert (exp_results_dir / "evidence" / "exact_duplicate_pairs.json").exists()


def test_copy_audit_artifacts_uses_workspace_dataset_card_when_missing_in_working_dir(tmp_path):
    agent = make_agent()
    workspace_root = tmp_path / "workspace"
    working_dir = workspace_root / "working"
    exp_results_dir = tmp_path / "experiment_results"
    working_dir.mkdir(parents=True, exist_ok=True)
    exp_results_dir.mkdir(parents=True, exist_ok=True)
    (workspace_root / "dataset_card.md").write_text("# Dataset Card\n")
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

    agent._copy_audit_artifacts(working_dir, exp_results_dir)

    assert (exp_results_dir / "dataset_card.md").exists()


def test_node_to_dict_preserves_absolute_result_paths_outside_repo(tmp_path):
    outside_path = tmp_path / "outside-results"
    node = Node(
        code="print('audit')",
        exp_results_dir=str(outside_path),
        plot_paths=[str(outside_path / "plot.png")],
        plot_analyses=[{"plot_path": str(outside_path / "plot.png"), "summary": "ok"}],
    )

    data = node.to_dict()

    assert data["exp_results_dir"] == str(outside_path.resolve())
    assert data["plot_paths"] == [str((outside_path / "plot.png").resolve())]
    assert data["plot_analyses"][0]["plot_path"] == str(
        (outside_path / "plot.png").resolve()
    )


def test_journal_good_nodes_accepts_plotless_audit_nodes():
    journal = Journal()
    journal.append(Node(code="print('ok')", is_buggy=False, is_buggy_plots=None))

    assert len(journal.good_nodes) == 1


def test_load_validated_audit_bundle_prefers_csv_when_parquet_is_broken(tmp_path):
    write_valid_audit_bundle(tmp_path)
    audit_results = build_example_audit_results()
    audit_results["detectors_run"] = [
        {
            "name": "exact_duplicate",
            "version": "1.0.0",
            "status": "completed",
            "finding_count": 1,
        }
    ]
    audit_results["findings_summary"] = {
        "total_findings": 1,
        "open_findings": 1,
        "by_severity": {"high": 1},
        "by_detector": {"exact_duplicate": 1},
    }
    audit_results["benchmark_summary"] = {
        "benchmark_name": "demo-benchmark",
        "dataset_name": "demo-dataset",
        "record_count": 120,
        "split_names": ["train", "test"],
    }
    (tmp_path / "audit_results.json").write_text(json.dumps(audit_results, indent=2))
    (tmp_path / "evidence").mkdir(parents=True, exist_ok=True)
    (tmp_path / "evidence" / "exact_duplicate_pairs.parquet").write_bytes(b"PAR1")
    (tmp_path / "findings.parquet").write_bytes(b"")

    bundle = load_validated_audit_bundle(tmp_path)

    assert bundle.findings_path.name == "findings.csv"
    assert len(bundle.findings) == 1
