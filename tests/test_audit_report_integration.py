import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ai_scientist.audits.schema import (
    build_example_audit_results,
    build_example_findings_columns,
    build_example_metrics_before_after,
    build_example_split_manifest,
)
from ai_scientist.treesearch.journal import Journal, Node
from ai_scientist.treesearch.utils.metric import MetricValue


def load_launcher(repo_root: Path):
    launcher_path = repo_root / "launch_scientist_bfts.py"
    spec = importlib.util.spec_from_file_location(
        "launch_scientist_bfts", launcher_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def test_run_audit_mode_generates_audit_and_study_outputs(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    launcher = load_launcher(repo_root)

    idea_dir = tmp_path / "audit-run"
    original_artifact_dir = (
        idea_dir
        / "logs"
        / "0-run"
        / "experiment_results"
        / "experiment_node-001_proc_12345"
    )
    original_artifact_dir.mkdir(parents=True, exist_ok=True)

    audit_results = build_example_audit_results()
    audit_results["findings_summary"] = {
        "total_findings": 1,
        "open_findings": 1,
        "by_severity": {"high": 1},
        "by_detector": {"exact_duplicate": 1},
    }
    split_manifest = build_example_split_manifest()
    audit_results["benchmark_summary"]["dataset_name"] = split_manifest["dataset_name"]
    audit_results["benchmark_summary"]["split_names"] = [
        split["name"] for split in split_manifest["splits"]
    ]
    audit_results["benchmark_summary"]["record_count"] = sum(
        split["record_count"] for split in split_manifest["splits"]
    )
    audit_results["detectors_run"] = [
        {
            "name": "exact_duplicate",
            "version": "1.0.0",
            "status": "completed",
            "finding_count": 1,
        },
        {
            "name": "temporal_overlap",
            "version": "1.0.0",
            "status": "completed",
            "finding_count": 0,
        },
    ]
    (original_artifact_dir / "audit_results.json").write_text(
        json.dumps(audit_results, indent=2)
    )
    (original_artifact_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2)
    )
    (original_artifact_dir / "metrics_before_after.json").write_text(
        json.dumps(build_example_metrics_before_after(), indent=2)
    )
    (original_artifact_dir / "dataset_card.md").write_text("# Dataset Card\n")
    (original_artifact_dir / "evidence").mkdir(parents=True, exist_ok=True)
    (original_artifact_dir / "evidence" / "exact_duplicate_pairs.parquet").write_text(
        "placeholder evidence"
    )
    write_findings_csv(original_artifact_dir / "findings.csv")

    node = Node(
        id="node-001",
        code="print('audit')",
        is_buggy=False,
        is_buggy_plots=False,
        metric=MetricValue(82.5, maximize=True, name="audit_score"),
        exp_results_dir=str(original_artifact_dir),
    )
    manager = SimpleNamespace(
        stages=[SimpleNamespace(name="4_final")],
        journals={"4_final": Journal(nodes=[node])},
        cfg=None,
    )

    def fake_prepare_audit_run(args):
        idea_json_path = idea_dir / "idea.json"
        idea_json_path.write_text(
            json.dumps(
                {
                    "Name": "audit-report-integration",
                    "Title": "Audit Report Integration",
                    "Abstract": "Synthetic test.",
                    "Short Hypothesis": "Synthetic test.",
                    "Experiments": ["Synthetic test."],
                    "Risk Factors and Limitations": "Synthetic test.",
                    "Audit Targets": ["exact_duplicate"],
                    "Leakage Taxonomy": ["duplicate leakage"],
                    "Acceptance Criteria": ["audit_report.md exists"],
                    "Benchmark Metadata": {
                        "benchmark_name": "demo-benchmark",
                        "dataset_name": "demo-dataset",
                        "dataset_fingerprint": "sha256:demo-dataset-fingerprint",
                    },
                    "Dataset Context": "Dataset card: dataset_card.md\nSplit manifest: split_manifest.json",
                },
                indent=2,
            )
        )
        return {
            "idea_dir": str(idea_dir),
            "idea_config_path": str(idea_dir / "bfts_config.yaml"),
            "idea_path_json": str(idea_json_path),
            "metadata_path": str(idea_dir / "audit_run_metadata.json"),
            "runtime_settings": {
                "skip_writeup": True,
                "skip_review": True,
                "run_plot_aggregation": False,
            },
        }

    def fake_perform_experiments_bfts(config_path):
        return manager

    fake_runner_module = types.ModuleType(
        "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager"
    )
    fake_runner_module.perform_experiments_bfts = fake_perform_experiments_bfts

    monkeypatch.setattr(launcher, "prepare_audit_run", fake_prepare_audit_run)
    monkeypatch.setattr(launcher, "save_token_tracker", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(
        sys.modules,
        "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager",
        fake_runner_module,
    )

    launcher.run_audit_mode(
        SimpleNamespace(
            dry_run=False,
            plan_review="skip",
            plan_review_mode="file",
            plan_feedback_file=None,
            plan_approval_file=None,
            approve_plan=False,
            max_plan_revisions=3,
            emit_study_zip=True,
        )
    )

    copied_artifact_dir = (
        idea_dir / "experiment_results" / "experiment_node-001_proc_12345"
    )
    report_path = copied_artifact_dir / "audit_report.md"

    assert report_path.exists()
    report_text = report_path.read_text()
    assert "Audit Report" in report_text
    assert "Workflow Guard" in report_text
    assert str(copied_artifact_dir / "audit_results.json") in report_text
    assert str(copied_artifact_dir / "findings.csv") in report_text
    assert not list(idea_dir.rglob("*.tex"))
    assert not list(idea_dir.rglob("*.pdf"))
    assert (idea_dir / "research_plan.json").exists()
    assert (idea_dir / "plan_approval.json").exists()
    assert (idea_dir / "audit_report_review.json").exists()
    assert (idea_dir / "study_report.md").exists()
    assert (idea_dir / "study_bundle_manifest.json").exists()
    assert (idea_dir / "study_figures.zip").exists()
