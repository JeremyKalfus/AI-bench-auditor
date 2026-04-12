import importlib.util
import json
import sys
import types
from pathlib import Path

import pandas as pd

from ai_scientist.audits import manuscript as manuscript_module
from tests.audit_fixture_utils import (
    make_manager_for_artifact_dir,
    write_references_bib,
    write_valid_audit_bundle,
)


def load_launcher(repo_root: Path):
    launcher_path = repo_root / "launch_scientist_bfts.py"
    spec = importlib.util.spec_from_file_location(
        "launch_scientist_bfts", launcher_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_benchmark_idea(ideas_path: Path, train_path: Path, test_path: Path) -> None:
    ideas_path.write_text(
        json.dumps(
            [
                {
                    "Name": "audit_single_command_demo",
                    "Title": "Single Command Flow Demo",
                    "Abstract": "Exercise the one-command audit flow.",
                    "Short Hypothesis": "A single approval gate should block research until approval and then allow the run to finish automatically.",
                    "Experiments": ["Audit the declared benchmark splits."],
                    "Risk Factors and Limitations": "Synthetic fixture only.",
                    "Audit Targets": ["exact_duplicate"],
                    "Leakage Taxonomy": ["duplicate leakage"],
                    "Acceptance Criteria": [
                        "split_manifest.json validates",
                        "audit_report.md is reviewed before manuscript generation",
                    ],
                    "Benchmark Metadata": {
                        "dataset_name": "demo-benchmark",
                        "files": [
                            {"path": str(train_path), "split": "train"},
                            {"path": str(test_path), "split": "test"},
                        ],
                        "candidate_key_columns": ["user_id"],
                        "target_column": "label",
                        "timestamp_columns": ["event_time"],
                    },
                }
            ],
            indent=2,
        )
    )


def test_audit_single_command_flow_runs_with_one_pre_research_gate(
    tmp_path, monkeypatch
):
    repo_root = Path(__file__).resolve().parents[1]
    launcher = load_launcher(repo_root)

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame(
        [
            {"user_id": 1, "feature": 0.1, "label": 0, "event_time": "2023-01-01"},
            {"user_id": 2, "feature": 0.2, "label": 1, "event_time": "2023-01-02"},
        ]
    ).to_csv(train_path, index=False)
    pd.DataFrame(
        [{"user_id": 3, "feature": 0.3, "label": 1, "event_time": "2023-01-03"}]
    ).to_csv(test_path, index=False)

    ideas_path = tmp_path / "ideas.json"
    write_benchmark_idea(ideas_path, train_path, test_path)
    run_dir = tmp_path / "single-command-run"
    references_path = tmp_path / "references.bib"
    write_references_bib(references_path)

    input_calls = {"count": 0}

    def fake_input(_prompt: str) -> str:
        input_calls["count"] += 1
        return "approve"

    def fake_perform_experiments_bfts(_config_path: str):
        artifact_dir = (
            run_dir
            / "logs"
            / "0-run"
            / "experiment_results"
            / "experiment_node-001_proc_12345"
        )
        write_valid_audit_bundle(artifact_dir)
        return make_manager_for_artifact_dir(artifact_dir)

    fake_runner_module = types.ModuleType(
        "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager"
    )
    fake_runner_module.perform_experiments_bfts = fake_perform_experiments_bfts

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(launcher, "cleanup_processes", lambda: None)
    monkeypatch.setattr(launcher, "save_token_tracker", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        manuscript_module,
        "_compile_pdf",
        lambda paper_dir, tex_name: {
            "requested": True,
            "attempted": True,
            "available": True,
            "succeeded": True,
            "pdf_path": str((paper_dir / "paper.pdf").resolve()),
            "log_path": str((paper_dir / "paper_build.log").resolve()),
            "error": None,
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager",
        fake_runner_module,
    )

    exit_code = launcher.main(
        [
            "--mode",
            "audit",
            "--benchmark",
            str(ideas_path),
            "--output_dir",
            str(run_dir),
            "--plan-review",
            "required",
            "--plan-review-mode",
            "interactive",
            "--paper-mode",
            "on_success",
            "--citation-mode",
            "provided",
            "--references-file",
            str(references_path),
        ]
    )

    assert exit_code == 0
    assert input_calls["count"] == 1
    assert (run_dir / "research_plan.json").exists()
    assert (run_dir / "plan_approval.json").exists()
    assert (run_dir / "audit_results.json").exists()
    assert (run_dir / "split_manifest.json").exists()
    assert (run_dir / "findings.csv").exists()
    assert (run_dir / "audit_report.md").exists()
    assert (run_dir / "audit_report_review.json").exists()
    assert (run_dir / "audit_report_review.md").exists()
    assert (run_dir / "paper" / "paper.tex").exists()
    assert (run_dir / "paper" / "paper_manifest.json").exists()
    assert (run_dir / "paper_bundle.zip").exists()


def test_unapproved_plan_stops_before_research_starts(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    launcher = load_launcher(repo_root)

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame(
        [{"user_id": 1, "feature": 0.1, "label": 0, "event_time": "2023-01-01"}]
    ).to_csv(train_path, index=False)
    pd.DataFrame(
        [{"user_id": 2, "feature": 0.2, "label": 1, "event_time": "2023-01-02"}]
    ).to_csv(test_path, index=False)
    ideas_path = tmp_path / "ideas.json"
    write_benchmark_idea(ideas_path, train_path, test_path)

    run_dir = tmp_path / "blocked-run"
    calls = {"count": 0}

    def fake_perform_experiments_bfts(_config_path: str):
        calls["count"] += 1
        raise AssertionError("Research should not start before plan approval.")

    fake_runner_module = types.ModuleType(
        "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager"
    )
    fake_runner_module.perform_experiments_bfts = fake_perform_experiments_bfts

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(launcher, "cleanup_processes", lambda: None)
    monkeypatch.setattr(launcher, "save_token_tracker", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(
        sys.modules,
        "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager",
        fake_runner_module,
    )

    exit_code = launcher.main(
        [
            "--mode",
            "audit",
            "--benchmark",
            str(ideas_path),
            "--output_dir",
            str(run_dir),
            "--plan-review",
            "required",
            "--plan-review-mode",
            "file",
            "--paper-mode",
            "off",
        ]
    )

    assert exit_code == 1
    assert calls["count"] == 0
    assert (run_dir / "research_plan.json").exists()
    assert (run_dir / "plan_approval.json").exists()
