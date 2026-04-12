import json
from pathlib import Path

import pytest

from ai_scientist.audits.plan_review import (
    PlanApprovalRequiredError,
    PlanRejectedError,
    ensure_plan_review,
)


def write_idea(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "Name": "audit_plan_review_demo",
                "Title": "Audit Plan Review Demo",
                "Abstract": "A tiny idea for testing the plan review gate.",
                "Short Hypothesis": "Plan approval should block audit execution.",
                "Experiments": ["Generate the plan and wait for human review."],
                "Risk Factors and Limitations": "Synthetic fixture only.",
                "Audit Targets": ["exact_duplicate", "group_overlap"],
                "Leakage Taxonomy": ["duplicate leakage", "group leakage"],
                "Acceptance Criteria": [
                    "research_plan.json exists",
                    "plan approval is recorded before research begins",
                ],
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


def write_dataset_context(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "dataset_card.md").write_text("# Dataset Card\n\nSynthetic context.\n")
    (run_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "demo-dataset",
                "file_paths_used": ["data/train.csv", "data/test.csv"],
                "splits": [
                    {
                        "name": "train",
                        "record_count": 10,
                        "file_paths": ["data/train.csv"],
                    },
                    {
                        "name": "test",
                        "record_count": 5,
                        "file_paths": ["data/test.csv"],
                    },
                ],
                "provenance": {
                    "schema_version": "0.1.0",
                    "git_sha": "96bd51617cfdbb494a9fc283af00fe090edfae48",
                    "dataset_fingerprint": "sha256:demo-dataset-fingerprint",
                    "seed": None,
                    "run_id": "demo-run",
                    "detector_versions": {"dataset_context": "0.1.0"},
                    "created_at": "2026-04-11T15:30:00Z",
                    "updated_at": "2026-04-11T15:30:00Z",
                },
            },
            indent=2,
        )
    )


def test_plan_artifacts_are_written_with_required_fields(tmp_path):
    idea_path = tmp_path / "idea.json"
    run_dir = tmp_path / "run"
    write_idea(idea_path)
    write_dataset_context(run_dir)

    result = ensure_plan_review(
        output_dir=run_dir,
        idea_path=idea_path,
        config_path=None,
        plan_review="required",
        plan_review_mode="file",
        plan_feedback_file=None,
        plan_approval_file=None,
        approve_plan=True,
        max_plan_revisions=3,
    )

    plan = json.loads((run_dir / "research_plan.json").read_text())
    approval = json.loads((run_dir / "plan_approval.json").read_text())
    state = json.loads((run_dir / "plan_review_state.json").read_text())

    assert (run_dir / "research_plan.md").exists()
    assert plan["benchmark_summary"]["dataset_name"] == "demo-dataset"
    assert "leakage_hypotheses" in plan
    assert "detector_plan" in plan
    assert "expected_artifacts" in plan
    assert approval["approved"] is True
    assert approval["final_plan_hash"] == plan["plan_fingerprint"]
    assert state["approved"] is True
    assert result.approval["approved"] is True


def test_plan_review_blocks_execution_until_approval(tmp_path):
    idea_path = tmp_path / "idea.json"
    run_dir = tmp_path / "run"
    write_idea(idea_path)
    write_dataset_context(run_dir)

    with pytest.raises(PlanApprovalRequiredError):
        ensure_plan_review(
            output_dir=run_dir,
            idea_path=idea_path,
            config_path=None,
            plan_review="required",
            plan_review_mode="file",
            plan_feedback_file=None,
            plan_approval_file=None,
            approve_plan=False,
            max_plan_revisions=3,
        )

    state = json.loads((run_dir / "plan_review_state.json").read_text())
    approval = json.loads((run_dir / "plan_approval.json").read_text())
    assert state["approved"] is False
    assert approval["approved"] is False


def test_feedback_triggers_a_revised_plan(tmp_path):
    idea_path = tmp_path / "idea.json"
    run_dir = tmp_path / "run"
    feedback_path = tmp_path / "feedback.md"
    approval_path = tmp_path / "approval.json"
    write_idea(idea_path)
    write_dataset_context(run_dir)

    feedback_path.write_text("- Add a stricter stop condition for unresolved evidence gaps.\n")
    approval_path.write_text(json.dumps({"approved": True}, indent=2))

    ensure_plan_review(
        output_dir=run_dir,
        idea_path=idea_path,
        config_path=None,
        plan_review="required",
        plan_review_mode="file",
        plan_feedback_file=feedback_path,
        plan_approval_file=approval_path,
        approve_plan=False,
        max_plan_revisions=3,
    )

    plan = json.loads((run_dir / "research_plan.json").read_text())
    assert plan["review_round"] == 2
    assert "stricter stop condition" in " ".join(plan["review_feedback_history"]).lower()
    assert "Address reviewer request" in " ".join(plan["evidence_strategy"])


def test_abort_stops_execution_cleanly(tmp_path):
    idea_path = tmp_path / "idea.json"
    run_dir = tmp_path / "run"
    write_idea(idea_path)
    write_dataset_context(run_dir)

    responses = iter(["abort"])

    with pytest.raises(PlanRejectedError):
        ensure_plan_review(
            output_dir=run_dir,
            idea_path=idea_path,
            config_path=None,
            plan_review="required",
            plan_review_mode="interactive",
            plan_feedback_file=None,
            plan_approval_file=None,
            approve_plan=False,
            max_plan_revisions=3,
            input_func=lambda _prompt: next(responses),
            is_tty=True,
        )

    approval = json.loads((run_dir / "plan_approval.json").read_text())
    assert approval["approved"] is False
