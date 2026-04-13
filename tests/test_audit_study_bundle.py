import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ai_scientist.audits.artifacts import AuditArtifactError
from ai_scientist.audits.report import generate_audit_report
from ai_scientist.audits.report_review import review_audit_report
from ai_scientist.audits.study import build_audit_study_bundle
from tests.audit_fixture_utils import write_valid_audit_bundle


def prepare_reviewed_run(run_dir: Path) -> tuple[Path, Path]:
    artifact_dir = write_valid_audit_bundle(run_dir)
    report_path = generate_audit_report(
        audit_results_path=artifact_dir / "audit_results.json",
        split_manifest_path=artifact_dir / "split_manifest.json",
        findings_path=artifact_dir / "findings.csv",
        metrics_before_after_path=artifact_dir / "metrics_before_after.json",
        output_path=artifact_dir / "audit_report.md",
    )
    review_audit_report(
        artifact_dir=artifact_dir,
        audit_report_path=report_path,
        output_json_path=run_dir / "audit_report_review.json",
        output_md_path=run_dir / "audit_report_review.md",
    )
    (run_dir / "research_plan.md").write_text(
        "# Research Plan\n\n- Audit the declared benchmark splits.\n- Validate detector outputs against evidence.\n"
    )
    (run_dir / "research_plan.json").write_text(
        json.dumps(
            {
                "plan_fingerprint": "sha256:test-plan",
                "review_round": 1,
                "audit_targets": ["exact_duplicate"],
            },
            indent=2,
        )
    )
    (run_dir / "audit_run_metadata.json").write_text(
        json.dumps(
            {
                "contract_version": 1,
                "mode": "audit",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "runtime_settings": {
                    "agent_num_workers": 1,
                    "skip_writeup": True,
                    "skip_review": True,
                },
                "review_and_output_settings": {
                    "output_surface": "study_bundle",
                    "emit_study_zip": True,
                    "plan_review": "required",
                    "plan_review_mode": "file",
                },
            },
            indent=2,
        )
    )
    return artifact_dir, run_dir / "audit_report_review.json"


def test_audit_study_bundle_emits_expected_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    artifact_dir, review_json_path = prepare_reviewed_run(run_dir)

    manifest = build_audit_study_bundle(
        run_dir=run_dir,
        artifact_dir=artifact_dir,
        audit_report_review_path=review_json_path,
        emit_figures_zip=True,
    )

    assert (run_dir / "study_report.md").exists()
    assert (run_dir / "study_bundle_manifest.json").exists()
    assert (run_dir / "study_figures").is_dir()
    assert (run_dir / "study_figures.zip").exists()
    assert manifest["figures_zip"]["emitted"] is True
    assert len(manifest["figures"]) >= 2

    report_text = (run_dir / "study_report.md").read_text()
    assert "Study Report" in report_text
    assert "Methodology Summary" in report_text
    assert "Data And Split Inventory" in report_text
    assert "Findings Inventory" in report_text
    assert "Study Figures" in report_text
    assert "Embedded Research Plan" in report_text
    assert "Embedded Dataset Card" in report_text


def test_audit_study_bundle_supports_optional_figure_zip(tmp_path):
    run_dir = tmp_path / "run"
    artifact_dir, review_json_path = prepare_reviewed_run(run_dir)

    manifest = build_audit_study_bundle(
        run_dir=run_dir,
        artifact_dir=artifact_dir,
        audit_report_review_path=review_json_path,
        emit_figures_zip=False,
    )

    assert (run_dir / "study_report.md").exists()
    assert (run_dir / "study_figures").is_dir()
    assert not (run_dir / "study_figures.zip").exists()
    assert manifest["figures_zip"]["requested"] is False
    assert manifest["figures_zip"]["emitted"] is False


def test_invalid_audit_artifacts_block_study_bundle_generation(tmp_path):
    run_dir = tmp_path / "run"
    artifact_dir, review_json_path = prepare_reviewed_run(run_dir)
    (artifact_dir / "evidence" / "exact_duplicate_pairs.parquet").unlink()

    with pytest.raises(AuditArtifactError):
        build_audit_study_bundle(
            run_dir=run_dir,
            artifact_dir=artifact_dir,
            audit_report_review_path=review_json_path,
            emit_figures_zip=True,
        )

    assert not (run_dir / "study_bundle_manifest.json").exists()
    assert not (run_dir / "study_figures.zip").exists()


def test_failed_report_review_blocks_study_bundle_generation(tmp_path):
    run_dir = tmp_path / "run"
    artifact_dir = write_valid_audit_bundle(run_dir)
    failed_review_path = run_dir / "audit_report_review.json"
    failed_review_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "issues": [
                    {
                        "code": "overclaiming_certainty",
                        "severity": "critical",
                        "message": "Unsupported certainty claim remained unresolved.",
                        "fixable": False,
                    }
                ],
            },
            indent=2,
        )
    )

    with pytest.raises(AuditArtifactError):
        build_audit_study_bundle(
            run_dir=run_dir,
            artifact_dir=artifact_dir,
            audit_report_review_path=failed_review_path,
            emit_figures_zip=True,
        )

    assert not (run_dir / "study_bundle_manifest.json").exists()
    assert not (run_dir / "study_figures.zip").exists()
