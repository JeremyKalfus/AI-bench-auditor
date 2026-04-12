import json
from pathlib import Path

from ai_scientist.audits.report import generate_audit_report
from ai_scientist.audits.report_review import review_audit_report
from tests.audit_fixture_utils import write_valid_audit_bundle


def write_report(bundle_dir: Path) -> Path:
    return generate_audit_report(
        audit_results_path=bundle_dir / "audit_results.json",
        split_manifest_path=bundle_dir / "split_manifest.json",
        findings_path=bundle_dir / "findings.csv",
        metrics_before_after_path=(
            bundle_dir / "metrics_before_after.json"
            if (bundle_dir / "metrics_before_after.json").exists()
            else None
        ),
        output_path=bundle_dir / "audit_report.md",
    )


def test_report_review_catches_unsupported_claims(tmp_path):
    bundle_dir = write_valid_audit_bundle(tmp_path / "bundle")
    report_path = write_report(bundle_dir)
    report_path.write_text(
        report_path.read_text()
        + "\nThis proves definitively that the benchmark is contaminated.\n"
    )

    review = review_audit_report(
        artifact_dir=bundle_dir,
        audit_report_path=report_path,
        output_json_path=bundle_dir / "audit_report_review.json",
        output_md_path=bundle_dir / "audit_report_review.md",
        regenerate_if_fixable=False,
    )

    assert review["status"] == "failed"
    assert any(issue["code"] == "overclaiming_certainty" for issue in review["issues"])


def test_report_review_catches_missing_evidence_references(tmp_path):
    bundle_dir = write_valid_audit_bundle(tmp_path / "bundle")
    report_path = write_report(bundle_dir)
    stripped = report_path.read_text().replace(
        "evidence/exact_duplicate_pairs.parquet", "evidence/missing_reference.parquet"
    )
    report_path.write_text(stripped)

    review = review_audit_report(
        artifact_dir=bundle_dir,
        audit_report_path=report_path,
        output_json_path=bundle_dir / "audit_report_review.json",
        output_md_path=bundle_dir / "audit_report_review.md",
        regenerate_if_fixable=False,
    )

    assert review["status"] == "failed"
    assert any(issue["code"] == "missing_expected_fragment" for issue in review["issues"])


def test_report_review_passes_for_clean_fixture_bundle_and_is_deterministic(tmp_path):
    bundle_dir = write_valid_audit_bundle(tmp_path / "bundle")
    report_path = write_report(bundle_dir)

    first = review_audit_report(
        artifact_dir=bundle_dir,
        audit_report_path=report_path,
        output_json_path=bundle_dir / "audit_report_review.json",
        output_md_path=bundle_dir / "audit_report_review.md",
    )
    second = review_audit_report(
        artifact_dir=bundle_dir,
        audit_report_path=report_path,
        output_json_path=bundle_dir / "audit_report_review_second.json",
        output_md_path=bundle_dir / "audit_report_review_second.md",
    )

    assert first["status"] == "passed"
    assert second["status"] == "passed"
    assert json.loads((bundle_dir / "audit_report_review.json").read_text()) == json.loads(
        (bundle_dir / "audit_report_review_second.json").read_text()
    )
    assert (bundle_dir / "audit_report_review.md").exists()
