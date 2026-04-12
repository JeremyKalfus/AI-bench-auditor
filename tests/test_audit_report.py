import json

import pandas as pd

from ai_scientist.audits.report import generate_audit_report
from ai_scientist.audits.schema import (
    build_example_audit_results,
    build_example_findings_columns,
    build_example_metrics_before_after,
    build_example_split_manifest,
)


def test_generate_audit_report_writes_markdown_with_artifact_paths(tmp_path):
    audit_results_path = tmp_path / "audit_results.json"
    split_manifest_path = tmp_path / "split_manifest.json"
    metrics_before_after_path = tmp_path / "metrics_before_after.json"
    findings_path = tmp_path / "findings.csv"
    output_path = tmp_path / "audit_report.md"
    evidence_path = tmp_path / "evidence" / "exact_duplicate_pairs.parquet"

    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text("placeholder evidence")

    audit_results = build_example_audit_results()
    audit_results["evidence_references"][0]["path"] = str(evidence_path)

    audit_results_path.write_text(json.dumps(audit_results, indent=2))
    split_manifest_path.write_text(json.dumps(build_example_split_manifest(), indent=2))
    metrics_before_after_path.write_text(
        json.dumps(build_example_metrics_before_after(), indent=2)
    )
    findings_columns = list(build_example_findings_columns())
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
        columns=findings_columns,
    ).to_csv(findings_path, index=False)

    generated_path = generate_audit_report(
        audit_results_path=audit_results_path,
        split_manifest_path=split_manifest_path,
        findings_path=findings_path,
        metrics_before_after_path=metrics_before_after_path,
        output_path=output_path,
    )

    assert generated_path == output_path
    assert output_path.exists()
    report_text = output_path.read_text()
    assert "Audit Report" in report_text
    assert str(audit_results_path) in report_text
    assert str(split_manifest_path) in report_text
    assert str(findings_path) in report_text
    assert str(metrics_before_after_path) in report_text
    assert "demo-dataset" in report_text
    assert "Detectors Run" in report_text
    assert "Evidence References" in report_text
    assert "finding-001" in report_text
    assert str(evidence_path) in report_text
    assert "Confidence and Limitations" in report_text
    assert "Workflow Guard" in report_text
