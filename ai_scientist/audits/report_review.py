from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .artifacts import AuditArtifactBundle, AuditArtifactError, load_validated_audit_bundle
from .report import SEVERITY_ORDER, generate_audit_report


REPORT_REVIEW_CONTRACT_VERSION = 1
OVERCLAIM_PATTERNS = (
    r"\bproves\b",
    r"\bproven\b",
    r"\bguarantees\b",
    r"\bdefinitive proof\b",
    r"\bcertainly\b",
    r"\bundeniable\b",
    r"\bconclusive\b",
)


def _load_text(path: Path) -> str:
    return path.read_text()


def _issue(
    *,
    code: str,
    severity: str,
    message: str,
    fixable: bool = False,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "fixable": fixable,
    }


def _expected_report_fragments(bundle: AuditArtifactBundle) -> list[str]:
    audit_results = bundle.audit_results
    fragments = [
        "# Audit Report",
        "## Summary",
        f"- Run ID: {audit_results['run_metadata']['run_id']}",
        f"- Benchmark: {audit_results['benchmark_summary']['benchmark_name']}",
        f"- Dataset: {audit_results['benchmark_summary']['dataset_name']}",
        f"- Status: {audit_results['run_metadata']['status']}",
        (
            f"- Audit score: {audit_results['audit_score']['value']} / "
            f"{audit_results['audit_score']['max_value']} ({audit_results['audit_score']['rating']})"
        ),
        "## Detectors Run",
        "## Findings Summary",
        f"- Total findings: {audit_results['findings_summary']['total_findings']}",
        f"- Open findings: {audit_results['findings_summary']['open_findings']}",
        "## Evidence References",
        "## Remediation Results",
        "## Confidence and Limitations",
    ]

    for detector in audit_results["detectors_run"]:
        fragments.append(
            f"- `{detector['name']}` version `{detector['version']}` finished with "
            f"status `{detector['status']}` and {detector['finding_count']} finding(s)."
        )

    for reference in audit_results["evidence_references"]:
        fragments.append(reference["path"])

    if not bundle.findings.empty:
        findings = bundle.findings.copy()
        findings["_severity_rank"] = findings["severity"].map(SEVERITY_ORDER).fillna(99)
        findings = findings.sort_values(
            ["_severity_rank", "confidence", "detector_name"],
            ascending=[True, False, True],
        )
        for _, row in findings.head(5).iterrows():
            fragments.append(f"Finding `{row['finding_id']}`")
            fragments.append(str(row["evidence_pointer"]))

    if bundle.metrics_before_after is not None:
        for delta in bundle.metrics_before_after["deltas"]:
            fragments.append(
                f"- `{delta['metric_name']}` on `{delta['split']}` changed from "
                f"`{delta['baseline_value']}` to `{delta['remediated_value']}` "
                f"(delta={delta['delta']})"
            )

    return fragments


def _figure_table_opportunities(bundle: AuditArtifactBundle) -> list[dict[str, Any]]:
    opportunities = [
        {
            "type": "table",
            "name": "split_summary_table",
            "reason": "Required protocol summary derived from split_manifest.json.",
        },
        {
            "type": "table",
            "name": "detector_summary_table",
            "reason": "Required detector coverage summary derived from audit_results.json.",
        },
        {
            "type": "table",
            "name": "findings_table",
            "reason": "Required findings table derived from findings.*.",
        },
        {
            "type": "figure",
            "name": "split_record_counts",
            "reason": "A simple split-size plot adds visual context without inventing evidence.",
        },
        {
            "type": "figure",
            "name": "detector_findings_counts",
            "reason": "Detector-level finding counts can be plotted directly from audit_results.json.",
        },
    ]
    if bundle.metrics_before_after is not None:
        opportunities.append(
            {
                "type": "table",
                "name": "remediation_comparison_table",
                "reason": "Required before/after remediation summary derived from metrics_before_after.json.",
            }
        )
        opportunities.append(
            {
                "type": "figure",
                "name": "remediation_delta_plot",
                "reason": "Metric deltas can be plotted directly from metrics_before_after.json.",
            }
        )
    return opportunities


def _citation_needs(bundle: AuditArtifactBundle) -> list[dict[str, Any]]:
    benchmark = bundle.audit_results["benchmark_summary"]
    needs = [
        {
            "query": benchmark["benchmark_name"],
            "reason": "Benchmark or task background citation for the paper introduction/protocol section.",
            "required": True,
        },
        {
            "query": benchmark["dataset_name"],
            "reason": "Dataset background citation for the protocol section.",
            "required": True,
        },
        {
            "query": "data leakage benchmark contamination machine learning audit",
            "reason": "Related-work citation for audit and contamination context.",
            "required": True,
        },
    ]
    for detector in bundle.audit_results["detectors_run"]:
        needs.append(
            {
                "query": f"{detector['name']} data leakage machine learning",
                "reason": f"Optional detector-specific background for {detector['name']}.",
                "required": False,
            }
        )
    return needs


def _evaluate_report_text(
    report_text: str,
    bundle: AuditArtifactBundle,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    lower_report = report_text.lower()

    for fragment in _expected_report_fragments(bundle):
        if fragment not in report_text:
            issues.append(
                _issue(
                    code="missing_expected_fragment",
                    severity="high",
                    message=f"Audit report is missing expected evidence-backed content: {fragment}",
                    fixable=True,
                )
            )

    if "## Confidence and Limitations" not in report_text or "- Notes:" not in report_text:
        issues.append(
            _issue(
                code="missing_limitations",
                severity="high",
                message="Audit report is missing its confidence/limitations explanation.",
                fixable=True,
            )
        )

    if "## Detectors Run" not in report_text:
        issues.append(
            _issue(
                code="missing_detector_coverage",
                severity="high",
                message="Audit report is missing the detector coverage summary.",
                fixable=True,
            )
        )

    findings_total = bundle.audit_results["findings_summary"]["total_findings"]
    if findings_total > 0 and bundle.metrics_before_after is None:
        issues.append(
            _issue(
                code="missing_remediation_artifact",
                severity="critical",
                message=(
                    "The audit still reports findings, but no metrics_before_after.json artifact exists "
                    "to support remediation or falsification claims."
                ),
                fixable=False,
            )
        )

    if findings_total > 0 and "## Remediation Results" not in report_text:
        issues.append(
            _issue(
                code="missing_remediation_section",
                severity="high",
                message="Audit report is missing the remediation-results section.",
                fixable=True,
            )
        )

    for pattern in OVERCLAIM_PATTERNS:
        if re.search(pattern, lower_report):
            issues.append(
                _issue(
                    code="overclaiming_certainty",
                    severity="critical",
                    message=(
                        "Audit report uses overclaiming certainty language that is not justified by "
                        "deterministic artifacts."
                    ),
                    fixable=False,
                )
            )
            break

    return issues


def _render_review_markdown(review: dict[str, Any]) -> str:
    lines = [
        "# Audit Report Review",
        "",
        f"- Review status: `{review['status']}`",
        f"- Regenerated report: `{review['regenerated_report']}`",
        f"- Blocking issues: {review['blocking_issue_count']}",
        "",
        "## Issues",
    ]
    if review["issues"]:
        for issue in review["issues"]:
            lines.append(
                f"- [{issue['severity']}] {issue['code']}: {issue['message']} "
                f"(fixable={issue['fixable']})"
            )
    else:
        lines.append("- No report-support issues were found.")

    lines.extend(["", "## Figure and Table Opportunities"])
    for opportunity in review["figure_table_opportunities"]:
        lines.append(
            f"- {opportunity['type']}: `{opportunity['name']}` because {opportunity['reason']}"
        )

    return "\n".join(lines) + "\n"


def review_audit_report(
    *,
    artifact_dir: str | Path,
    audit_report_path: str | Path,
    output_json_path: str | Path,
    output_md_path: str | Path,
    regenerate_if_fixable: bool = True,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    audit_report_path = Path(audit_report_path)
    output_json_path = Path(output_json_path)
    output_md_path = Path(output_md_path)

    bundle = load_validated_audit_bundle(artifact_dir)
    report_text = _load_text(audit_report_path)
    issues = _evaluate_report_text(report_text, bundle)
    regenerated = False

    if issues and regenerate_if_fixable and any(issue["fixable"] for issue in issues):
        generate_audit_report(
            audit_results_path=bundle.audit_results_path,
            split_manifest_path=bundle.split_manifest_path,
            findings_path=bundle.findings_path,
            metrics_before_after_path=bundle.metrics_before_after_path,
            output_path=audit_report_path,
        )
        report_text = _load_text(audit_report_path)
        issues = _evaluate_report_text(report_text, bundle)
        regenerated = True

    blocking_issues = [issue for issue in issues if issue["severity"] in {"critical", "high"}]
    review = {
        "contract_version": REPORT_REVIEW_CONTRACT_VERSION,
        "status": "passed" if not blocking_issues else "failed",
        "artifact_dir": str(artifact_dir.resolve()),
        "audit_report_path": str(audit_report_path.resolve()),
        "regenerated_report": regenerated,
        "issues": issues,
        "blocking_issue_count": len(blocking_issues),
        "figure_table_opportunities": _figure_table_opportunities(bundle),
        "citation_needs": _citation_needs(bundle),
    }

    output_json_path.write_text(json.dumps(review, indent=2))
    output_md_path.write_text(_render_review_markdown(review))
    return review


def ensure_review_passes(review: dict[str, Any]) -> None:
    if review["status"] != "passed":
        messages = "; ".join(issue["message"] for issue in review["issues"])
        raise AuditArtifactError(
            "Audit report review failed; study bundle generation is blocked. " + messages
        )
