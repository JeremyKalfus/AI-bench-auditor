from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import (
    validate_audit_results,
    validate_findings_columns,
    validate_metrics_before_after,
    validate_split_manifest,
)


SEVERITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _load_findings(path: str | Path) -> pd.DataFrame:
    findings_path = Path(path)
    if findings_path.suffix == ".parquet":
        findings = pd.read_parquet(findings_path)
    else:
        findings = pd.read_csv(findings_path)
    validate_findings_columns(findings.columns)
    return findings


def generate_audit_report(
    *,
    audit_results_path: str | Path,
    split_manifest_path: str | Path,
    findings_path: str | Path,
    output_path: str | Path,
    metrics_before_after_path: str | Path | None = None,
) -> Path:
    audit_results_path = Path(audit_results_path)
    split_manifest_path = Path(split_manifest_path)
    findings_path = Path(findings_path)
    output_path = Path(output_path)

    audit_results = _load_json(audit_results_path)
    split_manifest = _load_json(split_manifest_path)
    validate_audit_results(audit_results)
    validate_split_manifest(split_manifest)
    findings = _load_findings(findings_path)

    metrics_before_after = None
    if metrics_before_after_path is not None:
        metrics_before_after_path = Path(metrics_before_after_path)
        metrics_before_after = _load_json(metrics_before_after_path)
        validate_metrics_before_after(metrics_before_after)

    lines = [
        "# Audit Report",
        "",
        "## Summary",
        f"- Run ID: {audit_results['run_metadata']['run_id']}",
        f"- Benchmark: {audit_results['benchmark_summary']['benchmark_name']}",
        f"- Dataset: {audit_results['benchmark_summary']['dataset_name']}",
        f"- Record count: {audit_results['benchmark_summary']['record_count']}",
        f"- Split names: {', '.join(audit_results['benchmark_summary']['split_names'])}",
        f"- Status: {audit_results['run_metadata']['status']}",
        f"- Audit score: {audit_results['audit_score']['value']} / {audit_results['audit_score']['max_value']} ({audit_results['audit_score']['rating']})",
        f"- Overall confidence: {audit_results['confidence']['overall']:.2f}",
        f"- Evidence coverage: {audit_results['confidence']['evidence_coverage']:.2f}",
        f"- Audit results artifact: `{audit_results_path}`",
        f"- Split manifest artifact: `{split_manifest_path}`",
        f"- Findings artifact: `{findings_path}`",
    ]

    if metrics_before_after_path is not None:
        lines.append(f"- Metrics artifact: `{metrics_before_after_path}`")

    lines.extend(
        [
            "",
            "## Split Coverage",
            f"- Split names: {', '.join(split['name'] for split in split_manifest['splits'])}",
            f"- Dataset fingerprint: {audit_results['provenance']['dataset_fingerprint']}",
        ]
    )

    lines.extend(["", "## Detectors Run"])
    for detector_run in audit_results["detectors_run"]:
        lines.append(
            f"- `{detector_run['name']}` version `{detector_run['version']}` finished with "
            f"status `{detector_run['status']}` and {detector_run['finding_count']} finding(s)."
        )

    lines.extend(
        [
            "",
            "## Findings Summary",
            f"- Total findings: {audit_results['findings_summary']['total_findings']}",
            f"- Open findings: {audit_results['findings_summary']['open_findings']}",
            f"- By severity: {audit_results['findings_summary']['by_severity']}",
            f"- By detector: {audit_results['findings_summary']['by_detector']}",
        ]
    )

    lines.extend(["", "## Major Findings"])
    if findings.empty:
        lines.append(f"- No rows were present in `{findings_path}`.")
    else:
        findings = findings.copy()
        findings["_severity_rank"] = findings["severity"].map(SEVERITY_ORDER).fillna(99)
        findings = findings.sort_values(
            ["_severity_rank", "confidence", "detector_name"],
            ascending=[True, False, True],
        )
        for _, row in findings.head(5).iterrows():
            lines.append(
                f"- Finding `{row['finding_id']}` from `{row['detector_name']}` reported "
                f"`{row['severity']}` severity with confidence {row['confidence']:.2f}; "
                f"evidence file `{row['evidence_pointer']}`; source row lives in `{findings_path}`; "
                f"remediation status `{row['remediation_status']}`."
            )

    lines.extend(["", "## Evidence References"])
    if audit_results["evidence_references"]:
        for evidence_reference in audit_results["evidence_references"]:
            lines.append(
                f"- Evidence `{evidence_reference['evidence_id']}` points to "
                f"`{evidence_reference['path']}` ({evidence_reference['kind']}): "
                f"{evidence_reference['description']}"
            )
    else:
        lines.append("- No additional evidence references were recorded in audit_results.json.")
    lines.append(
        f"- Split-manifest support for file inventory and provenance is recorded in `{split_manifest_path}`."
    )

    lines.extend(["", "## Remediation Results"])
    if metrics_before_after is not None:
        for delta in metrics_before_after["deltas"]:
            lines.append(
                f"- `{delta['metric_name']}` on `{delta['split']}` changed from "
                f"`{delta['baseline_value']}` to `{delta['remediated_value']}` "
                f"(delta={delta['delta']}) using `{metrics_before_after_path}`."
            )
    else:
        lines.append("- No `metrics_before_after.json` artifact was present for this audit run.")

    lines.extend(
        [
            "",
            "## Confidence and Limitations",
            f"- Overall confidence: {audit_results['confidence']['overall']:.2f}",
            f"- Evidence coverage: {audit_results['confidence']['evidence_coverage']:.2f}",
            f"- Notes: {audit_results['confidence']['notes']}",
        ]
    )

    lines.extend(
        [
            "",
            "## Workflow Guard",
            "- Audit mode only generates deterministic audit artifacts and this markdown report.",
            "- `.tex` and `.pdf` outputs remain disabled until the later verification phases pass and paper mode is invoked on a validated audit run.",
        ]
    )

    lines.extend(["", "## Artifact References"])
    for artifact_path in [
        audit_results_path,
        split_manifest_path,
        findings_path,
        metrics_before_after_path,
    ]:
        if artifact_path is None:
            continue
        lines.append(f"- `{artifact_path}`")

    output_path.write_text("\n".join(lines) + "\n")
    return output_path
