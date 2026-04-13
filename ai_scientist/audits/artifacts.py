from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import (
    validate_audit_results,
    validate_findings_columns,
    validate_metrics_before_after,
    validate_split_manifest,
)


FINDINGS_ARTIFACT_NAMES = ("findings.csv", "findings.parquet")
OPEN_REMEDIATION_STATUSES = {"open", "pending", "unresolved", "needs_followup"}


class AuditArtifactError(ValueError):
    """Raised when an audit artifact bundle is missing or inconsistent."""


@dataclass(frozen=True)
class AuditArtifactBundle:
    artifact_dir: Path
    source_dir: Path
    audit_results_path: Path
    split_manifest_path: Path
    findings_path: Path
    metrics_before_after_path: Path | None
    audit_results: dict[str, Any]
    split_manifest: dict[str, Any]
    findings: pd.DataFrame
    metrics_before_after: dict[str, Any] | None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_findings(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        try:
            frame = pd.read_parquet(path)
        except Exception:
            csv_fallback = path.with_suffix(".csv")
            if not csv_fallback.exists():
                raise
            frame = pd.read_csv(csv_fallback)
    else:
        frame = pd.read_csv(path)
    validate_findings_columns(frame.columns)
    return frame


def find_findings_artifact(artifact_dir: str | Path) -> Path:
    artifact_dir = Path(artifact_dir)
    for name in FINDINGS_ARTIFACT_NAMES:
        candidate = artifact_dir / name
        if candidate.exists():
            return candidate
    raise AuditArtifactError(
        f"Missing required audit artifact: findings.csv or findings.parquet in {artifact_dir}"
    )


def _required_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise AuditArtifactError(f"Missing required audit artifact: {label}")
    return path


def _resolve_relative_to_source(source_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return source_dir / candidate


def _count_open_findings(findings: pd.DataFrame) -> int:
    if findings.empty:
        return 0
    statuses = findings["remediation_status"].fillna("").astype(str).str.lower()
    return int(statuses.isin(OPEN_REMEDIATION_STATUSES).sum())


def _validate_findings_summary(
    audit_results: dict[str, Any], findings: pd.DataFrame
) -> None:
    summary = audit_results["findings_summary"]
    total_findings = int(len(findings))
    if summary["total_findings"] != total_findings:
        raise AuditArtifactError(
            "findings_summary.total_findings does not match the findings artifact row count"
        )

    open_findings = _count_open_findings(findings)
    if summary["open_findings"] != open_findings:
        raise AuditArtifactError(
            "findings_summary.open_findings does not match open rows in the findings artifact"
        )

    by_detector = findings["detector_name"].value_counts().sort_index().to_dict()
    if summary["by_detector"] != by_detector:
        raise AuditArtifactError(
            "findings_summary.by_detector does not match the findings artifact"
        )

    by_severity = findings["severity"].value_counts().sort_index().to_dict()
    if summary["by_severity"] != by_severity:
        raise AuditArtifactError(
            "findings_summary.by_severity does not match the findings artifact"
        )


def _validate_detector_run_counts(
    audit_results: dict[str, Any], findings: pd.DataFrame
) -> None:
    by_detector = findings["detector_name"].value_counts().to_dict()
    for detector in audit_results["detectors_run"]:
        observed = int(by_detector.get(detector["name"], 0))
        if detector["finding_count"] != observed:
            raise AuditArtifactError(
                f"detectors_run entry for {detector['name']!r} reports {detector['finding_count']} "
                f"findings but the findings artifact contains {observed}"
            )


def _validate_benchmark_summary(
    audit_results: dict[str, Any], split_manifest: dict[str, Any]
) -> None:
    benchmark_summary = audit_results["benchmark_summary"]
    split_names = [split["name"] for split in split_manifest["splits"]]
    if benchmark_summary["split_names"] != split_names:
        raise AuditArtifactError(
            "benchmark_summary.split_names does not match split_manifest.json"
        )

    record_count = sum(int(split["record_count"]) for split in split_manifest["splits"])
    if benchmark_summary["record_count"] != record_count:
        raise AuditArtifactError(
            "benchmark_summary.record_count does not match split_manifest.json"
        )

    if benchmark_summary["dataset_name"] != split_manifest["dataset_name"]:
        raise AuditArtifactError(
            "benchmark_summary.dataset_name does not match split_manifest.json"
        )


def _validate_provenance_consistency(
    audit_results: dict[str, Any],
    split_manifest: dict[str, Any],
    metrics_before_after: dict[str, Any] | None,
    split_manifest_path: Path,
) -> None:
    audit_provenance = audit_results["provenance"]
    split_provenance = split_manifest["provenance"]
    if (
        audit_provenance["dataset_fingerprint"]
        != split_provenance["dataset_fingerprint"]
    ):
        raise AuditArtifactError(
            "Dataset fingerprint differs between audit_results.json and split_manifest.json"
        )

    if metrics_before_after is None:
        return

    metrics_provenance = metrics_before_after["provenance"]
    if (
        metrics_provenance["dataset_fingerprint"]
        != audit_provenance["dataset_fingerprint"]
    ):
        raise AuditArtifactError(
            "Dataset fingerprint differs between metrics_before_after.json and audit_results.json"
        )

    if metrics_provenance["run_id"] != audit_provenance["run_id"]:
        raise AuditArtifactError(
            "Run ID differs between metrics_before_after.json and audit_results.json"
        )

    manifest_name = Path(metrics_before_after["split_information"]["split_manifest_path"]).name
    if manifest_name != split_manifest_path.name:
        raise AuditArtifactError(
            "metrics_before_after.json references a different split manifest than the active audit bundle"
        )


def _validate_evidence_paths(
    source_dir: Path,
    audit_results: dict[str, Any],
    findings: pd.DataFrame,
) -> None:
    for reference in audit_results["evidence_references"]:
        resolved = _resolve_relative_to_source(source_dir, reference["path"])
        if not resolved.exists():
            raise AuditArtifactError(
                f"Referenced evidence file does not exist: {reference['path']}"
            )

    for evidence_pointer in findings["evidence_pointer"].fillna("").astype(str):
        if not evidence_pointer:
            raise AuditArtifactError("A findings row is missing its evidence_pointer")
        resolved = _resolve_relative_to_source(source_dir, evidence_pointer)
        if not resolved.exists():
            raise AuditArtifactError(
                f"Findings artifact references missing evidence: {evidence_pointer}"
            )


def load_validated_audit_bundle(artifact_dir: str | Path) -> AuditArtifactBundle:
    artifact_dir = Path(artifact_dir)
    audit_results_path = _required_file(
        artifact_dir / "audit_results.json", "audit_results.json"
    )
    split_manifest_path = _required_file(
        artifact_dir / "split_manifest.json", "split_manifest.json"
    )
    findings_path = find_findings_artifact(artifact_dir)
    metrics_before_after_path = artifact_dir / "metrics_before_after.json"
    if not metrics_before_after_path.exists():
        metrics_before_after_path = None

    source_dir = audit_results_path.resolve().parent

    audit_results = _load_json(audit_results_path)
    split_manifest = _load_json(split_manifest_path)
    findings = _load_findings(findings_path)
    metrics_before_after = None

    validate_audit_results(audit_results)
    validate_split_manifest(split_manifest)

    if metrics_before_after_path is not None:
        metrics_before_after = _load_json(metrics_before_after_path)
        validate_metrics_before_after(metrics_before_after)

    if audit_results["run_metadata"]["status"] != "completed":
        raise AuditArtifactError(
            "Audit bundle is not paper-eligible because run_metadata.status is not 'completed'"
        )

    _validate_findings_summary(audit_results, findings)
    _validate_detector_run_counts(audit_results, findings)
    _validate_benchmark_summary(audit_results, split_manifest)
    _validate_provenance_consistency(
        audit_results,
        split_manifest,
        metrics_before_after,
        split_manifest_path.resolve(),
    )
    _validate_evidence_paths(source_dir, audit_results, findings)

    return AuditArtifactBundle(
        artifact_dir=artifact_dir,
        source_dir=source_dir,
        audit_results_path=audit_results_path,
        split_manifest_path=split_manifest_path,
        findings_path=findings_path,
        metrics_before_after_path=metrics_before_after_path,
        audit_results=audit_results,
        split_manifest=split_manifest,
        findings=findings,
        metrics_before_after=metrics_before_after,
    )
