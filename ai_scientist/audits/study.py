from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .artifacts import AuditArtifactBundle, load_validated_audit_bundle
from .report_review import ensure_review_passes


class StudyBundleGenerationError(RuntimeError):
    """Raised when the audit-native study bundle cannot be generated."""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _maybe_relpath(path: Path, start: Path) -> str:
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return str(path)


def _format_float(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_optional_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text()


def _render_markdown_table(columns: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_None._"

    def normalize(value: Any) -> str:
        return str(value).replace("\n", " ").strip()

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rendered_rows = [
        "| " + " | ".join(normalize(value) for value in row) + " |" for row in rows
    ]
    return "\n".join([header, separator, *rendered_rows])


def _save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _generate_figures(bundle: AuditArtifactBundle, figures_dir: Path) -> list[dict[str, str]]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, str]] = []

    split_counts = pd.DataFrame(bundle.split_manifest["splits"])[["name", "record_count"]]
    split_plot = figures_dir / "split_record_counts.png"
    plt.figure(figsize=(6, 3.5))
    plt.bar(split_counts["name"], split_counts["record_count"], color="#4C78A8")
    plt.xlabel("Split")
    plt.ylabel("Records")
    plt.title("Benchmark Split Record Counts")
    _save_plot(split_plot)
    outputs.append(
        {
            "name": "split_record_counts",
            "path": split_plot.name,
            "caption": "Split record counts derived directly from split_manifest.json.",
            "source_artifact": "split_manifest.json",
        }
    )

    detector_counts = pd.DataFrame(bundle.audit_results["detectors_run"])[
        ["name", "finding_count"]
    ]
    detector_plot = figures_dir / "detector_findings_counts.png"
    plt.figure(figsize=(6, 3.5))
    plt.bar(detector_counts["name"], detector_counts["finding_count"], color="#F58518")
    plt.xlabel("Detector")
    plt.ylabel("Findings")
    plt.xticks(rotation=20, ha="right")
    plt.title("Detector Finding Counts")
    _save_plot(detector_plot)
    outputs.append(
        {
            "name": "detector_findings_counts",
            "path": detector_plot.name,
            "caption": "Detector-level finding counts derived directly from audit_results.json.",
            "source_artifact": "audit_results.json",
        }
    )

    if bundle.metrics_before_after is not None:
        delta_frame = pd.DataFrame(bundle.metrics_before_after["deltas"])
        delta_plot = figures_dir / "remediation_delta_plot.png"
        plt.figure(figsize=(6, 3.5))
        x_labels = [
            f"{row.metric_name} ({row.split})" for row in delta_frame.itertuples(index=False)
        ]
        plt.bar(x_labels, delta_frame["delta"], color="#54A24B")
        plt.xlabel("Metric")
        plt.ylabel("Delta")
        plt.xticks(rotation=20, ha="right")
        plt.title("Remediation Metric Deltas")
        _save_plot(delta_plot)
        outputs.append(
            {
                "name": "remediation_delta_plot",
                "path": delta_plot.name,
                "caption": "Metric deltas derived directly from metrics_before_after.json.",
                "source_artifact": "metrics_before_after.json",
            }
        )

    return outputs


def _zip_directory(source_dir: Path, destination_path: Path) -> Path:
    archive_base = destination_path.with_suffix("")
    created = shutil.make_archive(
        str(archive_base),
        "zip",
        root_dir=source_dir.parent,
        base_dir=source_dir.name,
    )
    return Path(created)


def _render_run_metadata(
    *,
    run_dir: Path,
    bundle: AuditArtifactBundle,
) -> str:
    metadata = _load_optional_json(run_dir / "audit_run_metadata.json") or {}
    runtime_settings = metadata.get("runtime_settings") or {}
    review_settings = metadata.get("review_and_output_settings") or {}

    lines = [
        "## Run Metadata",
        f"- Run ID: `{bundle.audit_results['run_metadata']['run_id']}`",
        f"- Run mode: `{bundle.audit_results['run_metadata']['mode']}`",
        f"- Run status: `{bundle.audit_results['run_metadata']['status']}`",
        f"- Seed: `{bundle.audit_results['run_metadata']['seed']}`",
        f"- Dataset fingerprint: `{bundle.audit_results['provenance']['dataset_fingerprint']}`",
        f"- Created at: `{bundle.audit_results['provenance']['created_at']}`",
        f"- Updated at: `{bundle.audit_results['provenance']['updated_at']}`",
        f"- Git SHA: `{bundle.audit_results['provenance']['git_sha']}`",
    ]

    if metadata:
        lines.append(f"- Audit metadata artifact: `{run_dir / 'audit_run_metadata.json'}`")
    if runtime_settings:
        lines.append("")
        lines.append("### Runtime Settings")
        for key, value in sorted(runtime_settings.items()):
            lines.append(f"- `{key}`: `{value}`")
    if review_settings:
        lines.append("")
        lines.append("### Review And Output Settings")
        for key, value in sorted(review_settings.items()):
            lines.append(f"- `{key}`: `{value}`")

    return "\n".join(lines)


def _render_split_inventory(bundle: AuditArtifactBundle) -> str:
    split_rows = []
    for split in bundle.split_manifest["splits"]:
        group_summary = split.get("group_key_summary") or {}
        temporal_coverage = split.get("temporal_coverage") or {}
        split_rows.append(
            [
                split["name"],
                split["record_count"],
                ", ".join(split["file_paths"]),
                ", ".join(group_summary.get("group_keys") or []),
                group_summary.get("unique_group_count", "n/a"),
                temporal_coverage.get("min_timestamp", "n/a"),
                temporal_coverage.get("max_timestamp", "n/a"),
            ]
        )

    lines = [
        "## Data And Split Inventory",
        f"- Benchmark: `{bundle.audit_results['benchmark_summary']['benchmark_name']}`",
        f"- Dataset: `{bundle.split_manifest['dataset_name']}`",
        f"- Total records: `{bundle.audit_results['benchmark_summary']['record_count']}`",
        f"- Split names: `{', '.join(bundle.audit_results['benchmark_summary']['split_names'])}`",
        f"- Source files used: `{', '.join(bundle.split_manifest['file_paths_used'])}`",
        "",
        _render_markdown_table(
            ["Split", "Rows", "Files", "Group Keys", "Unique Groups", "Min Time", "Max Time"],
            split_rows,
        ),
    ]
    return "\n".join(lines)


def _render_detector_coverage(bundle: AuditArtifactBundle) -> str:
    detector_rows = [
        [
            detector["name"],
            detector["version"],
            detector["status"],
            detector["finding_count"],
        ]
        for detector in bundle.audit_results["detectors_run"]
    ]
    lines = [
        "## Detector Coverage",
        _render_markdown_table(
            ["Detector", "Version", "Status", "Findings"],
            detector_rows,
        ),
    ]
    return "\n".join(lines)


def _render_findings_inventory(bundle: AuditArtifactBundle) -> str:
    lines = [
        "## Findings Inventory",
        f"- Total findings: `{bundle.audit_results['findings_summary']['total_findings']}`",
        f"- Open findings: `{bundle.audit_results['findings_summary']['open_findings']}`",
        f"- By severity: `{bundle.audit_results['findings_summary']['by_severity']}`",
        f"- By detector: `{bundle.audit_results['findings_summary']['by_detector']}`",
        "",
    ]

    if bundle.findings.empty:
        lines.append("_No findings rows were present in the validated artifact._")
        return "\n".join(lines)

    findings_rows = []
    findings = bundle.findings.sort_values(
        ["severity", "confidence", "detector_name"],
        ascending=[True, False, True],
    )
    for _, row in findings.iterrows():
        findings_rows.append(
            [
                row["finding_id"],
                row["detector_name"],
                row["severity"],
                _format_float(float(row["confidence"])),
                row["remediation_status"],
                row["evidence_pointer"],
            ]
        )
    lines.append(
        _render_markdown_table(
            ["Finding", "Detector", "Severity", "Confidence", "Status", "Evidence Pointer"],
            findings_rows,
        )
    )
    return "\n".join(lines)


def _render_evidence_registry(bundle: AuditArtifactBundle) -> str:
    rows = [
        [
            evidence["evidence_id"],
            evidence["kind"],
            evidence["path"],
            evidence["description"],
        ]
        for evidence in bundle.audit_results["evidence_references"]
    ]
    lines = ["## Evidence Registry"]
    if rows:
        lines.append(
            _render_markdown_table(
                ["Evidence ID", "Kind", "Path", "Description"],
                rows,
            )
        )
    else:
        lines.append("_No evidence references were recorded in audit_results.json._")
    return "\n".join(lines)


def _render_remediation_results(bundle: AuditArtifactBundle) -> str:
    lines = ["## Remediation Results"]
    if bundle.metrics_before_after is None:
        lines.append(
            "_No `metrics_before_after.json` artifact was present for this audit run._"
        )
        return "\n".join(lines)

    delta_rows = [
        [
            delta["metric_name"],
            delta["split"],
            _format_float(delta["baseline_value"]),
            _format_float(delta["remediated_value"]),
            _format_float(delta["delta"]),
        ]
        for delta in bundle.metrics_before_after["deltas"]
    ]
    lines.extend(
        [
            f"- Evaluated splits: `{', '.join(bundle.metrics_before_after['split_information']['evaluated_splits'])}`",
            f"- Notes: {bundle.metrics_before_after['split_information']['notes']}",
            "",
            _render_markdown_table(
                ["Metric", "Split", "Baseline", "Remediated", "Delta"],
                delta_rows,
            ),
        ]
    )
    return "\n".join(lines)


def _render_confidence(bundle: AuditArtifactBundle) -> str:
    confidence = bundle.audit_results["confidence"]
    lines = [
        "## Confidence And Limitations",
        f"- Overall confidence: `{confidence['overall']:.2f}`",
        f"- Evidence coverage: `{confidence['evidence_coverage']:.2f}`",
        f"- Notes: {confidence['notes']}",
    ]
    return "\n".join(lines)


def _render_review_summary(review: dict[str, Any]) -> str:
    lines = [
        "## Report Review",
        f"- Review status: `{review['status']}`",
        f"- Regenerated report: `{review['regenerated_report']}`",
        f"- Blocking issue count: `{review['blocking_issue_count']}`",
    ]
    issues = review.get("issues") or []
    if issues:
        lines.append("")
        lines.append("### Review Issues")
        for issue in issues:
            lines.append(
                f"- `{issue['severity']}` `{issue['code']}`: {issue['message']} (fixable={issue['fixable']})"
            )
    opportunities = review.get("figure_table_opportunities") or []
    if opportunities:
        lines.append("")
        lines.append("### Study Figures And Tables")
        for opportunity in opportunities:
            lines.append(
                f"- `{opportunity['type']}` `{opportunity['name']}`: {opportunity['reason']}"
            )
    return "\n".join(lines)


def _render_figure_inventory(
    *,
    run_dir: Path,
    figures_dir: Path,
    figures: list[dict[str, str]],
    figures_zip_path: Path | None,
) -> str:
    lines = [
        "## Study Figures",
        f"- Figures directory: `{_maybe_relpath(figures_dir, run_dir)}`",
    ]
    if figures_zip_path is not None:
        lines.append(f"- Figures zip: `{_maybe_relpath(figures_zip_path, run_dir)}`")
    lines.append("")
    for figure in figures:
        lines.append(
            f"- `{figure['name']}` from `{figure['source_artifact']}` -> "
            f"`{_maybe_relpath(figures_dir / figure['path'], run_dir)}`. {figure['caption']}"
        )
    return "\n".join(lines)


def _render_artifact_index(
    *,
    run_dir: Path,
    bundle: AuditArtifactBundle,
    figures_dir: Path,
    figures_zip_path: Path | None,
) -> str:
    candidate_paths = [
        bundle.audit_results_path,
        bundle.split_manifest_path,
        bundle.findings_path,
        bundle.metrics_before_after_path,
        run_dir / "dataset_card.md",
        run_dir / "research_plan.json",
        run_dir / "research_plan.md",
        run_dir / "audit_report.md",
        run_dir / "audit_report_review.json",
        run_dir / "audit_report_review.md",
        run_dir / "study_report.md",
        run_dir / "study_bundle_manifest.json",
        figures_dir,
        figures_zip_path,
    ]
    lines = ["## Artifact Index"]
    for path in candidate_paths:
        if path is None:
            continue
        path = Path(path)
        if path.exists():
            lines.append(f"- `{_maybe_relpath(path, run_dir)}`")
    return "\n".join(lines)


def _render_embedded_markdown(title: str, text: str | None) -> str:
    if not text:
        return f"## {title}\n_Not present in this run._"
    return f"## {title}\n```markdown\n{text.rstrip()}\n```"


def _render_study_report(
    *,
    run_dir: Path,
    bundle: AuditArtifactBundle,
    review: dict[str, Any],
    figures_dir: Path,
    figures: list[dict[str, str]],
    figures_zip_path: Path | None,
) -> str:
    lines = [
        "# Study Report",
        "",
        "This study report is the primary product surface for AI-bench-auditor. It is intentionally markdown-first and artifact-linked so another LLM or reviewer can read the methodology, data context, findings, remediation evidence, and figure inventory without going through LaTeX or PDF generation.",
        "",
        _render_run_metadata(run_dir=run_dir, bundle=bundle),
        "",
        "## Methodology Summary",
        "- Workflow: reproduce benchmark protocol, run leakage detectors, confirm findings with remediation or falsification, and finish with synthesis from validated artifacts.",
        f"- Audit score: `{bundle.audit_results['audit_score']['value']} / {bundle.audit_results['audit_score']['max_value']}` ({bundle.audit_results['audit_score']['rating']})",
        f"- Validated artifact directory: `{_maybe_relpath(bundle.artifact_dir, run_dir)}`",
        f"- Primary audit report: `{_maybe_relpath(run_dir / 'audit_report.md', run_dir)}`",
        "",
        _render_split_inventory(bundle),
        "",
        _render_detector_coverage(bundle),
        "",
        _render_findings_inventory(bundle),
        "",
        _render_evidence_registry(bundle),
        "",
        _render_remediation_results(bundle),
        "",
        _render_confidence(bundle),
        "",
        _render_review_summary(review),
        "",
        _render_figure_inventory(
            run_dir=run_dir,
            figures_dir=figures_dir,
            figures=figures,
            figures_zip_path=figures_zip_path,
        ),
        "",
        _render_artifact_index(
            run_dir=run_dir,
            bundle=bundle,
            figures_dir=figures_dir,
            figures_zip_path=figures_zip_path,
        ),
        "",
        _render_embedded_markdown("Embedded Research Plan", _load_optional_text(run_dir / "research_plan.md")),
        "",
        _render_embedded_markdown("Embedded Dataset Card", _load_optional_text(run_dir / "dataset_card.md")),
        "",
        _render_embedded_markdown(
            "Embedded Audit Report Review",
            _load_optional_text(run_dir / "audit_report_review.md"),
        ),
    ]
    return "\n".join(lines) + "\n"


def build_audit_study_bundle(
    *,
    run_dir: str | Path,
    artifact_dir: str | Path,
    audit_report_review_path: str | Path,
    emit_figures_zip: bool,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    artifact_dir = Path(artifact_dir)
    audit_report_review_path = Path(audit_report_review_path)

    bundle = load_validated_audit_bundle(artifact_dir)
    review = json.loads(audit_report_review_path.read_text())
    ensure_review_passes(review)

    figures_dir = run_dir / "study_figures"
    figures = _generate_figures(bundle, figures_dir)

    figures_zip_path = None
    if emit_figures_zip:
        figures_zip_path = _zip_directory(figures_dir, run_dir / "study_figures.zip")
        if not figures_zip_path.exists():
            raise StudyBundleGenerationError(
                "study_figures.zip was requested but was not created."
            )

    study_report_path = run_dir / "study_report.md"
    study_report = _render_study_report(
        run_dir=run_dir,
        bundle=bundle,
        review=review,
        figures_dir=figures_dir,
        figures=figures,
        figures_zip_path=figures_zip_path,
    )
    _write_text(study_report_path, study_report)

    manifest = {
        "contract_version": 1,
        "run_dir": str(run_dir.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "audit_report_path": str((run_dir / "audit_report.md").resolve()),
        "audit_report_review_path": str(audit_report_review_path.resolve()),
        "study_report_path": str(study_report_path.resolve()),
        "figures_dir": str(figures_dir.resolve()),
        "figures": figures,
        "figures_zip": {
            "requested": emit_figures_zip,
            "emitted": bool(figures_zip_path),
            "path": str(figures_zip_path.resolve()) if figures_zip_path else None,
        },
        "summaries": {
            "run_metadata": bundle.audit_results["run_metadata"],
            "benchmark_summary": bundle.audit_results["benchmark_summary"],
            "findings_summary": bundle.audit_results["findings_summary"],
            "confidence": bundle.audit_results["confidence"],
            "audit_score": bundle.audit_results["audit_score"],
        },
        "artifact_index": {
            "audit_results": str(bundle.audit_results_path.resolve()),
            "split_manifest": str(bundle.split_manifest_path.resolve()),
            "findings": str(bundle.findings_path.resolve()),
            "metrics_before_after": (
                str(bundle.metrics_before_after_path.resolve())
                if bundle.metrics_before_after_path is not None
                else None
            ),
            "dataset_card": (
                str((run_dir / "dataset_card.md").resolve())
                if (run_dir / "dataset_card.md").exists()
                else None
            ),
            "research_plan_json": (
                str((run_dir / "research_plan.json").resolve())
                if (run_dir / "research_plan.json").exists()
                else None
            ),
            "research_plan_md": (
                str((run_dir / "research_plan.md").resolve())
                if (run_dir / "research_plan.md").exists()
                else None
            ),
        },
    }
    (run_dir / "study_bundle_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
