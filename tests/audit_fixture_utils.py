import json
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


def write_findings_csv(
    path: Path,
    *,
    rows: list[dict] | None = None,
) -> None:
    columns = list(build_example_findings_columns())
    if rows is None:
        rows = [
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
                "provenance_detector_versions_json": '{"exact_duplicate": "1.0.0", "temporal_overlap": "1.0.0"}',
                "provenance_created_at": "2026-04-11T15:30:00Z",
                "provenance_updated_at": "2026-04-11T15:30:00Z",
            }
        ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def write_valid_audit_bundle(
    artifact_dir: Path,
    *,
    findings_rows: list[dict] | None = None,
    include_metrics: bool = True,
    high_confidence_clean: bool = False,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "dataset_card.md").write_text("# Dataset Card\n\nSynthetic fixture.\n")
    (artifact_dir / "evidence").mkdir(exist_ok=True)
    (artifact_dir / "evidence" / "exact_duplicate_pairs.parquet").write_text("placeholder")

    split_manifest = build_example_split_manifest()
    audit_results = build_example_audit_results()
    audit_results["benchmark_summary"]["dataset_name"] = split_manifest["dataset_name"]
    audit_results["benchmark_summary"]["split_names"] = [
        split["name"] for split in split_manifest["splits"]
    ]
    audit_results["benchmark_summary"]["record_count"] = sum(
        split["record_count"] for split in split_manifest["splits"]
    )

    if high_confidence_clean:
        findings_rows = []
        audit_results["findings_summary"] = {
            "total_findings": 0,
            "open_findings": 0,
            "by_severity": {},
            "by_detector": {},
        }
        audit_results["detectors_run"] = [
            {
                "name": "exact_duplicate",
                "version": "1.0.0",
                "status": "completed",
                "finding_count": 0,
            },
            {
                "name": "temporal_overlap",
                "version": "1.0.0",
                "status": "completed",
                "finding_count": 0,
            },
        ]
        audit_results["confidence"]["overall"] = 0.96
        audit_results["confidence"]["evidence_coverage"] = 0.97
        audit_results["confidence"]["notes"] = "No supported leakage findings were detected."
        audit_results["evidence_references"] = []
    else:
        if findings_rows is None:
            findings_rows = None
        total_findings = 0 if findings_rows == [] else 1
        audit_results["findings_summary"] = {
            "total_findings": total_findings,
            "open_findings": total_findings,
            "by_severity": ({"high": total_findings} if total_findings else {}),
            "by_detector": ({"exact_duplicate": total_findings} if total_findings else {}),
        }
        audit_results["detectors_run"] = [
            {
                "name": "exact_duplicate",
                "version": "1.0.0",
                "status": "completed",
                "finding_count": total_findings,
            },
            {
                "name": "temporal_overlap",
                "version": "1.0.0",
                "status": "completed",
                "finding_count": 0,
            },
        ]
        audit_results["evidence_references"] = (
            [
                {
                    "evidence_id": "evidence-001",
                    "path": "evidence/exact_duplicate_pairs.parquet",
                    "kind": "parquet",
                    "description": "Exact duplicate candidate pairs with hashes.",
                }
            ]
            if total_findings
            else []
        )

    (artifact_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2))
    (artifact_dir / "audit_results.json").write_text(json.dumps(audit_results, indent=2))
    write_findings_csv(artifact_dir / "findings.csv", rows=findings_rows)

    if include_metrics:
        metrics_before_after = build_example_metrics_before_after()
        metrics_before_after["provenance"]["dataset_fingerprint"] = audit_results[
            "provenance"
        ]["dataset_fingerprint"]
        metrics_before_after["provenance"]["run_id"] = audit_results["provenance"]["run_id"]
        (artifact_dir / "metrics_before_after.json").write_text(
            json.dumps(metrics_before_after, indent=2)
        )

    return artifact_dir


def write_references_bib(path: Path) -> None:
    path.write_text(
        """@article{Hinton06,
author = {Hinton, Geoffrey E. and Osindero, Simon and Teh, Yee Whye},
journal = {Neural Computation},
pages = {1527--1554},
title = {A Fast Learning Algorithm for Deep Belief Nets},
volume = {18},
year = {2006}
}

@book{goodfellow2016deep,
title={Deep learning},
author={Goodfellow, Ian and Bengio, Yoshua and Courville, Aaron and Bengio, Yoshua},
volume={1},
year={2016},
publisher={MIT Press}
}
"""
    )


def make_manager_for_artifact_dir(artifact_dir: Path):
    node = Node(
        id="node-001",
        code="print('audit')",
        is_buggy=False,
        is_buggy_plots=False,
        metric=MetricValue(82.5, maximize=True, name="audit_score"),
        exp_results_dir=str(artifact_dir),
    )
    return SimpleNamespace(
        stages=[SimpleNamespace(name="4_final")],
        journals={"4_final": Journal(nodes=[node])},
        cfg=None,
    )
