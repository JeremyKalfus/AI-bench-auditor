from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import load_validated_audit_bundle
from .canary_suite import load_canary_suite, write_canary_suite
from .detectors import (
    detect_exact_duplicates,
    detect_group_overlap,
    detect_near_duplicates,
    detect_preprocessing_leakage,
    detect_suspicious_feature_leakage,
    detect_temporal_leakage,
    empty_findings_dataframe,
)
from .report import generate_audit_report
from .schema import (
    build_provenance_block,
    validate_findings_columns,
    validate_metrics_before_after,
    validate_split_manifest,
)


VERIFICATION_STACK_SCHEMA_VERSION = "0.1.0"
VERIFICATION_BENCHMARK_SCHEMA_VERSION = "0.1.0"

STRATEGY_DETECTOR_PROFILES: dict[str, tuple[str, ...]] = {
    "detector_only": (
        "exact_duplicate",
        "group_overlap",
        "temporal_leakage",
    ),
    "one_shot_agent": (
        "exact_duplicate",
        "near_duplicate",
        "group_overlap",
        "temporal_leakage",
    ),
    "full_tree_search": (
        "exact_duplicate",
        "near_duplicate",
        "group_overlap",
        "temporal_leakage",
        "preprocessing_leakage",
        "suspicious_feature_leakage",
    ),
}

DETECTOR_VERSIONS = {
    "exact_duplicate": "0.1.0",
    "near_duplicate": "0.1.0",
    "group_overlap": "0.1.0",
    "temporal_leakage": "0.1.0",
    "preprocessing_leakage": "0.1.0",
    "suspicious_feature_leakage": "0.1.0",
}

OPEN_FINDING_STATUSES = {"open", "pending", "unresolved", "needs_followup"}

CANARY_REQUIRED_DETECTORS = {
    "exact_duplicate_leakage": ("exact_duplicate",),
    "near_duplicate_leakage": ("near_duplicate",),
    "group_entity_overlap": ("group_overlap",),
    "temporal_leakage": ("temporal_leakage",),
    "preprocessing_label_leakage": (
        "preprocessing_leakage",
        "suspicious_feature_leakage",
    ),
    "clean_negative_control": (),
}

DETECTOR_REMEDIATION_GUIDANCE = {
    "exact_duplicate": "Remove identical records that cross split boundaries before evaluation.",
    "near_duplicate": "Deduplicate or cluster semantically similar rows before assigning final splits.",
    "group_overlap": "Rebuild the split policy so entity identifiers stay confined to a single split.",
    "temporal_leakage": "Sort the data by event time and enforce a forward-only split boundary.",
    "preprocessing_leakage": "Move scaling or global preprocessing steps inside the train-only pipeline.",
    "suspicious_feature_leakage": "Drop or rebuild label-derived features before scoring the benchmark.",
}

SUMMARY_ARTIFACT_KEYS = {
    "canary_results.json": {"schema_version", "suite_dir", "cases", "summary"},
    "mutation_test_results.json": {
        "schema_version",
        "benchmark_id",
        "mutations",
        "summary",
    },
    "search_ablation_results.json": {
        "schema_version",
        "benchmarks",
        "strategies",
        "summary",
    },
    "reproducibility_summary.json": {
        "schema_version",
        "benchmarks",
        "reproducibility_score",
        "summary",
    },
    "acceptance_results.json": {
        "schema_version",
        "benchmarks",
        "summary",
    },
}


@dataclass(frozen=True)
class VerificationBenchmark:
    benchmark_dir: Path
    metadata: dict[str, Any]
    split_frames: dict[str, pd.DataFrame]
    file_entries: list[dict[str, str]]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_slug(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in value
    )


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _format_declared_issue_notes(metadata: dict[str, Any]) -> str | None:
    issues = metadata.get("known_or_suspected_issues", [])
    notes: list[str] = []
    for issue in issues:
        detector = str(issue.get("detector", "unknown"))
        status = str(issue.get("status", "unspecified"))
        description = str(issue.get("description", "")).strip()
        if description:
            notes.append(f"{detector} ({status}): {description}")
        else:
            notes.append(f"{detector} ({status})")
    return "; ".join(notes) if notes else None


def _acceptance_outcome(*, expected_detectors: set[str], passed: bool) -> str:
    if expected_detectors:
        if passed:
            return (
                "Recovered the expected detector pattern with deterministic evidence. "
                "This validates the acceptance contract, not broader benchmark-invalidity claims."
            )
        return "Did not fully recover the expected detector pattern."
    if passed:
        return "Emitted a clean audit without unsupported claims."
    return "Reported unsupported findings on a benchmark expected to be clean."


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    raise ValueError(f"Unsupported verification benchmark file type: {path}")


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
        git_sha = result.stdout.strip()
        if len(git_sha) >= 7:
            return git_sha
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return "0000000"


def _expected_issue_detectors(metadata: dict[str, Any]) -> list[str]:
    explicit = metadata.get("expected_issue_detectors")
    if explicit:
        return list(dict.fromkeys(str(item) for item in explicit))

    issue_specs = metadata.get("known_or_suspected_issues") or []
    detectors = []
    for item in issue_specs:
        if isinstance(item, dict) and item.get("detector"):
            detectors.append(str(item["detector"]))
    return list(dict.fromkeys(detectors))


def _build_dataset_fingerprint(benchmark: VerificationBenchmark) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "benchmark_id": benchmark.metadata["benchmark_id"],
                "dataset_name": benchmark.metadata["dataset_name"],
                "candidate_key_columns": benchmark.metadata.get("candidate_key_columns", []),
                "text_columns": benchmark.metadata.get("text_columns", []),
                "timestamp_columns": benchmark.metadata.get("timestamp_columns", []),
                "target_column": benchmark.metadata.get("target_column"),
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    for split_name in sorted(benchmark.split_frames):
        digest.update(split_name.encode("utf-8"))
        frame = benchmark.split_frames[split_name]
        stable_frame = frame.copy()
        stable_frame = stable_frame.reindex(sorted(stable_frame.columns), axis=1)
        digest.update(stable_frame.to_csv(index=False).encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


def _temporal_coverage(
    frame: pd.DataFrame, timestamp_columns: list[str]
) -> dict[str, Any] | None:
    for column in timestamp_columns:
        if column not in frame.columns:
            continue
        timestamps = pd.to_datetime(frame[column], errors="coerce", utc=True).dropna()
        if timestamps.empty:
            return {
                "timestamp_column": column,
                "min_timestamp": None,
                "max_timestamp": None,
            }
        return {
            "timestamp_column": column,
            "min_timestamp": timestamps.min().isoformat().replace("+00:00", "Z"),
            "max_timestamp": timestamps.max().isoformat().replace("+00:00", "Z"),
        }
    return None


def _group_key_summary(
    frame: pd.DataFrame, candidate_key_columns: list[str]
) -> dict[str, Any] | None:
    available_columns = [
        column for column in candidate_key_columns if column in frame.columns
    ]
    if not available_columns:
        return None
    return {
        "group_keys": available_columns,
        "unique_group_count": int(frame[available_columns].drop_duplicates().shape[0]),
    }


def default_registry_path() -> Path:
    return _repo_root() / "tests" / "fixtures" / "verification" / "registry.json"


def load_verification_registry(
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(registry_path or default_registry_path()).expanduser().resolve()
    registry = json.loads(path.read_text())
    if registry.get("schema_version") != VERIFICATION_STACK_SCHEMA_VERSION:
        raise ValueError(f"Unsupported verification registry schema: {path}")
    if "benchmarks" not in registry or not isinstance(registry["benchmarks"], list):
        raise ValueError(f"Verification registry is missing a `benchmarks` array: {path}")
    registry["registry_path"] = str(path)
    registry["registry_root"] = str(path.parent)
    return registry


def resolve_registered_benchmark_dirs(
    *,
    role: str | None = None,
    registry_path: str | Path | None = None,
) -> list[Path]:
    registry = load_verification_registry(registry_path)
    registry_root = Path(registry["registry_root"])
    benchmark_dirs = []
    for entry in registry["benchmarks"]:
        benchmark_dir = (registry_root / entry["relative_path"]).resolve()
        metadata = json.loads((benchmark_dir / "benchmark.json").read_text())
        benchmark_role = entry.get("role") or metadata.get("role")
        if role is not None and benchmark_role != role:
            continue
        benchmark_dirs.append(benchmark_dir)
    return benchmark_dirs


def load_verification_benchmark(benchmark_dir: str | Path) -> VerificationBenchmark:
    benchmark_dir = Path(benchmark_dir).expanduser().resolve()
    metadata_path = benchmark_dir / "benchmark.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != VERIFICATION_BENCHMARK_SCHEMA_VERSION:
        raise ValueError(f"Unsupported verification benchmark schema: {metadata_path}")

    file_entries: list[dict[str, str]] = []
    split_frames: dict[str, pd.DataFrame] = {}
    for file_info in metadata.get("files", []):
        relative_path = str(file_info["path"])
        split_name = str(file_info["split"])
        absolute_path = (benchmark_dir / relative_path).resolve()
        if not absolute_path.exists():
            raise FileNotFoundError(f"Verification benchmark file is missing: {absolute_path}")
        split_frames[split_name] = _read_table(absolute_path)
        file_entries.append(
            {
                "split": split_name,
                "path": str(absolute_path),
                "relative_path": relative_path,
            }
        )

    metadata.setdefault("candidate_key_columns", [])
    metadata.setdefault("text_columns", [])
    metadata.setdefault("timestamp_columns", [])
    metadata.setdefault("expected_issue_detectors", _expected_issue_detectors(metadata))
    return VerificationBenchmark(
        benchmark_dir=benchmark_dir,
        metadata=metadata,
        split_frames=split_frames,
        file_entries=file_entries,
    )


def _infer_case_metadata(case_name: str, split_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    shared_columns = set.intersection(
        *(set(frame.columns) for frame in split_frames.values())
    )
    preferred_key_columns = [
        column
        for column in ("group_id", "customer_id", "user_id", "account_id", "uuid")
        if column in shared_columns
    ]
    if preferred_key_columns:
        candidate_key_columns = preferred_key_columns
    else:
        candidate_key_columns = sorted(
            column
            for column in shared_columns
            if column == "id"
            or (
                column.endswith("_id")
                and column not in {"record_id", "row_id", "ticket_id"}
            )
        )
    text_columns = sorted(
        column
        for column in shared_columns
        if column in {"text", "prompt", "response", "content"}
    )
    timestamp_columns = sorted(
        column
        for column in shared_columns
        if "time" in column.lower() or "date" in column.lower()
    )
    target_column = "label" if "label" in shared_columns else None
    return {
        "schema_version": VERIFICATION_BENCHMARK_SCHEMA_VERSION,
        "benchmark_id": case_name,
        "benchmark_name": case_name.replace("_", " ").title(),
        "dataset_name": case_name,
        "description": f"Canary case {case_name}",
        "role": "canary",
        "candidate_key_columns": candidate_key_columns,
        "text_columns": text_columns,
        "timestamp_columns": timestamp_columns,
        "target_column": target_column,
        "expected_issue_detectors": list(CANARY_REQUIRED_DETECTORS[case_name]),
    }


def _benchmark_from_canary_case(case: dict[str, Any]) -> VerificationBenchmark:
    split_frames = {
        file_info["split"]: pd.read_csv(file_info["path"]) for file_info in case["files"]
    }
    first_file = Path(case["files"][0]["path"])
    file_entries = [
        {
            "split": file_info["split"],
            "path": file_info["path"],
            "relative_path": Path(file_info["path"]).name,
        }
        for file_info in case["files"]
    ]
    metadata = _infer_case_metadata(case["name"], split_frames)
    return VerificationBenchmark(
        benchmark_dir=first_file.parent,
        metadata=metadata,
        split_frames=split_frames,
        file_entries=file_entries,
    )


def _combine_findings(findings_frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in findings_frames if not frame.empty]
    if not non_empty:
        return empty_findings_dataframe()
    combined = pd.concat(non_empty, ignore_index=True)
    validate_findings_columns(combined.columns)
    return combined.sort_values(["detector_name", "finding_id"]).reset_index(drop=True)


def _detector_status_entry(
    detector_name: str, finding_count: int, status: str = "completed"
) -> dict[str, Any]:
    return {
        "name": detector_name,
        "version": DETECTOR_VERSIONS[detector_name],
        "status": status,
        "finding_count": int(finding_count),
    }


def _run_single_detector(
    detector_name: str,
    benchmark: VerificationBenchmark,
    provenance: dict[str, Any],
) -> pd.DataFrame:
    split_frames = benchmark.split_frames
    metadata = benchmark.metadata
    if detector_name == "exact_duplicate":
        return detect_exact_duplicates(
            split_frames,
            compare_columns=metadata.get("exact_duplicate_columns"),
            provenance=provenance,
        )
    if detector_name == "near_duplicate":
        return detect_near_duplicates(
            split_frames,
            text_columns=metadata.get("text_columns"),
            provenance=provenance,
            similarity_threshold=int(
                metadata.get("near_duplicate_similarity_threshold", 90)
            ),
        )
    if detector_name == "group_overlap":
        group_columns = list(metadata.get("candidate_key_columns") or [])
        if not group_columns:
            raise ValueError("group overlap requires candidate_key_columns")
        return detect_group_overlap(
            split_frames,
            group_columns=group_columns,
            provenance=provenance,
        )
    if detector_name == "temporal_leakage":
        timestamp_columns = list(metadata.get("timestamp_columns") or [])
        if not timestamp_columns:
            raise ValueError("temporal leakage requires timestamp_columns")
        return detect_temporal_leakage(
            split_frames,
            timestamp_column=timestamp_columns[0],
            provenance=provenance,
        )
    if detector_name == "preprocessing_leakage":
        return detect_preprocessing_leakage(
            split_frames,
            feature_columns=metadata.get("feature_columns"),
            provenance=provenance,
        )
    if detector_name == "suspicious_feature_leakage":
        target_column = metadata.get("target_column")
        if not target_column:
            raise ValueError("suspicious feature leakage requires target_column")
        return detect_suspicious_feature_leakage(
            split_frames,
            target_column=str(target_column),
            provenance=provenance,
            feature_columns=metadata.get("feature_columns"),
            match_threshold=float(metadata.get("label_match_threshold", 0.95)),
        )
    raise ValueError(f"Unsupported detector: {detector_name}")


def run_detector_profile(
    benchmark: VerificationBenchmark,
    *,
    strategy: str,
    run_id: str,
    seed: int | None = 7,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any], list[str]]:
    if strategy not in STRATEGY_DETECTOR_PROFILES:
        raise ValueError(f"Unsupported verification strategy: {strategy}")

    selected_detectors = STRATEGY_DETECTOR_PROFILES[strategy]
    provenance = build_provenance_block(
        git_sha=_git_sha(),
        dataset_fingerprint=_build_dataset_fingerprint(benchmark),
        seed=seed,
        run_id=run_id,
        detector_versions={
            detector_name: DETECTOR_VERSIONS[detector_name]
            for detector_name in selected_detectors
        },
    )

    detector_runs = []
    findings_frames = []
    skipped_notes: list[str] = []
    for detector_name in selected_detectors:
        try:
            findings = _run_single_detector(detector_name, benchmark, provenance)
            detector_runs.append(_detector_status_entry(detector_name, len(findings)))
            findings_frames.append(findings)
        except ValueError as exc:
            detector_runs.append(
                _detector_status_entry(detector_name, finding_count=0, status="skipped")
            )
            skipped_notes.append(f"{detector_name}: {exc}")

    return _combine_findings(findings_frames), detector_runs, provenance, skipped_notes


def _open_findings_count(findings: pd.DataFrame) -> int:
    if findings.empty:
        return 0
    statuses = findings["remediation_status"].fillna("").astype(str).str.lower()
    return int(statuses.isin(OPEN_FINDING_STATUSES).sum())


def _findings_summary(findings: pd.DataFrame) -> dict[str, Any]:
    if findings.empty:
        return {
            "total_findings": 0,
            "open_findings": 0,
            "by_severity": {},
            "by_detector": {},
        }
    return {
        "total_findings": int(len(findings)),
        "open_findings": _open_findings_count(findings),
        "by_severity": findings["severity"].value_counts().sort_index().to_dict(),
        "by_detector": findings["detector_name"].value_counts().sort_index().to_dict(),
    }


def _remediation_direction(detectors: list[str]) -> str:
    unique_detectors = sorted(set(detectors))
    if not unique_detectors:
        return (
            "No leakage indicators were detected; preserve the current split policy and "
            "keep the benchmark under routine regression checks."
        )
    return " ".join(
        DETECTOR_REMEDIATION_GUIDANCE[detector_name]
        for detector_name in unique_detectors
        if detector_name in DETECTOR_REMEDIATION_GUIDANCE
    )


def _audit_score(
    expected_detectors: list[str],
    observed_detectors: list[str],
) -> tuple[float, str]:
    expected = set(expected_detectors)
    observed = set(observed_detectors)
    if not expected:
        unexpected = len(observed)
        score = max(0.0, 100.0 - (unexpected * 20.0))
        rating = "clean" if unexpected == 0 else "warning"
        return round(score, 1), rating

    recovered = len(expected & observed)
    unexpected = len(observed - expected)
    score = (recovered / len(expected)) * 100.0 - (unexpected * 5.0)
    score = max(0.0, min(100.0, score))
    if recovered == len(expected) and unexpected == 0:
        rating = "pass"
    elif recovered > 0:
        rating = "warning"
    else:
        rating = "blocked"
    return round(score, 1), rating


def _confidence_block(
    *,
    strategy: str,
    expected_detectors: list[str],
    observed_detectors: list[str],
    skipped_notes: list[str],
) -> dict[str, Any]:
    expected = set(expected_detectors)
    observed = set(observed_detectors)
    if not expected:
        overall = 0.97 if not observed else 0.65
        notes = "Clean benchmark should remain free of supported leakage signals."
    else:
        recovered = len(expected & observed)
        if recovered == len(expected):
            overall = 0.97
        elif recovered > 0:
            overall = 0.79
        else:
            overall = 0.55
        notes = (
            f"Strategy `{strategy}` recovered {recovered} of {len(expected)} expected "
            "issue detectors."
        )
    if skipped_notes:
        notes += " Skipped detectors: " + "; ".join(skipped_notes) + "."
    return {
        "overall": overall,
        "evidence_coverage": 1.0,
        "notes": notes,
    }


def _build_metrics_before_after(
    metadata: dict[str, Any], provenance: dict[str, Any]
) -> dict[str, Any] | None:
    baseline_metrics = metadata.get("baseline_metrics")
    remediated_metrics = metadata.get("remediated_metrics")
    if not baseline_metrics or not remediated_metrics:
        return None
    if len(baseline_metrics) != len(remediated_metrics):
        raise ValueError("baseline_metrics and remediated_metrics must have the same length")

    deltas = []
    for baseline_metric, remediated_metric in zip(baseline_metrics, remediated_metrics):
        deltas.append(
            {
                "metric_name": baseline_metric["metric_name"],
                "split": baseline_metric["split"],
                "baseline_value": baseline_metric["value"],
                "remediated_value": remediated_metric["value"],
                "delta": round(
                    remediated_metric["value"] - baseline_metric["value"],
                    6,
                ),
            }
        )

    metrics = {
        "baseline_metrics": baseline_metrics,
        "remediated_metrics": remediated_metrics,
        "deltas": deltas,
        "split_information": {
            "evaluated_splits": sorted(
                {
                    metric["split"]
                    for metric in list(baseline_metrics) + list(remediated_metrics)
                }
            ),
            "split_manifest_path": "split_manifest.json",
            "notes": metadata.get(
                "metrics_notes",
                "Metrics compare the benchmark before and after the proposed remediation.",
            ),
        },
        "provenance": provenance,
    }
    validate_metrics_before_after(metrics)
    return metrics


def _build_split_manifest(
    benchmark: VerificationBenchmark, provenance: dict[str, Any]
) -> dict[str, Any]:
    metadata = benchmark.metadata
    split_manifest = {
        "dataset_name": metadata["dataset_name"],
        "file_paths_used": [
            file_entry["relative_path"] for file_entry in benchmark.file_entries
        ],
        "splits": [],
        "provenance": provenance,
    }
    candidate_key_columns = list(metadata.get("candidate_key_columns") or [])
    timestamp_columns = list(metadata.get("timestamp_columns") or [])

    for file_entry in benchmark.file_entries:
        split_name = file_entry["split"]
        frame = benchmark.split_frames[split_name]
        split_entry = {
            "name": split_name,
            "record_count": int(frame.shape[0]),
            "file_paths": [file_entry["relative_path"]],
        }
        group_key_summary = _group_key_summary(frame, candidate_key_columns)
        if group_key_summary is not None:
            split_entry["group_key_summary"] = group_key_summary
        temporal_coverage = _temporal_coverage(frame, timestamp_columns)
        if temporal_coverage is not None:
            split_entry["temporal_coverage"] = temporal_coverage
        split_manifest["splits"].append(split_entry)

    validate_split_manifest(split_manifest)
    return split_manifest


def _render_dataset_card(
    benchmark: VerificationBenchmark,
    *,
    dataset_fingerprint: str,
    strategy: str,
    findings_summary: dict[str, Any],
) -> str:
    lines = [
        "# Dataset Card",
        "",
        f"- Benchmark ID: {benchmark.metadata['benchmark_id']}",
        f"- Benchmark name: {benchmark.metadata['benchmark_name']}",
        f"- Dataset name: {benchmark.metadata['dataset_name']}",
        f"- Verification role: {benchmark.metadata.get('role', 'unknown')}",
        f"- Strategy: {strategy}",
        f"- Dataset fingerprint: {dataset_fingerprint}",
        f"- Expected issue detectors: {', '.join(_expected_issue_detectors(benchmark.metadata)) or 'none'}",
        f"- Candidate key columns: {', '.join(benchmark.metadata.get('candidate_key_columns', [])) or 'none'}",
        f"- Text columns: {', '.join(benchmark.metadata.get('text_columns', [])) or 'none'}",
        f"- Timestamp columns: {', '.join(benchmark.metadata.get('timestamp_columns', [])) or 'none'}",
        f"- Target column: {benchmark.metadata.get('target_column') or 'none'}",
        f"- Total findings: {findings_summary['total_findings']}",
        "",
        "## File Inventory",
    ]
    for file_entry in benchmark.file_entries:
        frame = benchmark.split_frames[file_entry["split"]]
        lines.append(
            f"- {file_entry['relative_path']} ({file_entry['split']}): "
            f"{int(frame.shape[0])} row(s), columns={', '.join(frame.columns)}"
        )
    description = benchmark.metadata.get("description")
    if description:
        lines.extend(["", "## Description", description])
    return "\n".join(lines) + "\n"


def _materialize_evidence(
    findings: pd.DataFrame,
    *,
    output_dir: Path,
    benchmark: VerificationBenchmark,
    strategy: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if findings.empty:
        return findings, []

    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    updated_findings = findings.copy()
    evidence_references = []
    for index, row in updated_findings.iterrows():
        evidence_name = (
            f"{row['detector_name']}_{_safe_slug(str(row['finding_id']))}.json"
        )
        relative_path = Path("evidence") / evidence_name
        evidence_payload = {
            "schema_version": VERIFICATION_STACK_SCHEMA_VERSION,
            "benchmark_id": benchmark.metadata["benchmark_id"],
            "strategy": strategy,
            "finding_id": row["finding_id"],
            "detector_name": row["detector_name"],
            "severity": row["severity"],
            "confidence": float(row["confidence"]),
            "source_descriptor": row["evidence_pointer"],
        }
        _json_dump(output_dir / relative_path, evidence_payload)
        updated_findings.at[index, "evidence_pointer"] = relative_path.as_posix()
        evidence_references.append(
            {
                "evidence_id": f"evidence-{index + 1:03d}",
                "path": relative_path.as_posix(),
                "kind": "json",
                "description": (
                    f"{row['detector_name']} evidence for "
                    f"{benchmark.metadata['benchmark_id']} under {strategy}"
                ),
            }
        )
    return updated_findings, evidence_references


def materialize_verification_audit_bundle(
    benchmark: VerificationBenchmark,
    *,
    output_dir: str | Path,
    strategy: str,
    run_id: str | None = None,
    seed: int | None = 7,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_run_id = run_id or f"{benchmark.metadata['benchmark_id']}-{strategy}"
    findings, detectors_run, provenance, skipped_notes = run_detector_profile(
        benchmark,
        strategy=strategy,
        run_id=effective_run_id,
        seed=seed,
    )
    findings, evidence_references = _materialize_evidence(
        findings,
        output_dir=output_dir,
        benchmark=benchmark,
        strategy=strategy,
    )

    findings_path = output_dir / "findings.csv"
    findings.to_csv(findings_path, index=False)

    split_manifest = _build_split_manifest(benchmark, provenance)
    _json_dump(output_dir / "split_manifest.json", split_manifest)

    findings_summary = _findings_summary(findings)
    dataset_card_text = _render_dataset_card(
        benchmark,
        dataset_fingerprint=provenance["dataset_fingerprint"],
        strategy=strategy,
        findings_summary=findings_summary,
    )
    (output_dir / "dataset_card.md").write_text(dataset_card_text)

    metrics_before_after = _build_metrics_before_after(benchmark.metadata, provenance)
    metrics_before_after_path: Path | None = None
    if metrics_before_after is not None:
        metrics_before_after_path = output_dir / "metrics_before_after.json"
        _json_dump(metrics_before_after_path, metrics_before_after)

    observed_detectors = (
        sorted(set(findings["detector_name"])) if not findings.empty else []
    )
    expected_detectors = _expected_issue_detectors(benchmark.metadata)
    audit_score_value, audit_rating = _audit_score(
        expected_detectors, observed_detectors
    )

    audit_results = {
        "run_metadata": {
            "run_id": effective_run_id,
            "mode": "audit",
            "seed": seed,
            "status": "completed",
        },
        "benchmark_summary": {
            "benchmark_name": benchmark.metadata["benchmark_name"],
            "dataset_name": benchmark.metadata["dataset_name"],
            "record_count": int(
                sum(frame.shape[0] for frame in benchmark.split_frames.values())
            ),
            "split_names": list(benchmark.split_frames.keys()),
        },
        "detectors_run": detectors_run,
        "findings_summary": findings_summary,
        "confidence": _confidence_block(
            strategy=strategy,
            expected_detectors=expected_detectors,
            observed_detectors=observed_detectors,
            skipped_notes=skipped_notes,
        ),
        "audit_score": {
            "value": audit_score_value,
            "max_value": 100.0,
            "rating": audit_rating,
        },
        "evidence_references": evidence_references,
        "provenance": provenance,
    }
    _json_dump(output_dir / "audit_results.json", audit_results)

    report_path = generate_audit_report(
        audit_results_path=output_dir / "audit_results.json",
        split_manifest_path=output_dir / "split_manifest.json",
        findings_path=findings_path,
        metrics_before_after_path=metrics_before_after_path,
        output_path=output_dir / "audit_report.md",
    )

    load_validated_audit_bundle(output_dir)

    missing_detectors = sorted(set(expected_detectors) - set(observed_detectors))
    unexpected_detectors = sorted(set(observed_detectors) - set(expected_detectors))

    return {
        "benchmark_id": benchmark.metadata["benchmark_id"],
        "benchmark_name": benchmark.metadata["benchmark_name"],
        "strategy": strategy,
        "artifact_dir": str(output_dir),
        "audit_report_path": str(report_path),
        "expected_detectors": expected_detectors,
        "observed_detectors": observed_detectors,
        "missing_detectors": missing_detectors,
        "unexpected_detectors": unexpected_detectors,
        "evidence_reference_count": len(evidence_references),
        "remediation_direction": _remediation_direction(
            observed_detectors or expected_detectors
        ),
        "audit_score": audit_score_value,
    }


def run_canary_suite_verification(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    manifest = write_canary_suite(output_dir)
    manifest = load_canary_suite(output_dir)

    case_results = []
    for case in manifest["cases"]:
        benchmark = _benchmark_from_canary_case(case)
        findings, detector_runs, _provenance, skipped_notes = run_detector_profile(
            benchmark,
            strategy="full_tree_search",
            run_id=f"canary-{case['name']}",
            seed=0,
        )
        observed_detectors = (
            sorted(set(findings["detector_name"])) if not findings.empty else []
        )
        required_detectors = list(CANARY_REQUIRED_DETECTORS[case["name"]])
        if required_detectors:
            passed = set(required_detectors).issubset(observed_detectors)
        else:
            passed = observed_detectors == []
        case_results.append(
            {
                "case_name": case["name"],
                "required_detectors": required_detectors,
                "observed_detectors": observed_detectors,
                "passed": passed,
                "detectors_run": detector_runs,
                "skipped_notes": skipped_notes,
            }
        )

    result = {
        "schema_version": VERIFICATION_STACK_SCHEMA_VERSION,
        "suite_dir": manifest["suite_dir"],
        "manifest_path": manifest["manifest_path"],
        "cases": case_results,
        "summary": {
            "total_cases": len(case_results),
            "passed_cases": sum(1 for case in case_results if case["passed"]),
            "failed_cases": sum(1 for case in case_results if not case["passed"]),
            "passed": all(case["passed"] for case in case_results),
        },
    }
    _json_dump(output_dir / "canary_results.json", result)
    _write_markdown(
        output_dir / "canary_results.md",
        [
            "# Canary Results",
            "",
            f"- Suite dir: `{manifest['suite_dir']}`",
            f"- Passed: `{result['summary']['passed']}`",
            "",
            "## Cases",
            *[
                (
                    f"- `{case['case_name']}`: passed={case['passed']}, "
                    f"required={case['required_detectors']}, observed={case['observed_detectors']}"
                )
                for case in case_results
            ],
        ],
    )
    return result


def _copy_split_frames(split_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        split_name: frame.copy(deep=True) for split_name, frame in split_frames.items()
    }


def _mutation_exact_duplicate(base: VerificationBenchmark) -> VerificationBenchmark:
    metadata = deepcopy(base.metadata)
    split_frames = _copy_split_frames(base.split_frames)
    duplicate_row = split_frames["train"].iloc[[0]].copy()
    split_frames["test"] = pd.concat(
        [split_frames["test"], duplicate_row],
        ignore_index=True,
    )
    metadata["benchmark_id"] = f"{base.metadata['benchmark_id']}_exact_duplicate"
    metadata["benchmark_name"] = "Exact Duplicate Mutation"
    metadata["expected_issue_detectors"] = ["exact_duplicate"]
    return VerificationBenchmark(base.benchmark_dir, metadata, split_frames, base.file_entries)


def _mutation_near_duplicate(base: VerificationBenchmark) -> VerificationBenchmark:
    metadata = deepcopy(base.metadata)
    split_frames = _copy_split_frames(base.split_frames)
    reference_text = str(split_frames["train"].iloc[0]["text"])
    split_frames["test"].loc[split_frames["test"].index[0], "text"] = (
        reference_text + " today"
    )
    metadata["benchmark_id"] = f"{base.metadata['benchmark_id']}_near_duplicate"
    metadata["benchmark_name"] = "Near Duplicate Mutation"
    metadata["expected_issue_detectors"] = ["near_duplicate"]
    return VerificationBenchmark(base.benchmark_dir, metadata, split_frames, base.file_entries)


def _mutation_group_overlap(base: VerificationBenchmark) -> VerificationBenchmark:
    metadata = deepcopy(base.metadata)
    split_frames = _copy_split_frames(base.split_frames)
    group_column = metadata["candidate_key_columns"][0]
    split_frames["test"].loc[split_frames["test"].index[0], group_column] = (
        split_frames["train"].iloc[0][group_column]
    )
    metadata["benchmark_id"] = f"{base.metadata['benchmark_id']}_group_overlap"
    metadata["benchmark_name"] = "Group Overlap Mutation"
    metadata["expected_issue_detectors"] = ["group_overlap"]
    return VerificationBenchmark(base.benchmark_dir, metadata, split_frames, base.file_entries)


def _mutation_temporal_leakage(base: VerificationBenchmark) -> VerificationBenchmark:
    metadata = deepcopy(base.metadata)
    split_frames = _copy_split_frames(base.split_frames)
    timestamp_column = metadata["timestamp_columns"][0]
    split_frames["train"][timestamp_column] = [
        "2024-09-01",
        "2024-09-02",
        "2024-09-03",
    ][: len(split_frames["train"])]
    split_frames["test"][timestamp_column] = [
        "2024-08-01",
        "2024-08-02",
        "2024-08-03",
    ][: len(split_frames["test"])]
    metadata["benchmark_id"] = f"{base.metadata['benchmark_id']}_temporal_leakage"
    metadata["benchmark_name"] = "Temporal Leakage Mutation"
    metadata["expected_issue_detectors"] = ["temporal_leakage"]
    return VerificationBenchmark(base.benchmark_dir, metadata, split_frames, base.file_entries)


def _mutation_preprocessing_leakage(base: VerificationBenchmark) -> VerificationBenchmark:
    metadata = deepcopy(base.metadata)
    split_frames = _copy_split_frames(base.split_frames)
    split_frames["train"]["global_scaled_feature"] = [0.11, 0.29, 0.44][
        : len(split_frames["train"])
    ]
    split_frames["test"]["global_scaled_feature"] = [0.18, 0.36, 0.52][
        : len(split_frames["test"])
    ]
    metadata["benchmark_id"] = f"{base.metadata['benchmark_id']}_preprocessing_leakage"
    metadata["benchmark_name"] = "Preprocessing Leakage Mutation"
    metadata["expected_issue_detectors"] = ["preprocessing_leakage"]
    return VerificationBenchmark(base.benchmark_dir, metadata, split_frames, base.file_entries)


def _mutation_suspicious_feature(base: VerificationBenchmark) -> VerificationBenchmark:
    metadata = deepcopy(base.metadata)
    split_frames = _copy_split_frames(base.split_frames)
    target_column = metadata["target_column"]
    split_frames["train"]["label_echo"] = split_frames["train"][target_column]
    split_frames["test"]["label_echo"] = split_frames["test"][target_column]
    metadata["benchmark_id"] = (
        f"{base.metadata['benchmark_id']}_suspicious_feature_leakage"
    )
    metadata["benchmark_name"] = "Suspicious Feature Leakage Mutation"
    metadata["expected_issue_detectors"] = ["suspicious_feature_leakage"]
    return VerificationBenchmark(base.benchmark_dir, metadata, split_frames, base.file_entries)


MUTATION_BUILDERS = {
    "exact_duplicate": {
        "builder": _mutation_exact_duplicate,
        "allowed_extra_detectors": ["group_overlap", "temporal_leakage"],
    },
    "near_duplicate": {
        "builder": _mutation_near_duplicate,
        "allowed_extra_detectors": [],
    },
    "group_overlap": {
        "builder": _mutation_group_overlap,
        "allowed_extra_detectors": [],
    },
    "temporal_leakage": {
        "builder": _mutation_temporal_leakage,
        "allowed_extra_detectors": [],
    },
    "preprocessing_leakage": {
        "builder": _mutation_preprocessing_leakage,
        "allowed_extra_detectors": [],
    },
    "suspicious_feature_leakage": {
        "builder": _mutation_suspicious_feature,
        "allowed_extra_detectors": [],
    },
}


def run_mutation_test_harness(
    benchmark_dir: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    base_benchmark = load_verification_benchmark(benchmark_dir)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_findings, _clean_runs, _provenance, _notes = run_detector_profile(
        base_benchmark,
        strategy="full_tree_search",
        run_id=f"{base_benchmark.metadata['benchmark_id']}-clean-control",
        seed=0,
    )
    clean_observed_detectors = (
        sorted(set(clean_findings["detector_name"])) if not clean_findings.empty else []
    )

    mutation_results = []
    recovered_mutations = 0
    unexpected_findings = 0
    for mutation_name, mutation_spec in MUTATION_BUILDERS.items():
        mutated_benchmark = mutation_spec["builder"](base_benchmark)
        findings, _detector_runs, _provenance, _notes = run_detector_profile(
            mutated_benchmark,
            strategy="full_tree_search",
            run_id=f"{mutated_benchmark.metadata['benchmark_id']}-mutation",
            seed=0,
        )
        observed_detectors = (
            sorted(set(findings["detector_name"])) if not findings.empty else []
        )
        required_detectors = mutated_benchmark.metadata["expected_issue_detectors"]
        allowed_extra_detectors = list(mutation_spec["allowed_extra_detectors"])
        missing_detectors = sorted(set(required_detectors) - set(observed_detectors))
        unexpected_detectors = sorted(
            set(observed_detectors)
            - set(required_detectors)
            - set(allowed_extra_detectors)
        )
        recovered = missing_detectors == []
        if recovered:
            recovered_mutations += 1
        unexpected_findings += len(unexpected_detectors)
        mutation_results.append(
            {
                "mutation_name": mutation_name,
                "required_detectors": required_detectors,
                "allowed_extra_detectors": allowed_extra_detectors,
                "observed_detectors": observed_detectors,
                "missing_detectors": missing_detectors,
                "unexpected_detectors": unexpected_detectors,
                "recovered": recovered,
            }
        )

    summary = {
        "total_mutations": len(mutation_results),
        "recovered_mutations": recovered_mutations,
        "overall_recall": recovered_mutations / len(mutation_results),
        "clean_false_positive_count": len(clean_observed_detectors),
        "clean_false_positive_rate": float(bool(clean_observed_detectors)),
        "unexpected_findings_per_mutation": unexpected_findings / len(mutation_results),
        "passed": (
            recovered_mutations == len(mutation_results)
            and clean_observed_detectors == []
            and unexpected_findings == 0
        ),
    }
    result = {
        "schema_version": VERIFICATION_STACK_SCHEMA_VERSION,
        "benchmark_id": base_benchmark.metadata["benchmark_id"],
        "clean_control_observed_detectors": clean_observed_detectors,
        "mutations": mutation_results,
        "summary": summary,
    }
    _json_dump(output_dir / "mutation_test_results.json", result)
    _write_markdown(
        output_dir / "mutation_test_results.md",
        [
            "# Mutation Test Results",
            "",
            f"- Benchmark ID: `{base_benchmark.metadata['benchmark_id']}`",
            f"- Overall recall: `{summary['overall_recall']:.2f}`",
            f"- Clean false positives: `{summary['clean_false_positive_count']}`",
            f"- Passed: `{summary['passed']}`",
            "",
            "## Mutation Cases",
            *[
                (
                    f"- `{item['mutation_name']}`: recovered={item['recovered']}, "
                    f"observed={item['observed_detectors']}, "
                    f"unexpected={item['unexpected_detectors']}"
                )
                for item in mutation_results
            ],
        ],
    )
    return result


def run_search_ablation(
    benchmark_dirs: list[Path], output_dir: str | Path
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_benchmarks = [load_verification_benchmark(benchmark_dir) for benchmark_dir in benchmark_dirs]
    strategies = {}
    benchmark_names = {
        benchmark.metadata["benchmark_id"]: benchmark.metadata["benchmark_name"]
        for benchmark in loaded_benchmarks
    }
    bundle_dirs: list[str] = []
    strategy_runs: dict[str, dict[str, dict[str, Any]]] = {}

    for strategy_name in STRATEGY_DETECTOR_PROFILES:
        total_expected = 0
        total_recovered = 0
        total_unexpected = 0
        benchmark_results = []
        strategy_runs[strategy_name] = {}
        for benchmark in loaded_benchmarks:
            artifact_dir = (
                output_dir / benchmark.metadata["benchmark_id"] / strategy_name
            )
            run = materialize_verification_audit_bundle(
                benchmark,
                output_dir=artifact_dir,
                strategy=strategy_name,
                run_id=f"{benchmark.metadata['benchmark_id']}-{strategy_name}",
                seed=0,
            )
            strategy_runs[strategy_name][benchmark.metadata["benchmark_id"]] = run
            bundle_dirs.append(run["artifact_dir"])
            expected_count = len(run["expected_detectors"])
            recovered_count = expected_count - len(run["missing_detectors"])
            total_expected += expected_count
            total_recovered += recovered_count
            total_unexpected += len(run["unexpected_detectors"])
            benchmark_results.append(
                {
                    "benchmark_id": run["benchmark_id"],
                    "benchmark_name": run["benchmark_name"],
                    "observed_detectors": run["observed_detectors"],
                    "missing_detectors": run["missing_detectors"],
                    "unexpected_detectors": run["unexpected_detectors"],
                    "audit_score": run["audit_score"],
                }
            )

        recall = total_recovered / total_expected if total_expected else 1.0
        strategies[strategy_name] = {
            "benchmark_results": benchmark_results,
            "overall_recall": recall,
            "unexpected_detector_count": total_unexpected,
        }

    ranked_strategies = sorted(
        strategies,
        key=lambda name: (
            -strategies[name]["overall_recall"],
            strategies[name]["unexpected_detector_count"],
            list(STRATEGY_DETECTOR_PROFILES).index(name),
        ),
    )
    summary = {
        "best_strategy": ranked_strategies[0],
        "full_tree_search_adds_value": (
            strategies["full_tree_search"]["overall_recall"]
            > strategies["one_shot_agent"]["overall_recall"]
            >= strategies["detector_only"]["overall_recall"]
        ),
        "passed": (
            ranked_strategies[0] == "full_tree_search"
            and strategies["full_tree_search"]["overall_recall"]
            > strategies["detector_only"]["overall_recall"]
        ),
    }
    result = {
        "schema_version": VERIFICATION_STACK_SCHEMA_VERSION,
        "benchmarks": [
            {
                "benchmark_id": benchmark_id,
                "benchmark_name": benchmark_name,
            }
            for benchmark_id, benchmark_name in benchmark_names.items()
        ],
        "strategies": strategies,
        "summary": summary,
        "bundle_dirs": bundle_dirs,
        "strategy_runs": strategy_runs,
    }
    _json_dump(output_dir / "search_ablation_results.json", result)
    _write_markdown(
        output_dir / "search_ablation_results.md",
        [
            "# Search Ablation Results",
            "",
            f"- Best strategy: `{summary['best_strategy']}`",
            f"- Full tree search adds value: `{summary['full_tree_search_adds_value']}`",
            f"- Passed: `{summary['passed']}`",
            "",
            "## Strategy Recall",
            *[
                (
                    f"- `{strategy_name}`: "
                    f"recall={strategy_payload['overall_recall']:.2f}, "
                    f"unexpected={strategy_payload['unexpected_detector_count']}"
                )
                for strategy_name, strategy_payload in strategies.items()
            ],
        ],
    )
    return result


def run_reproducibility_test(
    benchmark_dirs: list[Path],
    output_dir: str | Path,
    *,
    repeats: int = 3,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_summaries = []
    bundle_dirs: list[str] = []
    benchmark_scores = []
    for benchmark_dir in benchmark_dirs:
        benchmark = load_verification_benchmark(benchmark_dir)
        runs = []
        for index in range(1, repeats + 1):
            artifact_dir = (
                output_dir / benchmark.metadata["benchmark_id"] / f"run_{index}"
            )
            run = materialize_verification_audit_bundle(
                benchmark,
                output_dir=artifact_dir,
                strategy="full_tree_search",
                run_id=f"{benchmark.metadata['benchmark_id']}-repro-run-{index}",
                seed=0,
            )
            bundle_dirs.append(run["artifact_dir"])
            runs.append(run)

        detector_signatures = {
            tuple(run["observed_detectors"]) for run in runs
        }
        remediation_signatures = {
            run["remediation_direction"] for run in runs
        }
        score_signatures = {run["audit_score"] for run in runs}
        materially_consistent = (
            len(detector_signatures) == 1
            and len(remediation_signatures) == 1
            and len(score_signatures) == 1
        )
        reproducibility_score = 1.0 if materially_consistent else 0.0
        benchmark_scores.append(reproducibility_score)
        for run in runs:
            _json_dump(
                Path(run["artifact_dir"]) / "reproducibility.json",
                {
                    "schema_version": VERIFICATION_STACK_SCHEMA_VERSION,
                    "benchmark_id": run["benchmark_id"],
                    "reproducibility_score": reproducibility_score,
                    "materially_consistent": materially_consistent,
                    "runs_compared": repeats,
                },
            )

        benchmark_summaries.append(
            {
                "benchmark_id": benchmark.metadata["benchmark_id"],
                "benchmark_name": benchmark.metadata["benchmark_name"],
                "runs": runs,
                "materially_consistent": materially_consistent,
                "reproducibility_score": reproducibility_score,
            }
        )

    overall_score = (
        sum(benchmark_scores) / len(benchmark_scores) if benchmark_scores else 1.0
    )
    summary = {
        "runs_per_benchmark": repeats,
        "passed_benchmarks": sum(
            1 for benchmark in benchmark_summaries if benchmark["materially_consistent"]
        ),
        "failed_benchmarks": sum(
            1 for benchmark in benchmark_summaries if not benchmark["materially_consistent"]
        ),
        "passed": all(
            benchmark["materially_consistent"] for benchmark in benchmark_summaries
        ),
    }
    result = {
        "schema_version": VERIFICATION_STACK_SCHEMA_VERSION,
        "benchmarks": benchmark_summaries,
        "reproducibility_score": overall_score,
        "summary": summary,
        "bundle_dirs": bundle_dirs,
    }
    _json_dump(output_dir / "reproducibility_summary.json", result)
    _write_markdown(
        output_dir / "reproducibility_summary.md",
        [
            "# Reproducibility Summary",
            "",
            f"- Runs per benchmark: `{repeats}`",
            f"- Reproducibility score: `{overall_score:.2f}`",
            f"- Passed: `{summary['passed']}`",
            "",
            "## Benchmarks",
            *[
                (
                    f"- `{benchmark['benchmark_id']}`: "
                    f"materially_consistent={benchmark['materially_consistent']}, "
                    f"score={benchmark['reproducibility_score']:.2f}"
                )
                for benchmark in benchmark_summaries
            ],
        ],
    )
    return result


def run_acceptance_tests(
    *,
    benchmark_dirs: list[Path],
    full_tree_strategy_runs: dict[str, dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_results = []
    for benchmark_dir in benchmark_dirs:
        benchmark = load_verification_benchmark(benchmark_dir)
        run = full_tree_strategy_runs[benchmark.metadata["benchmark_id"]]
        expected_detectors = set(run["expected_detectors"])
        observed_detectors = set(run["observed_detectors"])
        declared_issue_notes = _format_declared_issue_notes(benchmark.metadata)
        if expected_detectors:
            passed = (
                run["missing_detectors"] == [] and run["unexpected_detectors"] == []
            )
        else:
            passed = observed_detectors == set()
        outcome = _acceptance_outcome(
            expected_detectors=expected_detectors,
            passed=passed,
        )

        report_lines = [
            f"# Acceptance Report: {benchmark.metadata['benchmark_name']}",
            "",
            f"- Benchmark ID: `{benchmark.metadata['benchmark_id']}`",
            f"- Audit bundle: `{run['artifact_dir']}`",
            f"- Audit report: `{run['audit_report_path']}`",
            f"- Expected detectors: `{sorted(expected_detectors)}`",
            *(
                [f"- Declared issue notes: {declared_issue_notes}"]
                if declared_issue_notes
                else []
            ),
            f"- Observed detectors: `{sorted(observed_detectors)}`",
            f"- Missing detectors: `{run['missing_detectors']}`",
            f"- Unexpected detectors: `{run['unexpected_detectors']}`",
            f"- Evidence files: `{run['evidence_reference_count']}`",
            f"- Remediation direction: {run['remediation_direction']}",
            f"- Outcome: {outcome}",
        ]
        report_path = output_dir / benchmark.metadata["benchmark_id"] / "acceptance_report.md"
        _write_markdown(report_path, report_lines)

        benchmark_results.append(
            {
                "benchmark_id": benchmark.metadata["benchmark_id"],
                "benchmark_name": benchmark.metadata["benchmark_name"],
                "audit_bundle": run["artifact_dir"],
                "audit_report_path": run["audit_report_path"],
                "expected_detectors": run["expected_detectors"],
                "observed_detectors": run["observed_detectors"],
                "missing_detectors": run["missing_detectors"],
                "unexpected_detectors": run["unexpected_detectors"],
                "acceptance_report_path": str(report_path),
                "passed": passed,
            }
        )

    summary = {
        "total_benchmarks": len(benchmark_results),
        "passed_benchmarks": sum(1 for benchmark in benchmark_results if benchmark["passed"]),
        "failed_benchmarks": sum(1 for benchmark in benchmark_results if not benchmark["passed"]),
        "passed": all(benchmark["passed"] for benchmark in benchmark_results),
    }
    result = {
        "schema_version": VERIFICATION_STACK_SCHEMA_VERSION,
        "benchmarks": benchmark_results,
        "summary": summary,
    }
    _json_dump(output_dir / "acceptance_results.json", result)
    return result


def run_schema_gate(
    *,
    bundle_dirs: list[str | Path],
    summary_artifacts: dict[str, str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validated_bundles = []
    for bundle_dir in sorted({str(Path(path).resolve()) for path in bundle_dirs}):
        bundle = load_validated_audit_bundle(bundle_dir)
        validated_bundles.append(
            {
                "artifact_dir": bundle_dir,
                "run_id": bundle.audit_results["run_metadata"]["run_id"],
                "dataset_name": bundle.audit_results["benchmark_summary"]["dataset_name"],
            }
        )

    validated_summaries = []
    for artifact_name, artifact_path in summary_artifacts.items():
        artifact_path = Path(artifact_path).expanduser().resolve()
        payload = json.loads(artifact_path.read_text())
        missing_keys = sorted(
            SUMMARY_ARTIFACT_KEYS[artifact_name] - set(payload.keys())
        )
        if payload.get("schema_version") != VERIFICATION_STACK_SCHEMA_VERSION:
            raise ValueError(f"Invalid summary artifact schema version: {artifact_path}")
        if missing_keys:
            raise ValueError(
                f"Summary artifact {artifact_path} is missing keys: {missing_keys}"
            )
        validated_summaries.append(
            {
                "artifact_name": artifact_name,
                "artifact_path": str(artifact_path),
            }
        )

    result = {
        "schema_version": VERIFICATION_STACK_SCHEMA_VERSION,
        "validated_bundles": validated_bundles,
        "validated_summary_artifacts": validated_summaries,
        "passed": True,
    }
    _json_dump(output_dir / "schema_gate_results.json", result)
    return result


def _render_verification_stack_summary(result: dict[str, Any]) -> list[str]:
    return [
        "# Verification Stack Summary",
        "",
        f"- Output dir: `{result['output_dir']}`",
        f"- Registry path: `{result['registry_path']}`",
        f"- Overall status: `{result['status']}`",
        "",
        "## Phases",
        f"- Canary suite: `{result['phases']['canary']['summary']['passed']}`",
        f"- Mutation tests: `{result['phases']['mutation']['summary']['passed']}`",
        f"- Search ablation: `{result['phases']['ablation']['summary']['passed']}`",
        f"- Reproducibility: `{result['phases']['reproducibility']['summary']['passed']}`",
        f"- Acceptance: `{result['phases']['acceptance']['summary']['passed']}`",
        f"- Schema gate: `{result['phases']['schema_gate']['passed']}`",
        "",
        "## Artifact Paths",
        *[
            f"- `{name}`: `{path}`"
            for name, path in sorted(result["artifact_paths"].items())
        ],
    ]


def run_verification_stack(
    *,
    output_dir: str | Path,
    registry_path: str | Path | None = None,
    reproducibility_repeats: int = 3,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = load_verification_registry(registry_path)
    acceptance_benchmark_dirs = resolve_registered_benchmark_dirs(
        role="acceptance", registry_path=registry["registry_path"]
    )
    mutation_benchmark_dirs = resolve_registered_benchmark_dirs(
        role="mutation_base", registry_path=registry["registry_path"]
    )
    if len(mutation_benchmark_dirs) != 1:
        raise ValueError("Verification stack expects exactly one mutation_base benchmark")

    canary = run_canary_suite_verification(output_dir / "canary")
    mutation = run_mutation_test_harness(
        mutation_benchmark_dirs[0],
        output_dir / "mutation",
    )
    ablation = run_search_ablation(
        acceptance_benchmark_dirs,
        output_dir / "search_ablation",
    )
    reproducibility = run_reproducibility_test(
        acceptance_benchmark_dirs,
        output_dir / "reproducibility",
        repeats=reproducibility_repeats,
    )
    acceptance = run_acceptance_tests(
        benchmark_dirs=acceptance_benchmark_dirs,
        full_tree_strategy_runs=ablation["strategy_runs"]["full_tree_search"],
        output_dir=output_dir / "acceptance",
    )
    summary_artifacts = {
        "canary_results.json": output_dir / "canary" / "canary_results.json",
        "mutation_test_results.json": output_dir / "mutation" / "mutation_test_results.json",
        "search_ablation_results.json": output_dir
        / "search_ablation"
        / "search_ablation_results.json",
        "reproducibility_summary.json": output_dir
        / "reproducibility"
        / "reproducibility_summary.json",
        "acceptance_results.json": output_dir
        / "acceptance"
        / "acceptance_results.json",
    }
    schema_gate = run_schema_gate(
        bundle_dirs=ablation["bundle_dirs"] + reproducibility["bundle_dirs"],
        summary_artifacts=summary_artifacts,
        output_dir=output_dir / "schema_gate",
    )

    status = "passed"
    if not (
        canary["summary"]["passed"]
        and mutation["summary"]["passed"]
        and ablation["summary"]["passed"]
        and reproducibility["summary"]["passed"]
        and acceptance["summary"]["passed"]
        and schema_gate["passed"]
    ):
        status = "failed"

    result = {
        "schema_version": VERIFICATION_STACK_SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "registry_path": registry["registry_path"],
        "status": status,
        "phases": {
            "canary": canary,
            "mutation": mutation,
            "ablation": ablation,
            "reproducibility": reproducibility,
            "acceptance": acceptance,
            "schema_gate": schema_gate,
        },
        "artifact_paths": {
            "canary_results": str(summary_artifacts["canary_results.json"]),
            "mutation_results": str(summary_artifacts["mutation_test_results.json"]),
            "search_ablation_results": str(summary_artifacts["search_ablation_results.json"]),
            "reproducibility_summary": str(summary_artifacts["reproducibility_summary.json"]),
            "acceptance_results": str(summary_artifacts["acceptance_results.json"]),
            "schema_gate_results": str(
                output_dir / "schema_gate" / "schema_gate_results.json"
            ),
        },
    }
    _json_dump(output_dir / "verification_stack_results.json", result)
    _write_markdown(
        output_dir / "verification_stack_summary.md",
        _render_verification_stack_summary(result),
    )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic Phase 11 verification stack."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="verification_results/latest",
        help="Directory where verification artifacts should be written.",
    )
    parser.add_argument(
        "--registry-path",
        type=str,
        default=None,
        help="Optional path to a custom verification benchmark registry.",
    )
    parser.add_argument(
        "--reproducibility-repeats",
        type=int,
        default=3,
        help="How many repeated full-tree-search runs to compare per benchmark.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    result = run_verification_stack(
        output_dir=args.output_dir,
        registry_path=args.registry_path,
        reproducibility_repeats=args.reproducibility_repeats,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
