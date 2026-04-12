from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone

import jsonschema


AUDIT_SCHEMA_VERSION = "0.1.0"


PROVENANCE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string", "minLength": 1},
        "git_sha": {"type": "string", "minLength": 7},
        "dataset_fingerprint": {"type": "string", "minLength": 1},
        "seed": {"type": ["integer", "null"]},
        "run_id": {"type": "string", "minLength": 1},
        "detector_versions": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": {"type": "string", "minLength": 1},
        },
        "created_at": {"type": "string", "format": "date-time"},
        "updated_at": {"type": "string", "format": "date-time"},
    },
    "required": [
        "schema_version",
        "git_sha",
        "dataset_fingerprint",
        "seed",
        "run_id",
        "detector_versions",
        "created_at",
        "updated_at",
    ],
}


EVIDENCE_REFERENCE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "evidence_id": {"type": "string", "minLength": 1},
        "path": {"type": "string", "minLength": 1},
        "kind": {"type": "string", "minLength": 1},
        "description": {"type": "string", "minLength": 1},
    },
    "required": ["evidence_id", "path", "kind", "description"],
}


DETECTOR_RUN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "version": {"type": "string", "minLength": 1},
        "status": {"type": "string", "enum": ["completed", "skipped", "failed"]},
        "finding_count": {"type": "integer", "minimum": 0},
    },
    "required": ["name", "version", "status", "finding_count"],
}


METRIC_SUMMARY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "metric_name": {"type": "string", "minLength": 1},
        "split": {"type": "string", "minLength": 1},
        "value": {"type": "number"},
        "higher_is_better": {"type": "boolean"},
    },
    "required": ["metric_name", "split", "value", "higher_is_better"],
}


METRIC_DELTA_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "metric_name": {"type": "string", "minLength": 1},
        "split": {"type": "string", "minLength": 1},
        "baseline_value": {"type": "number"},
        "remediated_value": {"type": "number"},
        "delta": {"type": "number"},
    },
    "required": [
        "metric_name",
        "split",
        "baseline_value",
        "remediated_value",
        "delta",
    ],
}


SPLIT_SUMMARY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "record_count": {"type": "integer", "minimum": 0},
        "file_paths": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "group_key_summary": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "group_keys": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "unique_group_count": {"type": ["integer", "null"], "minimum": 0},
            },
            "required": ["group_keys", "unique_group_count"],
        },
        "temporal_coverage": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "timestamp_column": {"type": "string", "minLength": 1},
                "min_timestamp": {"type": ["string", "null"]},
                "max_timestamp": {"type": ["string", "null"]},
            },
            "required": ["timestamp_column", "min_timestamp", "max_timestamp"],
        },
    },
    "required": ["name", "record_count", "file_paths"],
}


AUDIT_RESULTS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "run_metadata": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "run_id": {"type": "string", "minLength": 1},
                "mode": {"type": "string", "const": "audit"},
                "seed": {"type": ["integer", "null"]},
                "status": {
                    "type": "string",
                    "enum": ["completed", "failed", "blocked"],
                },
            },
            "required": ["run_id", "mode", "seed", "status"],
        },
        "benchmark_summary": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "benchmark_name": {"type": "string", "minLength": 1},
                "dataset_name": {"type": "string", "minLength": 1},
                "record_count": {"type": "integer", "minimum": 0},
                "split_names": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
            },
            "required": [
                "benchmark_name",
                "dataset_name",
                "record_count",
                "split_names",
            ],
        },
        "detectors_run": {
            "type": "array",
            "minItems": 1,
            "items": DETECTOR_RUN_SCHEMA,
        },
        "findings_summary": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "total_findings": {"type": "integer", "minimum": 0},
                "open_findings": {"type": "integer", "minimum": 0},
                "by_severity": {
                    "type": "object",
                    "additionalProperties": {"type": "integer", "minimum": 0},
                },
                "by_detector": {
                    "type": "object",
                    "additionalProperties": {"type": "integer", "minimum": 0},
                },
            },
            "required": [
                "total_findings",
                "open_findings",
                "by_severity",
                "by_detector",
            ],
        },
        "confidence": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "overall": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "evidence_coverage": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "notes": {"type": "string"},
            },
            "required": ["overall", "evidence_coverage", "notes"],
        },
        "audit_score": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "value": {"type": "number"},
                "max_value": {"type": "number"},
                "rating": {"type": "string", "minLength": 1},
            },
            "required": ["value", "max_value", "rating"],
        },
        "evidence_references": {
            "type": "array",
            "items": EVIDENCE_REFERENCE_SCHEMA,
        },
        "provenance": PROVENANCE_SCHEMA,
    },
    "required": [
        "run_metadata",
        "benchmark_summary",
        "detectors_run",
        "findings_summary",
        "confidence",
        "audit_score",
        "evidence_references",
        "provenance",
    ],
}


METRICS_BEFORE_AFTER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "baseline_metrics": {
            "type": "array",
            "minItems": 1,
            "items": METRIC_SUMMARY_SCHEMA,
        },
        "remediated_metrics": {
            "type": "array",
            "minItems": 1,
            "items": METRIC_SUMMARY_SCHEMA,
        },
        "deltas": {
            "type": "array",
            "minItems": 1,
            "items": METRIC_DELTA_SCHEMA,
        },
        "split_information": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "evaluated_splits": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
                "split_manifest_path": {"type": "string", "minLength": 1},
                "notes": {"type": "string"},
            },
            "required": ["evaluated_splits", "split_manifest_path", "notes"],
        },
        "provenance": PROVENANCE_SCHEMA,
    },
    "required": [
        "baseline_metrics",
        "remediated_metrics",
        "deltas",
        "split_information",
        "provenance",
    ],
}


SPLIT_MANIFEST_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "dataset_name": {"type": "string", "minLength": 1},
        "file_paths_used": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "splits": {
            "type": "array",
            "minItems": 1,
            "items": SPLIT_SUMMARY_SCHEMA,
        },
        "provenance": PROVENANCE_SCHEMA,
    },
    "required": ["dataset_name", "file_paths_used", "splits", "provenance"],
}


FINDINGS_COLUMN_CONTRACT = (
    "finding_id",
    "detector_name",
    "severity",
    "confidence",
    "evidence_pointer",
    "remediation_status",
    "provenance_schema_version",
    "provenance_git_sha",
    "provenance_dataset_fingerprint",
    "provenance_seed",
    "provenance_run_id",
    "provenance_detector_versions_json",
    "provenance_created_at",
    "provenance_updated_at",
)


def _validator(schema: Mapping) -> jsonschema.Draft7Validator:
    jsonschema.Draft7Validator.check_schema(schema)
    return jsonschema.Draft7Validator(schema, format_checker=jsonschema.FormatChecker())


def validate_provenance_block(provenance: Mapping) -> None:
    _validator(PROVENANCE_SCHEMA).validate(provenance)


def validate_audit_results(data: Mapping) -> None:
    _validator(AUDIT_RESULTS_SCHEMA).validate(data)


def validate_metrics_before_after(data: Mapping) -> None:
    _validator(METRICS_BEFORE_AFTER_SCHEMA).validate(data)


def validate_split_manifest(data: Mapping) -> None:
    _validator(SPLIT_MANIFEST_SCHEMA).validate(data)


def validate_findings_columns(columns: Sequence[str]) -> None:
    expected = set(FINDINGS_COLUMN_CONTRACT)
    observed = set(columns)

    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ValueError("Invalid findings columns: " + ", ".join(details))


def build_example_provenance() -> dict:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "git_sha": "96bd51617cfdbb494a9fc283af00fe090edfae48",
        "dataset_fingerprint": "sha256:demo-dataset-fingerprint",
        "seed": 7,
        "run_id": "audit-run-0001",
        "detector_versions": {
            "exact_duplicate": "1.0.0",
            "temporal_overlap": "1.0.0",
        },
        "created_at": "2026-04-11T15:30:00Z",
        "updated_at": "2026-04-11T15:30:00Z",
    }


def build_example_audit_results() -> dict:
    provenance = build_example_provenance()
    return {
        "run_metadata": {
            "run_id": provenance["run_id"],
            "mode": "audit",
            "seed": provenance["seed"],
            "status": "completed",
        },
        "benchmark_summary": {
            "benchmark_name": "demo-benchmark",
            "dataset_name": "demo-dataset",
            "record_count": 120,
            "split_names": ["train", "validation", "test"],
        },
        "detectors_run": [
            {
                "name": "exact_duplicate",
                "version": "1.0.0",
                "status": "completed",
                "finding_count": 2,
            },
            {
                "name": "temporal_overlap",
                "version": "1.0.0",
                "status": "completed",
                "finding_count": 1,
            },
        ],
        "findings_summary": {
            "total_findings": 3,
            "open_findings": 2,
            "by_severity": {"high": 1, "medium": 2},
            "by_detector": {"exact_duplicate": 2, "temporal_overlap": 1},
        },
        "confidence": {
            "overall": 0.91,
            "evidence_coverage": 1.0,
            "notes": "All findings are backed by deterministic evidence files.",
        },
        "audit_score": {
            "value": 82.5,
            "max_value": 100.0,
            "rating": "warning",
        },
        "evidence_references": [
            {
                "evidence_id": "evidence-001",
                "path": "evidence/exact_duplicate_pairs.parquet",
                "kind": "parquet",
                "description": "Exact duplicate candidate pairs with hashes.",
            }
        ],
        "provenance": provenance,
    }


def build_example_metrics_before_after() -> dict:
    provenance = build_example_provenance()
    return {
        "baseline_metrics": [
            {
                "metric_name": "accuracy",
                "split": "test",
                "value": 0.94,
                "higher_is_better": True,
            }
        ],
        "remediated_metrics": [
            {
                "metric_name": "accuracy",
                "split": "test",
                "value": 0.88,
                "higher_is_better": True,
            }
        ],
        "deltas": [
            {
                "metric_name": "accuracy",
                "split": "test",
                "baseline_value": 0.94,
                "remediated_value": 0.88,
                "delta": -0.06,
            }
        ],
        "split_information": {
            "evaluated_splits": ["train", "test"],
            "split_manifest_path": "split_manifest.json",
            "notes": "Metrics compare the original split against the remediated split.",
        },
        "provenance": provenance,
    }


def build_example_split_manifest() -> dict:
    provenance = build_example_provenance()
    return {
        "dataset_name": "demo-dataset",
        "file_paths_used": [
            "input/train.parquet",
            "input/test.parquet",
        ],
        "splits": [
            {
                "name": "train",
                "record_count": 100,
                "file_paths": ["input/train.parquet"],
                "group_key_summary": {
                    "group_keys": ["user_id"],
                    "unique_group_count": 80,
                },
                "temporal_coverage": {
                    "timestamp_column": "event_time",
                    "min_timestamp": "2023-01-01T00:00:00Z",
                    "max_timestamp": "2023-06-30T00:00:00Z",
                },
            },
            {
                "name": "test",
                "record_count": 20,
                "file_paths": ["input/test.parquet"],
                "group_key_summary": {
                    "group_keys": ["user_id"],
                    "unique_group_count": 20,
                },
                "temporal_coverage": {
                    "timestamp_column": "event_time",
                    "min_timestamp": "2023-07-01T00:00:00Z",
                    "max_timestamp": "2023-07-31T00:00:00Z",
                },
            },
        ],
        "provenance": provenance,
    }


def build_example_findings_columns() -> tuple[str, ...]:
    return deepcopy(FINDINGS_COLUMN_CONTRACT)


def build_provenance_block(
    *,
    git_sha: str,
    dataset_fingerprint: str,
    seed: int | None,
    run_id: str,
    detector_versions: Mapping[str, str],
    created_at: str | None = None,
    updated_at: str | None = None,
    schema_version: str = AUDIT_SCHEMA_VERSION,
) -> dict:
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if updated_at is None:
        updated_at = created_at

    provenance = {
        "schema_version": schema_version,
        "git_sha": git_sha,
        "dataset_fingerprint": dataset_fingerprint,
        "seed": seed,
        "run_id": run_id,
        "detector_versions": dict(detector_versions),
        "created_at": created_at,
        "updated_at": updated_at,
    }
    validate_provenance_block(provenance)
    return provenance
