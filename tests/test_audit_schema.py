from copy import deepcopy

import jsonschema
import pytest

from ai_scientist.audits.schema import (
    AUDIT_RESULTS_SCHEMA,
    METRICS_BEFORE_AFTER_SCHEMA,
    PROVENANCE_SCHEMA,
    SPLIT_MANIFEST_SCHEMA,
    build_example_audit_results,
    build_example_metrics_before_after,
    build_example_provenance,
    build_example_split_manifest,
    validate_audit_results,
    validate_findings_columns,
    validate_metrics_before_after,
    validate_provenance_block,
    validate_split_manifest,
)


def test_provenance_block_accepts_valid_example():
    validate_provenance_block(build_example_provenance())


def test_provenance_block_rejects_missing_required_field():
    provenance = build_example_provenance()
    provenance.pop("schema_version")

    with pytest.raises(jsonschema.ValidationError):
        validate_provenance_block(provenance)


def test_audit_results_example_matches_schema():
    artifact = build_example_audit_results()
    validate_audit_results(artifact)


def test_audit_results_rejects_missing_provenance():
    artifact = build_example_audit_results()
    artifact.pop("provenance")

    with pytest.raises(jsonschema.ValidationError):
        validate_audit_results(artifact)


def test_metrics_before_after_example_matches_schema():
    artifact = build_example_metrics_before_after()
    validate_metrics_before_after(artifact)


def test_metrics_before_after_rejects_missing_split_information():
    artifact = build_example_metrics_before_after()
    artifact.pop("split_information")

    with pytest.raises(jsonschema.ValidationError):
        validate_metrics_before_after(artifact)


def test_split_manifest_example_matches_schema():
    artifact = build_example_split_manifest()
    validate_split_manifest(artifact)


def test_split_manifest_rejects_split_without_file_paths():
    artifact = build_example_split_manifest()
    artifact["splits"][0].pop("file_paths")

    with pytest.raises(jsonschema.ValidationError):
        validate_split_manifest(artifact)


def test_findings_column_contract_accepts_exact_columns():
    columns = [
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
    ]

    validate_findings_columns(columns)


def test_findings_column_contract_rejects_missing_column():
    columns = [
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
        "provenance_created_at",
        "provenance_updated_at",
    ]

    with pytest.raises(ValueError):
        validate_findings_columns(columns)


def test_declared_json_schemas_are_valid():
    for schema in (
        PROVENANCE_SCHEMA,
        AUDIT_RESULTS_SCHEMA,
        METRICS_BEFORE_AFTER_SCHEMA,
        SPLIT_MANIFEST_SCHEMA,
    ):
        jsonschema.Draft7Validator.check_schema(schema)
