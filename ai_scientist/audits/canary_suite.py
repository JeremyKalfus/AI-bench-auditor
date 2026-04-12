from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


CANARY_SUITE_VERSION = "0.1.0"
CANARY_SUITE_MANIFEST_NAME = "canary_suite_manifest.json"


def _exact_duplicate_case() -> dict[str, Any]:
    train = pd.DataFrame(
        [
            {"record_id": 1, "group_id": "g1", "event_time": "2024-01-01", "text": "alpha", "label": 0, "feature": 0.10},
            {"record_id": 2, "group_id": "g2", "event_time": "2024-01-02", "text": "beta", "label": 1, "feature": 0.20},
        ]
    )
    test = pd.DataFrame(
        [
            {"record_id": 1, "group_id": "g1", "event_time": "2024-01-01", "text": "alpha", "label": 0, "feature": 0.10},
            {"record_id": 10, "group_id": "g10", "event_time": "2024-01-10", "text": "gamma", "label": 1, "feature": 0.30},
        ]
    )
    return {
        "name": "exact_duplicate_leakage",
        "description": "Train and test contain an identical row, exercising exact duplicate leakage detection.",
        "split_data": {"train": train, "test": test},
        "expected_properties": {"shared_exact_row": True},
    }


def _near_duplicate_case() -> dict[str, Any]:
    train = pd.DataFrame(
        [
            {"record_id": 1, "group_id": "g1", "event_time": "2024-02-01", "text": "The quick brown fox", "label": 0, "feature": 0.11},
            {"record_id": 2, "group_id": "g2", "event_time": "2024-02-02", "text": "Jumps over the lazy dog", "label": 1, "feature": 0.22},
        ]
    )
    test = pd.DataFrame(
        [
            {"record_id": 3, "group_id": "g3", "event_time": "2024-02-03", "text": "The quick brown box", "label": 0, "feature": 0.11},
            {"record_id": 4, "group_id": "g4", "event_time": "2024-02-04", "text": "A completely different sentence", "label": 1, "feature": 0.33},
        ]
    )
    return {
        "name": "near_duplicate_leakage",
        "description": "Train and test contain nearly identical text with a one-token change, exercising near-duplicate detection.",
        "split_data": {"train": train, "test": test},
        "expected_properties": {"near_duplicate_pair": ("The quick brown fox", "The quick brown box")},
    }


def _group_overlap_case() -> dict[str, Any]:
    train = pd.DataFrame(
        [
            {"record_id": 1, "customer_id": "cust_01", "event_time": "2024-03-01", "text": "train example one", "label": 0, "feature": 0.40},
            {"record_id": 2, "customer_id": "cust_02", "event_time": "2024-03-02", "text": "train example two", "label": 1, "feature": 0.50},
        ]
    )
    test = pd.DataFrame(
        [
            {"record_id": 3, "customer_id": "cust_02", "event_time": "2024-03-03", "text": "test example one", "label": 1, "feature": 0.60},
            {"record_id": 4, "customer_id": "cust_03", "event_time": "2024-03-04", "text": "test example two", "label": 0, "feature": 0.70},
        ]
    )
    return {
        "name": "group_entity_overlap",
        "description": "Train and test share a customer entity, exercising group overlap detection.",
        "split_data": {"train": train, "test": test},
        "expected_properties": {"shared_group_ids": ["cust_02"]},
    }


def _temporal_leakage_case() -> dict[str, Any]:
    train = pd.DataFrame(
        [
            {"record_id": 1, "event_time": "2024-05-03", "text": "future record one", "label": 1, "feature": 0.90},
            {"record_id": 2, "event_time": "2024-05-04", "text": "future record two", "label": 0, "feature": 0.80},
        ]
    )
    test = pd.DataFrame(
        [
            {"record_id": 3, "event_time": "2024-05-01", "text": "earlier record one", "label": 0, "feature": 0.10},
            {"record_id": 4, "event_time": "2024-05-02", "text": "earlier record two", "label": 1, "feature": 0.20},
        ]
    )
    return {
        "name": "temporal_leakage",
        "description": "Train timestamps occur after test timestamps, exercising temporal leakage checks.",
        "split_data": {"train": train, "test": test},
        "expected_properties": {"train_min_after_test_max": True},
    }


def _preprocessing_label_leakage_case() -> dict[str, Any]:
    train = pd.DataFrame(
        [
            {"record_id": 1, "event_time": "2024-06-01", "raw_signal": 12, "label": 0, "leaky_scaled_feature": 0},
            {"record_id": 2, "event_time": "2024-06-02", "raw_signal": 18, "label": 1, "leaky_scaled_feature": 1},
        ]
    )
    test = pd.DataFrame(
        [
            {"record_id": 3, "event_time": "2024-06-03", "raw_signal": 27, "label": 1, "leaky_scaled_feature": 1},
            {"record_id": 4, "event_time": "2024-06-04", "raw_signal": 33, "label": 0, "leaky_scaled_feature": 0},
        ]
    )
    return {
        "name": "preprocessing_label_leakage",
        "description": "A derived feature mirrors the label, exercising preprocessing or label leakage checks.",
        "split_data": {"train": train, "test": test},
        "expected_properties": {"label_derived_feature": "leaky_scaled_feature"},
    }


def _clean_negative_control_case() -> dict[str, Any]:
    train = pd.DataFrame(
        [
            {"record_id": 1, "group_id": "train_a", "event_time": "2024-07-01", "text": "clean train one", "label": 0, "feature": 0.15},
            {"record_id": 2, "group_id": "train_b", "event_time": "2024-07-02", "text": "clean train two", "label": 1, "feature": 0.25},
        ]
    )
    test = pd.DataFrame(
        [
            {"record_id": 3, "group_id": "test_a", "event_time": "2024-07-03", "text": "clean test one", "label": 1, "feature": 0.35},
            {"record_id": 4, "group_id": "test_b", "event_time": "2024-07-04", "text": "clean test two", "label": 0, "feature": 0.45},
        ]
    )
    return {
        "name": "clean_negative_control",
        "description": "No deliberate leakage is present, exercising the clean negative control path.",
        "split_data": {"train": train, "test": test},
        "expected_properties": {"shared_exact_row": False},
    }


def _canary_cases() -> list[dict[str, Any]]:
    return [
        _exact_duplicate_case(),
        _near_duplicate_case(),
        _group_overlap_case(),
        _temporal_leakage_case(),
        _preprocessing_label_leakage_case(),
        _clean_negative_control_case(),
    ]


def _write_case(case_dir: Path, case: dict[str, Any]) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[dict[str, Any]] = []
    for split_name, frame in case["split_data"].items():
        file_path = case_dir / f"{split_name}.csv"
        frame.to_csv(file_path, index=False)
        written_files.append(
            {
                "split": split_name,
                "path": str(file_path),
                "relative_path": str(file_path.relative_to(case_dir.parent.parent)),
                "row_count": int(frame.shape[0]),
                "column_names": list(frame.columns),
            }
        )

    return {
        "name": case["name"],
        "description": case["description"],
        "expected_properties": case["expected_properties"],
        "files": written_files,
    }


def write_canary_suite(output_dir: str | Path) -> dict[str, Any]:
    suite_dir = Path(output_dir).expanduser().resolve() / "canary_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": CANARY_SUITE_VERSION,
        "suite_name": "benchmark_audit_canary_suite",
        "suite_dir": str(suite_dir),
        "cases": [],
    }

    for case in _canary_cases():
        case_dir = suite_dir / case["name"]
        manifest["cases"].append(_write_case(case_dir, case))

    manifest_path = suite_dir / CANARY_SUITE_MANIFEST_NAME
    manifest["manifest_path"] = str(manifest_path)

    flat_written_files = []
    for case in manifest["cases"]:
        for file_info in case["files"]:
            flat_written_files.append(
                {
                    "case_name": case["name"],
                    "split": file_info["split"],
                    "path": file_info["path"],
                    "relative_path": file_info["relative_path"],
                    "row_count": file_info["row_count"],
                }
            )
    manifest["written_files"] = flat_written_files

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_canary_suite(output_dir: str | Path) -> dict[str, Any]:
    suite_dir = Path(output_dir).expanduser().resolve() / "canary_suite"
    manifest_path = suite_dir / CANARY_SUITE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())

    for case in manifest.get("cases", []):
        for file_info in case.get("files", []):
            file_path = Path(file_info["path"])
            if not file_path.exists():
                raise FileNotFoundError(f"Missing canary fixture file: {file_path}")

    manifest["suite_dir"] = str(suite_dir)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
