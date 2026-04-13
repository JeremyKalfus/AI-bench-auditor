from __future__ import annotations

import hashlib
import json
from itertools import combinations
from typing import Any, Mapping, Sequence

import pandas as pd
from rapidfuzz import fuzz

from .schema import FINDINGS_COLUMN_CONTRACT, validate_findings_columns


SPLIT_ORDER = {
    "train": 0,
    "validation": 1,
    "val": 1,
    "dev": 1,
    "test": 2,
    "holdout": 3,
}

SUSPICIOUS_PREPROCESSING_PATTERNS = (
    "scaled",
    "normalized",
    "target_encoded",
    "mean_encoded",
    "full_dataset",
    "global_",
    "_global",
    "leak",
)


def empty_findings_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(columns=FINDINGS_COLUMN_CONTRACT)
    validate_findings_columns(df.columns)
    return df


def _coerce_jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _finding_row(
    *,
    detector_name: str,
    severity: str,
    confidence: float,
    evidence_pointer: str,
    provenance: Mapping[str, Any],
    remediation_status: str = "open",
) -> dict[str, Any]:
    finding_id = f"{detector_name}:{_hash_text(evidence_pointer)}"
    return {
        "finding_id": finding_id,
        "detector_name": detector_name,
        "severity": severity,
        "confidence": float(confidence),
        "evidence_pointer": evidence_pointer,
        "remediation_status": remediation_status,
        "provenance_schema_version": provenance["schema_version"],
        "provenance_git_sha": provenance["git_sha"],
        "provenance_dataset_fingerprint": provenance["dataset_fingerprint"],
        "provenance_seed": provenance["seed"],
        "provenance_run_id": provenance["run_id"],
        "provenance_detector_versions_json": json.dumps(
            provenance["detector_versions"], sort_keys=True
        ),
        "provenance_created_at": provenance["created_at"],
        "provenance_updated_at": provenance["updated_at"],
    }


def _rows_to_findings(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return empty_findings_dataframe()

    df = pd.DataFrame(rows, columns=FINDINGS_COLUMN_CONTRACT)
    validate_findings_columns(df.columns)
    return df.sort_values(["detector_name", "finding_id"]).reset_index(drop=True)


def _ordered_split_pairs(
    split_frames: Mapping[str, pd.DataFrame],
) -> list[tuple[str, str]]:
    split_names = sorted(
        split_frames,
        key=lambda name: (SPLIT_ORDER.get(name.lower(), 99), name.lower()),
    )
    return list(combinations(split_names, 2))


def _shared_columns(
    split_frames: Mapping[str, pd.DataFrame], requested_columns: Sequence[str] | None
) -> list[str]:
    if requested_columns:
        columns = list(dict.fromkeys(requested_columns))
    else:
        shared = set.intersection(*(set(df.columns) for df in split_frames.values()))
        columns = sorted(shared)
    if not columns:
        raise ValueError("No shared columns available for detector comparison")
    return columns


def _default_text_columns(split_frames: Mapping[str, pd.DataFrame]) -> list[str]:
    shared = _shared_columns(split_frames, None)
    text_columns = []
    for column in shared:
        if all(
            pd.api.types.is_string_dtype(df[column])
            or pd.api.types.is_object_dtype(df[column])
            for df in split_frames.values()
        ):
            text_columns.append(column)
    return text_columns


def _normalize_row_signature(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    return df.loc[:, list(columns)].apply(
        lambda row: json.dumps(
            {column: _coerce_jsonable(row[column]) for column in columns},
            sort_keys=True,
        ),
        axis=1,
    )


def detect_exact_duplicates(
    split_frames: Mapping[str, pd.DataFrame],
    *,
    compare_columns: Sequence[str] | None = None,
    provenance: Mapping[str, Any],
) -> pd.DataFrame:
    columns = _shared_columns(split_frames, compare_columns)
    rows: list[dict[str, Any]] = []

    for left_name, right_name in _ordered_split_pairs(split_frames):
        left_df = split_frames[left_name]
        right_df = split_frames[right_name]
        left_signatures = set(_normalize_row_signature(left_df, columns))
        right_signatures = set(_normalize_row_signature(right_df, columns))
        for signature in sorted(left_signatures & right_signatures):
            evidence_pointer = (
                f"exact_duplicate:{left_name}:{right_name}:row_signature={_hash_text(signature)}"
            )
            rows.append(
                _finding_row(
                    detector_name="exact_duplicate",
                    severity="high",
                    confidence=0.99,
                    evidence_pointer=evidence_pointer,
                    provenance=provenance,
                )
            )

    return _rows_to_findings(rows)


def detect_near_duplicates(
    split_frames: Mapping[str, pd.DataFrame],
    *,
    text_columns: Sequence[str] | None = None,
    provenance: Mapping[str, Any],
    similarity_threshold: int = 95,
) -> pd.DataFrame:
    columns = list(text_columns) if text_columns else _default_text_columns(split_frames)
    if not columns:
        return empty_findings_dataframe()

    rows: list[dict[str, Any]] = []
    for left_name, right_name in _ordered_split_pairs(split_frames):
        left_df = split_frames[left_name].reset_index(drop=True)
        right_df = split_frames[right_name].reset_index(drop=True)
        left_text = left_df.loc[:, columns].fillna("").astype(str).agg(" ".join, axis=1)
        right_text = right_df.loc[:, columns].fillna("").astype(str).agg(" ".join, axis=1)

        for left_index, left_value in enumerate(left_text):
            for right_index, right_value in enumerate(right_text):
                if not left_value.strip() or not right_value.strip():
                    continue
                similarity = fuzz.token_set_ratio(left_value, right_value)
                if similarity < similarity_threshold or left_value == right_value:
                    continue
                evidence_pointer = (
                    f"near_duplicate:{left_name}[{left_index}]~{right_name}[{right_index}]:similarity={similarity}"
                )
                rows.append(
                    _finding_row(
                        detector_name="near_duplicate",
                        severity="medium",
                        confidence=similarity / 100.0,
                        evidence_pointer=evidence_pointer,
                        provenance=provenance,
                    )
                )

    return _rows_to_findings(rows)


def detect_group_overlap(
    split_frames: Mapping[str, pd.DataFrame],
    *,
    group_columns: Sequence[str],
    provenance: Mapping[str, Any],
) -> pd.DataFrame:
    columns = _shared_columns(split_frames, group_columns)
    rows: list[dict[str, Any]] = []

    for left_name, right_name in _ordered_split_pairs(split_frames):
        left_keys = {
            json.dumps(
                {column: _coerce_jsonable(row[column]) for column in columns},
                sort_keys=True,
            )
            for _, row in split_frames[left_name].loc[:, columns].drop_duplicates().iterrows()
        }
        right_keys = {
            json.dumps(
                {column: _coerce_jsonable(row[column]) for column in columns},
                sort_keys=True,
            )
            for _, row in split_frames[right_name].loc[:, columns].drop_duplicates().iterrows()
        }
        for shared_key in sorted(left_keys & right_keys):
            evidence_pointer = (
                f"group_overlap:{left_name}:{right_name}:group_key={_hash_text(shared_key)}"
            )
            rows.append(
                _finding_row(
                    detector_name="group_overlap",
                    severity="high",
                    confidence=0.97,
                    evidence_pointer=evidence_pointer,
                    provenance=provenance,
                )
            )

    return _rows_to_findings(rows)


def detect_temporal_leakage(
    split_frames: Mapping[str, pd.DataFrame],
    *,
    timestamp_column: str,
    provenance: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    ordered_splits = sorted(
        split_frames,
        key=lambda name: (SPLIT_ORDER.get(name.lower(), 99), name.lower()),
    )
    temporal_ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for split_name in ordered_splits:
        df = split_frames[split_name]
        if timestamp_column not in df.columns:
            raise ValueError(f"Timestamp column `{timestamp_column}` missing from split `{split_name}`")
        timestamps = pd.to_datetime(df[timestamp_column], utc=True, errors="coerce").dropna()
        if timestamps.empty:
            continue
        temporal_ranges[split_name] = (timestamps.min(), timestamps.max())

    for left_name, right_name in combinations(ordered_splits, 2):
        if left_name not in temporal_ranges or right_name not in temporal_ranges:
            continue
        left_min, left_max = temporal_ranges[left_name]
        right_min, right_max = temporal_ranges[right_name]
        if left_max >= right_min:
            evidence_pointer = (
                f"temporal_leakage:{left_name}[{left_min.isoformat()}..{left_max.isoformat()}]"
                f" overlaps {right_name}[{right_min.isoformat()}..{right_max.isoformat()}]"
            )
            rows.append(
                _finding_row(
                    detector_name="temporal_leakage",
                    severity="high",
                    confidence=0.96,
                    evidence_pointer=evidence_pointer,
                    provenance=provenance,
                )
            )

    return _rows_to_findings(rows)


def detect_preprocessing_leakage(
    split_frames: Mapping[str, pd.DataFrame],
    *,
    feature_columns: Sequence[str] | None = None,
    provenance: Mapping[str, Any],
    suspicious_patterns: Sequence[str] = SUSPICIOUS_PREPROCESSING_PATTERNS,
) -> pd.DataFrame:
    shared_columns = _shared_columns(split_frames, feature_columns)
    suspicious_columns = [
        column
        for column in shared_columns
        if any(pattern in column.lower() for pattern in suspicious_patterns)
    ]

    rows = [
        _finding_row(
            detector_name="preprocessing_leakage",
            severity="medium",
            confidence=0.8,
            evidence_pointer=f"preprocessing_leakage:column={column}",
            provenance=provenance,
        )
        for column in sorted(suspicious_columns)
    ]
    return _rows_to_findings(rows)


def detect_suspicious_feature_leakage(
    split_frames: Mapping[str, pd.DataFrame],
    *,
    target_column: str,
    provenance: Mapping[str, Any],
    feature_columns: Sequence[str] | None = None,
    match_threshold: float = 0.95,
) -> pd.DataFrame:
    shared_columns = _shared_columns(split_frames, feature_columns)
    candidate_columns = [column for column in shared_columns if column != target_column]
    combined = pd.concat(split_frames.values(), ignore_index=True)
    if target_column not in combined.columns:
        raise ValueError(f"Target column `{target_column}` missing from combined splits")

    target_values = combined[target_column].fillna("").astype(str)
    rows: list[dict[str, Any]] = []

    for column in candidate_columns:
        if column not in combined.columns:
            continue
        feature_values = combined[column].fillna("").astype(str)
        match_ratio = float((feature_values == target_values).mean())
        if match_ratio < match_threshold:
            continue
        evidence_pointer = (
            f"suspicious_feature_leakage:column={column}:match_ratio={match_ratio:.3f}"
        )
        rows.append(
            _finding_row(
                detector_name="suspicious_feature_leakage",
                severity="high",
                confidence=min(0.99, match_ratio),
                evidence_pointer=evidence_pointer,
                provenance=provenance,
            )
        )

    return _rows_to_findings(rows)
