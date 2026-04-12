import pandas as pd

from ai_scientist.audits.detectors import (
    detect_exact_duplicates,
    detect_group_overlap,
    detect_near_duplicates,
    detect_preprocessing_leakage,
    detect_suspicious_feature_leakage,
    detect_temporal_leakage,
)
from ai_scientist.audits.schema import build_provenance_block, validate_findings_columns


def make_provenance():
    return build_provenance_block(
        git_sha="96bd51617cfdbb494a9fc283af00fe090edfae48",
        dataset_fingerprint="sha256:test-detectors",
        seed=7,
        run_id="detector-test-run",
        detector_versions={
            "exact_duplicate": "1.0.0",
            "near_duplicate": "1.0.0",
            "group_overlap": "1.0.0",
            "temporal_leakage": "1.0.0",
            "preprocessing_leakage": "1.0.0",
            "suspicious_feature_leakage": "1.0.0",
        },
        created_at="2026-04-11T15:30:00Z",
        updated_at="2026-04-11T15:30:00Z",
    )


def test_detect_exact_duplicates_flags_shared_rows():
    provenance = make_provenance()
    split_frames = {
        "train": pd.DataFrame([{"id": 1, "text": "same row"}, {"id": 2, "text": "only train"}]),
        "test": pd.DataFrame([{"id": 1, "text": "same row"}, {"id": 3, "text": "only test"}]),
    }

    findings = detect_exact_duplicates(
        split_frames,
        compare_columns=["id", "text"],
        provenance=provenance,
    )

    validate_findings_columns(findings.columns)
    assert len(findings) == 1
    assert findings.iloc[0]["detector_name"] == "exact_duplicate"


def test_detect_near_duplicates_flags_similar_text():
    provenance = make_provenance()
    split_frames = {
        "train": pd.DataFrame([{"text": "A benchmark audit catches data leakage quickly"}]),
        "test": pd.DataFrame([{"text": "A benchmark audit catches data leakage very quickly"}]),
    }

    findings = detect_near_duplicates(
        split_frames,
        text_columns=["text"],
        provenance=provenance,
        similarity_threshold=90,
    )

    assert len(findings) == 1
    assert findings.iloc[0]["detector_name"] == "near_duplicate"


def test_detect_group_overlap_flags_shared_entities():
    provenance = make_provenance()
    split_frames = {
        "train": pd.DataFrame([{"user_id": "u1"}, {"user_id": "u2"}]),
        "test": pd.DataFrame([{"user_id": "u2"}, {"user_id": "u3"}]),
    }

    findings = detect_group_overlap(
        split_frames,
        group_columns=["user_id"],
        provenance=provenance,
    )

    assert len(findings) == 1
    assert findings.iloc[0]["detector_name"] == "group_overlap"


def test_detect_temporal_leakage_flags_overlapping_ranges():
    provenance = make_provenance()
    split_frames = {
        "train": pd.DataFrame([{"event_time": "2023-01-01"}, {"event_time": "2023-03-01"}]),
        "test": pd.DataFrame([{"event_time": "2023-02-15"}, {"event_time": "2023-04-01"}]),
    }

    findings = detect_temporal_leakage(
        split_frames,
        timestamp_column="event_time",
        provenance=provenance,
    )

    assert len(findings) == 1
    assert findings.iloc[0]["detector_name"] == "temporal_leakage"


def test_detect_preprocessing_leakage_flags_suspicious_column_names():
    provenance = make_provenance()
    split_frames = {
        "train": pd.DataFrame([{"globally_scaled_feature": 0.1, "plain_feature": 1.0}]),
        "test": pd.DataFrame([{"globally_scaled_feature": 0.2, "plain_feature": 2.0}]),
    }

    findings = detect_preprocessing_leakage(split_frames, provenance=provenance)

    assert len(findings) == 1
    assert findings.iloc[0]["detector_name"] == "preprocessing_leakage"


def test_detect_suspicious_feature_leakage_flags_target_copy():
    provenance = make_provenance()
    split_frames = {
        "train": pd.DataFrame(
            [{"label": 0, "leaky_label": 0}, {"label": 1, "leaky_label": 1}]
        ),
        "test": pd.DataFrame(
            [{"label": 0, "leaky_label": 0}, {"label": 1, "leaky_label": 1}]
        ),
    }

    findings = detect_suspicious_feature_leakage(
        split_frames,
        target_column="label",
        provenance=provenance,
    )

    assert len(findings) == 1
    assert findings.iloc[0]["detector_name"] == "suspicious_feature_leakage"
