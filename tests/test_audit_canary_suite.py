from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_scientist.audits.canary_suite import load_canary_suite, write_canary_suite


def test_canary_suite_writes_all_fixture_datasets(tmp_path):
    manifest = write_canary_suite(tmp_path)

    assert manifest["schema_version"] == "0.1.0"
    assert Path(manifest["suite_dir"]).exists()
    assert Path(manifest["manifest_path"]).exists()

    expected_case_names = {
        "exact_duplicate_leakage",
        "near_duplicate_leakage",
        "group_entity_overlap",
        "temporal_leakage",
        "preprocessing_label_leakage",
        "clean_negative_control",
    }
    assert {case["name"] for case in manifest["cases"]} == expected_case_names
    assert {file_info["case_name"] for file_info in manifest["written_files"]} == expected_case_names

    exact_case = next(case for case in manifest["cases"] if case["name"] == "exact_duplicate_leakage")
    exact_train = pd.read_csv(exact_case["files"][0]["path"])
    exact_test = pd.read_csv(exact_case["files"][1]["path"])
    assert exact_train.iloc[0].to_dict() == exact_test.iloc[0].to_dict()

    near_case = next(case for case in manifest["cases"] if case["name"] == "near_duplicate_leakage")
    near_train = pd.read_csv(next(file_info for file_info in near_case["files"] if file_info["split"] == "train")["path"])
    near_test = pd.read_csv(next(file_info for file_info in near_case["files"] if file_info["split"] == "test")["path"])
    assert near_train.loc[0, "text"] == "The quick brown fox"
    assert near_test.loc[0, "text"] == "The quick brown box"
    assert near_train.loc[0, "text"] != near_test.loc[0, "text"]

    group_case = next(case for case in manifest["cases"] if case["name"] == "group_entity_overlap")
    group_train = pd.read_csv(next(file_info for file_info in group_case["files"] if file_info["split"] == "train")["path"])
    group_test = pd.read_csv(next(file_info for file_info in group_case["files"] if file_info["split"] == "test")["path"])
    assert sorted(set(group_train["customer_id"]).intersection(group_test["customer_id"])) == ["cust_02"]

    temporal_case = next(case for case in manifest["cases"] if case["name"] == "temporal_leakage")
    temporal_train = pd.read_csv(next(file_info for file_info in temporal_case["files"] if file_info["split"] == "train")["path"])
    temporal_test = pd.read_csv(next(file_info for file_info in temporal_case["files"] if file_info["split"] == "test")["path"])
    assert pd.to_datetime(temporal_train["event_time"]).min() > pd.to_datetime(temporal_test["event_time"]).max()

    leakage_case = next(case for case in manifest["cases"] if case["name"] == "preprocessing_label_leakage")
    leakage_train = pd.read_csv(next(file_info for file_info in leakage_case["files"] if file_info["split"] == "train")["path"])
    leakage_test = pd.read_csv(next(file_info for file_info in leakage_case["files"] if file_info["split"] == "test")["path"])
    assert leakage_train["leaky_scaled_feature"].equals(leakage_train["label"])
    assert leakage_test["leaky_scaled_feature"].equals(leakage_test["label"])

    clean_case = next(case for case in manifest["cases"] if case["name"] == "clean_negative_control")
    clean_train = pd.read_csv(next(file_info for file_info in clean_case["files"] if file_info["split"] == "train")["path"])
    clean_test = pd.read_csv(next(file_info for file_info in clean_case["files"] if file_info["split"] == "test")["path"])
    assert clean_train.merge(clean_test, how="inner").empty
    assert set(clean_train["group_id"]).isdisjoint(clean_test["group_id"])


def test_canary_suite_manifest_can_be_loaded(tmp_path):
    written_manifest = write_canary_suite(tmp_path)
    loaded_manifest = load_canary_suite(tmp_path)

    assert loaded_manifest["manifest_path"] == written_manifest["manifest_path"]
    assert loaded_manifest["suite_dir"] == written_manifest["suite_dir"]
    assert [case["name"] for case in loaded_manifest["cases"]] == [case["name"] for case in written_manifest["cases"]]
