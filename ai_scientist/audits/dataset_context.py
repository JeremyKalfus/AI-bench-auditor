from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import build_provenance_block, validate_split_manifest


DATASET_CONTEXT_GENERATOR_VERSION = "0.1.0"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
    raise ValueError(f"Unsupported benchmark data file type: {path}")


def _infer_candidate_key_columns(
    all_columns: list[str], explicit_columns: list[str] | None
) -> list[str]:
    if explicit_columns:
        return list(explicit_columns)

    inferred = []
    for column in all_columns:
        column_lower = column.lower()
        if column_lower == "id" or column_lower.endswith("_id") or column_lower == "uuid":
            inferred.append(column)
    return sorted(set(inferred))


def _infer_target_column(
    all_columns: list[str], explicit_target_column: str | None
) -> str | None:
    if explicit_target_column:
        return explicit_target_column

    preferred = ["label", "target", "y", "class", "outcome"]
    column_lookup = {column.lower(): column for column in all_columns}
    for name in preferred:
        if name in column_lookup:
            return column_lookup[name]
    return None


def _infer_timestamp_columns(
    all_columns: list[str], explicit_timestamp_columns: list[str] | None
) -> list[str]:
    if explicit_timestamp_columns:
        return list(explicit_timestamp_columns)

    inferred = []
    for column in all_columns:
        column_lower = column.lower()
        if any(token in column_lower for token in ("time", "date", "timestamp")):
            inferred.append(column)
    return sorted(set(inferred))


def _sanitize_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in value)


def _stage_benchmark_files(
    benchmark_metadata: dict[str, Any], data_dir: Path
) -> list[dict[str, str]]:
    files = benchmark_metadata.get("files", [])
    if not files:
        return []

    data_dir.mkdir(parents=True, exist_ok=True)

    staged_files = []
    for index, file_info in enumerate(files):
        if not isinstance(file_info, dict) or "path" not in file_info:
            raise ValueError(
                "Each Benchmark Metadata file entry must be an object with at least a `path` key."
            )

        source_path = Path(file_info["path"]).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Benchmark source file does not exist: {source_path}")

        split_name = file_info.get("split") or source_path.stem
        staged_name = f"{index:02d}_{_sanitize_name(split_name)}_{source_path.name}"
        staged_path = data_dir / staged_name
        shutil.copy2(source_path, staged_path)

        staged_files.append(
            {
                "source_path": str(source_path),
                "split": split_name,
                "staged_path": str(staged_path),
                "relative_staged_path": str(Path("data") / staged_name),
            }
        )

    return staged_files


def _build_dataset_fingerprint(
    *,
    dataset_name: str,
    staged_files: list[dict[str, str]],
    candidate_key_columns: list[str],
    target_column: str | None,
    timestamp_columns: list[str],
) -> str:
    digest = hashlib.sha256()
    digest.update(dataset_name.encode("utf-8"))
    digest.update(json.dumps(candidate_key_columns, sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(target_column, sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(timestamp_columns, sort_keys=True).encode("utf-8"))

    for file_info in sorted(staged_files, key=lambda item: item["relative_staged_path"]):
        digest.update(file_info["relative_staged_path"].encode("utf-8"))
        digest.update(file_info["split"].encode("utf-8"))
        with open(file_info["staged_path"], "rb") as staged_file:
            for chunk in iter(lambda: staged_file.read(1024 * 1024), b""):
                digest.update(chunk)

    return f"sha256:{digest.hexdigest()}"


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_repo_root(),
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def _temporal_coverage(df: pd.DataFrame, timestamp_columns: list[str]) -> dict[str, Any] | None:
    for column in timestamp_columns:
        if column not in df.columns:
            continue
        converted = pd.to_datetime(df[column], errors="coerce", utc=True).dropna()
        if converted.empty:
            return {
                "timestamp_column": column,
                "min_timestamp": None,
                "max_timestamp": None,
            }
        return {
            "timestamp_column": column,
            "min_timestamp": converted.min().isoformat().replace("+00:00", "Z"),
            "max_timestamp": converted.max().isoformat().replace("+00:00", "Z"),
        }
    return None


def _group_key_summary(
    df: pd.DataFrame, candidate_key_columns: list[str]
) -> dict[str, Any] | None:
    available_columns = [column for column in candidate_key_columns if column in df.columns]
    if not available_columns:
        return None

    return {
        "group_keys": available_columns,
        "unique_group_count": int(df[available_columns].drop_duplicates().shape[0]),
    }


def _render_dataset_card(
    *,
    dataset_name: str,
    dataset_fingerprint: str,
    file_inventory: list[dict[str, Any]],
    split_names: list[str],
    candidate_key_columns: list[str],
    target_column: str | None,
    timestamp_columns: list[str],
) -> str:
    lines = [
        "# Dataset Card",
        "",
        f"- Dataset name: {dataset_name}",
        f"- Dataset fingerprint: {dataset_fingerprint}",
        f"- Split names: {', '.join(split_names) if split_names else 'unknown'}",
        f"- Candidate key columns: {', '.join(candidate_key_columns) if candidate_key_columns else 'none detected'}",
        f"- Target column: {target_column if target_column else 'unknown'}",
        f"- Timestamp columns: {', '.join(timestamp_columns) if timestamp_columns else 'none detected'}",
        "",
        "## File Inventory",
    ]

    for entry in file_inventory:
        lines.extend(
            [
                f"- File: {entry['relative_path']}",
                f"  Split: {entry['split']}",
                f"  Rows: {entry['row_count']}",
                f"  Columns: {', '.join(entry['columns'])}",
            ]
        )

    return "\n".join(lines) + "\n"


def _render_dataset_context_summary(
    *,
    dataset_card_path: str,
    split_manifest_path: str,
    split_names: list[str],
    candidate_key_columns: list[str],
    target_column: str | None,
    timestamp_columns: list[str],
    file_inventory: list[dict[str, Any]],
) -> str:
    lines = [
        f"Dataset card: {dataset_card_path}",
        f"Split manifest: {split_manifest_path}",
        f"Split names: {', '.join(split_names) if split_names else 'unknown'}",
        f"Candidate key columns: {', '.join(candidate_key_columns) if candidate_key_columns else 'none detected'}",
        f"Target column: {target_column if target_column else 'unknown'}",
        f"Timestamp columns: {', '.join(timestamp_columns) if timestamp_columns else 'none detected'}",
        "File inventory:",
    ]
    for entry in file_inventory:
        lines.append(
            f"- {entry['relative_path']} ({entry['split']}): {entry['row_count']} rows"
        )
    return "\n".join(lines)


def augment_idea_with_dataset_context(idea: dict[str, Any], idea_dir: str | Path) -> dict[str, Any]:
    updated_idea = deepcopy(idea)
    benchmark_metadata = updated_idea.get("Benchmark Metadata")
    if not isinstance(benchmark_metadata, dict):
        return updated_idea

    data_dir = Path(idea_dir) / "data"
    staged_files = _stage_benchmark_files(benchmark_metadata, data_dir)
    if not staged_files:
        return updated_idea

    frames = []
    file_inventory = []
    all_columns = []

    for staged_file in staged_files:
        staged_path = Path(staged_file["staged_path"])
        dataframe = _read_table(staged_path)
        frames.append((staged_file, dataframe))
        columns = [str(column) for column in dataframe.columns.tolist()]
        all_columns.extend(columns)
        file_inventory.append(
            {
                "relative_path": staged_file["relative_staged_path"],
                "split": staged_file["split"],
                "row_count": int(dataframe.shape[0]),
                "columns": columns,
            }
        )

    all_columns = list(dict.fromkeys(all_columns))
    dataset_name = benchmark_metadata.get("dataset_name") or updated_idea.get("Name") or "benchmark-dataset"
    candidate_key_columns = _infer_candidate_key_columns(
        all_columns, benchmark_metadata.get("candidate_key_columns")
    )
    target_column = _infer_target_column(all_columns, benchmark_metadata.get("target_column"))
    timestamp_columns = _infer_timestamp_columns(
        all_columns, benchmark_metadata.get("timestamp_columns")
    )
    dataset_fingerprint = _build_dataset_fingerprint(
        dataset_name=dataset_name,
        staged_files=staged_files,
        candidate_key_columns=candidate_key_columns,
        target_column=target_column,
        timestamp_columns=timestamp_columns,
    )

    provenance = build_provenance_block(
        git_sha=_git_sha(),
        dataset_fingerprint=dataset_fingerprint,
        seed=None,
        run_id=Path(idea_dir).name,
        detector_versions={"dataset_context": DATASET_CONTEXT_GENERATOR_VERSION},
    )

    split_manifest = {
        "dataset_name": dataset_name,
        "file_paths_used": [
            staged_file["relative_staged_path"] for staged_file in staged_files
        ],
        "splits": [],
        "provenance": provenance,
    }

    for staged_file, dataframe in frames:
        split_entry = {
            "name": staged_file["split"],
            "record_count": int(dataframe.shape[0]),
            "file_paths": [staged_file["relative_staged_path"]],
        }
        group_key_summary = _group_key_summary(dataframe, candidate_key_columns)
        if group_key_summary is not None:
            split_entry["group_key_summary"] = group_key_summary
        temporal_coverage = _temporal_coverage(dataframe, timestamp_columns)
        if temporal_coverage is not None:
            split_entry["temporal_coverage"] = temporal_coverage
        split_manifest["splits"].append(split_entry)

    validate_split_manifest(split_manifest)

    split_manifest_path = Path(idea_dir) / "split_manifest.json"
    with open(split_manifest_path, "w") as split_manifest_file:
        json.dump(split_manifest, split_manifest_file, indent=2)

    dataset_card_path = Path(idea_dir) / "dataset_card.md"
    dataset_card = _render_dataset_card(
        dataset_name=dataset_name,
        dataset_fingerprint=dataset_fingerprint,
        file_inventory=file_inventory,
        split_names=[entry["name"] for entry in split_manifest["splits"]],
        candidate_key_columns=candidate_key_columns,
        target_column=target_column,
        timestamp_columns=timestamp_columns,
    )
    dataset_card_path.write_text(dataset_card)

    updated_metadata = deepcopy(benchmark_metadata)
    updated_metadata["dataset_name"] = dataset_name
    updated_metadata["dataset_fingerprint"] = dataset_fingerprint
    updated_metadata["dataset_card_path"] = dataset_card_path.name
    updated_metadata["split_manifest_path"] = split_manifest_path.name
    updated_metadata["staged_files"] = [
        {
            "path": staged_file["relative_staged_path"],
            "split": staged_file["split"],
        }
        for staged_file in staged_files
    ]
    updated_idea["Benchmark Metadata"] = updated_metadata
    updated_idea["Dataset Context"] = _render_dataset_context_summary(
        dataset_card_path=dataset_card_path.name,
        split_manifest_path=split_manifest_path.name,
        split_names=[entry["name"] for entry in split_manifest["splits"]],
        candidate_key_columns=candidate_key_columns,
        target_column=target_column,
        timestamp_columns=timestamp_columns,
        file_inventory=file_inventory,
    )
    return updated_idea
