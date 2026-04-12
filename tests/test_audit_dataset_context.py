import importlib.util
import json
from pathlib import Path

import pandas as pd

from ai_scientist.audits.schema import validate_split_manifest
from ai_scientist.treesearch.agent_manager import AgentManager
from ai_scientist.treesearch.utils.config import load_cfg, load_task_desc


def load_launcher(repo_root: Path):
    launcher_path = repo_root / "launch_scientist_bfts.py"
    spec = importlib.util.spec_from_file_location(
        "launch_scientist_bfts", launcher_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_audit_dataset_context_is_generated_and_injected(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    launcher = load_launcher(repo_root)

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame(
        [
            {"user_id": 1, "feature": 0.1, "label": 0, "event_time": "2023-01-01"},
            {"user_id": 2, "feature": 0.2, "label": 1, "event_time": "2023-01-02"},
        ]
    ).to_csv(train_path, index=False)
    pd.DataFrame(
        [
            {"user_id": 3, "feature": 0.3, "label": 1, "event_time": "2023-01-03"},
        ]
    ).to_csv(test_path, index=False)

    ideas_path = tmp_path / "audit_ideas.json"
    ideas_path.write_text(
        json.dumps(
            [
                {
                    "Name": "audit_dataset_context_demo",
                    "Title": "Dataset Context Demo",
                    "Abstract": "A small audit-mode test idea.",
                    "Short Hypothesis": "Deterministic dataset context should be injected before code generation.",
                    "Experiments": ["Read manifests and audit the provided splits."],
                    "Risk Factors and Limitations": "This is a tiny synthetic benchmark for verification only.",
                    "Audit Targets": ["exact_duplicate", "temporal_overlap"],
                    "Leakage Taxonomy": [
                        "exact duplicate leakage",
                        "temporal leakage",
                    ],
                    "Acceptance Criteria": [
                        "dataset_card.md exists",
                        "split_manifest.json validates",
                    ],
                    "Benchmark Metadata": {
                        "dataset_name": "demo-benchmark",
                        "files": [
                            {"path": str(train_path), "split": "train"},
                            {"path": str(test_path), "split": "test"},
                        ],
                        "candidate_key_columns": ["user_id"],
                        "target_column": "label",
                        "timestamp_columns": ["event_time"],
                    },
                }
            ],
            indent=2,
        )
    )

    monkeypatch.chdir(tmp_path)
    exit_code = launcher.main(
        [
            "--mode",
            "audit",
            "--dry-run",
            "--attempt_id",
            "4343",
            "--config-path",
            str(repo_root / "bfts_config.yaml"),
            "--load_ideas",
            str(ideas_path),
        ]
    )
    assert exit_code == 0

    run_dirs = list((tmp_path / "experiments").glob("*_attempt_4343"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    dataset_card_path = run_dir / "dataset_card.md"
    split_manifest_path = run_dir / "split_manifest.json"
    assert dataset_card_path.exists()
    assert split_manifest_path.exists()

    split_manifest = json.loads(split_manifest_path.read_text())
    validate_split_manifest(split_manifest)

    dataset_card_text = dataset_card_path.read_text()
    assert "Dataset Card" in dataset_card_text
    assert "Candidate key columns: user_id" in dataset_card_text
    assert "Split names: train, test" in dataset_card_text

    generated_idea = json.loads((run_dir / "idea.json").read_text())
    assert "Audit Targets" in generated_idea
    assert "Leakage Taxonomy" in generated_idea
    assert "Acceptance Criteria" in generated_idea
    assert generated_idea["Benchmark Metadata"]["dataset_card_path"] == "dataset_card.md"
    assert generated_idea["Benchmark Metadata"]["split_manifest_path"] == "split_manifest.json"
    assert "Dataset Context" in generated_idea

    cfg = load_cfg(run_dir / "bfts_config.yaml")
    task_desc = load_task_desc(cfg)
    manager = AgentManager(task_desc=task_desc, cfg=cfg, workspace_dir=Path(cfg.workspace_dir))
    rendered_task_desc = manager._curate_task_desc(manager.current_stage)

    assert "Dataset Context" in rendered_task_desc
    assert "Split manifest: split_manifest.json" in rendered_task_desc
    assert "Candidate key columns: user_id" in rendered_task_desc
    assert "Audit Targets" in rendered_task_desc
