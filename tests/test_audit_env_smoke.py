import importlib.util
import json
from pathlib import Path

import pandas as pd


def load_launcher(repo_root: Path):
    launcher_path = repo_root / "launch_scientist_bfts.py"
    spec = importlib.util.spec_from_file_location(
        "launch_scientist_bfts", launcher_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_benchmark_idea(ideas_path: Path, train_path: Path, test_path: Path) -> None:
    ideas_path.write_text(
        json.dumps(
            [
                {
                    "Name": "autodiscovered_demo",
                    "Title": "Autodiscovered Demo",
                    "Abstract": "Exercise the default discovery-first launcher path.",
                    "Short Hypothesis": "Automatic benchmark discovery should select a draft spec before audit preparation begins.",
                    "Experiments": ["Audit the discovered benchmark splits."],
                    "Risk Factors and Limitations": "Synthetic fixture only.",
                    "Benchmark Metadata": {
                        "dataset_name": "demo-benchmark",
                        "files": [
                            {"path": str(train_path), "split": "train"},
                            {"path": str(test_path), "split": "test"},
                        ],
                    },
                }
            ],
            indent=2,
        )
    )


def test_audit_mode_dry_run_smoke(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    launcher = load_launcher(repo_root)

    monkeypatch.chdir(tmp_path)

    exit_code = launcher.main(
        [
            "--mode",
            "audit",
            "--dry-run",
            "--attempt_id",
            "4242",
            "--config-path",
            str(repo_root / "bfts_config.yaml"),
            "--load_ideas",
            str(repo_root / "ai_scientist/ideas/i_cant_believe_its_not_better.json"),
        ]
    )

    assert exit_code == 0

    run_dirs = list((tmp_path / "experiments").glob("*_attempt_4242"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    metadata = json.loads((run_dir / "audit_run_metadata.json").read_text())
    assert metadata["mode"] == "audit"
    assert metadata["runtime_settings"]["skip_writeup"] is True
    assert metadata["runtime_settings"]["skip_review"] is True
    assert metadata["runtime_settings"]["run_plot_aggregation"] is False

    run_config_text = (run_dir / "bfts_config.yaml").read_text()
    assert "num_workers: 1" in run_config_text
    assert "num_seeds: 1" in run_config_text


def test_select_discovered_spec_prefers_dataset_backed_candidates():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = load_launcher(repo_root)

    candidate, spec_path, selection_strategy = launcher.select_discovered_spec(
        {
            "candidates": [
                {
                    "benchmark_name": "Paper Benchmark",
                    "dataset_name": "paper_only_benchmark",
                    "source_type": "paper_only",
                    "source_id": None,
                    "source_url": "https://example.com/paper",
                },
                {
                    "benchmark_name": "GLUE",
                    "dataset_name": "nyu-mll/glue",
                    "source_type": "huggingface_dataset",
                    "source_id": "nyu-mll/glue",
                    "source_url": "https://huggingface.co/datasets/nyu-mll/glue",
                },
            ],
            "spec_paths": {
                "paper_only_benchmark": "/tmp/paper_only_benchmark.json",
                "nyu_mll_glue": "/tmp/nyu_mll_glue.json",
            },
        }
    )

    assert candidate["dataset_name"] == "nyu-mll/glue"
    assert spec_path == "/tmp/nyu_mll_glue.json"
    assert selection_strategy == "dataset_backed_first"


def test_audit_mode_dry_run_autodiscovers_by_default(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    launcher = load_launcher(repo_root)

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame([{"row_id": 1, "text": "alpha"}]).to_csv(train_path, index=False)
    pd.DataFrame([{"row_id": 2, "text": "beta"}]).to_csv(test_path, index=False)

    ideas_path = tmp_path / "autodiscovered_spec.json"
    write_benchmark_idea(ideas_path, train_path, test_path)

    calls = {"count": 0}

    def fake_autodiscover(args):
        calls["count"] += 1
        args.load_ideas = str(ideas_path)
        args.idea_idx = 0
        args.output_dir = str(tmp_path / "autodiscovered-run")
        args.benchmark_discovery = {
            "topic": "default discovery topic",
            "selected_spec_path": str(ideas_path),
            "selection_strategy": "dataset_backed_first",
        }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher, "autodiscover_benchmark_spec", fake_autodiscover)

    exit_code = launcher.main(["--mode", "audit", "--dry-run", "--attempt_id", "73"])

    assert exit_code == 0
    assert calls["count"] == 1

    run_dir = tmp_path / "autodiscovered-run"
    metadata = json.loads((run_dir / "audit_run_metadata.json").read_text())
    assert metadata["benchmark_discovery"]["selected_spec_path"] == str(ideas_path)
    assert metadata["runtime_settings"]["used_benchmark_discovery"] is True
