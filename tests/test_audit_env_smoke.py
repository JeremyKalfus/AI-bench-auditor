import importlib.util
import json
from pathlib import Path


def load_launcher(repo_root: Path):
    launcher_path = repo_root / "launch_scientist_bfts.py"
    spec = importlib.util.spec_from_file_location(
        "launch_scientist_bfts", launcher_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
