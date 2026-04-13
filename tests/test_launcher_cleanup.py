import builtins
import importlib.util
from pathlib import Path


def load_launcher():
    repo_root = Path(__file__).resolve().parents[1]
    launcher_path = repo_root / "launch_scientist_bfts.py"
    spec = importlib.util.spec_from_file_location(
        "launch_scientist_bfts", launcher_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cleanup_processes_skips_cleanly_when_psutil_is_missing(
    monkeypatch, capsys
):
    launcher = load_launcher()
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ModuleNotFoundError("No module named 'psutil'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    launcher.cleanup_processes()

    captured = capsys.readouterr()
    assert "Skipping process cleanup because `psutil` is not installed." in captured.out
