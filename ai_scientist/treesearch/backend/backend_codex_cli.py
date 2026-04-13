import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .utils import FunctionSpec, OutputType


@dataclass
class CodexCLIClient:
    model: str


def _resolve_codex_model(model: str) -> str:
    if not model.startswith("codex-cli/"):
        raise ValueError(f"Unsupported Codex CLI model identifier: {model}")
    return model.split("/", 1)[1]


def get_ai_client(model: str, **_model_kwargs) -> CodexCLIClient:
    return CodexCLIClient(model=_resolve_codex_model(model))


def _build_prompt(system_message: str | None, user_message: str | None) -> str:
    if system_message and user_message:
        return (
            "System instructions:\n"
            f"{system_message}\n\n"
            "User message:\n"
            f"{user_message}\n"
        )
    if system_message:
        return system_message
    if user_message:
        return user_message
    return ""


def _strip_nonstandard_schema_keys(value):
    if isinstance(value, dict):
        return {
            key: _strip_nonstandard_schema_keys(inner)
            for key, inner in value.items()
            if key != "strict"
        }
    if isinstance(value, list):
        return [_strip_nonstandard_schema_keys(item) for item in value]
    return value


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    model = model_kwargs.get("model")
    if not isinstance(model, str):
        raise ValueError("Codex CLI backend requires a string model name")

    codex_model = _resolve_codex_model(model)
    prompt = _build_prompt(system_message, user_message)
    cwd = Path(os.getcwd())

    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as output_file:
        output_path = Path(output_file.name)

    schema_path: Path | None = None
    if func_spec is not None:
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as schema_file:
            schema_path = Path(schema_file.name)
            json.dump(_strip_nonstandard_schema_keys(func_spec.json_schema), schema_file)
            schema_file.flush()

    cmd = [
        "codex",
        "exec",
        "--ephemeral",
        "--color",
        "never",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "-m",
        codex_model,
        "-o",
        str(output_path),
    ]
    if schema_path is not None:
        cmd.extend(["--output-schema", str(schema_path)])
    cmd.append("-")

    try:
        t0 = time.time()
        result = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            cwd=cwd,
            check=False,
        )
        req_time = time.time() - t0

        if result.returncode != 0:
            raise RuntimeError(
                "Codex CLI request failed "
                f"(exit {result.returncode}) for model {codex_model}.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        content = output_path.read_text().strip()
        if not content:
            content = result.stdout.strip()

        if func_spec is None:
            output: OutputType = content
        else:
            try:
                output = json.loads(content)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Codex CLI returned non-JSON output for a structured request.\n"
                    f"output:\n{content}\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                ) from exc

        info = {
            "provider": "codex-cli",
            "model": codex_model,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        return output, req_time, 0, 0, info
    finally:
        output_path.unlink(missing_ok=True)
        if schema_path is not None:
            schema_path.unlink(missing_ok=True)
