from types import SimpleNamespace

from ai_scientist.treesearch import parallel_agent as parallel_agent_module
from ai_scientist.treesearch.backend import compile_prompt_to_md
from ai_scientist.treesearch.journal import Node
from ai_scientist.treesearch.parallel_agent import (
    MinimalAgent,
    ParallelAgent,
    _populate_process_workspace_inputs,
)


def make_audit_task_desc() -> str:
    return "\n".join(
        [
            "Title:",
            "Benchmark Leakage Audit",
            "Audit Targets:",
            "- exact_duplicate",
            "Leakage Taxonomy:",
            "- duplicate leakage",
            "Acceptance Criteria:",
            "- emit audit artifacts",
            "Benchmark Metadata:",
            '{"dataset_name": "demo-dataset"}',
            "Dataset Context:",
            "Split names: train, test",
        ]
    )


def make_cfg():
    return SimpleNamespace(
        exec=SimpleNamespace(timeout=3600),
        experiment=SimpleNamespace(num_syn_datasets=1),
        agent=SimpleNamespace(
            data_preview=False,
            k_fold_validation=1,
            code=SimpleNamespace(model="test-model", temp=0.0),
            feedback=SimpleNamespace(model="test-model", temp=0.0),
        ),
    )


def test_audit_draft_prompt_prefers_audit_stack_and_required_artifacts():
    agent = MinimalAgent(
        task_desc=make_audit_task_desc(),
        cfg=make_cfg(),
        memory_summary="Previous audit attempts found no blockers.",
        evaluation_metrics="overall audit confidence",
        stage_name="1_initial_implementation_1_preliminary",
    )

    prompt_md = compile_prompt_to_md(agent._build_draft_prompt())

    assert "Preferred Audit Stack" in prompt_md
    assert "pandas" in prompt_md
    assert "scikit-learn" in prompt_md
    assert "duckdb" in prompt_md
    assert "rapidfuzz" in prompt_md
    assert "pyarrow" in prompt_md
    assert "audit_results.json" in prompt_md
    assert "split_manifest.json" in prompt_md
    assert "metrics_before_after.json" in prompt_md
    assert "findings.csv or findings.parquet" in prompt_md
    assert "Schema inspection, split inspection" in prompt_md
    assert "validate_audit_results" in prompt_md
    assert "empty_findings_dataframe" in prompt_md
    assert "Do not invent a custom top-level schema" in prompt_md
    assert "Don't suggest to do EDA." not in prompt_md
    assert "Track and print validation loss at each epoch" not in prompt_md
    assert "torch.device" not in prompt_md
    assert "synthetic data if needed" not in prompt_md.lower()


def test_audit_debug_prompt_allows_schema_and_split_inspection():
    agent = MinimalAgent(
        task_desc=make_audit_task_desc(),
        cfg=make_cfg(),
        memory_summary=None,
        evaluation_metrics="overall audit confidence",
        stage_name="1_initial_implementation_1_preliminary",
    )
    parent_node = Node(
        code="print('broken audit')",
        _term_out=["Traceback: broken artifact writer"],
        vlm_feedback_summary="Artifacts were incomplete.",
        exec_time_feedback="",
        is_buggy=True,
    )

    prompt_md = compile_prompt_to_md(agent._build_debug_prompt(parent_node))

    assert "debugging a benchmark leakage audit implementation" in prompt_md
    assert "schemas, split boundaries, and artifact contents" in prompt_md
    assert "Don't suggest to do EDA." not in prompt_md
    assert "Track and print validation loss at each epoch" not in prompt_md
    assert "torch.device" not in prompt_md


def test_audit_stage_two_idea_prompt_is_detector_focused(monkeypatch):
    captured = {}

    def fake_query(system_message=None, user_message=None, model=None, temperature=None, func_spec=None):
        captured["prompt"] = system_message
        return (
            "AUDIT STEP NAME: Exact Duplicate Scan\n"
            "DESCRIPTION: Run a deterministic exact-duplicate detector across benchmark split boundaries."
        )

    monkeypatch.setattr(parallel_agent_module, "query", fake_query)

    agent = ParallelAgent.__new__(ParallelAgent)
    agent.task_desc = make_audit_task_desc()
    agent.cfg = make_cfg()
    agent.best_stage1_node = Node(code="print('baseline audit')")
    agent._hyperparam_tuning_state = {"tried_hyperparams": set()}
    agent.is_audit_task = True

    idea = agent._generate_hyperparam_tuning_idea()
    prompt_md = compile_prompt_to_md(captured["prompt"])

    assert idea is not None
    assert idea.name == "Exact Duplicate Scan"
    assert "deterministic detector or evidence-gathering step" in prompt_md
    assert "training longer" not in prompt_md.lower()
    assert "learning rate" not in prompt_md.lower()


def test_audit_stage_four_idea_prompt_is_robustness_focused(monkeypatch):
    captured = {}

    def fake_query(system_message=None, user_message=None, model=None, temperature=None, func_spec=None):
        captured["prompt"] = system_message
        return (
            "ROBUSTNESS CHECK NAME: Temporal Negative Control\n"
            "DESCRIPTION: Tighten the temporal split boundary and check whether the reported leakage signal survives."
        )

    monkeypatch.setattr(parallel_agent_module, "query", fake_query)

    agent = ParallelAgent.__new__(ParallelAgent)
    agent.task_desc = make_audit_task_desc()
    agent.cfg = make_cfg()
    agent.best_stage3_node = Node(code="print('stage3 audit')")
    agent._ablation_state = {"completed_ablations": set()}
    agent.is_audit_task = True

    idea = agent._generate_ablation_idea()
    prompt_md = compile_prompt_to_md(captured["prompt"])

    assert idea is not None
    assert idea.name == "Temporal Negative Control"
    assert "robustness, falsification, or negative-control step" in prompt_md
    assert "multiple synthetic datasets" not in prompt_md.lower()


def test_populate_process_workspace_inputs_copies_benchmark_context_only(tmp_path):
    source_dir = tmp_path / "source"
    source_dir_2 = tmp_path / "source-2"
    workspace_dir = tmp_path / "workspace"
    source_dir.mkdir()
    source_dir_2.mkdir()
    workspace_dir.mkdir()

    (source_dir / "dataset_card.md").write_text("# Dataset Card\n")
    (source_dir / "audit_results.json").write_text("{}")
    data_dir = source_dir_2 / "data"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text("x\n1\n")

    _populate_process_workspace_inputs([source_dir, source_dir_2], workspace_dir)

    assert (workspace_dir / "dataset_card.md").exists()
    assert (workspace_dir / "data" / "train.csv").exists()
    assert not (workspace_dir / "audit_results.json").exists()
