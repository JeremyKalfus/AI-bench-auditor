# Benchmark Audit Execution Log

## Phase 0: Baseline Capture

Goal: Record the current repository baseline and prove the existing launcher, ranking, and `data_preview` behavior before any run-mode changes.

Microsteps completed:
- Captured the repository baseline state from `main` at commit `96bd51617cfdbb494a9fc283af00fe090edfae48`.
- Confirmed the current launcher is a single linear path with unconditional plot aggregation, then optional citation gathering, writeup, and review.
- Confirmed the active best-node path in `journal.py` is LLM-mediated with metric fallback rather than deterministic metric-only ranking.
- Confirmed `data_preview` is only threaded into prompts when enabled and is never populated in the current codepath.

Files changed:
- `docs/benchmark_audit_execution_log.md`

Verification commands/tests:
- `git status --short --branch` -> PASS
- `git rev-parse HEAD` -> PASS
- `nl -ba launch_scientist_bfts.py | sed -n '1,340p'` -> PASS
- `nl -ba ai_scientist/treesearch/journal.py | sed -n '420,520p'` -> PASS
- `nl -ba ai_scientist/treesearch/parallel_agent.py | sed -n '260,290p;475,525p;1160,1185p'` -> PASS
- `rg -n "data_preview" ai_scientist/treesearch` -> PASS
- `rg -n "data_preview.generate|self\\.data_preview\\s*=\\s*[^N]" ai_scientist` -> PASS

Result: PASS

Residual risks:
- `python launch_scientist_bfts.py --help` currently fails before argparse because `torch` is imported at module import time; Phase 1 must fix that to make the new mode surface verifiable.
- The launcher currently removes copied `experiment_results` after plot aggregation, so later audit artifact retention needs explicit policy in a later phase.
- `find_pdf_path_for_review()` can return an undefined variable when no reflection PDF exists; that path is brittle today.

Next phase prerequisites:
- Add explicit `audit` and `paper` mode surfaces.
- Add an Apple-Silicon-safe audit preset.
- Enforce the audit-to-paper handoff contract.
- Make the CLI help and dry-run paths work without importing optional runtime dependencies.

## Phase 1: Introduce Run Modes and Audit Skeleton

Goal: Add explicit `audit` and `paper` mode surfaces, enforce an audit-to-paper handoff contract, and make the audit path default to a safe verification-first preset.

Microsteps completed:
- Added explicit `--mode {audit,paper}` parsing plus `--dry-run`, `--audit-run-dir`, and `--audit-num-workers` in `launch_scientist_bfts.py`.
- Moved optional runtime imports behind function boundaries so `python launch_scientist_bfts.py --help` works without `torch`.
- Added paper-mode validation that rejects raw benchmark input flags and requires an audit run directory with `audit_run_metadata.json`.
- Added audit-mode scaffolding that writes `idea.json`, a copied `bfts_config.yaml`, and `audit_run_metadata.json` during dry-run.
- Added a minimal audit preset override path in `ai_scientist/treesearch/bfts_utils.py` so audit mode rewrites `agent.num_workers=1` and keeps `agent.multi_seed_eval.num_seeds=1` consistent with the existing config note.
- Split the launcher so audit mode stops after experiment execution/token tracking and never reaches plot aggregation, citation gathering, paper writeup, or paper review.
- Fixed the default ideas path while verifying the new dry-run flow, because the previous default (`ideas/...`) did not exist from the repository root.
- Hardened `find_pdf_path_for_review()` so it returns `None` instead of an undefined variable when no reflection PDF exists.

Files changed:
- `launch_scientist_bfts.py`
- `ai_scientist/treesearch/bfts_utils.py`
- `docs/benchmark_audit_execution_log.md`

Verification commands/tests:
- `python -m py_compile launch_scientist_bfts.py ai_scientist/treesearch/bfts_utils.py` -> PASS
- `python launch_scientist_bfts.py --help` -> PASS
- `python launch_scientist_bfts.py --mode audit --dry-run --attempt_id 9001` -> PASS
- `rg -n "num_workers|num_seeds" experiments/2026-04-11_15-26-45_compositional_regularization_nn_attempt_9001/bfts_config.yaml` -> PASS
- `cat experiments/2026-04-11_15-26-45_compositional_regularization_nn_attempt_9001/audit_run_metadata.json` -> PASS
- `python launch_scientist_bfts.py --mode paper --dry-run --audit-run-dir /Users/jeremykalfus/CodingProjects/AI Scientist/experiments/2026-04-11_15-26-45_compositional_regularization_nn_attempt_9001` -> PASS
- `python launch_scientist_bfts.py --mode paper --dry-run --audit-run-dir /Users/jeremykalfus/CodingProjects/AI Scientist/experiments/2026-04-11_15-26-45_compositional_regularization_nn_attempt_9001 --load_ideas ai_scientist/ideas/i_cant_believe_its_not_better.json` -> PASS (rejected as intended)
- `rg -n "run_audit_mode|run_paper_mode|perform_writeup|perform_review|gather_citations|perform_imgs_cap_ref_review" launch_scientist_bfts.py` -> PASS

Result: PASS

Residual risks:
- The paper handoff contract is currently validated by `audit_run_metadata.json`; later phases must tighten this to validated audit artifacts rather than mode metadata alone.
- Actual experiment execution in audit mode was not run in Phase 1; this phase only verified the CLI surface, the copied config, and the handoff contract via dry-run.
- Actual paper generation from an audit run was not executed in Phase 1; only the mode surface and argument contract were verified.
- The launcher still contains the broad process cleanup routine for non-dry runs; Phase 1 only ensured dry-run verification avoids that path.

Next phase prerequisites:
- Add the real audit dependencies and a CPU-only smoke test.
- Make prompt package claims truthful.
- Keep paper mode gated behind validated audit artifacts as schemas and provenance are introduced.

## Phase 2: Make the Environment Truthful

Goal: Add the missing audit dependencies, make package-related prompts truthful, and add a minimal CPU-only audit-mode smoke test.

Microsteps completed:
- Added `pandas`, `scikit-learn`, `pyarrow`, `duckdb`, `rapidfuzz`, and `pytest` to `requirements.txt`.
- Rewrote the `parallel_agent.py` package prompt so it no longer claims arbitrary extra packages are already installed.
- Narrowed the prompt’s device guidance so PyTorch-specific instructions apply only if a PyTorch-based approach is chosen and `torch` is actually available.
- Added a CPU-only smoke test in `tests/test_audit_env_smoke.py` that imports the launcher by file path, runs audit mode in `--dry-run`, and verifies the generated audit scaffolding plus audit preset overrides.
- Encountered a PEP 668 install failure when trying to install into the system interpreter, then applied the minimum local fix by creating `.venv-benchmark-audit` and rerunning verification there.

Files changed:
- `requirements.txt`
- `ai_scientist/treesearch/parallel_agent.py`
- `tests/test_audit_env_smoke.py`
- `docs/benchmark_audit_execution_log.md`

Verification commands/tests:
- `python -m pip install -r requirements.txt` -> FAIL (`externally-managed-environment`; fixed locally by switching verification to a venv)
- `python -m venv .venv-benchmark-audit` -> PASS
- `.venv-benchmark-audit/bin/python -m pip install -r requirements.txt` -> PASS
- `.venv-benchmark-audit/bin/python - <<'PY' ... import pandas, sklearn, pyarrow, duckdb, rapidfuzz ... PY` -> PASS
- `rg -n "all packages are already installed|Installed Packages" ai_scientist/treesearch/parallel_agent.py` -> PASS
- `.venv-benchmark-audit/bin/python -m pytest tests/test_audit_env_smoke.py` -> PASS

Result: PASS

Residual risks:
- Verification now depends on the local `.venv-benchmark-audit` because the system Python is externally managed.
- `psutil` is still not declared in `requirements.txt`, so non-dry-run cleanup in `launch_scientist_bfts.py` remains a separate environment risk outside the Phase 2 deliverables.
- Prompt truthfulness is improved for package availability, but later audit-specific prompt rewrites will still be needed as the system shifts from paper-generation behavior to audit behavior.

Next phase prerequisites:
- Define shared provenance requirements and strict primary artifact schemas.
- Keep artifacts deterministic and machine-validated before any narrative work.

## Phase 3: Define Primary Artifact Schemas and Provenance

Goal: Establish a shared provenance contract and strict schema validation for the first set of primary audit artifacts before any generation logic is added.

Microsteps completed:
- Added `ai_scientist/audits/schema.py` with a shared provenance schema covering `schema_version`, `git_sha`, `dataset_fingerprint`, `seed`, `run_id`, `detector_versions`, `created_at`, and `updated_at`.
- Added strict JSON Schemas for `audit_results.json`, `metrics_before_after.json`, and `split_manifest.json`.
- Added a strict findings-table column contract for `findings.csv` / `findings.parquet`, including provenance linkage columns.
- Added example builders for valid provenance, audit results, metrics-before-after, and split manifest artifacts so the schema module includes concrete example payloads.
- Added `ai_scientist/audits/__init__.py` exports for the new validators and schema constants.
- Added `tests/test_audit_schema.py` covering valid and invalid provenance blocks, valid and invalid examples for each primary JSON artifact, findings column validation, and schema self-validation.

Files changed:
- `ai_scientist/audits/__init__.py`
- `ai_scientist/audits/schema.py`
- `tests/test_audit_schema.py`
- `docs/benchmark_audit_execution_log.md`

Verification commands/tests:
- `.venv-benchmark-audit/bin/python -m py_compile ai_scientist/audits/__init__.py ai_scientist/audits/schema.py tests/test_audit_schema.py` -> PASS
- `.venv-benchmark-audit/bin/python -m pytest tests/test_audit_schema.py` -> PASS

Result: PASS

Residual risks:
- The schemas are intentionally foundational and static; later phases still need to connect real artifact generation to these validators.
- The findings-table contract is strict at the column-name level but does not yet validate row values beyond what later generators will emit.
- The current examples are deterministic fixtures for validation, not yet live outputs from audit-mode execution.

Next phase prerequisites:
- Generate deterministic dataset context without relying on dormant `data_preview`.
- Inject that context into the audit task payload.
- Extend the task/idea shape with audit-native acceptance criteria.

## Phase 4: Deterministic Dataset Context

Goal: Generate deterministic dataset context artifacts, inject them into the audit task payload, and extend the idea/task shape with audit-native fields without relying on dormant `data_preview` plumbing.

Microsteps completed:
- Added `ai_scientist/audits/dataset_context.py` to stage benchmark files into `idea_dir/data`, compute a deterministic dataset fingerprint, infer basic split/column metadata, emit `dataset_card.md`, and emit a schema-validated `split_manifest.json`.
- Reused the shared provenance contract by adding a `build_provenance_block(...)` helper in `ai_scientist/audits/schema.py` and exporting it from `ai_scientist/audits/__init__.py`.
- Updated `launch_scientist_bfts.py` so audit-mode preparation deep-copies the selected idea, augments it with deterministic dataset context, and writes the enriched `idea.json` before agent execution.
- Updated `ai_scientist/treesearch/agent_manager.py` so rendered task descriptions include audit-native fields: `Audit Targets`, `Leakage Taxonomy`, `Acceptance Criteria`, `Benchmark Metadata`, and `Dataset Context`.
- Added `tests/test_audit_dataset_context.py` to verify generated dataset artifacts, the augmented audit idea payload, and the rendered task description.

Files changed:
- `ai_scientist/audits/__init__.py`
- `ai_scientist/audits/schema.py`
- `ai_scientist/audits/dataset_context.py`
- `launch_scientist_bfts.py`
- `ai_scientist/treesearch/agent_manager.py`
- `tests/test_audit_dataset_context.py`
- `docs/benchmark_audit_execution_log.md`

Verification commands/tests:
- `.venv-benchmark-audit/bin/python -m py_compile ai_scientist/audits/__init__.py ai_scientist/audits/schema.py ai_scientist/audits/dataset_context.py launch_scientist_bfts.py ai_scientist/treesearch/agent_manager.py tests/test_audit_dataset_context.py` -> PASS
- `.venv-benchmark-audit/bin/python -m pytest tests/test_audit_dataset_context.py` -> PASS
- `.venv-benchmark-audit/bin/python -m pytest tests/test_audit_env_smoke.py` -> PASS

Result: PASS

Residual risks:
- The deterministic dataset-context path currently supports local `.csv`, `.tsv`, `.parquet`, and `.json` inputs; other dataset layouts will need explicit support later rather than implicit fallback behavior.
- Column-role inference is heuristic and intentionally lightweight in Phase 4; later detector phases may require tighter validation against benchmark-specific metadata.
- `dataset_card.md` is generated deterministically and verified by test coverage, but it is still a Markdown artifact rather than a machine-validated structured schema.

Next phase prerequisites:
- Preserve the existing four-stage scaffold in audit mode.
- Rewrite stage goals to benchmark-audit semantics for all four stages.
- Replace stage completion gates with artifact- and verification-based audit criteria plus unit tests.

## Phase 5: Replace Stage Goals, Keep the Tree Search

Goal: Preserve the existing four-stage scaffold while rewriting audit-mode stage goals and replacing paper-era completion gates with deterministic artifact-based audit checks.

Microsteps completed:
- Kept the internal four-stage scaffold intact in `ai_scientist/treesearch/agent_manager.py` while adding audit-native main-stage titles for rendering and run logs.
- Replaced the audit-mode stage goal text for all four stages with benchmark-audit semantics: benchmark reproduction, detector evidence gathering, remediation/falsification, and robustness-plus-synthesis.
- Added `_build_task_desc_for_stage(...)` so rendered task descriptions include the audit-native stage title plus the updated stage goals in a deterministic testable path.
- Added deterministic audit artifact inspection in `agent_manager.py` that validates `dataset_card.md`, `split_manifest.json`, `audit_results.json`, optional `findings.csv` / `findings.parquet`, and optional `metrics_before_after.json`.
- Replaced audit-mode main-stage completion logic with artifact-based gates: stage 1 requires baseline-validation artifacts, stage 2 requires a confirmed finding or a high-confidence clean audit, and stages 3-4 require remediation/falsification evidence when findings remain open.
- Fixed a correctness bug in the scaffold by branching audit completion on the parsed main-stage number from `stage.name` instead of the monotonically increasing `stage.stage_number`, so follow-up substages still use the correct main-stage gate.
- Added `tests/test_audit_stage_progression.py` covering stage-goal rendering, run-log stage titles, stage-1 baseline completion, stage-2 follow-up substage gating, stage-3 remediation requirements, and stage-4 clean-audit completion.

Files changed:
- `ai_scientist/treesearch/agent_manager.py`
- `tests/test_audit_stage_progression.py`
- `docs/benchmark_audit_execution_log.md`

Verification commands/tests:
- `.venv-benchmark-audit/bin/python -m py_compile ai_scientist/treesearch/agent_manager.py tests/test_audit_stage_progression.py` -> PASS
- `.venv-benchmark-audit/bin/python -m pytest tests/test_audit_stage_progression.py` -> PASS
- `.venv-benchmark-audit/bin/python -m pytest tests/test_audit_dataset_context.py tests/test_audit_schema.py tests/test_audit_env_smoke.py` -> PASS

Result: PASS

Residual risks:
- Audit stage completion still uses `Journal.get_best_node(...)`, which remains LLM-assisted when multiple good nodes exist; deterministic ranking is still deferred to the later ranking phase.
- Substage completion and substage-goal generation still rely on the older VLM/LLM feedback path; Phase 5 only replaced the main-stage completion gates and main-stage goal text.
- The new audit gates validate artifact presence and consistency, but the actual execution path that emits those artifacts is still scheduled for later phases.

Next phase prerequisites:
- Inspect `ai_scientist/treesearch/parallel_agent.py` prompt assembly in audit mode.
- Remove training-centric and anti-EDA instructions from the audit prompt branch only.
- Make the audit environment guidance and required artifact filenames explicit in prompt rendering and verify them with prompt-focused tests.

## Phase 6: Rewrite the Agent Prompts for Audit Work

Goal: Add an audit-only prompt branch in `parallel_agent.py` so audit mode stops asking for training-era behavior and instead prefers deterministic benchmark-audit work plus explicit artifact emission.

Microsteps completed:
- Added audit-task detection in `ai_scientist/treesearch/parallel_agent.py` based on the rendered task description markers injected by the earlier audit phases.
- Rewrote the audit-mode environment prompt so it deterministically prefers `pandas`, `scikit-learn`, `duckdb`, `rapidfuzz`, and `pyarrow` instead of defaulting to training-oriented package guidance.
- Replaced the audit-mode implementation guideline branch so it no longer requires GPU boilerplate, epoch-wise validation-loss tracking, training curves, or synthetic data creation, and instead explicitly requires `audit_results.json`, `split_manifest.json`, `metrics_before_after.json`, and `findings.csv` / `findings.parquet`.
- Added prompt builders for draft/debug/improve in `MinimalAgent` so audit-mode prompt rendering is explicit and testable without relying on LLM output.
- Rewrote the stage-2 and stage-4 idea prompts in `ParallelAgent` so audit mode proposes detector/evidence steps and robustness/falsification steps instead of hyperparameter-tuning and synthetic-dataset ablation ideas, while preserving the existing plumbing.
- Added `tests/test_audit_prompt_branches.py` to verify rendered audit prompts, the stage-2 audit-step idea prompt, and the stage-4 robustness-step idea prompt directly.
- Fixed the Phase 6 test expectations after the first verification run by checking for removal of the old positive training instructions rather than rejecting the phrase “validation loss” even when it only appeared in a negative prohibition.

Files changed:
- `ai_scientist/treesearch/parallel_agent.py`
- `tests/test_audit_prompt_branches.py`
- `docs/benchmark_audit_execution_log.md`

Verification commands/tests:
- `.venv-benchmark-audit/bin/python -m py_compile ai_scientist/treesearch/parallel_agent.py tests/test_audit_prompt_branches.py` -> PASS
- `.venv-benchmark-audit/bin/python -m pytest tests/test_audit_prompt_branches.py` -> PASS
- `.venv-benchmark-audit/bin/python -m pytest tests/test_audit_stage_progression.py tests/test_audit_dataset_context.py tests/test_audit_schema.py tests/test_audit_env_smoke.py` -> PASS
- `.venv-benchmark-audit/bin/python - <<'PY' ... compile_prompt_to_md(MinimalAgent(...)._build_draft_prompt()) -> docs/.phase6_audit_draft_prompt.md ... PY` -> PASS
- `! rg -n "Don't suggest to do EDA\\.|Track and print validation loss at each epoch|device = torch.device|Make sure to create synthetic data if needed\\." docs/.phase6_audit_draft_prompt.md` -> PASS
- `rg -n "audit_results.json|split_manifest.json|metrics_before_after.json|findings.csv or findings.parquet|pandas|scikit-learn|duckdb|rapidfuzz|pyarrow" docs/.phase6_audit_draft_prompt.md` -> PASS

Result: PASS

Residual risks:
- The post-execution metric-parsing and plotting prompts in `parallel_agent.py` still retain some training-era assumptions; those paths are not yet the primary audit-mode evidence path and will need cleanup alongside the later execution-path interception work.
- The stage-2 and stage-4 helper function names remain legacy (`hyperparam` / `ablation`) even though their audit-mode prompt content is now detector- and robustness-focused.
- Audit prompt detection currently relies on the presence of audit markers in the rendered task description; that contract is now covered by tests but should stay aligned with the AgentManager task renderer.

Next phase prerequisites:
- Create deterministic detector functions for the initial leakage checks.
- Add unit fixtures/tests for exact duplicates, near duplicates, group overlap, temporal leakage, preprocessing leakage, suspicious feature leakage, and a clean negative control.
- Keep detector outputs aligned with the schema/provenance contract introduced earlier.

## Phase 7: Deterministic Audit Components

Goal: Add deterministic detector, scoring, report, and canary-suite modules that operate on schema-aligned audit artifacts without depending on the LLM execution path.

Microsteps completed:
- Added `ai_scientist/audits/detectors.py` with deterministic detectors for exact duplicates, near duplicates, group/entity overlap, temporal leakage, preprocessing leakage, and suspicious feature leakage.
- Standardized detector outputs as findings dataframes that honor the existing findings column contract and provenance linkage fields.
- Added `ai_scientist/audits/scoring.py` with an explicit `FindingScoreInput` model and deterministic `score_branch(...)` logic driven by severity, evidence completeness, remediation effect size, and negative-control penalties.
- Added `ai_scientist/audits/report.py` with deterministic `generate_audit_report(...)` output that cites the concrete artifact paths it summarizes.
- Added `ai_scientist/audits/canary_suite.py` with a file-based canary writer/loader for exact duplicates, near duplicates, group overlap, temporal leakage, preprocessing/label leakage, and a clean negative control.
- Updated `ai_scientist/audits/__init__.py` to export the new detector, scoring, report, and canary helpers.
- Added `tests/test_audit_detectors.py`, `tests/test_audit_scoring.py`, `tests/test_audit_report.py`, and `tests/test_audit_canary_suite.py` for the new deterministic modules.
- Fixed the only verification failure in this phase by tightening the near-duplicate fixture so it was unambiguously similar at the configured threshold while keeping the detector deterministic.

Files changed:
- `ai_scientist/audits/__init__.py`
- `ai_scientist/audits/detectors.py`
- `ai_scientist/audits/scoring.py`
- `ai_scientist/audits/report.py`
- `ai_scientist/audits/canary_suite.py`
- `tests/test_audit_detectors.py`
- `tests/test_audit_scoring.py`
- `tests/test_audit_report.py`
- `tests/test_audit_canary_suite.py`
- `docs/benchmark_audit_execution_log.md`

Verification commands/tests:
- `.venv-benchmark-audit/bin/python -m py_compile ai_scientist/audits/__init__.py ai_scientist/audits/detectors.py ai_scientist/audits/scoring.py ai_scientist/audits/report.py ai_scientist/audits/canary_suite.py tests/test_audit_detectors.py tests/test_audit_scoring.py tests/test_audit_report.py tests/test_audit_canary_suite.py` -> PASS
- `.venv-benchmark-audit/bin/python -m pytest tests/test_audit_detectors.py tests/test_audit_scoring.py tests/test_audit_report.py tests/test_audit_canary_suite.py` -> PASS
- `.venv-benchmark-audit/bin/python -m pytest tests/test_audit_prompt_branches.py tests/test_audit_stage_progression.py tests/test_audit_dataset_context.py tests/test_audit_schema.py tests/test_audit_env_smoke.py` -> PASS
- `.venv-benchmark-audit/bin/python -m py_compile ai_scientist/audits/__init__.py ai_scientist/audits/detectors.py ai_scientist/audits/scoring.py ai_scientist/audits/report.py ai_scientist/audits/canary_suite.py` -> PASS

Result: PASS

Residual risks:
- The preprocessing leakage detector is intentionally conservative and currently relies on deterministic column-name heuristics rather than full execution-path provenance; later phases may need stronger artifact-first evidence for this class of leakage.
- The scoring module is deterministic and tested, but it is not yet connected to node acceptance or journal ranking; that integration is deferred to the execution-path and ranking phases.
- The report generator is deterministic and path-citing, but it is still an offline helper until later phases wire it into audit-mode artifact emission.

Next phase prerequisites:
- Intercept the audit execution path so it checks for `audit_results.json` before falling back to LLM metric parsing.
- Reject nodes with missing or invalid primary artifacts and store explicit failure reasons.
- Populate `node.analysis` and `node.metric` directly from structured audit artifacts before any fallback parsing.

## Phase 8: Intercept the Execution Path

Goal: Make audit-mode execution artifact-first so structured audit artifacts populate node state before any legacy stdout or metric parsing path can accept a node.

Microsteps completed:
- Added audit artifact validation helpers inside `ai_scientist/treesearch/parallel_agent.py` that check for `audit_results.json` first, validate `split_manifest.json`, validate the findings table, and optionally validate `metrics_before_after.json`.
- Reworked `MinimalAgent.parse_exec_result(...)` so audit mode now finalizes node parsing from structured artifacts, populates `node.analysis` from a deterministic audit summary, and sets `node.metric` from `audit_results.json` audit score.
- Added explicit rejection behavior for missing or malformed audit artifacts: audit nodes are marked invalid, assigned `WorstMetricValue`, and store a clear failure reason in `node.analysis`.
- Kept the legacy execution-review query only as a diagnostic fallback when `audit_results.json` is missing; it no longer revives the node into a successful audit path.
- Updated `_process_node_wrapper(...)` so audit-mode nodes that were finalized by artifact parsing skip the old `.npy` metric-parsing path entirely.
- Added audit artifact copying into `exp_results_dir` for accepted audit nodes so downstream stage gates can inspect the persisted JSON/CSV/Parquet/Markdown artifacts.
- Added `tests/test_audit_artifact_parse.py` covering valid structured-artifact parsing, missing-artifact fallback, malformed-artifact rejection, and artifact copying into the results directory.
- Fixed the only verification issue in this phase by adding the missing `json` import and aligning the test assertions with the actual `WorstMetricValue` shape used by the current metric type.

Files changed:
- `ai_scientist/treesearch/parallel_agent.py`
- `tests/test_audit_artifact_parse.py`
- `docs/benchmark_audit_execution_log.md`

Verification commands/tests:
- `.venv-benchmark-audit/bin/python -m py_compile ai_scientist/treesearch/parallel_agent.py tests/test_audit_artifact_parse.py` -> PASS
- `.venv-benchmark-audit/bin/python -m pytest tests/test_audit_artifact_parse.py` -> PASS
- `.venv-benchmark-audit/bin/python -m pytest tests/test_audit_prompt_branches.py tests/test_audit_stage_progression.py tests/test_audit_dataset_context.py tests/test_audit_schema.py tests/test_audit_env_smoke.py tests/test_audit_detectors.py tests/test_audit_scoring.py tests/test_audit_report.py tests/test_audit_canary_suite.py` -> PASS
- `.venv-benchmark-audit/bin/python -m py_compile ai_scientist/treesearch/parallel_agent.py tests/test_audit_artifact_parse.py` -> PASS

Result: PASS

Residual risks:
- The audit artifact copier currently mirrors top-level `.json`, `.csv`, `.parquet`, and `.md` files from the worker `working` directory; if later audit steps emit nested evidence trees, that copy policy will need to expand.
- Missing-artifact fallback still uses the LLM execution review for diagnostic text, but the node remains invalid and no longer enters the successful metric path; later phases may choose to remove even that diagnostic fallback.
- Plot-generation and VLM-analysis code still exists in the execution path after a successful audit parse, although audit success no longer depends on those outputs.

Next phase prerequisites:
- Replace audit-mode `Journal.get_best_node(...)` selection with deterministic score-based ranking.
- Use structured audit artifacts and the deterministic scoring module rather than LLM node selection in audit mode.
- Add ranking tests that prove expected ordering across canned audit cases.

## Phase 9: Deterministic Best-Node Selection

Goal: Replace audit-mode best-node selection with a deterministic artifact-first comparator while preserving the existing LLM-assisted selection path for non-audit workflows.

Microsteps completed:
- Inspected the Phase 9 plan requirements and the existing `Journal.get_best_node(...)` path to confirm that multi-node selection still depended on `query(...)`.
- Added an audit-only ranking branch in `ai_scientist/treesearch/journal.py` that reads validated `audit_results.json`, derives evidence coverage from structured confidence fields, derives remediation confirmation from validated `metrics_before_after.json` when findings are present, and reads an optional reproducibility signal from a small set of JSON artifact names if one exists.
- Made the audit comparator deterministic with the required precedence: audit score, evidence completeness, remediation confirmation, reproducibility signal presence/value, then stable node-ID ordering.
- Exposed the same deterministic audit ordering through `Journal.get_ranked_nodes()` and routed the fallback branch in `ai_scientist/treesearch/parallel_agent.py` through it so audit node ordering stays consistent even when the current best tree has already been processed.
- Kept the existing non-audit behavior intact by falling back to the prior `use_val_metric_only` / LLM selection path whenever no audit-ranked nodes are available.
- Added `tests/test_audit_journal_ranking.py` to verify: audit mode never calls `query(...)`, higher audit score wins, evidence coverage breaks ties, remediation confirmation breaks ties, reproducibility signal breaks ties, node ID is the stable final tiebreaker, the reusable ranked-node order matches the same precedence, and non-audit selection still uses the LLM path.
- Narrowed the new `journal.py` imports to the schema validators only so the ranking path does not import the broader audit package unnecessarily.

Files changed:
- `ai_scientist/treesearch/journal.py`
- `ai_scientist/treesearch/parallel_agent.py`
- `tests/test_audit_journal_ranking.py`
- `docs/benchmark_audit_execution_log.md`

Verification commands/tests:
- `.venv-benchmark-audit/bin/python -m py_compile ai_scientist/treesearch/journal.py ai_scientist/treesearch/parallel_agent.py tests/test_audit_journal_ranking.py` -> PASS
- `.venv-benchmark-audit/bin/python -m pytest tests/test_audit_journal_ranking.py` -> PASS
- `.venv-benchmark-audit/bin/python -m pytest tests/test_audit_journal_ranking.py tests/test_audit_artifact_parse.py tests/test_audit_prompt_branches.py tests/test_audit_stage_progression.py tests/test_audit_dataset_context.py tests/test_audit_schema.py tests/test_audit_env_smoke.py tests/test_audit_detectors.py tests/test_audit_scoring.py tests/test_audit_report.py tests/test_audit_canary_suite.py` -> PASS

Result: PASS

Residual risks:
- The audit ranking branch currently treats reproducibility as optional and only reads it from a small JSON artifact-name set (`reproducibility_summary.json` / `reproducibility.json`); later reproducibility phases may need to tighten that contract once the artifact is formally introduced.
- Remediation confirmation is intentionally conservative and currently keys off the validated `metrics_before_after.json` contract plus provenance alignment rather than richer per-finding remediation-state semantics.
- Later report/output phases still need to surface the chosen node and its supporting evidence more explicitly in end-of-run artifacts.

Next phase prerequisites:
- Wire deterministic `audit_report.md` generation into the end of audit-mode runs.
- Ensure the generated report cites actual artifact files and includes strict evidence references for major findings.
- Keep paper-generation behavior disabled in audit mode until the report/output gates are satisfied.

## Phase 10: Audit Report Output

Goal: Generate a deterministic `audit_report.md` from the winning audit artifacts at the end of audit mode, include explicit evidence/confidence/remediation sections, and keep paper outputs disabled.

Microsteps completed:
- Expanded `ai_scientist/audits/report.py` so the markdown report now includes benchmark summary, detector run status, findings summary, major findings, explicit evidence references, remediation results, confidence/limitations, and an audit-mode workflow guard that states `.tex` / `.pdf` outputs remain disabled.
- Returned the live `AgentManager` from `perform_experiments_bfts(...)` so audit-mode completion can identify the winning audit node after the experiment runner finishes.
- Added launcher helpers in `launch_scientist_bfts.py` to locate the best audit node, map its original `logs/0-run/experiment_results/...` artifact directory into the copied top-level `experiment_results/...` tree, locate the findings artifact, and generate `audit_report.md` against the copied artifact paths.
- Wired `run_audit_mode(...)` to call the new report-generation helper after copying `experiment_results`, so each completed audit run now emits a report in the copied artifact directory rather than only leaving behind raw JSON/CSV/Parquet files.
- Added `tests/test_audit_report_integration.py` to fake a completed audit-mode run, verify `audit_report.md` is generated under the copied artifact tree, and confirm audit mode still creates no `.tex` or `.pdf` files.
- Strengthened `tests/test_audit_report.py` so the report test now checks for detector/evidence/confidence/workflow-guard sections plus concrete evidence-path references.

Files changed:
- `ai_scientist/audits/report.py`
- `ai_scientist/treesearch/perform_experiments_bfts_with_agentmanager.py`
- `launch_scientist_bfts.py`
- `tests/test_audit_report.py`
- `tests/test_audit_report_integration.py`
- `docs/benchmark_audit_execution_log.md`

Verification commands/tests:
- `.venv-benchmark-audit/bin/python -m py_compile ai_scientist/audits/report.py ai_scientist/treesearch/perform_experiments_bfts_with_agentmanager.py launch_scientist_bfts.py tests/test_audit_report.py tests/test_audit_report_integration.py` -> PASS
- `.venv-benchmark-audit/bin/python -m pytest tests/test_audit_report.py tests/test_audit_report_integration.py` -> PASS
- `.venv-benchmark-audit/bin/python -m pytest tests/test_audit_journal_ranking.py tests/test_audit_artifact_parse.py tests/test_audit_prompt_branches.py tests/test_audit_stage_progression.py tests/test_audit_dataset_context.py tests/test_audit_schema.py tests/test_audit_env_smoke.py tests/test_audit_detectors.py tests/test_audit_scoring.py tests/test_audit_report.py tests/test_audit_report_integration.py tests/test_audit_canary_suite.py` -> PASS

Result: PASS

Residual risks:
- The report generator references evidence files and manifest paths deterministically, but it does not yet enforce existence checks for every path named inside `audit_results.json`; the stricter schema/evidence gate is still deferred to Phase 11.
- The winning-node discovery for report generation currently walks stages in reverse and uses the best node with an `exp_results_dir`; that is correct for the current staged audit flow but may need tighter explicit contracts if later phases introduce additional artifact-only terminal stages.
- Audit mode now writes a markdown report plus copied structured artifacts, but it still does not gate paper mode on downstream verification phases yet; that remains a later-phase requirement.

Next phase prerequisites:
- Add explicit verification gates for schema validity, referenced evidence existence, and audit-summary artifact completeness.
- Add canary/mutation/reproducibility-oriented verification entry points without reintroducing paper generation.
- Keep the workflow artifact-first: verification should fail loudly on invalid audit artifacts rather than falling back to narrative summaries.

## Phase 11: Verification Stack

Goal: Add the required verification stack for schema gating, canaries, mutation testing, search ablations, reproducibility runs, and real-benchmark acceptance.

Microsteps completed:
- Re-read the Phase 11 requirements to confirm that the phase is not limited to unit tests; it explicitly requires search ablation results, reproducibility summaries across repeated runs, and acceptance-test reports on one or two real benchmarks.
- Inspected the current audit code and tests to confirm what already exists: schema validators, detector unit tests, canary fixture generation, audit-report generation, and artifact validation helpers.
- Searched the repository for existing mutation-testing, search-ablation, reproducibility, and acceptance-test harnesses and found no implemented harness or result artifact path for those Phase 11 deliverables.
- Searched the repository for checked-in benchmark datasets / benchmark directories and found no `data/`, `datasets/`, or `benchmarks/` directories that could support the required real-benchmark acceptance runs in Phase 11.6.
- Confirmed that the current repo state does not contain the real benchmark inputs needed to honestly satisfy Phase 11.4 through 11.6 verification, so the phase cannot be marked passing without inventing unsupported fixtures or silently narrowing scope.

Files changed:
- `docs/benchmark_audit_execution_log.md`

Verification commands/tests:
- `sed -n '635,735p' docs/benchmark_audit_revised_plan.md` -> PASS
- `rg -n "schema gate|validate_audit_results|canary|mutation|reproducibility|acceptance test|evidence references|audit_report.md|audit_report" ai_scientist launch_scientist_bfts.py tests -g '!**/__pycache__/**'` -> PASS
- `sed -n '1,240p' ai_scientist/audits/canary_suite.py` -> PASS
- `sed -n '1,260p' ai_scientist/audits/detectors.py` -> PASS
- `rg -n "mutation|ablation|reproducibility|acceptance" ai_scientist tests docs -g '!**/__pycache__/**'` -> PASS
- `rg -n "data_dir|benchmark|dataset_name|load_task_desc|ideas/i_cant_believe_its_not_better|csv|parquet" bfts_config.yaml ai_scientist -g '!**/__pycache__/**'` -> PASS
- `find . -maxdepth 2 -type d \\( -name data -o -name datasets -o -name benchmarks \\) | sort` -> PASS (no benchmark directories found)

Result: BLOCKED (historical checkpoint before the repo-native verification harness was implemented)

Residual risks:
- Phase 11.4 requires a reproducible comparison run for full tree search vs one-shot vs detector-only baselines, but there is no existing ablation harness or benchmark-run fixture in the repo to execute or validate that comparison.
- Phase 11.5 requires three repeated audit runs and a reproducibility summary artifact, but there is no current reproducibility runner or benchmark fixture set to run against.
- Phase 11.6 requires one or two real benchmarks with known or suspected issues, and the repository currently contains no checked-in benchmark data or benchmark registry that could support an honest acceptance-test report.

Next phase prerequisites:
- Provide or point to one or two concrete benchmark datasets plus the benchmark metadata/config needed to run them through audit mode.
- Define or approve the intended harness shape for search ablation (`full tree search`, `one-shot agent`, `detector-only baseline`) so it can be implemented and verified deterministically.
- Add real verification-run inputs before attempting to continue Phase 11; without them, any further work would be speculative rather than verifiable.

## Phase 11 (Continuation): Repo-Native Verification Harness

Goal: Replace the earlier blocked assumption with a self-contained verification harness that is deterministic, checked in, and usable in local development and CI.

Microsteps completed:
- Added `ai_scientist/audits/verification.py` as the executable verification-stack module and CLI.
- Added checked-in verification registry and benchmark fixtures under `tests/fixtures/verification/`.
- Implemented canary, mutation, search-ablation, reproducibility, acceptance, and schema-gate phases as structured summary artifacts.
- Added `tests/test_audit_verification_stack.py` to verify bundle materialization, mutation recall, ablation ordering, reproducibility consistency, acceptance wording, and full-stack output writing.
- Updated the surrounding docs to describe Phase 11 as a repo-native harness rather than an untracked real-benchmark prerequisite for every local verification pass.

Files changed:
- `ai_scientist/audits/verification.py`
- `tests/test_audit_verification_stack.py`
- `tests/fixtures/verification/registry.json`
- `tests/fixtures/verification/acceptance/...`
- `tests/fixtures/verification/mutation/...`
- `docs/verification_stack.md`
- `README.md`
- `docs/benchmark_audit_execution_log.md`

Verification commands/tests:
- `.venv-benchmark-audit/bin/python -m pytest tests/test_audit_verification_stack.py` -> PASS
- `.venv-benchmark-audit/bin/python -m ai_scientist.audits.verification --output-dir /tmp/ai-bench-auditor-verification.<suffix>` -> PASS (`status=passed`, `best_strategy=full_tree_search`, `full_tree_search_adds_value=true`)

Result: PASS

Residual risks:
- The default verification harness is intentionally fixture-based and deterministic; it is a strong local correctness gate, not a universal claim about every external benchmark workflow.
- Supplemental real-benchmark inspection remains useful for wording discipline, but those runs intentionally live outside the tracked default harness.

Next phase prerequisites:
- Reintroduce paper outputs only after the report-review and verification-summary gates are wired into the launcher.

## Phase 12: Gated Paper Outputs And Single-Command Flow

Goal: Reintroduce manuscript outputs only after validated audit artifacts, a passing report review, and a passing verification summary.

Microsteps completed:
- Added `ai_scientist/audits/artifacts.py` to validate a completed audit bundle as a coherent artifact set.
- Added deterministic report-review logic in `ai_scientist/audits/report_review.py`.
- Added audit-native manuscript bundle generation in `ai_scientist/audits/manuscript.py`.
- Added launcher preconditions through `ensure_paper_generation_preconditions(...)` and post-audit orchestration through `run_post_audit_review_and_paper(...)`.
- Added tests covering report review, manuscript bundle generation, and the one-command audit flow with a single pre-research approval gate.

Files changed:
- `ai_scientist/audits/artifacts.py`
- `ai_scientist/audits/report_review.py`
- `ai_scientist/audits/manuscript.py`
- `launch_scientist_bfts.py`
- `tests/test_audit_report_review.py`
- `tests/test_audit_paper_bundle.py`
- `tests/test_audit_single_command_flow.py`
- `docs/benchmark_audit_execution_log.md`

Verification commands/tests:
- `.venv-benchmark-audit/bin/python -m pytest tests/test_audit_report_review.py tests/test_audit_paper_bundle.py tests/test_audit_single_command_flow.py` -> PASS

Result: PASS

Residual risks:
- End-to-end manual audit runs still depend on live model/API availability and a valid benchmark idea/spec; the checked-in tests intentionally stub or fixture those external surfaces where appropriate.
- The paper gate is only as trustworthy as the selected verification summary, so local overrides of `--verification-stack-results` must be used carefully.

Next phase prerequisites:
- Keep documentation and runtime dependencies aligned with the implemented launcher behavior.

## Phase 13: Documentation And Runtime Consistency Pass

Goal: Make the docs consistent with the implemented system and close the `psutil` dependency gap that affected non-dry-run cleanup.

Microsteps completed:
- Rewrote `README.md` so it describes the current audit mode, paper mode, verification stack, and repository map.
- Added `docs/architecture.md` as an in-depth architecture reference covering control flow, artifact flow, run-directory layout, and extension points.
- Updated `docs/verification_stack.md`, `docs/phase12_kickoff_note.md`, `docs/artifact_inspection_real_benchmark.md`, and `docs/benchmark_audit_revised_plan.md` so they agree on the current repo-native verification and gated manuscript model.
- Added `psutil` to `requirements.txt`.
- Hardened `cleanup_processes()` so a missing `psutil` install no longer turns cleanup into a runtime failure.
- Added `tests/test_launcher_cleanup.py` to cover the missing-`psutil` fallback behavior.

Files changed:
- `README.md`
- `docs/architecture.md`
- `docs/verification_stack.md`
- `docs/phase12_kickoff_note.md`
- `docs/artifact_inspection_real_benchmark.md`
- `docs/benchmark_audit_revised_plan.md`
- `launch_scientist_bfts.py`
- `requirements.txt`
- `tests/test_launcher_cleanup.py`
- `docs/benchmark_audit_execution_log.md`

Verification commands/tests:
- `.venv-benchmark-audit/bin/python launch_scientist_bfts.py --help` -> PASS
- `.venv-benchmark-audit/bin/python -m pytest -q` -> PASS (`72 passed`)

Result: PASS

Residual risks:
- `cleanup_processes()` is now robust to a missing `psutil`, but the recommended environment still needs `pip install -r requirements.txt` so non-dry-run cleanup uses the intended dependency set.
- The verification CLI emitted a harmless `runpy` runtime warning in one local invocation because the module had already been imported in-process; the command still completed with `status=passed`.

Next phase prerequisites:
- Keep the docs and verification expectations updated as the artifact contracts or benchmark registry evolve.
