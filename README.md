# AI-bench-auditor

AI-bench-auditor is a benchmark leakage auditing workflow built on top of the AI Scientist v2 codebase and its agentic tree-search spine. The product surface is audit-first: it prepares deterministic dataset context, generates a research plan, enforces a human approval gate before research begins, prefers structured audit artifacts over prose, reviews the final audit report automatically, and packages the validated run as a markdown-first study bundle for LLM or human consumption.

The repository still uses the internal Python package path `ai_scientist` for compatibility with the upstream codebase. The intended user-facing workflow, CLI, and documentation in this repository are for AI-bench-auditor.

## Current Status

The tracked repository currently implements:

- `audit` mode for benchmark-input preparation, dataset-context staging, plan review, tree-search execution, artifact-first validation, audit-report generation, report review, and study-bundle generation.
- `study` mode that only consumes a prepared audit run directory and refuses raw benchmark input.
- deterministic audit ranking in the tree-search journal based on structured artifacts rather than LLM selection.
- a repo-native verification stack with schema gating, canaries, mutation tests, search ablations, reproducibility checks, and acceptance checks over checked-in verification fixtures.
- a markdown-first study bundle that writes `study_report.md`, `study_bundle_manifest.json`, `study_figures/`, and `study_figures.zip`.

The repository does not claim that every external benchmark workflow has already been empirically validated end to end. The checked-in verification harness is deliberately small and deterministic; real external benchmark staging and generated outputs should remain outside the tracked source tree.

## What The System Does

The main audit workflow is:

1. Read a benchmark idea/spec JSON file.
2. Stage benchmark files and generate deterministic dataset context.
3. Write `research_plan.json` and `research_plan.md`.
4. Pause for the required human plan-review gate unless review is explicitly skipped or pre-approved.
5. Run the adapted four-stage tree-search scaffold in audit mode.
6. Accept or reject branch results from structured audit artifacts, not just terminal text.
7. Choose the best audit branch deterministically.
8. Generate `audit_report.md` from the validated artifact bundle.
9. Run an automated report review against the artifact bundle.
10. Build the markdown-first study bundle for downstream LLM or human review.

The system is intentionally conservative:

- It does not allow research to begin when plan review is required and approval is missing.
- It does not allow study-bundle generation from invalid audit artifacts or a failing report review.
- It does not fabricate figures or artifact-backed claims.
- It prefers deterministic artifacts over LLM prose whenever both exist.

## Installation

This workflow is Python-first and was patched with CPU-safe audit defaults in mind. A local virtual environment is strongly recommended.

```bash
python -m venv .venv-benchmark-audit
source .venv-benchmark-audit/bin/activate
pip install -r requirements.txt
```

`requirements.txt` now includes `psutil`, which is used by the launcher for non-dry-run process cleanup.

Model and API keys depend on which LLM backends you use. Typical environment variables include:

```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
export S2_API_KEY="YOUR_SEMANTIC_SCHOLAR_KEY"
```

Notes:

- A real manual audit run still requires a working model backend and a valid benchmark idea/spec.
- The checked-in tests use deterministic fixtures and targeted monkeypatching where external models would otherwise be required.

## Run Modes

### Audit Mode

`audit` mode consumes a raw benchmark idea/spec JSON file and produces a run directory containing dataset context, plan-review artifacts, copied experiment results, promoted audit artifacts, and the final study bundle.

Example:

```bash
.venv-benchmark-audit/bin/python launch_scientist_bfts.py \
  --mode audit \
  --benchmark path/to/benchmark_spec.json \
  --output_dir path/to/run_dir \
  --plan-review required
```

Useful audit flags:

- `--plan-review {required,skip}`
- `--plan-review-mode {interactive,file}`
- `--plan-feedback-file PATH`
- `--plan-approval-file PATH`
- `--approve-plan`
- `--max-plan-revisions N`
- `--emit-study-zip` or `--no-emit-study-zip`

Behavioral contract:

- `--plan-review required` is the default in audit mode.
- In interactive terminals, the launcher can pause for approval, requested changes, or abort.
- In non-interactive settings, approval must come from explicit approval flags or files.
- Once the plan is approved, the rest of the run proceeds automatically.

### Study Mode

`study` mode consumes a previously prepared audit run directory. It refuses raw benchmark input and reruns the report-review plus study-bundle stages against the resolved audit artifact bundle.

Example:

```bash
.venv-benchmark-audit/bin/python launch_scientist_bfts.py \
  --mode study \
  --audit-run-dir path/to/run_dir
```

## Expected Run Outputs

A successful audit-mode run should leave behind at least:

- `idea.json`
- `bfts_config.yaml`
- `audit_run_metadata.json`
- `dataset_card.md`
- `research_plan.json`
- `research_plan.md`
- `plan_review_state.json`
- `plan_approval.json`
- `audit_results.json`
- `split_manifest.json`
- `findings.csv` or `findings.parquet`
- `metrics_before_after.json` when remediation or falsification is required
- `audit_report.md`
- `audit_report_review.json`
- `audit_report_review.md`
- `experiment_results/`
- `evidence/` when evidence files exist
- `study_report.md`
- `study_bundle_manifest.json`
- `study_figures/`
- `study_figures.zip` when zip emission is enabled

Top-level promoted artifacts may be symlinked or copied from the winning bundle under `experiment_results/`.

## Verification And Testing

Run the full local test suite:

```bash
.venv-benchmark-audit/bin/python -m pytest -q
```

Run the deterministic verification stack:

```bash
.venv-benchmark-audit/bin/python -m ai_scientist.audits.verification \
  --output-dir verification_results/latest
```

Scout public benchmark candidates:

```bash
.venv-benchmark-audit/bin/python -m ai_scientist.discover_benchmarks \
  --topic "reasoning LLM benchmarks" \
  --output-dir benchmark_discovery/reasoning_llms
```

This discovery tool searches broadly across Semantic Scholar and Hugging Face datasets, writes
`benchmark_discovery_results.json` plus `benchmark_discovery_report.md`, and emits draft
single-candidate spec files under `specs/`. Those specs are intentionally review-first: they keep
`Benchmark Metadata.files` empty until you stage local benchmark files for a real audit run.

What the verification stack covers:

- schema gating for generated audit bundles and summary artifacts
- canary execution over deterministic leakage fixtures
- mutation testing from a clean benchmark base
- search ablation across `detector_only`, `one_shot_agent`, and `full_tree_search`
- reproducibility checks across repeated `full_tree_search` runs
- acceptance checks over checked-in verification benchmarks

The verification stack remains a repo-native correctness signal for the audit workflow and its checked-in fixtures. It is no longer a LaTeX/paper gate because the final product surface is the study bundle.

## Repository Map

Key implementation areas:

- `launch_scientist_bfts.py`: top-level CLI, run-mode validation, plan-review orchestration, artifact promotion, report review, and study-bundle handoff.
- `ai_scientist/audits/`: audit-native schemas, dataset context, research-plan generation, plan-review logic, artifact validation, detectors, scoring, reporting, report review, study-bundle generation, and verification stack.
- `ai_scientist/treesearch/`: reused tree-search engine adapted for audit prompts, artifact-first execution parsing, audit-stage completion, and deterministic audit ranking.
- `tests/`: fixture-backed unit and integration coverage for the audit path, verification stack, report review, study bundle, and single-command flow.
- `tests/fixtures/verification/`: checked-in registry plus deterministic acceptance and mutation datasets used by the verification harness.

## Supporting Docs

- Architecture spec: [docs/architecture.md](docs/architecture.md)
- Revised implementation plan: [docs/benchmark_audit_revised_plan.md](docs/benchmark_audit_revised_plan.md)
- Execution log: [docs/benchmark_audit_execution_log.md](docs/benchmark_audit_execution_log.md)
- Verification stack guide: [docs/verification_stack.md](docs/verification_stack.md)
- Supplemental real-benchmark inspection memo: [docs/artifact_inspection_real_benchmark.md](docs/artifact_inspection_real_benchmark.md)
- Phase 12 status note: [docs/phase12_kickoff_note.md](docs/phase12_kickoff_note.md)

## Acknowledgements

AI-bench-auditor is built on top of the AI Scientist v2 repository and continues to reuse pieces of the AIDE-based tree-search infrastructure. Credit remains due to the upstream authors and maintainers whose code made this adaptation possible.

## License And Responsible Use

This repository remains subject to the upstream project license unless and until that license is replaced explicitly. Because the system executes LLM-written code and can produce reviewer-facing audit artifacts, it should be run in a controlled environment with careful human oversight.
