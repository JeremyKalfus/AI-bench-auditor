# AI-bench-auditor

AI-bench-auditor is a benchmark leakage auditing workflow built on top of the AI Scientist v2 codebase and its agentic tree-search spine. The product surface is audit-first: it generates a research plan, requires a human approval gate before any audit execution begins, produces deterministic audit artifacts, reviews the final audit report automatically, and only then packages the validated run as a LaTeX paper source bundle with real figures, tables, and citations after the verification stack gate passes.

The repository still uses the internal Python package path `ai_scientist` for compatibility with the upstream codebase. The intended user-facing workflow, CLI, and documentation in this repository are for AI-bench-auditor.

## What It Does

AI-bench-auditor is designed for benchmark leakage and protocol-failure investigations where structured artifacts must remain the source of truth.

Core workflow:

1. Prepare benchmark and dataset context from the supplied audit idea/spec plus dataset metadata.
2. Generate `research_plan.json` and `research_plan.md`.
3. Pause for the required human plan review gate.
4. Run the existing four-stage audit search scaffold.
5. Validate primary audit artifacts.
6. Generate `audit_report.md`.
7. Run an automated audit-report review against real artifacts.
8. Build a LaTeX paper source bundle and optional PDF, but only when the Phase 12 paper-generation gate passes.
9. Stop with no additional human checkpoint after plan approval.

The system is intentionally conservative:

- It does not allow research to begin when plan review is required and approval is missing.
- It does not allow paper generation from invalid audit artifacts or a failing verification-stack gate.
- It does not fabricate citations, figures, or artifact-backed claims.
- It prefers deterministic artifacts over LLM prose whenever both exist.

## Expected Outputs

A successful audit-mode run with paper packaging enabled should leave behind at least:

- `research_plan.json`
- `research_plan.md`
- `plan_review_state.json`
- `plan_approval.json`
- `audit_results.json`
- `split_manifest.json`
- `metrics_before_after.json`
- `findings.csv` or `findings.parquet`
- `audit_report.md`
- `audit_report_review.json`
- `audit_report_review.md`
- `paper/paper.tex`
- `paper/references.bib`
- `paper/figures/`
- `paper/tables/`
- `paper/appendix/`
- `paper/paper_manifest.json`
- optional `paper/paper.pdf` when a LaTeX toolchain is available and compilation succeeds
- `paper_bundle.zip`

## Installation

This workflow is Python-first and was patched with CPU-safe audit defaults in mind. A local virtual environment is strongly recommended.

```bash
python -m venv .venv-benchmark-audit
source .venv-benchmark-audit/bin/activate
pip install -r requirements.txt
```

Optional but recommended for PDF compilation:

- macOS with Homebrew: `brew install texlive`
- Linux: install a TeX distribution that provides `pdflatex` and `bibtex`

Model and API keys depend on which LLMs you use. Typical environment variables include:

```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
export S2_API_KEY="YOUR_SEMANTIC_SCHOLAR_KEY"
```

Notes:

- `citation-mode=provided` is the most reproducible option because it uses a user-supplied bibliography file.
- `citation-mode=auto` is supported, but it depends on honest external citation resolution and can fail if required references cannot be resolved.

## One-Command Audit Flow

The intended top-level command is:

```bash
python launch_scientist_bfts.py \
  --mode audit \
  --benchmark path/to/benchmark_spec.json \
  --plan-review required \
  --paper-mode on_success \
  --output_dir path/to/run_dir
```

Useful flags:

- `--plan-review {required,skip}`
- `--plan-review-mode {interactive,file}`
- `--plan-feedback-file PATH`
- `--plan-approval-file PATH`
- `--approve-plan`
- `--max-plan-revisions N`
- `--paper-mode {off,on_success,always_if_valid}`
- `--emit-paper-zip` or `--no-emit-paper-zip`
- `--compile-paper-pdf` or `--no-compile-paper-pdf`
- `--allow-source-only`
- `--citation-mode {auto,provided,off}`
- `--references-file PATH`
- `--verification-stack-results PATH`

Behavioral contract:

- `--plan-review required` is the default in audit mode.
- In interactive terminals, the launcher can pause for approval, requested changes, or abort.
- In non-interactive settings, approval must come from explicit approval flags or files.
- Once the plan is approved, the rest of the run proceeds automatically.

## Paper Generation Rules

The paper stage is audit-native, not the old generic AI Scientist paper path.

The manuscript builder:

- consumes validated audit artifacts plus the reviewed audit report
- requires a passed `verification_stack_results.json` summary before paper mode can proceed
- derives figures and tables from real artifact files
- emits an evidence map in the appendix
- fails rather than inventing references when citations cannot be resolved honestly
- records PDF compilation success or failure in `paper_manifest.json`

If PDF compilation is requested and fails, the run exits nonzero unless `--allow-source-only` is explicitly set.

By default, paper mode looks for `verification_results/latest/verification_stack_results.json` under the repo root. Use `--verification-stack-results PATH` to point the gate at a different verification run summary.

## Current Scope

This repository now supports:

- a required pre-research human plan-review gate
- automated post-audit report review
- automated LaTeX source bundle generation after a validated audit run and a passed verification-stack gate
- a deterministic Phase 11 verification stack with schema gating, canaries, mutation tests, search ablations, reproducibility summaries, and acceptance checks over checked-in verification benchmarks

This repository still does not claim final empirical validation of every external benchmark workflow. The self-contained Phase 11 harness lives in `ai_scientist.audits.verification` and uses tiny checked-in deterministic fixtures under `tests/fixtures/verification/`; real external benchmarks should be staged outside tracked source trees, with generated run outputs kept outside the repo’s tracked surface as well.

## Repository Notes

- The internal package layout remains under `ai_scientist/` to minimize patch risk and preserve compatibility with the upstream base.
- The audit search still reuses the existing four-stage scaffold where possible.
- The implementation favors the smallest trustworthy patch set over a broad rename or architectural fork.

## Supporting Docs

- Revised implementation plan: [docs/benchmark_audit_revised_plan.md](docs/benchmark_audit_revised_plan.md)
- Execution log: [docs/benchmark_audit_execution_log.md](docs/benchmark_audit_execution_log.md)
- Verification stack guide: [docs/verification_stack.md](docs/verification_stack.md)
- Real-benchmark artifact inspection memo: [docs/artifact_inspection_real_benchmark.md](docs/artifact_inspection_real_benchmark.md)
- Phase 12 kickoff note: [docs/phase12_kickoff_note.md](docs/phase12_kickoff_note.md)

## Acknowledgements

AI-bench-auditor is built on top of the AI Scientist v2 repository and continues to reuse pieces of the AIDE-based tree-search infrastructure. Credit remains due to the upstream authors and maintainers whose code made this adaptation possible.

## License and Responsible Use

This repository remains subject to the upstream project license unless and until that license is replaced explicitly. Because the system executes LLM-written code and can produce publication-ready artifacts, it should be run in a controlled environment with careful human oversight.
