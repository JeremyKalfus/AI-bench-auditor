# Benchmark Audit Revised Plan

Status note as of 2026-04-13: the repository now implements the core audit flow, the repo-native verification harness, and the markdown-first study bundle output surface. This document records the current plan in terms of the product surface that exists today.

## Current Product Surface

The repository currently implements:

- `audit` mode for benchmark staging, dataset context, plan review, audit search, artifact validation, audit report generation, report review, and study-bundle generation
- `study` mode for rebuilding the study bundle from an existing validated audit run
- a deterministic verification harness for local development and CI
- structured outputs such as `audit_results.json`, `split_manifest.json`, `findings.csv`, `metrics_before_after.json`, `audit_report.md`, `audit_report_review.json`, `study_report.md`, and `study_bundle_manifest.json`

## Product Principles

The plan is guided by these principles:

- the benchmark audit should be artifact-first
- the final output should be easy for another LLM to ingest directly
- markdown and structured JSON/CSV outputs are preferred over document-compilation workflows
- the tracked repository should remain deterministic and testable without large external benchmark downloads

## What Is Explicitly Out Of Scope

The repository no longer treats any of the following as the primary product surface:

- LaTeX generation
- PDF compilation
- bibliography gathering for formal related-work sections
- document review flows that depend on PDF rendering

If such outputs ever return, they should be strictly derivative of the study bundle rather than the primary interface.

## Current Output Contract

A successful audit run should leave behind at least:

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
- `study_report.md`
- `study_bundle_manifest.json`
- `study_figures/`
- optional `study_figures.zip`

## Remaining Near-Term Work

The next work should focus on strengthening the audit system itself rather than reviving document-generation layers.

### 1. Improve the study report as an LLM handoff artifact

- add more explicit provenance summaries
- make embedded sections easier to chunk programmatically
- consider a companion machine-readable summary optimized for tool calling and downstream agents

### 2. Improve figure coverage

- add figures that summarize evidence density, detector coverage, and remediation impact more clearly
- keep every figure derived directly from tracked artifacts
- ensure captions stay factual and non-interpretive

### 3. Expand verification fixtures carefully

- add more deterministic benchmark fixtures with distinct failure modes
- strengthen reproducibility and acceptance coverage
- keep the checked-in verification harness small enough for local runs and CI

### 4. Keep the real-run story honest

- continue to stage real external benchmarks outside tracked source trees
- treat supplemental inspection memos as calibration aids, not the default regression surface
- avoid making broad claims that exceed the evidence in the tracked harness

## Success Criteria For The Current Plan

The repository is in a healthy state if all of the following remain true:

- `audit` and `study` modes are usable and well-documented
- the final output surface is `study_report.md` plus structured artifacts and figure zip
- full local tests remain green
- verification runs remain deterministic
- report review blocks unsupported or overclaimed outputs before the study bundle is emitted

## References For Current Behavior

For current implementation details, see:

- [../README.md](../README.md)
- [architecture.md](architecture.md)
- [verification_stack.md](verification_stack.md)
- [benchmark_audit_execution_log.md](benchmark_audit_execution_log.md)
