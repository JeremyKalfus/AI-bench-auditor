# Phase 12 Status Note

This file now serves as the current status note for the output-surface phase.

Phase 12 is no longer only a kickoff item. The repository now contains the implemented markdown-first study bundle path.

## What Is Implemented

- `launch_scientist_bfts.py` runs deterministic report review after artifact promotion and then builds the study bundle.
- `ai_scientist.audits.study.build_audit_study_bundle(...)` generates:
  - `study_report.md`
  - `study_bundle_manifest.json`
  - `study_figures/`
  - optional `study_figures.zip`
- `study` mode consumes a validated audit run directory and refuses raw benchmark input.

The tracked tests covering this phase include:

- `tests/test_audit_report_review.py`
- `tests/test_audit_study_bundle.py`
- `tests/test_audit_single_command_flow.py`

## What The Output Surface Optimizes For

The study bundle is meant to be easy for another LLM or human reviewer to consume directly. That means:

- markdown instead of LaTeX
- explicit artifact references instead of bibliography plumbing
- embedded methodology and run context instead of formal publication-style prose sections
- figure PNGs plus a zip bundle instead of PDF compilation

## Relationship To Verification

The tracked repository still uses the deterministic verification harness under `tests/fixtures/verification/` to validate the audit pipeline and search-quality regressions. That harness remains valuable, but it is no longer a document-generation gate for LaTeX or PDF output.

## Relationship To Real-Benchmark Inspection

Supplemental real-benchmark inspection remains documented separately in [artifact_inspection_real_benchmark.md](artifact_inspection_real_benchmark.md) and is kept outside the tracked source surface under `downloads/`.

That separation is intentional:

- the tracked harness keeps development and CI deterministic
- supplemental real-benchmark inspection helps calibrate wording and honesty on external benchmarks

## Why The MRPC Note Still Matters

The local GLUE MRPC inspection remains useful as a wording guardrail:

- it supports a narrow claim of overlap risk, not blanket benchmark invalidity
- its local ablation tie means MRPC should not be used as evidence that tree search clearly outperforms one-shot search

That lesson still matters even though the final output is now a study bundle rather than the earlier document bundle concept.
