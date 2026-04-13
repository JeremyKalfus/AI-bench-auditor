# Phase 12 Status Note

This file now serves as the current status note for the paper-output phase.

Phase 12 is no longer only a kickoff item. The repository now contains the implemented manuscript gate and bundle-generation path.

## What Is Implemented

- `launch_scientist_bfts.py` enforces paper-generation preconditions through `ensure_paper_generation_preconditions(...)`.
- The post-audit path always runs deterministic report review before manuscript generation through `review_audit_report(...)`.
- `ai_scientist.audits.manuscript.build_audit_manuscript_bundle(...)` generates:
  - `paper/paper.tex`
  - `paper/references.bib`
  - `paper/figures/`
  - `paper/tables/`
  - `paper/appendix/`
  - `paper/paper_manifest.json`
  - optional `paper/paper.pdf`
  - optional `paper_bundle.zip`
- `paper` mode consumes a validated audit run directory and refuses raw benchmark input.

The tracked tests covering this phase include:

- `tests/test_audit_report_review.py`
- `tests/test_audit_paper_bundle.py`
- `tests/test_audit_single_command_flow.py`

## What The Gate Requires

Paper generation is blocked unless the selected `verification_stack_results.json` reports:

- overall status `passed`
- schema gate passed
- canary suite passed
- mutation thresholds passed
- search ablation passed
- `full_tree_search_adds_value = true`
- reproducibility passed

This means the manuscript path depends on the repo-native verification harness, not on a handwritten claim in documentation.

## Relationship To Real-Benchmark Inspection

The tracked repository uses a small deterministic verification harness under `tests/fixtures/verification/`. Supplemental real-benchmark inspection remains documented separately in [artifact_inspection_real_benchmark.md](artifact_inspection_real_benchmark.md) and is kept outside the tracked source surface under `downloads/`.

That separation is intentional:

- the tracked harness keeps development and CI deterministic
- supplemental real-benchmark inspection helps calibrate wording and honesty on external benchmarks

## Why The MRPC Note Still Matters

The local GLUE MRPC inspection remains useful as a wording guardrail:

- it supports a narrow claim of overlap risk, not blanket benchmark invalidity
- its local ablation tie means MRPC should not be used as evidence that tree search clearly outperforms one-shot search

That historical lesson is now enforced in code by the gate: the launcher looks at the selected verification summary and blocks manuscript generation if the ablation summary does not show that tree search adds value.
