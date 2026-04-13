# Verification Stack

The repository includes a deterministic, repo-native verification harness in `ai_scientist.audits.verification`. Its job is to prove that the audit pipeline, artifact contracts, and manuscript gate behave consistently in local development and CI.

## Scope

The default verification stack covers:

- schema gating for generated audit bundles and phase-summary artifacts
- canary execution over deterministic leakage fixtures
- mutation testing from a clean benchmark base
- search ablation across `detector_only`, `one_shot_agent`, and `full_tree_search`
- repeated-run reproducibility summaries
- acceptance checks over checked-in verification benchmarks

The checked-in registry lives at `tests/fixtures/verification/registry.json`.

## How To Run It

Use the project virtual environment when available:

```bash
.venv-benchmark-audit/bin/python -m ai_scientist.audits.verification \
  --output-dir verification_results/latest
```

Optional flags:

- `--registry-path` to point at a custom benchmark registry
- `--reproducibility-repeats` to change the number of repeated runs per benchmark

Useful companion checks:

```bash
.venv-benchmark-audit/bin/python -m pytest tests/test_audit_verification_stack.py
.venv-benchmark-audit/bin/python -m pytest -q
```

## Phase Outputs

The verification command writes a structured bundle under the chosen output directory:

- `canary/canary_results.json`
- `mutation/mutation_test_results.json`
- `search_ablation/search_ablation_results.json`
- `reproducibility/reproducibility_summary.json`
- `acceptance/acceptance_results.json`
- `schema_gate/schema_gate_results.json`
- `verification_stack_results.json`
- `verification_stack_summary.md`

Search-ablation and reproducibility phases also materialize validated audit bundles containing:

- `audit_results.json`
- `split_manifest.json`
- `findings.csv`
- `audit_report.md`
- `evidence/`

## Paper-Generation Gate

`launch_scientist_bfts.py` consumes `verification_stack_results.json` when paper packaging is enabled. By default it looks at `verification_results/latest/verification_stack_results.json`.

Paper generation is blocked unless all of the following are true in the chosen summary:

- `status == "passed"`
- `phases.schema_gate.passed == true`
- `phases.canary.summary.passed == true`
- `phases.mutation.summary.passed == true`
- `phases.ablation.summary.passed == true`
- `phases.ablation.summary.full_tree_search_adds_value == true`
- `phases.reproducibility.summary.passed == true`

This is intentionally stricter than "a summary file exists." The manuscript path requires both artifact correctness and evidence that tree search is adding value in the checked-in verification harness.

## Default Benchmarks

The default registry currently includes:

- `support_ticket_overlap` for group-overlap plus near-duplicate recovery
- `loan_default_temporal_proxy` for temporal leakage plus suspicious-feature recovery
- `clean_customer_churn` as the clean base used for mutation testing

These are repo-native verification benchmarks intended to keep the stack self-contained and deterministic. They are not a claim that every external benchmark workflow has already been validated.

## Fixture Versus Real Runs

- `tests/fixtures/verification/` contains tiny deterministic datasets used by tests and by the default verification harness.
- real external benchmarks should be staged outside tracked source trees, for example under `downloads/real_benchmarks/`
- generated acceptance, ablation, reproducibility, and other local outputs should likewise stay outside tracked source trees, for example under `downloads/verification_results/`
- `docs/artifact_inspection_real_benchmark.md` documents a supplemental real-benchmark inspection memo for a local ignored run; it is not part of the default tracked verification stack
