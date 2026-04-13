# Verification Stack

The repository now includes a deterministic Phase 11 verification harness in
`ai_scientist.audits.verification`.

## What It Covers

- schema gating for generated audit bundles
- canary execution against deterministic leakage fixtures
- mutation testing from a clean benchmark base
- search ablation across `detector_only`, `one_shot_agent`, and `full_tree_search`
- three-run reproducibility summaries
- acceptance checks over checked-in benchmark directories

The checked-in deterministic fixture registry lives at
`tests/fixtures/verification/registry.json`.

## How To Run It

Use the project virtual environment when available:

```bash
.venv-benchmark-audit/bin/python -m ai_scientist.audits.verification \
  --output-dir verification_results/latest
```

Optional flags:

- `--registry-path` to point at a custom benchmark registry
- `--reproducibility-repeats` to change the number of repeated runs per benchmark

## Output Artifacts

The verification command writes a structured bundle under the chosen output directory:

- `canary/canary_results.json`
- `mutation/mutation_test_results.json`
- `search_ablation/search_ablation_results.json`
- `reproducibility/reproducibility_summary.json`
- `acceptance/acceptance_results.json`
- `schema_gate/schema_gate_results.json`
- `verification_stack_results.json`
- `verification_stack_summary.md`

Search ablation and reproducibility phases also emit validated audit bundles with:

- `audit_results.json`
- `split_manifest.json`
- `findings.csv`
- `audit_report.md`
- `evidence/`

## Benchmarks

The default registry currently includes:

- `support_ticket_overlap` for group-overlap plus near-duplicate recovery
- `loan_default_temporal_proxy` for temporal leakage plus label-copy recovery
- `clean_customer_churn` as the clean base used for single-mode mutation testing

These are repo-native verification benchmarks intended to keep Phase 11 runnable
and deterministic in local development and CI. They are not a claim that every
external benchmark workflow has already been validated.

## Fixture vs Real Runs

- `tests/fixtures/verification/` contains tiny deterministic fixture datasets used by tests
  and by the self-contained verification harness.
- real external benchmarks should be staged outside tracked source trees, for example under
  `downloads/real_benchmarks/`
- generated acceptance, ablation, and reproducibility outputs should likewise stay outside
  tracked source trees, for example under `downloads/verification_results/`
