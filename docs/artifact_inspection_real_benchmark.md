# Real Benchmark Artifact Inspection Memo

## Scope

- Run directory inspected: `downloads/verification_results/glue_mrpc_audit_bundle/`
- Acceptance directory inspected: `downloads/verification_results/glue_mrpc_acceptance/glue_mrpc_real/`
- Ablation directory inspected: `downloads/verification_results/glue_mrpc_ablation/`
- Reproducibility directory inspected: `downloads/verification_results/glue_mrpc_repro/`
- Benchmark staged under: `downloads/real_benchmarks/glue_mrpc/`
- Benchmark used: `GLUE MRPC (train/validation)` from Hugging Face `datasets`

## Artifact Files Opened

- `downloads/real_benchmarks/glue_mrpc/benchmark.json`
- `downloads/verification_results/glue_mrpc_audit_bundle/audit_report.md`
- `downloads/verification_results/glue_mrpc_audit_bundle/audit_results.json`
- `downloads/verification_results/glue_mrpc_audit_bundle/findings.csv`
- Representative evidence files:
  - `downloads/verification_results/glue_mrpc_audit_bundle/evidence/near_duplicate_near_duplicate_01d2acfe5ecebbf6.json`
  - `downloads/verification_results/glue_mrpc_audit_bundle/evidence/near_duplicate_near_duplicate_0ddbe7593ef74ff6.json`
  - `downloads/verification_results/glue_mrpc_audit_bundle/evidence/near_duplicate_near_duplicate_1461fe327d2ca218.json`
  - `downloads/verification_results/glue_mrpc_audit_bundle/evidence/near_duplicate_near_duplicate_1c53b4e03296e4be.json`
  - `downloads/verification_results/glue_mrpc_audit_bundle/evidence/near_duplicate_near_duplicate_2402b525bc38fcff.json`
- `downloads/verification_results/glue_mrpc_acceptance/glue_mrpc_real/acceptance_report.md`
- `downloads/verification_results/glue_mrpc_ablation/search_ablation_results.json`
- `downloads/verification_results/glue_mrpc_repro/reproducibility_summary.json`
- Underlying ablation bundles:
  - `downloads/verification_results/glue_mrpc_ablation/glue_mrpc_real/detector_only/audit_results.json`
  - `downloads/verification_results/glue_mrpc_ablation/glue_mrpc_real/one_shot_agent/audit_results.json`
  - `downloads/verification_results/glue_mrpc_ablation/glue_mrpc_real/full_tree_search/audit_results.json`
- Underlying reproducibility bundles:
  - `downloads/verification_results/glue_mrpc_repro/glue_mrpc_real/run_1/audit_results.json`
  - `downloads/verification_results/glue_mrpc_repro/glue_mrpc_real/run_2/audit_results.json`
  - `downloads/verification_results/glue_mrpc_repro/glue_mrpc_real/run_3/audit_results.json`

## Supported Claims

- The real MRPC run supports a narrow claim of cross-split near-duplicate or paraphrase-overlap risk.
- The `near_duplicate` detector fired with deterministic evidence on the real benchmark.
- The full-tree audit bundle contains `48` findings, all from `near_duplicate`, with `48` evidence references and `48` evidence files present.
- The acceptance contract is satisfied for the declared detector expectation: `expected_detectors == observed_detectors == ['near_duplicate']`.
- The reproducibility summary is materially consistent with the underlying runs. Each of the three repeated runs produced `48` findings, `48` evidence references, `audit_score = 100.0`, and `observed_detectors = ['near_duplicate']`.
- The ablation artifact supports a strict comparison result: `detector_only` failed to recover the expected detector, while `one_shot_agent` and `full_tree_search` both recovered it with the same observed detector set and the same finding count.

## Unsupported or Overstated Claims

- The inspected artifacts do not justify a blanket claim that MRPC is invalid due to leakage.
- The evidence supports overlap or near-duplicate structure across the train and validation boundary, not a stronger claim of exact benchmark invalidity or target leakage.
- The ablation artifacts do not support any claim that `full_tree_search` outperformed `one_shot_agent` on MRPC. They tie at `overall_recall = 1.0`, and `search_ablation_results.json` records `best_strategy = "one_shot_agent"` with `full_tree_search_adds_value = false`.

## Consistency Checks

- `audit_report.md` reports `48` total findings and `{'near_duplicate': 48}`. Those counts match `audit_results.json` and `findings.csv`.
- Every evidence path named in `audit_results.json` and `findings.csv` exists on disk.
- Representative evidence descriptors map back to real MRPC train and validation rows. The inspected examples show reused sentence pairs or paraphrase-heavy pair variants across splits.
- Aggregate evidence inspection showed:
  - `48` findings total
  - similarity scores ranging from `90.0` to `100.0`
  - average similarity `95.7715`
  - `44 / 48` findings with matching labels
  - `23 / 48` findings where `sentence2` matched exactly across train and validation
  - `22 / 48` findings with a sentence reused cross-position between the pair members

## Evidence Gaps and Limits

- `group_overlap` and `temporal_leakage` were skipped on MRPC because the benchmark metadata does not declare `candidate_key_columns` or `timestamp_columns`.
- No `metrics_before_after.json` artifact was present for the real MRPC run, so the report cannot support claims about remediation deltas.
- The detector is a fuzzy token-set similarity detector over concatenated text columns. It is evidence for overlap risk, not a direct estimate of downstream model inflation.

## Exact Remediation Edits Made

- Moved checked-in synthetic verification fixtures from `benchmarks/verification/` to `tests/fixtures/verification/` so the tracked repo surface no longer presents test fixtures as real benchmark corpora.
- Updated fixture references in `ai_scientist/audits/verification.py`, `tests/test_audit_verification_stack.py`, `docs/verification_stack.md`, and `README.md`.
- Narrowed acceptance-report wording in `ai_scientist/audits/verification.py` from a generic "known or suspected issue set" claim to detector-contract wording that explicitly avoids broader benchmark-invalidity claims.
- Regenerated `downloads/verification_results/glue_mrpc_acceptance/glue_mrpc_real/acceptance_report.md` after the wording fix.

## Remaining Uncertainty

- The MRPC artifacts show real cross-split overlap structure, but they do not on their own quantify how much any specific model evaluation is inflated.
- Because `one_shot_agent` tied `full_tree_search` on this benchmark, MRPC is a useful acceptance target for detector recovery but not a positive example for proving tree-search advantage.
