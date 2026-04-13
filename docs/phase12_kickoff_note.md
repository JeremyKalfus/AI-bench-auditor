# Phase 12 Kickoff Note

Phase 12 is defined in `docs/benchmark_audit_revised_plan.md` at lines 705-720. The first continuation point is Phase `12.1 Define paper-generation preconditions`, which requires an explicit gate and verifies that attempted paper generation fails early when those preconditions are not satisfied.

## What Cleanup Changed

- Synthetic verification fixtures were moved from `benchmarks/verification/` to `tests/fixtures/verification/` so tracked test assets are clearly labeled as fixtures rather than real benchmark corpora.
- `docs/verification_stack.md` and `README.md` now distinguish tracked deterministic fixtures from ignored local real-benchmark staging under `downloads/real_benchmarks/` and ignored generated outputs under `downloads/verification_results/`.
- No third-party benchmark corpus was committed. The real MRPC staging area and generated run outputs remain outside the tracked source surface.

## What The Real Artifact Inspection Concluded

- The latest real benchmark run inspected was the local GLUE MRPC audit under `downloads/verification_results/`.
- The artifacts support a narrow claim of cross-split near-duplicate or paraphrase-overlap risk with `48` evidence-backed findings.
- The artifacts do not support a broader claim that MRPC is invalid due to leakage.
- The real ablation ties `one_shot_agent` and `full_tree_search`, so the repo must not imply that tree search clearly outperformed one-shot on this benchmark.

## Why The Repo Is Better Positioned For Phase 12

- The tracked repo surface is cleaner and more intentional: deterministic fixtures are labeled as fixtures, real benchmark downloads remain ignored, and generated outputs no longer blur together with checked-in assets.
- The acceptance wording now reflects the narrower truth supported by the structured MRPC artifacts.
- Because the real ablation did not show tree-search advantage, the honest next Phase 12 step is to add a paper-generation gate that blocks manuscript output unless the verification stack actually passes the required preconditions.
