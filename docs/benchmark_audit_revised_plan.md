# Benchmark Leakage Audit: Revised Implementation Plan

## Context

We want to build `AI-bench-auditor`, a benchmark leakage auditing system built on top of `AI Scientist v2`, that:

- runs on a CPU-first machine, including Apple Silicon Macs
- uses agentic tree search where that search is genuinely useful
- separates operational concerns into two explicit run modes:
  - `audit` mode for validated artifact production
  - `paper` mode for paper generation from a previously validated audit run
- audits benchmarks for exact-duplicate leakage, near-duplicate leakage, group/entity overlap, temporal leakage, preprocessing leakage, suspicious feature leakage, and related protocol failures
- emits deterministic, machine-validated audit artifacts before any narrative writeup is considered valid
- produces a human-auditable `audit_report.md`
- can later generate full paper artifacts including `.tex` and `.pdf` outputs, but only after the audit loop itself is stable and verified

The system must not "pretend success" through plausible prose. A run is only successful if it emits valid structured artifacts, passes schema validation, and survives the verification suite.

## Desired End State

At the end of this project, an audit run should produce all of the following:

- `audit_results.json`
- `split_manifest.json`
- `metrics_before_after.json`
- `findings.csv` or `findings.parquet`
- `audit_report.md`
- copied supporting evidence files referenced by the report
- optional plots
- optional paper outputs later in the project:
  - `paper.tex`
  - compiled paper `.pdf`
  - review outputs only after audit-mode correctness is already established

The `paper` mode must consume only a previously validated audit run. It must not accept raw benchmark input directly.

## Non-Negotiable Constraints

- No placeholder artifacts count as progress.
- No TODO-based "future parser" counts as done.
- No LLM-only interpretation of stdout counts as the primary source of truth when a deterministic artifact can be emitted instead.
- No stage may be marked complete on style, polish, or plots alone.
- No paper PDF counts as valid if the audit artifacts are missing, invalid, or unsupported.
- No agent should expand scope silently.
- Paper mode may only consume a previously validated audit run.
- Every primary artifact must carry provenance fields from day one.

## Agent Operating Rules

These rules apply to every implementation pass.

### Subagents

- Use subagents liberally for bounded tasks.
- Prefer many small, verifiable subtasks over one long monolithic implementation pass.
- Typical subagent task shapes:
  - inspect one file or subsystem
  - implement one isolated schema or scoring function
  - add one test module
  - validate one launcher or config path
- Do not hand critical-path ambiguity to a subagent without first deciding the intended shape locally.
- Every subagent must return:
  - files changed
  - commands run
  - verification status
  - remaining uncertainty

### Context Hygiene

- Keep the main context window clean.
- Summarize findings from subagents instead of pasting raw transcripts.
- When exploring code, capture conclusions and file references, not giant code dumps.
- When a decision is settled, stop revisiting the same context unless a failing test forces it.
- Prefer deterministic artifacts and tests over repeated natural-language reasoning.

### No-Workaround Policy

- If a verification step is missing, add it before proceeding.
- If a schema is needed, define it before generating artifacts.
- If a dependency is claimed in prompts, it must exist in the environment.
- If ranking is supposed to be deterministic, no LLM should remain in the ranking path.

## High-Level Strategy

Keep the existing tree-search, workspace, execution, and journal plumbing where possible. Replace the training-centric protocol, artifact model, completion gates, and branch ranking with audit-native logic.

The smallest trustworthy adaptation is:

1. add explicit `audit` and `paper` modes
2. make the environment truthful
3. define deterministic artifact schemas and provenance before prompt rewrites
4. inject deterministic dataset context
5. require deterministic audit artifacts
6. score and rank branches deterministically
7. verify with canaries, mutation tests, reproducibility checks, and search ablations
8. only then reintroduce report polishing and paper generation

## Implementation Plan

Each microstep below must be completed and verified before the next dependent step begins.

## Phase 0: Baseline Capture

### 0.1 Record current repository baseline

- Capture current branch, commit SHA, and dirty-state.
- Deliverable:
  - a short baseline note in the implementation log or task notes
- Verification:
  - `git status --short --branch`
  - `git rev-parse HEAD`

### 0.2 Prove the current launcher behavior

- Run a dry inspection of the current launch path and document:
  - whether plot aggregation runs unconditionally
  - where writeup starts
  - where review starts
- Deliverable:
  - a note citing the exact file and lines
- Verification:
  - direct file references to the relevant launcher lines

### 0.3 Prove the current ranking path

- Confirm whether best-node selection is deterministic or LLM-mediated.
- Deliverable:
  - a note citing the exact ranking path in `journal.py`
- Verification:
  - direct file reference showing the selection prompt and fallback

### 0.4 Prove the current data-preview path

- Confirm whether `data_preview` is actually populated anywhere.
- Deliverable:
  - a note citing the exact file references
- Verification:
  - search results showing prompt hooks and lack of assignment

## Phase 1: Introduce Run Modes and Audit Skeleton

### 1.1 Add audit-mode configuration surface

- Add a dedicated audit-mode config path.
- Acceptable shapes:
  - `launch_benchmark_audit.py`
  - or `launch_scientist_bfts.py --mode audit`
- Deliverable:
  - one explicit audit entrypoint
- Verification:
  - `python <entrypoint> --help`
  - help output contains audit-specific arguments

### 1.2 Add paper-mode configuration surface

- Add a dedicated paper-mode config path.
- Acceptable shapes:
  - `launch_paper_from_audit.py`
  - or `launch_scientist_bfts.py --mode paper`
- Deliverable:
  - one explicit paper entrypoint or mode
- Verification:
  - `python <entrypoint> --help`
  - help output contains paper-specific arguments

### 1.3 Define the handoff contract between audit mode and paper mode

- Paper mode must accept a path to a validated audit run.
- Paper mode must refuse raw benchmark input.
- Deliverable:
  - documented and enforced handoff contract
- Verification:
  - a test or dry run showing:
    - valid audit-run path is accepted
    - raw benchmark path is rejected

### 1.4 Add an audit preset safe for Apple Silicon

- Add a config preset or defaults that set:
  - `num_workers = 1`
  - writeup disabled by default
  - review disabled by default
  - plot aggregation disabled by default
- Deliverable:
  - one audit preset committed to code or config
- Verification:
  - run help or config printout showing the audit defaults

### 1.5 Prevent accidental writeup/review/plotting in audit mode

- Ensure audit mode does not call:
  - plot aggregation
  - citation gathering
  - paper writeup
  - paper review
- Deliverable:
  - launcher control flow that short-circuits these steps in audit mode
- Verification:
  - a dry run or unit test proving those functions are not reached

## Phase 2: Make the Environment Truthful

### 2.1 Add actual audit dependencies

- Update `requirements.txt` or the chosen environment file to include:
  - `pandas`
  - `scikit-learn`
  - `pyarrow`
  - `duckdb`
  - `rapidfuzz`
  - `pytest`
- Keep `xgboost` optional in the first pass.
- Deliverable:
  - committed dependency file changes
- Verification:
  - environment install succeeds
  - `python -c "import pandas, sklearn, pyarrow, duckdb, rapidfuzz"`

### 2.2 Make prompt package claims truthful

- Update any prompt text that currently implies unavailable packages are already installed.
- Deliverable:
  - prompt text matches real environment state
- Verification:
  - grep for stale package claims
  - manual inspection of prompt strings

### 2.3 Add minimal CPU-only smoke test

- Create a tiny environment smoke test for audit mode.
- Deliverable:
  - one test file or script that imports the audit stack
- Verification:
  - `pytest <smoke_test>`

## Phase 3: Define Primary Artifact Schemas and Provenance

### 3.1 Create the shared provenance block

- Define a provenance structure required in every primary artifact.
- Minimum provenance fields:
  - `schema_version`
  - `git_sha`
  - `dataset_fingerprint`
  - `seed`
  - `run_id`
  - `detector_versions`
  - `created_at`
  - `updated_at`
- Deliverable:
  - one shared provenance schema or helper
- Verification:
  - unit tests for valid and invalid provenance blocks

### 3.2 Create `ai_scientist/audits/schema.py`

- Define strict schemas for:
  - `audit_results.json`
  - `split_manifest.json`
  - `metrics_before_after.json`
  - `findings.csv` or `findings.parquet` column contract
- Every primary schema must include the shared provenance block.
- Deliverable:
  - schema module
- Verification:
  - unit tests for valid and invalid examples of each artifact

### 3.3 Define `audit_results.json`

- Minimum fields:
  - run metadata
  - benchmark summary
  - detectors run
  - findings summary
  - confidence fields
  - audit score
  - evidence references
  - provenance block
- Deliverable:
  - stable `audit_results.json` schema and example
- Verification:
  - JSON schema validation passes on a sample artifact

### 3.4 Define `metrics_before_after.json`

- Minimum fields:
  - baseline metrics
  - remediated metrics
  - deltas
  - split information
  - provenance block
- Deliverable:
  - stable `metrics_before_after.json` schema and example
- Verification:
  - JSON schema validation passes on a sample artifact

### 3.5 Define `split_manifest.json`

- Add a deterministic split manifest schema.
- Minimum fields:
  - split names
  - record counts
  - group-key summary if available
  - temporal coverage if available
  - file paths used
  - provenance block
- Deliverable:
  - schema plus generation contract
- Verification:
  - JSON schema validation passes on a sample manifest

### 3.6 Define findings table contract

- Define the required columns for `findings.csv` or `findings.parquet`.
- Minimum columns:
  - finding ID
  - detector name
  - severity
  - confidence
  - evidence pointer
  - remediation status
  - provenance linkage
- Deliverable:
  - stable findings-table contract
- Verification:
  - unit tests for schema or column validation

## Phase 4: Deterministic Dataset Context

### 4.1 Define a dataset-card artifact

- Add a deterministic dataset context artifact such as `dataset_card.md`.
- Minimum contents:
  - file inventory
  - row counts
  - split names
  - candidate key columns
  - target column if known
  - timestamp columns if present
- Deliverable:
  - schema or generator for `dataset_card.md`
- Verification:
  - generated card exists for a sample input dataset

### 4.2 Inject dataset context into task description

- In audit mode, inject `dataset_card.md` and `split_manifest.json` summaries into the task description before the agent writes code.
- Do not depend on dormant `data_preview` plumbing for the first version.
- Deliverable:
  - task description contains deterministic dataset context
- Verification:
  - captured prompt or rendered task description shows dataset context

### 4.3 Add explicit audit acceptance criteria to the task payload

- Extend the idea/task schema with optional audit-native fields:
  - `Audit Targets`
  - `Leakage Taxonomy`
  - `Acceptance Criteria`
  - `Benchmark Metadata`
- Keep existing idea fields compatible.
- Deliverable:
  - updated task schema accepted by audit mode
- Verification:
  - a sample audit idea file loads successfully

## Phase 5: Replace Stage Goals, Keep the Tree Search

### 5.1 Preserve the 4-stage scaffold

- Keep the existing stage count to minimize plumbing changes.
- Deliverable:
  - audit mode still runs through 4 stages
- Verification:
  - log output shows four audit-native stages

### 5.2 Rewrite Stage 1 goals

- Stage 1 must mean:
  - reproduce the benchmark protocol
  - validate baseline scoring
  - emit manifests and initial artifacts
- Deliverable:
  - stage-goal text updated in code
- Verification:
  - task description for stage 1 contains the new goals

### 5.3 Rewrite Stage 2 goals

- Stage 2 must mean:
  - run leakage detectors
  - gather concrete evidence
  - avoid speculative conclusions
- Deliverable:
  - stage-goal text updated in code
- Verification:
  - task description for stage 2 contains the new goals

### 5.4 Rewrite Stage 3 goals

- Stage 3 must mean:
  - confirm findings via remediations and falsification
  - compare before/after metrics
  - rule out obvious benign explanations
- Deliverable:
  - stage-goal text updated in code
- Verification:
  - task description for stage 3 contains the new goals

### 5.5 Rewrite Stage 4 goals

- Stage 4 must mean:
  - robustness checks
  - audit summary synthesis
  - evidence-complete reporting
- Deliverable:
  - stage-goal text updated in code
- Verification:
  - task description for stage 4 contains the new goals

### 5.6 Replace stage completion gates

- Remove dependence on:
  - training curves
  - plot quality
  - added HuggingFace datasets
- Require:
  - valid artifacts
  - baseline reproduction or validation
  - at least one confirmed finding or a high-confidence clean audit
  - at least one remediation or falsification attempt where applicable
- Deliverable:
  - audit-mode completion logic
- Verification:
  - unit tests for stage completion on mocked node artifacts

## Phase 6: Rewrite the Agent Prompts for Audit Work

### 6.1 Remove training-centric requirements in audit mode

- Audit mode prompts must not require:
  - epoch-wise validation loss
  - GPU boilerplate
  - training curves as primary evidence
  - synthetic data creation unless explicitly requested
- Deliverable:
  - audit prompt branch in `parallel_agent.py`
- Verification:
  - prompt snapshot test or direct prompt rendering check

### 6.2 Remove anti-EDA instructions in audit mode

- Delete or bypass prompt lines that say:
  - "Don't suggest to do EDA."
- Deliverable:
  - audit prompt permits schema inspection and split inspection
- Verification:
  - grep confirms those lines are absent from audit-mode prompt assembly

### 6.3 Replace prompt environment with audit stack guidance

- Audit mode should prefer:
  - `pandas`
  - `scikit-learn`
  - `duckdb`
  - `rapidfuzz`
  - `pyarrow`
- Deliverable:
  - audit-mode environment prompt text
- Verification:
  - prompt snapshot or direct rendered prompt inspection

### 6.4 Make artifact emission mandatory

- Audit mode must instruct the agent to emit:
  - `audit_results.json`
  - `split_manifest.json`
  - `metrics_before_after.json`
  - `findings.csv` or `findings.parquet`
- Deliverable:
  - hard prompt requirements
- Verification:
  - prompt snapshot shows required file names explicitly

## Phase 7: Deterministic Audit Components

### 7.1 Create `ai_scientist/audits/detectors.py`

- Implement the initial detector set:
  - exact duplicates
  - near duplicates
  - group/entity overlap
  - temporal leakage
  - preprocessing leakage
  - suspicious feature leakage
- Deliverable:
  - detector module with deterministic functions
- Verification:
  - unit tests for each detector on tiny fixtures

### 7.2 Create `ai_scientist/audits/scoring.py`

- Define deterministic scoring for a branch.
- Minimum inputs:
  - confirmed findings
  - severity
  - evidence completeness
  - remediation effect size
  - negative-control penalties
- Deliverable:
  - scoring module
- Verification:
  - tests for expected score ordering across canned cases

### 7.3 Create `ai_scientist/audits/report.py`

- Generate `audit_report.md` from deterministic artifacts.
- The report must cite artifact paths, not just summarize claims.
- Deliverable:
  - report-generation module
- Verification:
  - run report generation on a fixture and inspect the output file

### 7.4 Create `ai_scientist/audits/canary_suite.py`

- Add a canary fixture generator or loader for:
  - exact duplicate leakage
  - near-duplicate leakage
  - group/entity overlap
  - temporal leakage
  - preprocessing or label leakage
  - clean negative control
- Deliverable:
  - canary suite module
- Verification:
  - canary suite can be invoked and writes fixture datasets

## Phase 8: Intercept the Execution Path

### 8.1 Audit-mode parse path must check artifacts first

- Update the execution result path so audit mode first looks for `audit_results.json`.
- Deliverable:
  - deterministic artifact-first parse path
- Verification:
  - test where a valid artifact bypasses LLM metric parsing

### 8.2 Validate artifacts before accepting a node

- If `audit_results.json` is missing or invalid:
  - mark the node invalid
  - store a clear failure reason
- Deliverable:
  - schema-gated node acceptance
- Verification:
  - unit tests for missing and malformed artifact cases

### 8.3 Populate `node.analysis` from structured audit summary

- Do not derive core analysis from free-form stdout if structured summary exists.
- Deliverable:
  - deterministic analysis population path
- Verification:
  - node fields match structured artifact contents in tests

### 8.4 Populate `node.metric` from deterministic `audit_score`

- Convert the audit score into the node metric path used elsewhere.
- Deliverable:
  - `node.metric` set from deterministic audit score
- Verification:
  - tests show score propagation into the node

### 8.5 Keep LLM parsing only as an explicit fallback

- Fallback is allowed only when deterministic artifacts are missing.
- Deliverable:
  - clear conditional fallback
- Verification:
  - tests cover both primary and fallback paths

## Phase 9: Deterministic Best-Node Selection

### 9.1 Add audit-mode comparator in `journal.py`

- In audit mode, bypass LLM node selection entirely.
- Deliverable:
  - audit-specific deterministic selection branch
- Verification:
  - tests show no LLM selection call in audit mode

### 9.2 Define exact sort precedence

- Order nodes by:
  - audit score
  - evidence completeness
  - remediation confirmation
  - reproducibility signal if present
  - stable tiebreaker such as node ID or creation time
- Deliverable:
  - deterministic comparator implementation
- Verification:
  - tests with equal-score and near-equal-score cases

### 9.3 Preserve old behavior outside audit mode

- Non-audit workflows should remain compatible.
- Deliverable:
  - no regression in default mode selection path
- Verification:
  - tests or smoke checks for non-audit code path

## Phase 10: Audit Report Output

### 10.1 Add `audit_report.md` generation at the end of audit mode

- The report must include:
  - benchmark summary
  - detectors run
  - findings
  - evidence references
  - remediation results
  - confidence and limitations
- Deliverable:
  - generated report per run
- Verification:
  - report exists and references actual artifact files

### 10.2 Add a strict evidence section

- Every major finding in the report must reference:
  - supporting rows, files, or manifest entries
- Deliverable:
  - evidence-linked report template
- Verification:
  - report lint or tests confirm references exist

### 10.3 Do not re-enable paper generation yet

- Paper generation is deferred until the audit loop is correct.
- Deliverable:
  - explicit project note or guard in audit mode
- Verification:
  - audit runs do not attempt `.tex` or `.pdf` generation yet

## Phase 11: Verification Stack

These verification phases are required, not optional.

### 11.1 Schema gate

- Define success as:
  - valid `audit_results.json`
  - valid referenced evidence files
  - valid summary artifacts
- Deliverable:
  - schema validation command or test target
- Verification:
  - automated test pass

### 11.2 Canary suite

- Run every detector against the deterministic canaries.
- Deliverable:
  - test output for all canary fixtures
- Verification:
  - exact leakage canaries detected every run
  - negative control remains clean

### 11.3 Mutation testing

- Inject one leakage mode at a time into a clean benchmark.
- Deliverable:
  - mutation-test harness
- Verification:
  - reported recall and false-positive metrics

### 11.4 Search ablation

- Compare:
  - full tree search
  - one-shot agent
  - detector-only baseline
- Deliverable:
  - ablation results artifact and summary
- Verification:
  - reproducible comparison run

### 11.5 Reproducibility test

- Run the same audit three times.
- Deliverable:
  - reproducibility summary
- Verification:
  - top findings and remediation direction remain materially consistent

### 11.6 Real-benchmark acceptance test

- Run on one or two real benchmarks with known or suspected issues.
- Deliverable:
  - acceptance-test reports
- Verification:
  - system either recovers the known issue with evidence or emits a clean audit without unsupported claims

## Phase 12: Reintroduce Paper Outputs

This phase starts only after Phase 11 passes.

### 12.1 Define paper-generation preconditions

- Paper generation may start only if:
  - schema gate passes
  - canary suite passes
  - mutation tests pass thresholds
  - search ablation shows tree search adds value
  - reproducibility is acceptable
- Deliverable:
  - explicit gate in code or process documentation
- Verification:
  - attempted paper generation without passing gates fails early

### 12.2 Generate `.tex` from deterministic audit artifacts

- The paper draft must be derived from structured artifacts and the audit report.
- Paper mode must consume only a previously validated audit run.
- Deliverable:
  - `paper.tex`
- Verification:
  - `paper.tex` exists and references actual findings

### 12.3 Compile the paper PDF

- Compile a `.pdf` only after `.tex` generation succeeds.
- Deliverable:
  - paper `.pdf`
- Verification:
  - LaTeX compilation succeeds

### 12.4 Add review only after paper generation is stable

- Re-enable LLM review only as a secondary critique layer.
- Deliverable:
  - optional review outputs
- Verification:
  - review runs only after `.pdf` exists

## Required Pass Criteria

The first version is not "working" until all of the following are true:

- zero schema failures on accepted audit runs
- every accepted primary artifact contains valid provenance fields
- exact duplicate leakage canaries are detected every time
- group/entity leakage canaries are detected every time
- near-duplicate, temporal, and preprocessing leakage canaries are detected at high rate
- negative-control false positives stay below the agreed threshold
- remediation direction is correct in the large majority of mutation tests
- full tree search outperforms the one-shot and detector-only baselines on confirmed-findings quality or evidence completeness
- reproducibility is materially stable across repeated runs

## Implementation Order

Follow this order exactly unless a failing verification step forces a local reorder.

1. Phase 1: run modes and audit skeleton
2. Phase 2: truthful environment
3. Phase 3: schemas and provenance
4. Phase 4: deterministic dataset context
5. Phase 5: stage-goal rewrite
6. Phase 6: audit prompts
7. Phase 7: detectors, scoring, report, canaries
8. Phase 8: artifact-first execution parse
9. Phase 9: deterministic journal ranking
10. Phase 10: audit report output
11. Phase 11: verification stack
12. Phase 12: reintroduce `.tex` and `.pdf` outputs

## Stop Conditions

Stop and re-evaluate if any of these happen:

- audit mode still depends on plots to declare success
- journal selection still calls an LLM in audit mode
- prompt text claims dependencies that do not exist
- nodes can be marked successful without valid audit artifacts
- primary artifacts are emitted without provenance fields
- the canary suite shows regressions after a patch
- paper generation is attempted before verification gates pass

## Definition of Done

The project is done when:

- the repo contains a CPU-safe benchmark leakage audit mode
- the repo contains an explicit paper mode that only consumes validated audit runs
- runs emit deterministic, schema-validated audit artifacts
- branch selection is deterministic in audit mode
- the verification stack passes
- the system can generate a grounded `audit_report.md`
- and only after all of that, the system can optionally generate `.tex` and `.pdf` outputs from the validated audit artifacts
