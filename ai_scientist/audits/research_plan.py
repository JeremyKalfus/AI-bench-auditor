from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_scientist.treesearch.utils.config import load_cfg


PLAN_CONTRACT_VERSION = 1

DEFAULT_EXPECTED_ARTIFACTS = [
    "dataset_card.md",
    "split_manifest.json",
    "audit_results.json",
    "findings.csv or findings.parquet",
    "metrics_before_after.json (when remediation is required)",
    "audit_report.md",
]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_optional_text(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return path.read_text().strip() or None


def _read_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def _normalize_feedback_items(feedback_items: list[str] | None) -> list[str]:
    if not feedback_items:
        return []
    normalized = []
    for item in feedback_items:
        stripped = item.strip()
        if stripped:
            normalized.append(stripped)
    return normalized


def _feedback_to_items(feedback_text: str | None) -> list[str]:
    if not feedback_text:
        return []
    items = []
    for raw_line in feedback_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("-", "*")):
            line = line[1:].strip()
        items.append(line)
    return _normalize_feedback_items(items)


def _detector_plan_entry(target: str, split_names: list[str]) -> dict[str, Any]:
    target_lower = target.lower()
    evidence_expectation = "findings artifact rows plus evidence pointers"
    confirmation = "confirm or refute the finding using deterministic evidence"

    if "duplicate" in target_lower:
        evidence_expectation = "duplicate pair evidence across benchmark splits"
        confirmation = "remove duplicate-linked records and recompute metrics"
    elif "group" in target_lower:
        evidence_expectation = "cross-split overlap on declared group identifiers"
        confirmation = "repartition by group and compare before/after metrics"
    elif "temporal" in target_lower:
        evidence_expectation = "timestamp overlap or chronology violations"
        confirmation = "enforce temporal ordering and compare before/after metrics"
    elif "feature" in target_lower or "leakage" in target_lower:
        evidence_expectation = "feature-level leakage evidence tied to split boundaries"
        confirmation = "drop or mask suspect features and compare metrics"

    return {
        "detector_name": target,
        "planned_splits": split_names,
        "required_outputs": [
            "audit_results.json",
            "findings.csv or findings.parquet",
        ],
        "evidence_expectation": evidence_expectation,
        "confirmation_step": confirmation,
    }


def _build_runtime_budget(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {
            "search_budget_steps": None,
            "num_workers": None,
            "stage_iteration_limits": {},
            "multi_seed_num_seeds": None,
        }

    try:
        cfg = load_cfg(Path(config_path))
    except Exception:
        return {
            "search_budget_steps": None,
            "num_workers": None,
            "stage_iteration_limits": {},
            "multi_seed_num_seeds": None,
        }
    stage_limits = {}
    for stage_number in range(1, 5):
        key = f"stage{stage_number}_max_iters"
        stage_limits[key] = getattr(cfg.agent.stages, key, None)

    return {
        "search_budget_steps": getattr(cfg.agent, "steps", None),
        "num_workers": getattr(cfg.agent, "num_workers", None),
        "stage_iteration_limits": stage_limits,
        "multi_seed_num_seeds": getattr(
            getattr(cfg.agent, "multi_seed_eval", object()),
            "num_seeds",
            None,
        ),
    }


def _build_leakage_hypotheses(
    audit_targets: list[str],
    taxonomy: list[str],
    split_manifest: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    split_names = []
    if split_manifest is not None:
        split_names = [split["name"] for split in split_manifest["splits"]]

    hypothesis_inputs = audit_targets or taxonomy or ["benchmark leakage"]
    hypotheses = []
    for index, item in enumerate(hypothesis_inputs, start=1):
        label = str(item)
        lower = label.lower()
        rationale = (
            "Audit the declared benchmark splits for evidence-backed leakage or contamination."
        )
        if "duplicate" in lower:
            rationale = "Cross-split duplicates can inflate benchmark performance without reflecting generalization."
        elif "group" in lower:
            rationale = "Shared entities across train/test style splits can leak identity information."
        elif "temporal" in lower:
            rationale = "Temporal ordering mistakes can leak future information into earlier splits."
        elif "feature" in lower:
            rationale = "Direct or proxy target leakage can be hidden in benchmark features."

        hypotheses.append(
            {
                "hypothesis_id": f"H{index}",
                "target": label,
                "suspect_splits": split_names,
                "rationale": rationale,
            }
        )
    return hypotheses


def _build_review_adjustments(feedback_items: list[str]) -> list[str]:
    return [f"Address reviewer request: {item}" for item in feedback_items]


def fingerprint_research_plan(plan_data: dict[str, Any]) -> str:
    payload = deepcopy(plan_data)
    payload.pop("plan_fingerprint", None)
    normalized = json.dumps(payload, indent=2, sort_keys=True)
    return f"sha256:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"


def build_research_plan(
    *,
    idea_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    review_round: int = 1,
    feedback_items: list[str] | None = None,
) -> dict[str, Any]:
    idea_path = Path(idea_path)
    output_dir = Path(output_dir)
    idea = json.loads(idea_path.read_text())

    dataset_card_path = output_dir / "dataset_card.md"
    split_manifest_path = output_dir / "split_manifest.json"
    dataset_card_text = _read_optional_text(dataset_card_path)
    split_manifest = _read_optional_json(split_manifest_path)
    benchmark_metadata = idea.get("Benchmark Metadata") or {}
    split_names = []
    if split_manifest is not None:
        split_names = [split["name"] for split in split_manifest["splits"]]

    audit_targets = [str(item) for item in idea.get("Audit Targets", [])]
    leakage_taxonomy = [str(item) for item in idea.get("Leakage Taxonomy", [])]
    acceptance_criteria = [str(item) for item in idea.get("Acceptance Criteria", [])]
    feedback_items = _normalize_feedback_items(feedback_items)
    review_adjustments = _build_review_adjustments(feedback_items)

    benchmark_summary = {
        "title": idea.get("Title", idea.get("Name", "Audit Run")),
        "benchmark_name": benchmark_metadata.get(
            "benchmark_name", idea.get("Name", "benchmark-audit")
        ),
        "dataset_name": benchmark_metadata.get("dataset_name"),
        "dataset_fingerprint": benchmark_metadata.get("dataset_fingerprint"),
        "split_names": split_names,
        "dataset_card_path": dataset_card_path.name if dataset_card_path.exists() else None,
        "split_manifest_path": (
            split_manifest_path.name if split_manifest_path.exists() else None
        ),
    }

    detector_plan = [
        _detector_plan_entry(target, split_names) for target in audit_targets
    ] or [_detector_plan_entry("benchmark leakage scan", split_names)]

    plan = {
        "contract_version": PLAN_CONTRACT_VERSION,
        "generated_at": _timestamp(),
        "review_round": review_round,
        "source_paths": {
            "idea_json": str(idea_path.resolve()),
            "dataset_card": (
                str(dataset_card_path.resolve()) if dataset_card_path.exists() else None
            ),
            "split_manifest": (
                str(split_manifest_path.resolve())
                if split_manifest_path.exists()
                else None
            ),
            "config_path": str(Path(config_path).resolve()) if config_path else None,
        },
        "benchmark_summary": benchmark_summary,
        "audit_targets": audit_targets,
        "leakage_hypotheses": _build_leakage_hypotheses(
            audit_targets, leakage_taxonomy, split_manifest
        ),
        "detector_plan": detector_plan,
        "confirmation_remediation_plan": [
            "Validate each candidate issue against deterministic evidence before escalating a claim.",
            "When a finding remains plausible, run remediation or falsification work and emit metrics_before_after.json.",
            "Document whether each finding stays confirmed, weakens under controls, or is refuted.",
        ]
        + review_adjustments,
        "success_failure_criteria": {
            "success": acceptance_criteria
            + [
                "Every major audit claim must trace back to deterministic artifacts and evidence files.",
                "Research stops immediately if artifact validation fails or unsupported claims remain unresolved.",
            ],
            "failure": [
                "A required audit artifact is missing or invalid.",
                "Plan approval is withheld or the audit exceeds stop conditions without evidence-backed progress.",
                "The run cannot produce honest citations or evidence-backed manuscript tables/figures.",
            ],
        },
        "expected_artifacts": DEFAULT_EXPECTED_ARTIFACTS,
        "evidence_strategy": [
            "Use dataset_card.md and split_manifest.json as the benchmark context of record.",
            "Use audit_results.json and findings.* as the primary source of truth for detector outcomes.",
            "Use metrics_before_after.json only for remediation conclusions and before/after comparisons.",
            "Treat markdown narrative as secondary to structured artifacts whenever they disagree.",
        ]
        + review_adjustments,
        "risks_and_benign_alternatives": [
            str(idea.get("Risk Factors and Limitations", "")).strip(),
            "Apparent leakage may reflect benign duplication, grouping, or timestamp formatting issues rather than benchmark contamination.",
            "Small synthetic or incomplete fixtures can validate the pipeline while still underrepresenting real benchmark complexity.",
        ]
        + [f"Reviewer concern: {item}" for item in feedback_items],
        "runtime_budget": _build_runtime_budget(config_path),
        "stop_conditions": [
            "Do not begin experiments until plan approval is recorded.",
            "Stop if the best candidate branch cannot emit valid deterministic audit artifacts.",
            "Stop paper generation if the post-audit review still finds unsupported claims after regeneration.",
            "Stop manuscript generation rather than inventing missing citations, evidence, figures, or tables.",
        ],
        "dataset_context_excerpt": dataset_card_text,
        "dataset_context_summary": idea.get("Dataset Context"),
        "review_feedback_history": feedback_items,
    }
    plan["plan_fingerprint"] = fingerprint_research_plan(plan)
    return plan


def render_research_plan_markdown(plan: dict[str, Any]) -> str:
    benchmark = plan["benchmark_summary"]
    lines = [
        "# Research Plan",
        "",
        "## Benchmark Summary",
        f"- Title: {benchmark['title']}",
        f"- Benchmark: {benchmark['benchmark_name']}",
        f"- Dataset: {benchmark.get('dataset_name') or 'unknown'}",
        f"- Split names: {', '.join(benchmark['split_names']) if benchmark['split_names'] else 'unknown'}",
        f"- Plan fingerprint: `{plan['plan_fingerprint']}`",
        "",
        "## Audit Targets",
    ]

    if plan["audit_targets"]:
        lines.extend(f"- {target}" for target in plan["audit_targets"])
    else:
        lines.append("- No explicit audit targets were supplied; use the leakage hypotheses below.")

    lines.extend(["", "## Leakage Hypotheses"])
    for hypothesis in plan["leakage_hypotheses"]:
        suspect_splits = ", ".join(hypothesis["suspect_splits"]) or "declared benchmark splits"
        lines.append(
            f"- {hypothesis['hypothesis_id']}: {hypothesis['target']} across {suspect_splits}. "
            f"Rationale: {hypothesis['rationale']}"
        )

    lines.extend(["", "## Detector Plan"])
    for detector in plan["detector_plan"]:
        lines.append(
            f"- `{detector['detector_name']}` on {', '.join(detector['planned_splits']) or 'all splits'}; "
            f"expect {detector['evidence_expectation']}."
        )

    lines.extend(["", "## Confirmation and Remediation"])
    lines.extend(f"- {item}" for item in plan["confirmation_remediation_plan"])

    lines.extend(["", "## Success Criteria"])
    lines.extend(f"- {item}" for item in plan["success_failure_criteria"]["success"])

    lines.extend(["", "## Failure Criteria"])
    lines.extend(f"- {item}" for item in plan["success_failure_criteria"]["failure"])

    lines.extend(["", "## Expected Artifacts"])
    lines.extend(f"- {item}" for item in plan["expected_artifacts"])

    lines.extend(["", "## Evidence Strategy"])
    lines.extend(f"- {item}" for item in plan["evidence_strategy"])

    lines.extend(["", "## Risks and Benign Alternatives"])
    lines.extend(
        f"- {item}" for item in plan["risks_and_benign_alternatives"] if item
    )

    lines.extend(["", "## Runtime Budget"])
    runtime_budget = plan["runtime_budget"]
    lines.append(f"- Search steps: {runtime_budget['search_budget_steps']}")
    lines.append(f"- Workers: {runtime_budget['num_workers']}")
    lines.append(f"- Multi-seed runs: {runtime_budget['multi_seed_num_seeds']}")
    for key, value in runtime_budget["stage_iteration_limits"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Stop Conditions"])
    lines.extend(f"- {item}" for item in plan["stop_conditions"])

    if plan.get("review_feedback_history"):
        lines.extend(["", "## Review Feedback Incorporated"])
        lines.extend(f"- {item}" for item in plan["review_feedback_history"])

    return "\n".join(lines) + "\n"


def write_research_plan(
    *,
    idea_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    review_round: int = 1,
    feedback_text: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feedback_items = _feedback_to_items(feedback_text)
    plan = build_research_plan(
        idea_path=idea_path,
        output_dir=output_dir,
        config_path=config_path,
        review_round=review_round,
        feedback_items=feedback_items,
    )

    (output_dir / "research_plan.json").write_text(json.dumps(plan, indent=2))
    (output_dir / "research_plan.md").write_text(render_research_plan_markdown(plan))
    return plan
