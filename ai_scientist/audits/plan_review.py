from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .research_plan import write_research_plan


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _summarize_feedback(feedback_text: str | None) -> str:
    if not feedback_text:
        return ""
    lines = [line.strip() for line in feedback_text.splitlines() if line.strip()]
    return " | ".join(lines[:5])


class PlanReviewError(RuntimeError):
    """Base error for plan-review failures."""


class PlanApprovalRequiredError(PlanReviewError):
    """Raised when a required plan approval has not been granted."""


class PlanRejectedError(PlanReviewError):
    """Raised when the reviewer aborts or explicitly rejects the plan."""


@dataclass(frozen=True)
class PlanReviewResult:
    plan: dict[str, Any]
    approval: dict[str, Any]
    state: dict[str, Any]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_feedback(path: Path, feedback_text: str | None) -> None:
    if feedback_text is None:
        return
    path.write_text(feedback_text.rstrip() + "\n")


def _review_state_payload(
    *,
    status: str,
    review_mode: str,
    review_round: int,
    max_plan_revisions: int,
    plan: dict[str, Any],
    approved: bool,
    feedback_summary: str,
) -> dict[str, Any]:
    return {
        "contract_version": 1,
        "status": status,
        "review_mode": review_mode,
        "review_round": review_round,
        "max_plan_revisions": max_plan_revisions,
        "approved": approved,
        "feedback_summary": feedback_summary,
        "plan_fingerprint": plan["plan_fingerprint"],
        "updated_at": _timestamp(),
    }


def _approval_payload(
    *,
    approved: bool,
    reviewer_mode: str,
    review_round: int,
    feedback_summary: str,
    plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "approved": approved,
        "reviewer_mode": reviewer_mode,
        "review_round": review_round,
        "timestamp": _timestamp(),
        "summarized_feedback": feedback_summary,
        "final_plan_hash": plan["plan_fingerprint"],
    }


def _load_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_optional_text(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    text = path.read_text().strip()
    return text or None


def ensure_plan_review(
    *,
    output_dir: str | Path,
    idea_path: str | Path,
    config_path: str | Path | None,
    plan_review: str,
    plan_review_mode: str,
    plan_feedback_file: str | Path | None,
    plan_approval_file: str | Path | None,
    approve_plan: bool,
    max_plan_revisions: int,
    dry_run: bool = False,
    input_func: Callable[[str], str] | None = None,
    is_tty: bool | None = None,
) -> PlanReviewResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_feedback_path = output_dir / "plan_feedback.md"
    canonical_state_path = output_dir / "plan_review_state.json"
    canonical_approval_path = output_dir / "plan_approval.json"
    if input_func is None:
        input_func = input

    review_round = 1
    feedback_text = None
    plan = write_research_plan(
        idea_path=idea_path,
        output_dir=output_dir,
        config_path=config_path,
        review_round=review_round,
        feedback_text=None,
    )

    if plan_review == "skip":
        approval = _approval_payload(
            approved=True,
            reviewer_mode="skip",
            review_round=review_round,
            feedback_summary="Plan review explicitly skipped.",
            plan=plan,
        )
        state = _review_state_payload(
            status="skipped",
            review_mode="skip",
            review_round=review_round,
            max_plan_revisions=max_plan_revisions,
            plan=plan,
            approved=True,
            feedback_summary=approval["summarized_feedback"],
        )
        _write_json(canonical_approval_path, approval)
        _write_json(canonical_state_path, state)
        return PlanReviewResult(plan=plan, approval=approval, state=state)

    initial_state = _review_state_payload(
        status="awaiting_review",
        review_mode=plan_review_mode,
        review_round=review_round,
        max_plan_revisions=max_plan_revisions,
        plan=plan,
        approved=False,
        feedback_summary="",
    )
    _write_json(canonical_state_path, initial_state)

    if dry_run:
        approval = _approval_payload(
            approved=False,
            reviewer_mode=plan_review_mode,
            review_round=review_round,
            feedback_summary="Dry run generated the plan but did not request approval.",
            plan=plan,
        )
        _write_json(canonical_approval_path, approval)
        return PlanReviewResult(plan=plan, approval=approval, state=initial_state)

    if approve_plan:
        approval = _approval_payload(
            approved=True,
            reviewer_mode="approve_plan",
            review_round=review_round,
            feedback_summary="Plan approved via --approve-plan.",
            plan=plan,
        )
        state = _review_state_payload(
            status="approved",
            review_mode="approve_plan",
            review_round=review_round,
            max_plan_revisions=max_plan_revisions,
            plan=plan,
            approved=True,
            feedback_summary=approval["summarized_feedback"],
        )
        _write_json(canonical_approval_path, approval)
        _write_json(canonical_state_path, state)
        return PlanReviewResult(plan=plan, approval=approval, state=state)

    if is_tty is None:
        is_tty = sys.stdin.isatty()

    interactive_mode = plan_review_mode == "interactive" and is_tty

    if interactive_mode:
        while True:
            prompt = (
                f"Research plan written to {output_dir / 'research_plan.md'}.\n"
                "Approve plan, request changes, or abort? [approve/changes/abort]: "
            )
            response = input_func(prompt).strip().lower()
            if response in {"approve", "a"}:
                approval = _approval_payload(
                    approved=True,
                    reviewer_mode="interactive",
                    review_round=review_round,
                    feedback_summary=_summarize_feedback(feedback_text),
                    plan=plan,
                )
                state = _review_state_payload(
                    status="approved",
                    review_mode="interactive",
                    review_round=review_round,
                    max_plan_revisions=max_plan_revisions,
                    plan=plan,
                    approved=True,
                    feedback_summary=approval["summarized_feedback"],
                )
                _write_json(canonical_approval_path, approval)
                _write_json(canonical_state_path, state)
                return PlanReviewResult(plan=plan, approval=approval, state=state)

            if response in {"abort", "x"}:
                summary = _summarize_feedback(feedback_text) or "Reviewer aborted the run before approval."
                approval = _approval_payload(
                    approved=False,
                    reviewer_mode="interactive",
                    review_round=review_round,
                    feedback_summary=summary,
                    plan=plan,
                )
                state = _review_state_payload(
                    status="aborted",
                    review_mode="interactive",
                    review_round=review_round,
                    max_plan_revisions=max_plan_revisions,
                    plan=plan,
                    approved=False,
                    feedback_summary=summary,
                )
                _write_json(canonical_approval_path, approval)
                _write_json(canonical_state_path, state)
                raise PlanRejectedError("Plan review aborted before research execution.")

            if response in {"changes", "change", "c"}:
                if review_round >= max_plan_revisions:
                    summary = "Reviewer requested another revision after reaching the configured revision limit."
                    approval = _approval_payload(
                        approved=False,
                        reviewer_mode="interactive",
                        review_round=review_round,
                        feedback_summary=summary,
                        plan=plan,
                    )
                    state = _review_state_payload(
                        status="revision_limit_reached",
                        review_mode="interactive",
                        review_round=review_round,
                        max_plan_revisions=max_plan_revisions,
                        plan=plan,
                        approved=False,
                        feedback_summary=summary,
                    )
                    _write_json(canonical_approval_path, approval)
                    _write_json(canonical_state_path, state)
                    raise PlanRejectedError(summary)

                feedback_text = input_func("Enter plan feedback: ").strip()
                _write_feedback(canonical_feedback_path, feedback_text)
                review_round += 1
                plan = write_research_plan(
                    idea_path=idea_path,
                    output_dir=output_dir,
                    config_path=config_path,
                    review_round=review_round,
                    feedback_text=feedback_text,
                )
                state = _review_state_payload(
                    status="revised",
                    review_mode="interactive",
                    review_round=review_round,
                    max_plan_revisions=max_plan_revisions,
                    plan=plan,
                    approved=False,
                    feedback_summary=_summarize_feedback(feedback_text),
                )
                _write_json(canonical_state_path, state)
                continue

            continue

    feedback_text = _load_optional_text(plan_feedback_file)
    if feedback_text:
        if review_round >= max_plan_revisions:
            summary = "A plan feedback file was provided after reaching the revision limit."
            approval = _approval_payload(
                approved=False,
                reviewer_mode="file",
                review_round=review_round,
                feedback_summary=summary,
                plan=plan,
            )
            state = _review_state_payload(
                status="revision_limit_reached",
                review_mode="file",
                review_round=review_round,
                max_plan_revisions=max_plan_revisions,
                plan=plan,
                approved=False,
                feedback_summary=summary,
            )
            _write_json(canonical_approval_path, approval)
            _write_json(canonical_state_path, state)
            raise PlanRejectedError(summary)

        _write_feedback(canonical_feedback_path, feedback_text)
        review_round += 1
        plan = write_research_plan(
            idea_path=idea_path,
            output_dir=output_dir,
            config_path=config_path,
            review_round=review_round,
            feedback_text=feedback_text,
        )
        revised_state = _review_state_payload(
            status="revised",
            review_mode="file",
            review_round=review_round,
            max_plan_revisions=max_plan_revisions,
            plan=plan,
            approved=False,
            feedback_summary=_summarize_feedback(feedback_text),
        )
        _write_json(canonical_state_path, revised_state)

    approval_request = _load_optional_json(plan_approval_file)
    if approval_request is not None:
        approved = bool(approval_request.get("approved"))
        provided_hash = approval_request.get("final_plan_hash") or approval_request.get(
            "plan_fingerprint"
        )
        if provided_hash and provided_hash != plan["plan_fingerprint"]:
            summary = (
                "The supplied approval file targeted a different plan fingerprint than the current plan."
            )
            approval = _approval_payload(
                approved=False,
                reviewer_mode="file",
                review_round=review_round,
                feedback_summary=summary,
                plan=plan,
            )
            state = _review_state_payload(
                status="approval_hash_mismatch",
                review_mode="file",
                review_round=review_round,
                max_plan_revisions=max_plan_revisions,
                plan=plan,
                approved=False,
                feedback_summary=summary,
            )
            _write_json(canonical_approval_path, approval)
            _write_json(canonical_state_path, state)
            raise PlanApprovalRequiredError(summary)

        summary = approval_request.get("summarized_feedback") or _summarize_feedback(
            feedback_text
        )
        approval = _approval_payload(
            approved=approved,
            reviewer_mode="file",
            review_round=review_round,
            feedback_summary=summary,
            plan=plan,
        )
        state = _review_state_payload(
            status="approved" if approved else "rejected",
            review_mode="file",
            review_round=review_round,
            max_plan_revisions=max_plan_revisions,
            plan=plan,
            approved=approved,
            feedback_summary=summary,
        )
        _write_json(canonical_approval_path, approval)
        _write_json(canonical_state_path, state)
        if not approved:
            raise PlanRejectedError("Plan approval file marked the plan as not approved.")
        return PlanReviewResult(plan=plan, approval=approval, state=state)

    summary = _summarize_feedback(feedback_text)
    approval = _approval_payload(
        approved=False,
        reviewer_mode="file" if plan_review_mode == "file" or not is_tty else "interactive",
        review_round=review_round,
        feedback_summary=summary,
        plan=plan,
    )
    state = _review_state_payload(
        status="awaiting_review",
        review_mode="file" if plan_review_mode == "file" or not is_tty else "interactive",
        review_round=review_round,
        max_plan_revisions=max_plan_revisions,
        plan=plan,
        approved=False,
        feedback_summary=summary,
    )
    _write_json(canonical_approval_path, approval)
    _write_json(canonical_state_path, state)
    raise PlanApprovalRequiredError(
        "Plan approval is required before audit execution can start. "
        f"Review {output_dir / 'research_plan.md'} and rerun with --approve-plan or --plan-approval-file."
    )
