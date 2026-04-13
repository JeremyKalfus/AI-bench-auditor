import argparse
import json
import os
import os.path as osp
import re
import shutil
import sys
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


MODE_AUDIT = "audit"
MODE_PAPER = "paper"
DEFAULT_IDEAS_PATH = "ai_scientist/ideas/i_cant_believe_its_not_better.json"
DEFAULT_CONFIG_PATH = "bfts_config.yaml"
AUDIT_RUN_METADATA_FILE = "audit_run_metadata.json"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def set_ai_scientist_root() -> None:
    os.environ["AI_SCIENTIST_ROOT"] = os.path.dirname(os.path.abspath(__file__))
    print(f"Set AI_SCIENTIST_ROOT to {os.environ['AI_SCIENTIST_ROOT']}")


def save_token_tracker(idea_dir):
    from ai_scientist.utils.token_tracker import token_tracker

    with open(osp.join(idea_dir, "token_tracker.json"), "w") as f:
        json.dump(token_tracker.get_summary(), f)
    with open(osp.join(idea_dir, "token_tracker_interactions.json"), "w") as f:
        json.dump(token_tracker.get_interactions(), f)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Run AI-bench-auditor in explicit audit or paper mode."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[MODE_AUDIT, MODE_PAPER],
        default=MODE_AUDIT,
        help="Run mode. `audit` consumes raw benchmark inputs; `paper` consumes a prepared audit run directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments, materialize audit-mode scaffolding, print effective settings, and exit without running experiments or writeup.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the base BFTS config used when preparing an audit run.",
    )
    parser.add_argument(
        "--audit-run-dir",
        type=str,
        default=None,
        help="Paper-mode only: path to a prepared audit run directory.",
    )
    parser.add_argument(
        "--audit-num-workers",
        type=positive_int,
        default=1,
        help="Audit-mode only: override `agent.num_workers` in the copied BFTS config. Defaults to 1 for Apple Silicon safety.",
    )
    parser.add_argument(
        "--writeup-type",
        type=str,
        default="icbinb",
        choices=["normal", "icbinb"],
        help="Type of writeup to generate (normal=8 page, icbinb=4 page).",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Audit-mode alias for --load_ideas. Accepts the benchmark idea/spec JSON file.",
    )
    parser.add_argument(
        "--load_ideas",
        type=str,
        default=None,
        help="Audit-mode only: path to a JSON file containing pregenerated ideas.",
    )
    parser.add_argument(
        "--load_code",
        action="store_true",
        help="Audit-mode only: load a Python file with the same name as the ideas file but a .py extension.",
    )
    parser.add_argument(
        "--idea_idx",
        type=int,
        default=0,
        help="Audit-mode only: index of the idea to run.",
    )
    parser.add_argument(
        "--add_dataset_ref",
        action="store_true",
        help="Audit-mode only: add a HF dataset reference to the idea.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Audit-mode only: explicit output directory for the full run bundle.",
    )
    parser.add_argument(
        "--plan-review",
        type=str,
        choices=["required", "skip"],
        default="required",
        help="Audit-mode only: require a human research-plan review before experiments begin.",
    )
    parser.add_argument(
        "--plan-review-mode",
        type=str,
        choices=["interactive", "file"],
        default="interactive",
        help="Audit-mode only: how to collect plan review decisions.",
    )
    parser.add_argument(
        "--plan-feedback-file",
        type=str,
        default=None,
        help="Audit-mode only: optional file containing requested plan changes for file-based review.",
    )
    parser.add_argument(
        "--plan-approval-file",
        type=str,
        default=None,
        help="Audit-mode only: optional approval JSON for file-based review.",
    )
    parser.add_argument(
        "--approve-plan",
        action="store_true",
        help="Audit-mode only: explicitly approve the generated plan without pausing for review.",
    )
    parser.add_argument(
        "--max-plan-revisions",
        type=positive_int,
        default=3,
        help="Audit-mode only: maximum number of plan review rounds including the initial plan.",
    )
    parser.add_argument(
        "--paper-mode",
        type=str,
        choices=["off", "on_success", "always_if_valid"],
        default="on_success",
        help="Audit-mode post-processing policy for audit-native manuscript generation.",
    )
    parser.add_argument(
        "--emit-paper-zip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable emission of paper_bundle.zip.",
    )
    parser.add_argument(
        "--compile-paper-pdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attempt PDF compilation when a LaTeX toolchain is available.",
    )
    parser.add_argument(
        "--allow-source-only",
        action="store_true",
        help="Permit the manuscript stage to succeed without paper.pdf if compilation fails after being attempted.",
    )
    parser.add_argument(
        "--citation-mode",
        type=str,
        choices=["auto", "provided", "off"],
        default="auto",
        help="How the audit-native manuscript stage should source citations.",
    )
    parser.add_argument(
        "--references-file",
        type=str,
        default=None,
        help="Optional BibTeX file used by the audit-native manuscript stage.",
    )
    parser.add_argument(
        "--verification-stack-results",
        type=str,
        default=None,
        help=(
            "Path to verification_stack_results.json used to gate paper generation. "
            "Defaults to verification_results/latest/verification_stack_results.json under the repo root."
        ),
    )
    parser.add_argument(
        "--writeup-retries",
        type=int,
        default=3,
        help="Number of writeup attempts to try.",
    )
    parser.add_argument(
        "--attempt_id",
        type=int,
        default=0,
        help="Attempt ID, used to distinguish same idea in different attempts in parallel runs.",
    )
    parser.add_argument(
        "--model_agg_plots",
        type=str,
        default="o3-mini-2025-01-31",
        help="Model to use for plot aggregation.",
    )
    parser.add_argument(
        "--model_writeup",
        type=str,
        default="o1-preview-2024-09-12",
        help="Model to use for writeup.",
    )
    parser.add_argument(
        "--model_citation",
        type=str,
        default="gpt-4o-2024-11-20",
        help="Model to use for citation gathering.",
    )
    parser.add_argument(
        "--num_cite_rounds",
        type=int,
        default=20,
        help="Number of citation rounds to perform.",
    )
    parser.add_argument(
        "--model_writeup_small",
        type=str,
        default="gpt-4o-2024-05-13",
        help="Smaller model to use for writeup.",
    )
    parser.add_argument(
        "--model_review",
        type=str,
        default="gpt-4o-2024-11-20",
        help="Model to use for review main text and captions.",
    )
    parser.add_argument(
        "--skip_writeup",
        action="store_true",
        help="If set, skip the writeup process.",
    )
    parser.add_argument(
        "--skip_review",
        action="store_true",
        help="If set, skip the review process.",
    )
    return parser.parse_args(argv), parser


def get_available_gpus(gpu_ids=None):
    try:
        import torch
    except ModuleNotFoundError:
        return []

    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    return list(range(torch.cuda.device_count()))


def find_pdf_path_for_review(idea_dir):
    pdf_path = None
    pdf_files = [f for f in os.listdir(idea_dir) if f.endswith(".pdf")]
    reflection_pdfs = [f for f in pdf_files if "reflection" in f]
    if reflection_pdfs:
        final_pdfs = [f for f in reflection_pdfs if "final" in f.lower()]
        if final_pdfs:
            pdf_path = osp.join(idea_dir, final_pdfs[0])
        else:
            reflection_nums = []
            for f in reflection_pdfs:
                match = re.search(r"reflection[_.]?(\d+)", f)
                if match:
                    reflection_nums.append((int(match.group(1)), f))

            if reflection_nums:
                highest_reflection = max(reflection_nums, key=lambda x: x[0])
                pdf_path = osp.join(idea_dir, highest_reflection[1])
            else:
                pdf_path = osp.join(idea_dir, reflection_pdfs[0])
    return pdf_path


@contextmanager
def redirect_stdout_stderr_to_file(log_file_path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log = open(log_file_path, "a")
    sys.stdout = log
    sys.stderr = log
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log.close()


def validate_arguments(args, parser):
    if args.mode == MODE_AUDIT:
        if args.benchmark is not None:
            if args.load_ideas is not None and args.load_ideas != args.benchmark:
                parser.error("--benchmark conflicts with --load_ideas")
            args.load_ideas = args.benchmark
        if args.audit_run_dir is not None:
            parser.error("audit mode does not accept --audit-run-dir")
        if args.load_ideas is None:
            args.load_ideas = DEFAULT_IDEAS_PATH
    else:
        if not args.audit_run_dir:
            parser.error("paper mode requires --audit-run-dir")
        if args.load_ideas is not None or args.benchmark is not None:
            parser.error(
                "paper mode rejects raw benchmark input such as --load_ideas/--benchmark; pass --audit-run-dir instead"
            )
        if args.load_code:
            parser.error("paper mode rejects --load_code")
        if args.add_dataset_ref:
            parser.error("paper mode rejects --add_dataset_ref")
        if args.output_dir is not None:
            parser.error("paper mode rejects --output_dir")

    return args


def build_audit_config_overrides(args) -> dict:
    return {
        "agent": {
            "num_workers": args.audit_num_workers,
            # The existing config notes that num_seeds should match num_workers
            # when num_workers < 3, so keep the audit preset internally consistent.
            "multi_seed_eval": {
                "num_seeds": args.audit_num_workers,
            },
        },
    }


def write_audit_run_metadata(
    idea_dir: str,
    args,
    runtime_settings: dict,
    idea_path_json: str,
    idea_config_path: str,
) -> str:
    metadata_path = osp.join(idea_dir, AUDIT_RUN_METADATA_FILE)
    metadata = {
        "contract_version": 1,
        "mode": MODE_AUDIT,
        "created_at": datetime.now().isoformat(),
        "attempt_id": args.attempt_id,
        "idea_idx": args.idea_idx,
        "idea_source": args.load_ideas,
        "idea_json_path": idea_path_json,
        "run_config_path": idea_config_path,
        "runtime_settings": runtime_settings,
        "review_and_paper_settings": {
            "plan_review": getattr(args, "plan_review", "required"),
            "plan_review_mode": getattr(args, "plan_review_mode", "interactive"),
            "paper_mode": getattr(args, "paper_mode", "on_success"),
            "citation_mode": getattr(args, "citation_mode", "auto"),
        },
        "paper_handoff_contract": {
            "paper_mode_requires_audit_run_dir": True,
            "paper_mode_rejects_raw_benchmark_input": True,
        },
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def prepare_audit_run(args) -> dict:
    from ai_scientist.audits.dataset_context import augment_idea_with_dataset_context
    from ai_scientist.treesearch.bfts_utils import (
        edit_bfts_config_file,
        idea_to_markdown,
    )

    set_ai_scientist_root()

    available_gpus = get_available_gpus()
    print(f"Using GPUs: {available_gpus}")

    with open(args.load_ideas, "r") as f:
        ideas = json.load(f)
        print(f"Loaded {len(ideas)} pregenerated ideas from {args.load_ideas}")

    idea = ideas[args.idea_idx]

    if args.output_dir:
        idea_dir = str(Path(args.output_dir))
    else:
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        idea_dir = f"experiments/{date}_{idea['Name']}_attempt_{args.attempt_id}"
    print(f"Results will be saved in {idea_dir}")
    os.makedirs(idea_dir, exist_ok=True)

    idea_path_md = osp.join(idea_dir, "idea.md")

    code = None
    if args.load_code:
        code_path = args.load_ideas.rsplit(".", 1)[0] + ".py"
        if os.path.exists(code_path):
            with open(code_path, "r") as f:
                code = f.read()
        else:
            print(f"Warning: Code file {code_path} not found")
    else:
        code_path = None

    idea_to_markdown(ideas[args.idea_idx], idea_path_md, code_path)

    dataset_ref_code = None
    if args.add_dataset_ref:
        dataset_ref_path = "hf_dataset_reference.py"
        if os.path.exists(dataset_ref_path):
            with open(dataset_ref_path, "r") as f:
                dataset_ref_code = f.read()
        else:
            print(f"Warning: Dataset reference file {dataset_ref_path} not found")

    if dataset_ref_code is not None and code is not None:
        added_code = dataset_ref_code + "\n" + code
    elif dataset_ref_code is not None and code is None:
        added_code = dataset_ref_code
    elif dataset_ref_code is None and code is not None:
        added_code = code
    else:
        added_code = None

    print(added_code)

    prepared_idea = deepcopy(ideas[args.idea_idx])
    if added_code is not None:
        prepared_idea["Code"] = added_code

    prepared_idea = augment_idea_with_dataset_context(prepared_idea, idea_dir)
    ideas[args.idea_idx] = prepared_idea

    idea_path_json = osp.join(idea_dir, "idea.json")
    with open(idea_path_json, "w") as f:
        json.dump(prepared_idea, f, indent=4)

    runtime_settings = {
        "skip_writeup": True,
        "skip_review": True,
        "run_plot_aggregation": False,
        "agent_num_workers": args.audit_num_workers,
        "agent_multi_seed_num_seeds": args.audit_num_workers,
    }
    idea_config_path = edit_bfts_config_file(
        args.config_path,
        idea_dir,
        idea_path_json,
        config_overrides=build_audit_config_overrides(args),
    )
    metadata_path = write_audit_run_metadata(
        idea_dir=idea_dir,
        args=args,
        runtime_settings=runtime_settings,
        idea_path_json=idea_path_json,
        idea_config_path=idea_config_path,
    )

    return {
        "idea_dir": idea_dir,
        "idea_name": idea["Name"],
        "idea_config_path": idea_config_path,
        "idea_path_json": idea_path_json,
        "metadata_path": metadata_path,
        "runtime_settings": runtime_settings,
    }


def validate_audit_run_dir(audit_run_dir: str) -> tuple[str, dict]:
    resolved_dir = str(Path(audit_run_dir).resolve())
    if not osp.isdir(resolved_dir):
        raise ValueError(
            f"paper mode requires an audit run directory, but got: {audit_run_dir}"
        )

    metadata_path = osp.join(resolved_dir, AUDIT_RUN_METADATA_FILE)
    if not osp.exists(metadata_path):
        raise ValueError(
            f"paper mode requires {AUDIT_RUN_METADATA_FILE} in {resolved_dir}"
        )

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    if metadata.get("mode") != MODE_AUDIT:
        raise ValueError(
            f"paper mode only accepts audit runs, but {metadata_path} declared mode={metadata.get('mode')!r}"
        )

    return resolved_dir, metadata


def run_paper_writeup_and_review(args, audit_run_dir: str) -> None:
    from ai_scientist.llm import create_client
    from ai_scientist.perform_icbinb_writeup import (
        gather_citations,
        perform_writeup as perform_icbinb_writeup,
    )
    from ai_scientist.perform_llm_review import load_paper, perform_review
    from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
    from ai_scientist.perform_writeup import perform_writeup

    if not args.skip_writeup:
        writeup_success = False
        citations_text = gather_citations(
            audit_run_dir,
            num_cite_rounds=args.num_cite_rounds,
            small_model=args.model_citation,
        )
        for attempt in range(args.writeup_retries):
            print(f"Writeup attempt {attempt + 1} of {args.writeup_retries}")
            if args.writeup_type == "normal":
                writeup_success = perform_writeup(
                    base_folder=audit_run_dir,
                    small_model=args.model_writeup_small,
                    big_model=args.model_writeup,
                    page_limit=8,
                    citations_text=citations_text,
                )
            else:
                writeup_success = perform_icbinb_writeup(
                    base_folder=audit_run_dir,
                    small_model=args.model_writeup_small,
                    big_model=args.model_writeup,
                    page_limit=4,
                    citations_text=citations_text,
                )
            if writeup_success:
                break

        if not writeup_success:
            print("Writeup process did not complete successfully after all retries.")

    save_token_tracker(audit_run_dir)

    if not args.skip_review and not args.skip_writeup:
        pdf_path = find_pdf_path_for_review(audit_run_dir)
        if pdf_path and os.path.exists(pdf_path):
            print("Paper found at: ", pdf_path)
            paper_content = load_paper(pdf_path)
            client, client_model = create_client(args.model_review)
            review_text = perform_review(paper_content, client_model, client)
            review_img_cap_ref = perform_imgs_cap_ref_review(
                client, client_model, pdf_path
            )
            with open(osp.join(audit_run_dir, "review_text.txt"), "w") as f:
                f.write(json.dumps(review_text, indent=4))
            with open(osp.join(audit_run_dir, "review_img_cap_ref.json"), "w") as f:
                json.dump(review_img_cap_ref, f, indent=4)
            print("Paper review completed.")
        else:
            print("No paper PDF found for review.")


def _find_best_audit_report_node(manager):
    if manager is None:
        raise ValueError("audit mode did not return a manager state")

    stages = list(getattr(manager, "stages", []))
    journals = getattr(manager, "journals", {})
    cfg = getattr(manager, "cfg", None)

    for stage in reversed(stages):
        journal = journals.get(stage.name)
        if journal is None:
            continue
        best_node = journal.get_best_node(cfg=cfg)
        if best_node and best_node.exp_results_dir:
            return best_node

    raise ValueError("audit mode could not locate a best node with experiment results")


def _resolve_copied_audit_artifact_dir(idea_dir: str, original_artifact_dir: str) -> Path:
    idea_dir_path = Path(idea_dir).resolve()
    original_results_root = idea_dir_path / "logs" / "0-run" / "experiment_results"
    copied_results_root = idea_dir_path / "experiment_results"
    original_artifact_dir_path = Path(original_artifact_dir).resolve()

    try:
        relative_artifact_dir = original_artifact_dir_path.relative_to(
            original_results_root
        )
    except ValueError as exc:
        raise ValueError(
            "best audit artifact directory did not live under logs/0-run/experiment_results: "
            f"{original_artifact_dir_path}"
        ) from exc

    copied_artifact_dir = copied_results_root / relative_artifact_dir
    if not copied_artifact_dir.exists():
        raise ValueError(
            f"copied audit artifact directory does not exist: {copied_artifact_dir}"
        )
    return copied_artifact_dir


def _find_findings_artifact(artifact_dir: Path) -> Path:
    for artifact_name in ("findings.csv", "findings.parquet"):
        candidate = artifact_dir / artifact_name
        if candidate.exists():
            return candidate
    raise ValueError(f"missing findings.csv or findings.parquet in {artifact_dir}")


def _resolve_best_audit_artifact_dir(idea_dir: str, manager) -> Path:
    best_node = _find_best_audit_report_node(manager)
    return _resolve_copied_audit_artifact_dir(idea_dir, best_node.exp_results_dir)


def _link_or_copy_path(source_path: Path, destination_path: Path) -> None:
    if destination_path.exists() or destination_path.is_symlink():
        if destination_path.is_dir() and not destination_path.is_symlink():
            shutil.rmtree(destination_path)
        else:
            destination_path.unlink()

    try:
        relative_target = os.path.relpath(source_path, destination_path.parent)
        os.symlink(relative_target, destination_path, target_is_directory=source_path.is_dir())
    except OSError:
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, destination_path)


def publish_audit_artifacts_to_run_dir(idea_dir: str, artifact_dir: Path, report_path: Path) -> None:
    run_dir = Path(idea_dir)
    promoted_paths = [
        artifact_dir / "dataset_card.md",
        artifact_dir / "audit_results.json",
        artifact_dir / "split_manifest.json",
        report_path,
    ]
    findings_path = _find_findings_artifact(artifact_dir)
    promoted_paths.append(findings_path)

    metrics_path = artifact_dir / "metrics_before_after.json"
    if metrics_path.exists():
        promoted_paths.append(metrics_path)

    evidence_dir = artifact_dir / "evidence"
    if evidence_dir.exists():
        promoted_paths.append(evidence_dir)

    for source_path in promoted_paths:
        if source_path.exists():
            _link_or_copy_path(source_path, run_dir / source_path.name)


def generate_audit_report_for_run(idea_dir: str, manager) -> tuple[Path, Path]:
    from ai_scientist.audits.report import generate_audit_report

    artifact_dir = _resolve_best_audit_artifact_dir(idea_dir, manager)
    audit_results_path = artifact_dir / "audit_results.json"
    split_manifest_path = artifact_dir / "split_manifest.json"
    findings_path = _find_findings_artifact(artifact_dir)
    metrics_before_after_path = artifact_dir / "metrics_before_after.json"
    if not metrics_before_after_path.exists():
        metrics_before_after_path = None

    report_path = generate_audit_report(
        audit_results_path=audit_results_path,
        split_manifest_path=split_manifest_path,
        findings_path=findings_path,
        metrics_before_after_path=metrics_before_after_path,
        output_path=artifact_dir / "audit_report.md",
    )
    print(f"Generated audit report at {report_path}")
    publish_audit_artifacts_to_run_dir(idea_dir, artifact_dir, report_path)
    return artifact_dir, report_path


def _resolve_run_artifact_dir(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    if (run_dir / "audit_results.json").exists() and (run_dir / "audit_report.md").exists():
        return run_dir

    candidate_dirs = []
    experiment_results_dir = run_dir / "experiment_results"
    if experiment_results_dir.exists():
        for candidate in experiment_results_dir.rglob("audit_report.md"):
            if (candidate.parent / "audit_results.json").exists():
                candidate_dirs.append(candidate.parent)

    if len(candidate_dirs) == 1:
        return candidate_dirs[0]

    if not candidate_dirs:
        raise ValueError(
            f"Could not locate audit artifacts inside {run_dir}; expected top-level artifacts or a single experiment_results bundle."
        )

    raise ValueError(
        f"Ambiguous audit artifact bundles in {run_dir}; found {len(candidate_dirs)} candidates."
    )


def _default_verification_stack_results_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "verification_results"
        / "latest"
        / "verification_stack_results.json"
    )


def ensure_paper_generation_preconditions(
    verification_stack_results_path: str | Path | None,
) -> dict:
    results_path = (
        Path(verification_stack_results_path).expanduser().resolve()
        if verification_stack_results_path
        else _default_verification_stack_results_path().resolve()
    )
    if not results_path.exists():
        raise ValueError(
            "paper generation is blocked until Phase 12.1 preconditions pass; "
            f"missing verification stack summary at {results_path}"
        )

    verification = json.loads(results_path.read_text())
    phases = verification.get("phases", {})
    ablation_summary = phases.get("ablation", {}).get("summary", {})
    failures = []
    if verification.get("status") != "passed":
        failures.append("overall verification stack status is not `passed`")
    if not phases.get("schema_gate", {}).get("passed"):
        failures.append("schema gate has not passed")
    if not phases.get("canary", {}).get("summary", {}).get("passed"):
        failures.append("canary suite has not passed")
    if not phases.get("mutation", {}).get("summary", {}).get("passed"):
        failures.append("mutation thresholds have not passed")
    if not ablation_summary.get("passed"):
        failures.append("search ablation has not passed")
    if not ablation_summary.get("full_tree_search_adds_value"):
        failures.append("search ablation does not show tree search adds value")
    if not phases.get("reproducibility", {}).get("summary", {}).get("passed"):
        failures.append("reproducibility has not passed")

    if failures:
        raise ValueError(
            "paper generation is blocked until Phase 12.1 preconditions pass: "
            + "; ".join(failures)
            + f". See {results_path}"
        )
    return {"path": str(results_path), "verification": verification}


def run_post_audit_review_and_paper(
    *,
    run_dir: str,
    artifact_dir: Path,
    args,
) -> dict:
    from ai_scientist.audits.manuscript import build_audit_manuscript_bundle
    from ai_scientist.audits.report_review import (
        ensure_review_passes,
        review_audit_report,
    )

    review_json_path = Path(run_dir) / "audit_report_review.json"
    review_md_path = Path(run_dir) / "audit_report_review.md"
    review = review_audit_report(
        artifact_dir=artifact_dir,
        audit_report_path=Path(run_dir) / "audit_report.md",
        output_json_path=review_json_path,
        output_md_path=review_md_path,
    )
    ensure_review_passes(review)

    if getattr(args, "paper_mode", "on_success") == "off":
        return review

    ensure_paper_generation_preconditions(
        getattr(args, "verification_stack_results", None)
    )
    build_audit_manuscript_bundle(
        run_dir=run_dir,
        artifact_dir=artifact_dir,
        audit_report_review_path=review_json_path,
        citation_mode=getattr(args, "citation_mode", "auto"),
        references_file=getattr(args, "references_file", None),
        compile_pdf=getattr(args, "compile_paper_pdf", True),
        allow_source_only=getattr(args, "allow_source_only", False),
        emit_paper_zip=getattr(args, "emit_paper_zip", True),
    )
    return review


def run_audit_mode(args) -> None:
    from ai_scientist.audits.plan_review import ensure_plan_review
    prepared = prepare_audit_run(args)
    ensure_plan_review(
        output_dir=prepared["idea_dir"],
        idea_path=prepared["idea_path_json"],
        config_path=prepared["idea_config_path"],
        plan_review=getattr(args, "plan_review", "required"),
        plan_review_mode=getattr(args, "plan_review_mode", "interactive"),
        plan_feedback_file=getattr(args, "plan_feedback_file", None),
        plan_approval_file=getattr(args, "plan_approval_file", None),
        approve_plan=getattr(args, "approve_plan", False),
        max_plan_revisions=getattr(args, "max_plan_revisions", 3),
        dry_run=getattr(args, "dry_run", False),
    )
    if args.dry_run:
        print_json(
            {
                "audit_run_dir": prepared["idea_dir"],
                "dry_run": True,
                "metadata_path": prepared["metadata_path"],
                "mode": MODE_AUDIT,
                "plan_approval_path": str(Path(prepared["idea_dir"]) / "plan_approval.json"),
                "plan_review_state_path": str(Path(prepared["idea_dir"]) / "plan_review_state.json"),
                "run_config_path": prepared["idea_config_path"],
                "runtime_settings": prepared["runtime_settings"],
            }
        )
        return

    from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import (
        perform_experiments_bfts,
    )

    try:
        manager = perform_experiments_bfts(prepared["idea_config_path"])
        experiment_results_dir = osp.join(
            prepared["idea_dir"], "logs/0-run/experiment_results"
        )
        copied_results_dir = osp.join(prepared["idea_dir"], "experiment_results")
        if os.path.exists(experiment_results_dir):
            shutil.copytree(
                experiment_results_dir,
                copied_results_dir,
                dirs_exist_ok=True,
            )

        artifact_dir, _report_path = generate_audit_report_for_run(
            prepared["idea_dir"], manager
        )
        run_post_audit_review_and_paper(
            run_dir=prepared["idea_dir"],
            artifact_dir=artifact_dir,
            args=args,
        )
    finally:
        save_token_tracker(prepared["idea_dir"])


def run_paper_mode(args) -> None:
    set_ai_scientist_root()

    audit_run_dir, metadata = validate_audit_run_dir(args.audit_run_dir)
    artifact_dir = _resolve_run_artifact_dir(audit_run_dir)
    if args.dry_run:
        print_json(
            {
                "audit_run_dir": audit_run_dir,
                "artifact_dir": str(artifact_dir),
                "dry_run": True,
                "metadata_mode": metadata.get("mode"),
                "mode": MODE_PAPER,
                "paper_mode": args.paper_mode,
                "citation_mode": args.citation_mode,
            }
        )
        return

    if artifact_dir != Path(audit_run_dir):
        publish_audit_artifacts_to_run_dir(
            audit_run_dir, artifact_dir, artifact_dir / "audit_report.md"
        )
        artifact_dir = Path(audit_run_dir)

    run_post_audit_review_and_paper(
        run_dir=audit_run_dir,
        artifact_dir=artifact_dir,
        args=args,
    )


def cleanup_processes() -> None:
    print("Start cleaning up processes")

    import signal
    try:
        import psutil
    except ModuleNotFoundError:
        print("Skipping process cleanup because `psutil` is not installed.")
        return

    current_process = psutil.Process()
    children = current_process.children(recursive=True)

    for child in children:
        try:
            child.send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    gone, alive = psutil.wait_procs(children, timeout=3)

    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    keywords = ["python", "torch", "mp", "bfts", "experiment"]
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            cmdline = " ".join(proc.cmdline()).lower()
            if any(keyword in cmdline for keyword in keywords):
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=3)
                if proc.is_running():
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            continue


def main(argv=None) -> int:
    from ai_scientist.audits.artifacts import AuditArtifactError
    from ai_scientist.audits.manuscript import ManuscriptGenerationError
    from ai_scientist.audits.plan_review import PlanReviewError

    args, parser = parse_arguments(argv)
    args = validate_arguments(args, parser)

    try:
        if args.mode == MODE_AUDIT:
            run_audit_mode(args)
        else:
            run_paper_mode(args)
    except (AuditArtifactError, ManuscriptGenerationError, PlanReviewError, ValueError) as exc:
        if not args.dry_run:
            cleanup_processes()
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1

    if not args.dry_run:
        cleanup_processes()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
