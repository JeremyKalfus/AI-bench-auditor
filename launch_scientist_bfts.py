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
MODE_STUDY = "study"
DEFAULT_IDEAS_PATH = "ai_scientist/ideas/i_cant_believe_its_not_better.json"
DEFAULT_CONFIG_PATH = "bfts_config.yaml"
AUDIT_RUN_METADATA_FILE = "audit_run_metadata.json"
DEFAULT_DISCOVERY_MODEL = "gpt-4.1-mini"
DISCOVERY_ARTIFACT_DIRNAME = "benchmark_discovery"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_repo_relative_path(path: str | None) -> str | None:
    if path is None:
        return None

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    if candidate.exists():
        return str(candidate.resolve())

    repo_candidate = repo_root() / candidate
    if repo_candidate.exists():
        return str(repo_candidate.resolve())

    return str(candidate)


def slugify_identifier(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return value.strip("_") or "benchmark"


def load_ideas_from_path(path: str) -> list[dict]:
    with open(path, "r") as f:
        ideas = json.load(f)

    if not isinstance(ideas, list) or not ideas:
        raise ValueError(f"Expected a non-empty idea/spec list in {path}")

    return ideas


def derive_discovery_topic(idea: dict) -> str:
    for field_name in ("Discovery Topic", "Title", "Short Hypothesis", "Name"):
        value = idea.get(field_name)
        if isinstance(value, str) and value.strip():
            topic = value.strip()
            break
    else:
        topic = "machine learning benchmarks"

    topic_lower = topic.lower()
    if "benchmark" not in topic_lower and "dataset" not in topic_lower:
        topic = f"{topic} benchmarks and datasets"
    return topic


def select_discovered_spec(report: dict) -> tuple[dict, str, str]:
    candidates = report.get("candidates") or []
    spec_paths = report.get("spec_paths") or {}
    ranked_candidates = []

    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue

        candidate_key = slugify_identifier(
            str(candidate.get("dataset_name") or candidate.get("benchmark_name") or "")
        )
        spec_path = spec_paths.get(candidate_key)
        if not spec_path:
            continue

        source_type = str(candidate.get("source_type") or "").lower()
        source_url = str(candidate.get("source_url") or "").lower()
        dataset_backed = bool(candidate.get("source_id")) or source_type in {
            "huggingface_dataset",
            "mixed",
        }
        if "huggingface.co/datasets/" in source_url:
            dataset_backed = True

        ranked_candidates.append(
            (
                (0 if dataset_backed else 1, candidate_index),
                candidate,
                spec_path,
                dataset_backed,
            )
        )

    if not ranked_candidates:
        raise ValueError(
            "Benchmark discovery completed but did not emit a selectable draft spec."
        )

    _, candidate, spec_path, dataset_backed = min(
        ranked_candidates, key=lambda item: item[0]
    )
    selection_strategy = (
        "dataset_backed_first" if dataset_backed else "top_candidate_fallback"
    )
    return candidate, spec_path, selection_strategy


def ensure_audit_output_dir(args, topic: str) -> None:
    if args.output_dir:
        return

    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    topic_slug = slugify_identifier(topic)[:80]
    args.output_dir = f"experiments/{date}_{topic_slug}_attempt_{args.attempt_id}"


def autodiscover_benchmark_spec(args) -> None:
    from ai_scientist.discover_benchmarks import discover_benchmarks

    seed_ideas_path = resolve_repo_relative_path(DEFAULT_IDEAS_PATH)
    assert seed_ideas_path is not None
    seed_ideas = load_ideas_from_path(seed_ideas_path)
    if args.idea_idx >= len(seed_ideas):
        raise ValueError(
            f"--idea_idx {args.idea_idx} is out of range for default discovery ideas at {seed_ideas_path}"
        )

    seed_idea_idx = args.idea_idx
    topic = args.discover_topic or derive_discovery_topic(seed_ideas[seed_idea_idx])
    ensure_audit_output_dir(args, topic)

    discovery_output_dir = Path(
        args.discovery_output_dir or Path(args.output_dir) / DISCOVERY_ARTIFACT_DIRNAME
    )
    discovery_kwargs = {
        "topic": topic,
        "output_dir": discovery_output_dir,
        "model": args.discovery_model,
        "max_queries": args.discovery_max_queries,
        "max_candidates": args.discovery_max_candidates,
        "use_llm": True,
    }

    try:
        result = discover_benchmarks(**discovery_kwargs)
        used_llm = True
    except Exception as exc:
        print(
            "Automatic benchmark discovery could not use LLM-assisted query expansion "
            f"({exc}). Retrying with deterministic query expansion."
        )
        result = discover_benchmarks(
            **{
                **discovery_kwargs,
                "model": None,
                "use_llm": False,
            }
        )
        used_llm = False

    report = json.loads(Path(result["json_path"]).read_text())
    selected_candidate, spec_path, selection_strategy = select_discovered_spec(report)

    args.load_ideas = spec_path
    args.idea_idx = 0
    args.benchmark_discovery = {
        "topic": topic,
        "source_idea_idx": seed_idea_idx,
        "selection_strategy": selection_strategy,
        "used_llm": used_llm,
        "report_json_path": result["json_path"],
        "report_markdown_path": result["markdown_path"],
        "selected_spec_path": spec_path,
        "selected_candidate": {
            "benchmark_name": selected_candidate.get("benchmark_name"),
            "dataset_name": selected_candidate.get("dataset_name"),
            "source_type": selected_candidate.get("source_type"),
            "source_url": selected_candidate.get("source_url"),
        },
    }

    print(f"Auto-discovery topic: {topic}")
    print(f"Discovery report: {result['markdown_path']}")
    print(f"Selected draft spec: {spec_path}")


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
        description="Run AI-bench-auditor in explicit audit or study mode."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[MODE_AUDIT, MODE_STUDY],
        default=MODE_AUDIT,
        help=(
            "Run mode. `audit` consumes raw benchmark inputs; `study` consumes a prepared "
            "audit run directory and rebuilds the markdown-first study bundle."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments, materialize audit-mode scaffolding, print effective settings, and exit without running experiments.",
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
        help="Study-mode only: path to a prepared audit run directory.",
    )
    parser.add_argument(
        "--audit-num-workers",
        type=positive_int,
        default=1,
        help="Audit-mode only: override `agent.num_workers` in the copied BFTS config. Defaults to 1 for Apple Silicon safety.",
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
        "--discover-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Audit-mode only: when no benchmark spec is provided, scout public benchmarks "
            "first and auto-select a draft spec. Enabled by default."
        ),
    )
    parser.add_argument(
        "--discover-topic",
        type=str,
        default=None,
        help=(
            "Audit-mode only: override the topic used for automatic benchmark discovery. "
            "If omitted, discovery is seeded from the bundled AI Scientist ideas."
        ),
    )
    parser.add_argument(
        "--discovery-output-dir",
        type=str,
        default=None,
        help=(
            "Audit-mode only: directory where discovery artifacts should be written. "
            "Defaults to <output_dir>/benchmark_discovery for auto-discovery runs."
        ),
    )
    parser.add_argument(
        "--discovery-model",
        type=str,
        default=DEFAULT_DISCOVERY_MODEL,
        help="Audit-mode only: model used for automatic benchmark discovery.",
    )
    parser.add_argument(
        "--discovery-max-queries",
        type=positive_int,
        default=12,
        help="Audit-mode only: number of discovery queries to run.",
    )
    parser.add_argument(
        "--discovery-max-candidates",
        type=positive_int,
        default=8,
        help="Audit-mode only: maximum number of discovered benchmark candidates to keep.",
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
        "--emit-study-zip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable emission of study_figures.zip.",
    )
    parser.add_argument(
        "--attempt_id",
        type=int,
        default=0,
        help="Attempt ID, used to distinguish same idea in different attempts in parallel runs.",
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
    args.config_path = resolve_repo_relative_path(args.config_path)
    if args.mode == MODE_AUDIT:
        if args.benchmark is not None:
            if args.load_ideas is not None and args.load_ideas != args.benchmark:
                parser.error("--benchmark conflicts with --load_ideas")
            args.load_ideas = args.benchmark
        if args.audit_run_dir is not None:
            parser.error("audit mode does not accept --audit-run-dir")
        args.load_ideas = resolve_repo_relative_path(args.load_ideas)
        if args.load_ideas is None and not args.discover_first:
            args.load_ideas = resolve_repo_relative_path(DEFAULT_IDEAS_PATH)
    else:
        if not args.audit_run_dir:
            parser.error("study mode requires --audit-run-dir")
        if args.load_ideas is not None or args.benchmark is not None:
            parser.error(
                "study mode rejects raw benchmark input such as --load_ideas/--benchmark; pass --audit-run-dir instead"
            )
        if args.load_code:
            parser.error("study mode rejects --load_code")
        if args.add_dataset_ref:
            parser.error("study mode rejects --add_dataset_ref")
        if args.output_dir is not None:
            parser.error("study mode rejects --output_dir")
        if args.discover_topic is not None:
            parser.error("study mode rejects --discover-topic")
        if args.discovery_output_dir is not None:
            parser.error("study mode rejects --discovery-output-dir")

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
        "review_and_output_settings": {
            "output_surface": "study_bundle",
            "plan_review": getattr(args, "plan_review", "required"),
            "plan_review_mode": getattr(args, "plan_review_mode", "interactive"),
            "emit_study_zip": getattr(args, "emit_study_zip", True),
        },
        "study_handoff_contract": {
            "study_mode_requires_audit_run_dir": True,
            "study_mode_rejects_raw_benchmark_input": True,
        },
    }
    if getattr(args, "benchmark_discovery", None) is not None:
        metadata["benchmark_discovery"] = args.benchmark_discovery
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

    ideas = load_ideas_from_path(args.load_ideas)
    print(f"Loaded {len(ideas)} pregenerated ideas from {args.load_ideas}")
    if args.idea_idx >= len(ideas):
        raise ValueError(
            f"--idea_idx {args.idea_idx} is out of range for {args.load_ideas}"
        )

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

    benchmark_metadata = prepared_idea.get("Benchmark Metadata")
    benchmark_files = []
    if isinstance(benchmark_metadata, dict):
        benchmark_files = benchmark_metadata.get("files") or []
    if not benchmark_files:
        print(
            "Warning: Benchmark Metadata.files is empty, so no local dataset files were "
            "staged. Auto-discovered draft specs still need benchmark splits added before "
            "a real audit can be trusted."
        )

    prepared_idea = augment_idea_with_dataset_context(prepared_idea, idea_dir)
    ideas[args.idea_idx] = prepared_idea
    dataset_context_staged = "Dataset Context" in prepared_idea

    idea_path_json = osp.join(idea_dir, "idea.json")
    with open(idea_path_json, "w") as f:
        json.dump(prepared_idea, f, indent=4)

    runtime_settings = {
        "skip_writeup": True,
        "skip_review": True,
        "run_plot_aggregation": False,
        "agent_num_workers": args.audit_num_workers,
        "agent_multi_seed_num_seeds": args.audit_num_workers,
        "dataset_context_staged": dataset_context_staged,
        "used_benchmark_discovery": bool(getattr(args, "benchmark_discovery", None)),
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
            f"study mode requires an audit run directory, but got: {audit_run_dir}"
        )

    metadata_path = osp.join(resolved_dir, AUDIT_RUN_METADATA_FILE)
    if not osp.exists(metadata_path):
        raise ValueError(
            f"study mode requires {AUDIT_RUN_METADATA_FILE} in {resolved_dir}"
        )

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    if metadata.get("mode") != MODE_AUDIT:
        raise ValueError(
            f"study mode only accepts audit runs, but {metadata_path} declared mode={metadata.get('mode')!r}"
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
    run_dir = Path(idea_dir).resolve()
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


def _resolve_audit_report_path(
    *,
    run_dir: str | Path,
    artifact_dir: Path,
    audit_report_path: str | Path | None = None,
) -> Path:
    run_dir_path = Path(run_dir).resolve()
    artifact_dir = artifact_dir.resolve()

    candidates: list[Path] = []
    if audit_report_path is not None:
        candidates.append(Path(audit_report_path).resolve())
    candidates.extend([run_dir_path / "audit_report.md", artifact_dir / "audit_report.md"])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate audit_report.md in the run directory or artifact directory. "
        f"Checked: {', '.join(str(path) for path in candidates)}"
    )


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


def run_post_audit_review_and_study_bundle(
    *,
    run_dir: str,
    artifact_dir: Path,
    args,
    audit_report_path: str | Path | None = None,
) -> dict:
    from ai_scientist.audits.report_review import (
        ensure_review_passes,
        review_audit_report,
    )
    from ai_scientist.audits.study import build_audit_study_bundle

    run_dir_path = Path(run_dir).resolve()
    resolved_report_path = _resolve_audit_report_path(
        run_dir=run_dir_path,
        artifact_dir=artifact_dir,
        audit_report_path=audit_report_path,
    )
    review_json_path = run_dir_path / "audit_report_review.json"
    review_md_path = run_dir_path / "audit_report_review.md"
    review = review_audit_report(
        artifact_dir=artifact_dir,
        audit_report_path=resolved_report_path,
        output_json_path=review_json_path,
        output_md_path=review_md_path,
    )
    ensure_review_passes(review)
    build_audit_study_bundle(
        run_dir=run_dir_path,
        artifact_dir=artifact_dir,
        audit_report_review_path=review_json_path,
        emit_figures_zip=getattr(args, "emit_study_zip", True),
    )
    return review


def run_audit_mode(args) -> None:
    from ai_scientist.audits.plan_review import ensure_plan_review

    if hasattr(args, "load_ideas") and args.load_ideas is None:
        autodiscover_benchmark_spec(args)

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

        artifact_dir, report_path = generate_audit_report_for_run(
            prepared["idea_dir"], manager
        )
        run_post_audit_review_and_study_bundle(
            run_dir=prepared["idea_dir"],
            artifact_dir=artifact_dir,
            args=args,
            audit_report_path=report_path,
        )
    finally:
        save_token_tracker(prepared["idea_dir"])


def run_study_mode(args) -> None:
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
                "mode": MODE_STUDY,
                "emit_study_zip": args.emit_study_zip,
            }
        )
        return

    if artifact_dir != Path(audit_run_dir):
        publish_audit_artifacts_to_run_dir(
            audit_run_dir, artifact_dir, artifact_dir / "audit_report.md"
        )
        artifact_dir = Path(audit_run_dir)

    run_post_audit_review_and_study_bundle(
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
    from ai_scientist.audits.plan_review import PlanReviewError
    from ai_scientist.audits.study import StudyBundleGenerationError

    args, parser = parse_arguments(argv)
    args = validate_arguments(args, parser)

    try:
        if args.mode == MODE_AUDIT:
            run_audit_mode(args)
        else:
            run_study_mode(args)
    except (AuditArtifactError, PlanReviewError, StudyBundleGenerationError, ValueError) as exc:
        if not args.dry_run:
            cleanup_processes()
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1

    if not args.dry_run:
        cleanup_processes()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
