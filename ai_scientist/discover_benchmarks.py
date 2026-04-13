from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from ai_scientist.llm import (
    AVAILABLE_LLMS,
    create_client,
    extract_json_between_markers,
    get_response_from_llm,
)
from ai_scientist.tools.huggingface_datasets import HuggingFaceDatasetSearchTool
from ai_scientist.tools.semantic_scholar import SemanticScholarSearchTool


QUERY_SYSTEM_PROMPT = """You are a benchmark scout for an audit-first ML tooling repository.
Generate diverse search queries that help discover public benchmarks and datasets worth auditing.

Rules:
- Focus on benchmark names, dataset names, leaderboard terms, and evaluation-task phrases.
- Mix broad and narrow queries.
- Prefer benchmarks that are likely to be public and reproducible.
- Return only JSON in a fenced ```json block.

JSON schema:
{
  "queries": ["...", "..."]
}
"""


CANDIDATE_SYSTEM_PROMPT = """You are turning search evidence into benchmark candidates for an audit workflow.
Use only the provided evidence. Do not invent precise file names, columns, or split details unless they appear in the evidence.

Return only JSON in a fenced ```json block using this schema:
{
  "candidates": [
    {
      "benchmark_name": "string",
      "dataset_name": "string",
      "task": "string",
      "summary": "string",
      "why_promising": "string",
      "auditability": "high|medium|low",
      "confidence": "high|medium|low",
      "source_type": "huggingface_dataset|paper_only|mixed",
      "source_id": "string or null",
      "source_url": "string or null",
      "suggested_audit_targets": ["string"],
      "blocking_unknowns": ["string"],
      "supporting_papers": [
        {
          "title": "string",
          "year": 2024,
          "reason": "string"
        }
      ]
    }
  ]
}
"""


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "benchmark_candidate"


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        key = value.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(value.strip())
    return deduped


def default_queries_for_topic(topic: str, max_queries: int) -> list[str]:
    base = topic.strip()
    queries = [
        base,
        f"{base} benchmark",
        f"{base} dataset",
        f"{base} leaderboard benchmark",
        f"{base} shared task dataset",
        f"{base} evaluation benchmark",
        f"{base} huggingface dataset",
        f"{base} train validation test splits",
        f"{base} public benchmark",
        f"{base} academic benchmark dataset",
        f"{base} reproducible evaluation benchmark",
        f"{base} standard benchmark",
    ]
    return dedupe_preserve_order(queries)[:max_queries]


def generate_queries(
    *,
    topic: str,
    max_queries: int,
    client: Any | None,
    model: str | None,
) -> list[str]:
    fallback = default_queries_for_topic(topic, max_queries)
    if client is None or model is None:
        return fallback

    prompt = (
        f"Topic: {topic}\n"
        f"Generate {max_queries} strong search queries to discover public benchmarks worth auditing."
    )
    response, _history = get_response_from_llm(
        prompt=prompt,
        client=client,
        model=model,
        system_message=QUERY_SYSTEM_PROMPT,
        temperature=0.3,
    )
    payload = extract_json_between_markers(response) or {}
    queries = payload.get("queries")
    if not isinstance(queries, list):
        return fallback

    combined = dedupe_preserve_order(
        [str(query) for query in queries if str(query).strip()] + fallback
    )
    return combined[:max_queries]


def normalize_dataset_hit(result: dict[str, Any], query: str) -> dict[str, Any]:
    dataset_id = result.get("id", "")
    tags = result.get("tags") or []
    card_data = result.get("cardData") or {}
    task_tags = [
        tag.split(":", 1)[1]
        for tag in tags
        if isinstance(tag, str) and tag.startswith("task_ids:")
    ]
    category_tags = [
        tag.split(":", 1)[1]
        for tag in tags
        if isinstance(tag, str) and tag.startswith("task_categories:")
    ]
    return {
        "query": query,
        "id": dataset_id,
        "url": f"https://huggingface.co/datasets/{dataset_id}" if dataset_id else None,
        "author": result.get("author"),
        "downloads": int(result.get("downloads") or 0),
        "likes": int(result.get("likes") or 0),
        "description": (result.get("description") or "").strip(),
        "tags": tags,
        "task_tags": task_tags,
        "category_tags": category_tags,
        "paperswithcode_id": result.get("paperswithcode_id"),
        "card_data": card_data,
    }


def normalize_paper_hit(result: dict[str, Any], query: str) -> dict[str, Any]:
    authors = result.get("authors") or []
    return {
        "query": query,
        "title": (result.get("title") or "").strip(),
        "year": result.get("year"),
        "venue": result.get("venue"),
        "citation_count": int(result.get("citationCount") or 0),
        "abstract": (result.get("abstract") or "").strip(),
        "authors": [author.get("name", "Unknown") for author in authors],
        "url": result.get("url"),
    }


def collect_search_results(
    *,
    queries: list[str],
    paper_tool: SemanticScholarSearchTool,
    dataset_tool: HuggingFaceDatasetSearchTool,
    papers_per_query: int,
    datasets_per_query: int,
) -> dict[str, list[dict[str, Any]]]:
    paper_hits: dict[tuple[str, int | None], dict[str, Any]] = {}
    dataset_hits: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    for query in queries:
        try:
            paper_results = paper_tool.search_for_papers(query) or []
        except Exception as exc:
            paper_results = []
            errors.append(f"Semantic Scholar query failed for {query!r}: {exc}")

        for result in paper_results:
            normalized = normalize_paper_hit(result, query)
            key = (normalized["title"].lower(), normalized["year"])
            existing = paper_hits.get(key)
            if existing is None:
                paper_hits[key] = normalized
            else:
                existing["citation_count"] = max(
                    existing["citation_count"], normalized["citation_count"]
                )
                existing["query"] = existing["query"] + f" | {query}"

        try:
            dataset_results = dataset_tool.search_datasets(query, limit=datasets_per_query) or []
        except Exception as exc:
            dataset_results = []
            errors.append(f"Hugging Face query failed for {query!r}: {exc}")

        for result in dataset_results:
            normalized = normalize_dataset_hit(result, query)
            key = normalized["id"]
            existing = dataset_hits.get(key)
            if existing is None:
                dataset_hits[key] = normalized
                dataset_hits[key]["query_hits"] = [query]
            else:
                existing["downloads"] = max(existing["downloads"], normalized["downloads"])
                existing["likes"] = max(existing["likes"], normalized["likes"])
                existing["query_hits"] = dedupe_preserve_order(
                    existing["query_hits"] + [query]
                )

    papers = sorted(
        paper_hits.values(),
        key=lambda item: (-item["citation_count"], item["title"].lower()),
    )[: max(10, papers_per_query * len(queries))]
    datasets = sorted(
        dataset_hits.values(),
        key=lambda item: (
            -(item["downloads"] + item["likes"] * 10 + len(item.get("query_hits", [])) * 25),
            item["id"].lower(),
        ),
    )
    return {"papers": papers, "datasets": datasets, "errors": errors}


def compact_search_context(
    *,
    topic: str,
    queries: list[str],
    search_results: dict[str, list[dict[str, Any]]],
    max_papers: int = 20,
    max_datasets: int = 20,
) -> str:
    lines = [
        f"Topic: {topic}",
        "Queries:",
        *[f"- {query}" for query in queries],
        "",
        "Top datasets:",
    ]
    for dataset in search_results["datasets"][:max_datasets]:
        lines.extend(
            [
                f"- Dataset: {dataset['id']}",
                f"  URL: {dataset['url']}",
                f"  Downloads: {dataset['downloads']}, Likes: {dataset['likes']}",
                f"  Tasks: {', '.join(dataset['task_tags'] or dataset['category_tags']) or 'unknown'}",
                f"  Description: {dataset['description'][:280] or 'none'}",
            ]
        )

    lines.extend(["", "Top papers:"])
    for paper in search_results["papers"][:max_papers]:
        lines.extend(
            [
                f"- Paper: {paper['title']}",
                f"  Year: {paper['year']}, Venue: {paper['venue']}, Citations: {paper['citation_count']}",
                f"  Abstract: {paper['abstract'][:280] or 'none'}",
            ]
        )

    return "\n".join(lines)


def infer_audit_targets(dataset: dict[str, Any], related_papers: list[dict[str, Any]]) -> list[str]:
    text = " ".join(
        [
            dataset.get("description") or "",
            " ".join(dataset.get("tags") or []),
            " ".join(paper.get("abstract") or "" for paper in related_papers),
        ]
    ).lower()
    targets = ["exact_duplicate"]
    if "text" in text or "paraphrase" in text or "semantic similarity" in text:
        targets.append("near_duplicate")
    if "time" in text or "temporal" in text or "forecast" in text or "event" in text:
        targets.append("temporal_leakage")
    if "user" in text or "speaker" in text or "patient" in text or "entity" in text:
        targets.append("group_overlap")
    if any(token in text for token in ("feature", "tabular", "classification", "regression", "label")):
        targets.append("suspicious_feature_leakage")
    return dedupe_preserve_order(targets)


def build_fallback_candidates(
    *,
    topic: str,
    search_results: dict[str, list[dict[str, Any]]],
    max_candidates: int,
) -> list[dict[str, Any]]:
    candidates = []
    papers = search_results["papers"]
    for dataset in search_results["datasets"][:max_candidates]:
        task = ", ".join(dataset["task_tags"] or dataset["category_tags"]) or "unknown"
        related_papers = [
            paper
            for paper in papers
            if any(
                token in (paper["title"] + " " + paper["abstract"]).lower()
                for token in dataset["id"].lower().replace("-", " ").replace("/", " ").split()
                if len(token) > 3
            )
        ][:3]
        audit_targets = infer_audit_targets(dataset, related_papers)
        confidence = "high" if dataset["downloads"] >= 10000 else "medium"
        auditability = "high" if dataset["downloads"] >= 1000 else "medium"
        candidates.append(
            {
                "benchmark_name": dataset["id"].split("/")[-1].replace("-", " "),
                "dataset_name": dataset["id"],
                "task": task,
                "summary": dataset["description"] or f"Candidate benchmark for {topic}.",
                "why_promising": (
                    f"Found through multiple benchmark-focused searches with "
                    f"{dataset['downloads']} downloads and task tags {task or 'unknown'}."
                ),
                "auditability": auditability,
                "confidence": confidence,
                "source_type": "huggingface_dataset",
                "source_id": dataset["id"],
                "source_url": dataset["url"],
                "suggested_audit_targets": audit_targets,
                "blocking_unknowns": [
                    "Local benchmark files and exact split exports still need to be staged."
                ],
                "supporting_papers": [
                    {
                        "title": paper["title"],
                        "year": paper["year"],
                        "reason": "Likely related benchmark or task reference from discovery search results.",
                    }
                    for paper in related_papers
                ],
            }
        )
    return candidates


def synthesize_candidates(
    *,
    topic: str,
    queries: list[str],
    search_results: dict[str, list[dict[str, Any]]],
    client: Any | None,
    model: str | None,
    max_candidates: int,
) -> list[dict[str, Any]]:
    fallback = build_fallback_candidates(
        topic=topic,
        search_results=search_results,
        max_candidates=max_candidates,
    )
    if client is None or model is None:
        return fallback

    prompt = (
        f"Produce at most {max_candidates} benchmark candidates.\n\n"
        + compact_search_context(
            topic=topic,
            queries=queries,
            search_results=search_results,
        )
    )
    response, _history = get_response_from_llm(
        prompt=prompt,
        client=client,
        model=model,
        system_message=CANDIDATE_SYSTEM_PROMPT,
        temperature=0.2,
    )
    payload = extract_json_between_markers(response) or {}
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return fallback

    normalized_candidates = []
    for candidate in candidates[:max_candidates]:
        if not isinstance(candidate, dict):
            continue
        candidate.setdefault("benchmark_name", candidate.get("dataset_name") or "unknown benchmark")
        candidate.setdefault("dataset_name", candidate.get("benchmark_name") or "unknown dataset")
        candidate.setdefault("task", "unknown")
        candidate.setdefault("summary", "")
        candidate.setdefault("why_promising", "")
        candidate.setdefault("auditability", "medium")
        candidate.setdefault("confidence", "medium")
        candidate.setdefault("source_type", "mixed")
        candidate.setdefault("source_id", None)
        candidate.setdefault("source_url", None)
        candidate["suggested_audit_targets"] = dedupe_preserve_order(
            [str(item) for item in candidate.get("suggested_audit_targets", [])]
        ) or ["exact_duplicate"]
        candidate["blocking_unknowns"] = [
            str(item) for item in candidate.get("blocking_unknowns", [])
        ]
        candidate["supporting_papers"] = [
            paper
            for paper in candidate.get("supporting_papers", [])
            if isinstance(paper, dict) and paper.get("title")
        ]
        normalized_candidates.append(candidate)
    return normalized_candidates or fallback


def candidate_to_idea_spec(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    benchmark_name = candidate["benchmark_name"]
    dataset_name = candidate["dataset_name"]
    source_url = candidate.get("source_url")
    audit_targets = candidate.get("suggested_audit_targets", [])
    supporting_titles = [
        paper["title"]
        for paper in candidate.get("supporting_papers", [])
        if isinstance(paper, dict) and paper.get("title")
    ]
    return [
        {
            "Name": slugify(dataset_name),
            "Title": f"Benchmark Leakage Audit of {benchmark_name}",
            "Abstract": candidate.get("summary")
            or f"Audit the benchmark {benchmark_name} for contamination and split leakage risks.",
            "Short Hypothesis": (
                f"The benchmark {benchmark_name} may contain artifact-level leakage risks that "
                "can be surfaced through deterministic split and evidence analysis."
            ),
            "Experiments": [
                "Stage the benchmark splits locally and emit deterministic dataset context artifacts.",
                "Run the recommended leakage detectors and collect evidence-backed findings.",
                "If issues remain plausible, run remediation or falsification checks and record before/after metrics.",
            ],
            "Risk Factors and Limitations": [
                "This draft spec was produced from discovery results and still requires local data staging.",
                *candidate.get("blocking_unknowns", []),
            ],
            "Audit Targets": audit_targets,
            "Leakage Taxonomy": [
                target.replace("_", " ") for target in audit_targets
            ],
            "Acceptance Criteria": [
                "dataset_card.md exists",
                "split_manifest.json validates",
                "audit_report.md is grounded in deterministic artifacts",
            ],
            "Benchmark Metadata": {
                "benchmark_name": benchmark_name,
                "dataset_name": dataset_name,
                "source_url": source_url,
                "files": [],
            },
            "Discovery Notes": {
                "task": candidate.get("task"),
                "why_promising": candidate.get("why_promising"),
                "auditability": candidate.get("auditability"),
                "confidence": candidate.get("confidence"),
                "supporting_papers": supporting_titles,
            },
        }
    ]


def render_markdown_report(
    *,
    topic: str,
    queries: list[str],
    search_results: dict[str, list[dict[str, Any]]],
    candidates: list[dict[str, Any]],
    spec_paths: dict[str, str],
) -> str:
    lines = [
        "# Benchmark Discovery Report",
        "",
        f"- Topic: {topic}",
        f"- Queries run: {len(queries)}",
        f"- Unique paper hits: {len(search_results['papers'])}",
        f"- Unique dataset hits: {len(search_results['datasets'])}",
        f"- Search warnings: {len(search_results.get('errors', []))}",
        "",
        "## Queries",
        *[f"- {query}" for query in queries],
        "",
    ]
    if search_results.get("errors"):
        lines.extend(["## Search Warnings"])
        lines.extend([f"- {warning}" for warning in search_results["errors"][:20]])
        lines.append("")

    lines.extend([
        "## Top Dataset Hits",
    ])
    for dataset in search_results["datasets"][:15]:
        lines.append(
            f"- `{dataset['id']}`: downloads={dataset['downloads']}, likes={dataset['likes']}, "
            f"tasks={', '.join(dataset['task_tags'] or dataset['category_tags']) or 'unknown'}"
        )

    lines.extend(["", "## Top Paper Hits"])
    for paper in search_results["papers"][:15]:
        lines.append(
            f"- {paper['title']} ({paper['year']}, citations={paper['citation_count']})"
        )

    lines.extend(["", "## Candidate Benchmarks"])
    for candidate in candidates:
        key = slugify(candidate["dataset_name"])
        lines.extend(
            [
                f"### {candidate['benchmark_name']}",
                f"- Dataset: `{candidate['dataset_name']}`",
                f"- Task: {candidate.get('task') or 'unknown'}",
                f"- Source: {candidate.get('source_url') or 'unknown'}",
                f"- Auditability: `{candidate.get('auditability', 'medium')}`",
                f"- Confidence: `{candidate.get('confidence', 'medium')}`",
                f"- Suggested audit targets: {', '.join(candidate.get('suggested_audit_targets', [])) or 'none'}",
                f"- Why promising: {candidate.get('why_promising') or candidate.get('summary') or 'n/a'}",
                f"- Draft spec: `{spec_paths.get(key, 'not written')}`",
            ]
        )
        blocking_unknowns = candidate.get("blocking_unknowns", [])
        if blocking_unknowns:
            lines.append("- Blocking unknowns:")
            lines.extend([f"  - {item}" for item in blocking_unknowns[:4]])
        supporting_papers = candidate.get("supporting_papers", [])
        if supporting_papers:
            lines.append("- Supporting papers:")
            lines.extend(
                [
                    f"  - {paper.get('title')} ({paper.get('year')})"
                    for paper in supporting_papers[:3]
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def discover_benchmarks(
    *,
    topic: str,
    output_dir: str | Path,
    model: str | None = None,
    max_queries: int = 12,
    max_candidates: int = 8,
    papers_per_query: int = 8,
    datasets_per_query: int = 8,
    use_llm: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs_dir = output_dir / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)

    client = None
    client_model = None
    if use_llm and model is not None:
        client, client_model = create_client(model)

    queries = generate_queries(
        topic=topic,
        max_queries=max_queries,
        client=client,
        model=client_model,
    )

    paper_tool = SemanticScholarSearchTool(max_results=papers_per_query)
    dataset_tool = HuggingFaceDatasetSearchTool(max_results=datasets_per_query)
    search_results = collect_search_results(
        queries=queries,
        paper_tool=paper_tool,
        dataset_tool=dataset_tool,
        papers_per_query=papers_per_query,
        datasets_per_query=datasets_per_query,
    )
    candidates = synthesize_candidates(
        topic=topic,
        queries=queries,
        search_results=search_results,
        client=client,
        model=client_model,
        max_candidates=max_candidates,
    )

    spec_paths: dict[str, str] = {}
    for candidate in candidates:
        slug = slugify(candidate["dataset_name"])
        spec_path = specs_dir / f"{slug}.json"
        spec_path.write_text(json.dumps(candidate_to_idea_spec(candidate), indent=2) + "\n")
        spec_paths[slug] = str(spec_path.resolve())

    report = {
        "topic": topic,
        "queries": queries,
        "paper_count": len(search_results["papers"]),
        "dataset_count": len(search_results["datasets"]),
        "search_errors": search_results.get("errors", []),
        "papers": search_results["papers"],
        "datasets": search_results["datasets"],
        "candidates": candidates,
        "spec_paths": spec_paths,
    }

    json_path = output_dir / "benchmark_discovery_results.json"
    md_path = output_dir / "benchmark_discovery_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(
        render_markdown_report(
            topic=topic,
            queries=queries,
            search_results=search_results,
            candidates=candidates,
            spec_paths=spec_paths,
        )
    )
    return {
        "json_path": str(json_path.resolve()),
        "markdown_path": str(md_path.resolve()),
        "spec_dir": str(specs_dir.resolve()),
        "queries": queries,
        "candidate_count": len(candidates),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search broadly for benchmark candidates and write draft audit specs."
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Topic or benchmark area to scout, for example 'reasoning LLM benchmarks'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where discovery artifacts and draft specs should be written.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        choices=AVAILABLE_LLMS,
        help="Model used for query expansion and candidate synthesis.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=12,
        help="How many search queries to run.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=8,
        help="Maximum number of candidate benchmarks to keep.",
    )
    parser.add_argument(
        "--papers-per-query",
        type=int,
        default=8,
        help="Maximum Semantic Scholar paper hits per query.",
    )
    parser.add_argument(
        "--datasets-per-query",
        type=int,
        default=8,
        help="Maximum Hugging Face dataset hits per query.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM-assisted query expansion and candidate synthesis.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = discover_benchmarks(
        topic=args.topic,
        output_dir=args.output_dir,
        model=None if args.no_llm else args.model,
        max_queries=args.max_queries,
        max_candidates=args.max_candidates,
        papers_per_query=args.papers_per_query,
        datasets_per_query=args.datasets_per_query,
        use_llm=not args.no_llm,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
