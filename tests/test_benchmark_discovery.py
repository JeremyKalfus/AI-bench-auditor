import json

from ai_scientist.discover_benchmarks import (
    build_fallback_candidates,
    candidate_to_idea_spec,
    default_queries_for_topic,
    discover_benchmarks,
)


def test_default_queries_for_topic_expand_search_space():
    queries = default_queries_for_topic("medical imaging fairness", max_queries=12)

    assert queries[0] == "medical imaging fairness"
    assert "medical imaging fairness benchmark" in queries
    assert "medical imaging fairness dataset" in queries
    assert len(queries) == 12


def test_candidate_to_idea_spec_produces_audit_shaped_json():
    candidate = {
        "benchmark_name": "GLUE MRPC",
        "dataset_name": "nyu-mll/glue",
        "task": "paraphrase-detection",
        "summary": "A classic paraphrase benchmark.",
        "why_promising": "It is well-known and easy to stage.",
        "auditability": "high",
        "confidence": "high",
        "source_url": "https://huggingface.co/datasets/nyu-mll/glue",
        "suggested_audit_targets": ["exact_duplicate", "near_duplicate"],
        "blocking_unknowns": ["Need local split exports."],
        "supporting_papers": [{"title": "GLUE", "year": 2018}],
    }

    spec = candidate_to_idea_spec(candidate)

    assert isinstance(spec, list)
    assert spec[0]["Name"] == "nyu_mll_glue"
    assert spec[0]["Benchmark Metadata"]["dataset_name"] == "nyu-mll/glue"
    assert spec[0]["Benchmark Metadata"]["files"] == []
    assert spec[0]["Audit Targets"] == ["exact_duplicate", "near_duplicate"]


def test_build_fallback_candidates_prefers_dataset_hits():
    search_results = {
        "papers": [
            {
                "title": "GLUE: A Multi-Task Benchmark and Analysis Platform",
                "year": 2018,
                "citation_count": 10000,
                "abstract": "GLUE is a benchmark for natural language understanding.",
                "venue": "ICLR",
                "authors": ["Alex Wang"],
                "url": None,
                "query": "glue benchmark",
            }
        ],
        "datasets": [
            {
                "id": "nyu-mll/glue",
                "url": "https://huggingface.co/datasets/nyu-mll/glue",
                "downloads": 200000,
                "likes": 100,
                "description": "General Language Understanding Evaluation benchmark.",
                "tags": ["task_ids:paraphrase-detection", "modality:text"],
                "task_tags": ["paraphrase-detection"],
                "category_tags": [],
                "query_hits": ["glue benchmark"],
            }
        ],
    }

    candidates = build_fallback_candidates(
        topic="nlp benchmarks",
        search_results=search_results,
        max_candidates=3,
    )

    assert len(candidates) == 1
    assert candidates[0]["dataset_name"] == "nyu-mll/glue"
    assert "near_duplicate" in candidates[0]["suggested_audit_targets"]


def test_discover_benchmarks_writes_report_without_llm(tmp_path, monkeypatch):
    fake_search_results = {
        "papers": [
            {
                "title": "MMLU: Massive Multitask Language Understanding",
                "year": 2021,
                "citation_count": 5000,
                "abstract": "A benchmark for multitask evaluation.",
                "venue": "ICLR",
                "authors": ["Dan Hendrycks"],
                "url": None,
                "query": "mmlu benchmark",
            }
        ],
        "datasets": [
            {
                "id": "cais/mmlu",
                "url": "https://huggingface.co/datasets/cais/mmlu",
                "downloads": 50000,
                "likes": 300,
                "description": "MMLU benchmark dataset.",
                "tags": ["task_categories:text-classification", "modality:text"],
                "task_tags": [],
                "category_tags": ["text-classification"],
                "query_hits": ["mmlu benchmark"],
            }
        ],
    }

    monkeypatch.setattr(
        "ai_scientist.discover_benchmarks.collect_search_results",
        lambda **_kwargs: fake_search_results,
    )

    result = discover_benchmarks(
        topic="reasoning benchmarks",
        output_dir=tmp_path,
        model=None,
        use_llm=False,
        max_queries=5,
        max_candidates=4,
        papers_per_query=3,
        datasets_per_query=3,
    )

    report_json = tmp_path / "benchmark_discovery_results.json"
    report_md = tmp_path / "benchmark_discovery_report.md"
    spec_files = list((tmp_path / "specs").glob("*.json"))

    assert result["candidate_count"] == 1
    assert report_json.exists()
    assert report_md.exists()
    assert len(spec_files) == 1

    payload = json.loads(report_json.read_text())
    assert payload["candidates"][0]["dataset_name"] == "cais/mmlu"
