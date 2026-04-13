import math
from typing import Any

import backoff
import requests

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.tools.semantic_scholar import on_backoff


class HuggingFaceDatasetSearchTool(BaseTool):
    def __init__(
        self,
        name: str = "SearchHuggingFaceDatasets",
        description: str = (
            "Search Hugging Face datasets to find benchmark and dataset candidates."
        ),
        max_results: int = 10,
    ):
        parameters = [
            {
                "name": "query",
                "type": "str",
                "description": "The search query to find relevant benchmark datasets.",
            }
        ]
        super().__init__(name, description, parameters)
        self.max_results = max_results

    def use_tool(self, query: str) -> str:
        datasets = self.search_datasets(query)
        if not datasets:
            return "No datasets found."
        return self.format_datasets(datasets)

    @backoff.on_exception(
        backoff.expo,
        (
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ),
        on_backoff=on_backoff,
        max_tries=3,
    )
    def search_datasets(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        if not query:
            return []

        effective_limit = limit or self.max_results
        response = requests.get(
            "https://huggingface.co/api/datasets",
            params={"search": query, "limit": effective_limit},
            timeout=30,
        )
        response.raise_for_status()
        results = response.json()
        if not isinstance(results, list):
            return []

        enriched_results = []
        for result in results:
            dataset_id = result.get("id")
            if not dataset_id:
                continue
            details = self.get_dataset_details(dataset_id)
            merged = dict(result)
            if details:
                merged.update(details)
            enriched_results.append(merged)

        enriched_results.sort(key=self._ranking_score, reverse=True)
        return enriched_results[:effective_limit]

    @backoff.on_exception(
        backoff.expo,
        (
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ),
        on_backoff=on_backoff,
        max_tries=3,
    )
    def get_dataset_details(self, dataset_id: str) -> dict[str, Any]:
        response = requests.get(
            f"https://huggingface.co/api/datasets/{dataset_id}",
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return {}
        return payload

    @staticmethod
    def _ranking_score(dataset: dict[str, Any]) -> float:
        downloads = max(int(dataset.get("downloads") or 0), 0)
        likes = max(int(dataset.get("likes") or 0), 0)
        tags = dataset.get("tags") or []
        description = (dataset.get("description") or "").lower()

        score = math.log1p(downloads) * 4.0 + likes * 1.5
        tag_boost_terms = (
            "task_categories:",
            "task_ids:",
            "benchmark",
            "leaderboard",
            "evaluation",
        )
        score += sum(
            1.5 for tag in tags if any(term in str(tag).lower() for term in tag_boost_terms)
        )
        if "benchmark" in description or "leaderboard" in description:
            score += 3.0
        return score

    def format_datasets(self, datasets: list[dict[str, Any]]) -> str:
        entries = []
        for index, dataset in enumerate(datasets, start=1):
            dataset_id = dataset.get("id", "unknown")
            tags = dataset.get("tags") or []
            top_tags = ", ".join(str(tag) for tag in tags[:6]) or "none"
            entries.append(
                "\n".join(
                    [
                        f"{index}: {dataset_id}",
                        f"URL: https://huggingface.co/datasets/{dataset_id}",
                        f"Downloads: {dataset.get('downloads', 'N/A')}",
                        f"Likes: {dataset.get('likes', 'N/A')}",
                        f"Tags: {top_tags}",
                        f"Description: {(dataset.get('description') or 'No description available.').strip()}",
                    ]
                )
            )
        return "\n\n".join(entries)
