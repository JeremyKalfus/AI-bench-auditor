from pathlib import Path

from ai_scientist.audits.artifacts import load_validated_audit_bundle
from ai_scientist.audits.verification import (
    load_verification_benchmark,
    materialize_verification_audit_bundle,
    run_acceptance_tests,
    run_mutation_test_harness,
    run_reproducibility_test,
    run_search_ablation,
    run_verification_stack,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def acceptance_benchmark_dirs() -> list[Path]:
    root = repo_root() / "tests" / "fixtures" / "verification" / "acceptance"
    return [
        root / "support_ticket_overlap",
        root / "loan_default_temporal_proxy",
    ]


def mutation_benchmark_dir() -> Path:
    return (
        repo_root()
        / "tests"
        / "fixtures"
        / "verification"
        / "mutation"
        / "clean_customer_churn"
    )


def test_materialized_verification_bundle_validates_and_detects_expected_issues(tmp_path):
    benchmark = load_verification_benchmark(
        acceptance_benchmark_dirs()[0]
    )

    result = materialize_verification_audit_bundle(
        benchmark,
        output_dir=tmp_path / "bundle",
        strategy="full_tree_search",
        run_id="support-ticket-bundle",
        seed=0,
    )
    bundle = load_validated_audit_bundle(tmp_path / "bundle")

    assert set(result["observed_detectors"]) == {"group_overlap", "near_duplicate"}
    assert result["missing_detectors"] == []
    assert result["unexpected_detectors"] == []
    assert bundle.audit_results["findings_summary"]["total_findings"] == 2
    assert bundle.metrics_before_after is not None
    assert (tmp_path / "bundle" / "audit_report.md").exists()


def test_mutation_harness_reports_full_recall_with_clean_negative_control(tmp_path):
    result = run_mutation_test_harness(
        mutation_benchmark_dir(),
        tmp_path / "mutation",
    )

    assert result["summary"]["passed"] is True
    assert result["summary"]["overall_recall"] == 1.0
    assert result["summary"]["clean_false_positive_count"] == 0
    assert {item["mutation_name"] for item in result["mutations"]} == {
        "exact_duplicate",
        "near_duplicate",
        "group_overlap",
        "temporal_leakage",
        "preprocessing_leakage",
        "suspicious_feature_leakage",
    }


def test_search_ablation_prefers_full_tree_search(tmp_path):
    result = run_search_ablation(
        acceptance_benchmark_dirs(),
        tmp_path / "ablation",
    )

    assert result["summary"]["best_strategy"] == "full_tree_search"
    assert result["summary"]["passed"] is True
    assert (
        result["strategies"]["full_tree_search"]["overall_recall"]
        > result["strategies"]["one_shot_agent"]["overall_recall"]
        >= result["strategies"]["detector_only"]["overall_recall"]
    )


def test_reproducibility_is_materially_consistent_across_repeated_runs(tmp_path):
    result = run_reproducibility_test(
        acceptance_benchmark_dirs(),
        tmp_path / "reproducibility",
        repeats=3,
    )

    assert result["summary"]["passed"] is True
    assert result["reproducibility_score"] == 1.0
    assert all(
        benchmark["materially_consistent"] for benchmark in result["benchmarks"]
    )
    assert all(
        (Path(run["artifact_dir"]) / "reproducibility.json").exists()
        for benchmark in result["benchmarks"]
        for run in benchmark["runs"]
    )


def test_acceptance_report_uses_contract_wording_without_overclaiming(tmp_path):
    ablation = run_search_ablation(
        acceptance_benchmark_dirs(),
        tmp_path / "ablation",
    )
    result = run_acceptance_tests(
        benchmark_dirs=acceptance_benchmark_dirs(),
        full_tree_strategy_runs=ablation["strategy_runs"]["full_tree_search"],
        output_dir=tmp_path / "acceptance",
    )

    report_text = Path(result["benchmarks"][0]["acceptance_report_path"]).read_text()
    assert (
        "Recovered the expected detector pattern with deterministic evidence."
        in report_text
    )
    assert (
        "This validates the acceptance contract, not broader benchmark-invalidity claims."
        in report_text
    )
    assert "known or suspected issue set" not in report_text


def test_full_verification_stack_writes_phase_artifacts(tmp_path):
    result = run_verification_stack(output_dir=tmp_path / "verification")

    assert result["status"] == "passed"
    for artifact_path in result["artifact_paths"].values():
        assert Path(artifact_path).exists()
    assert (tmp_path / "verification" / "verification_stack_results.json").exists()
    assert (tmp_path / "verification" / "verification_stack_summary.md").exists()
