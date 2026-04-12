import json
from pathlib import Path

import pytest

from ai_scientist.audits.artifacts import AuditArtifactError
from ai_scientist.audits import manuscript as manuscript_module
from ai_scientist.audits.manuscript import (
    ManuscriptGenerationError,
    build_audit_manuscript_bundle,
)
from ai_scientist.audits.report import generate_audit_report
from ai_scientist.audits.report_review import review_audit_report
from tests.audit_fixture_utils import write_references_bib, write_valid_audit_bundle


def prepare_reviewed_run(run_dir: Path) -> tuple[Path, Path]:
    artifact_dir = write_valid_audit_bundle(run_dir)
    report_path = generate_audit_report(
        audit_results_path=artifact_dir / "audit_results.json",
        split_manifest_path=artifact_dir / "split_manifest.json",
        findings_path=artifact_dir / "findings.csv",
        metrics_before_after_path=artifact_dir / "metrics_before_after.json",
        output_path=artifact_dir / "audit_report.md",
    )
    review_audit_report(
        artifact_dir=artifact_dir,
        audit_report_path=report_path,
        output_json_path=run_dir / "audit_report_review.json",
        output_md_path=run_dir / "audit_report_review.md",
    )
    return artifact_dir, run_dir / "audit_report_review.json"


def test_audit_paper_bundle_emits_expected_source_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    artifact_dir, review_json_path = prepare_reviewed_run(run_dir)
    references_path = tmp_path / "references.bib"
    write_references_bib(references_path)

    manifest = build_audit_manuscript_bundle(
        run_dir=run_dir,
        artifact_dir=artifact_dir,
        audit_report_review_path=review_json_path,
        citation_mode="provided",
        references_file=references_path,
        compile_pdf=False,
        allow_source_only=False,
        emit_paper_zip=True,
    )

    paper_dir = run_dir / "paper"
    assert (paper_dir / "paper.tex").exists()
    assert (paper_dir / "references.bib").exists()
    assert (paper_dir / "figures").is_dir()
    assert (paper_dir / "tables").is_dir()
    assert (paper_dir / "appendix").is_dir()
    assert (paper_dir / "paper_manifest.json").exists()
    assert (run_dir / "paper_bundle.zip").exists()
    assert manifest["zip_bundle"]["emitted"] is True
    assert len(manifest["figures"]) >= 2
    assert len(manifest["tables"]) >= 4


def test_audit_paper_bundle_reports_optional_pdf_behavior(tmp_path):
    run_dir = tmp_path / "run"
    artifact_dir, review_json_path = prepare_reviewed_run(run_dir)
    references_path = tmp_path / "references.bib"
    write_references_bib(references_path)

    manifest = build_audit_manuscript_bundle(
        run_dir=run_dir,
        artifact_dir=artifact_dir,
        audit_report_review_path=review_json_path,
        citation_mode="provided",
        references_file=references_path,
        compile_pdf=True,
        allow_source_only=True,
        emit_paper_zip=True,
    )

    pdf_info = manifest["pdf_compilation"]
    if pdf_info["available"] and pdf_info["succeeded"]:
        assert (run_dir / "paper" / "paper.pdf").exists()
    else:
        assert pdf_info["error"] is not None
        assert manifest["source_only_fallback_used"] is True


def test_requested_pdf_without_source_only_fails_when_toolchain_unavailable(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    artifact_dir, review_json_path = prepare_reviewed_run(run_dir)
    references_path = tmp_path / "references.bib"
    write_references_bib(references_path)

    monkeypatch.setattr(
        manuscript_module,
        "_compile_pdf",
        lambda *_args, **_kwargs: {
            "requested": True,
            "attempted": False,
            "available": False,
            "succeeded": False,
            "pdf_path": str((run_dir / "paper" / "paper.pdf").resolve()),
            "log_path": str((run_dir / "paper" / "paper_build.log").resolve()),
            "error": "LaTeX toolchain unavailable.",
        },
    )

    with pytest.raises(ManuscriptGenerationError):
        build_audit_manuscript_bundle(
            run_dir=run_dir,
            artifact_dir=artifact_dir,
            audit_report_review_path=review_json_path,
            citation_mode="provided",
            references_file=references_path,
            compile_pdf=True,
            allow_source_only=False,
            emit_paper_zip=True,
        )

    manifest = json.loads((run_dir / "paper" / "paper_manifest.json").read_text())
    assert manifest["pdf_compilation"]["requested"] is True
    assert manifest["pdf_compilation"]["succeeded"] is False
    assert manifest["zip_bundle"]["emitted"] is False


def test_audit_paper_bundle_supports_auto_citation_mode_with_honest_results(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    artifact_dir, review_json_path = prepare_reviewed_run(run_dir)

    def fake_search(query, result_limit=5):
        key = {
            "demo-benchmark": "demoBenchmark2024",
            "demo-dataset": "demoDataset2023",
            "data leakage benchmark contamination machine learning audit": "auditSurvey2022",
            "exact_duplicate data leakage machine learning": "dupDetector2021",
            "temporal_overlap data leakage machine learning": "temporalLeakage2020",
        }.get(query, "fallback2024")
        title = {
            "demo-benchmark": "Demo Benchmark: A Reference Benchmark for Leakage Audits",
            "demo-dataset": "Demo Dataset Card and Benchmark Construction",
            "data leakage benchmark contamination machine learning audit": "A Survey of Benchmark Contamination and Data Leakage in Machine Learning",
            "exact_duplicate data leakage machine learning": "Detecting Exact Duplicates in Machine Learning Benchmarks",
            "temporal_overlap data leakage machine learning": "Temporal Leakage Detection for Machine Learning Benchmarks",
        }.get(query, f"Reference for {query}")
        return [
            {
                "title": title,
                "authors": [{"name": "Test Author"}],
                "venue": "TestConf",
                "year": 2024,
                "abstract": "Synthetic search result for citation-mode=auto tests.",
                "citationCount": 42,
                "citationStyles": {
                    "bibtex": (
                        f"@article{{{key},\n"
                        f"  title={{" + title + r"}},"
                        "\n  author={Test Author},\n"
                        "  journal={TestConf},\n"
                        "  year={2024}\n}"
                    )
                },
            }
        ]

    monkeypatch.setattr(manuscript_module, "search_for_papers", fake_search)

    manifest = build_audit_manuscript_bundle(
        run_dir=run_dir,
        artifact_dir=artifact_dir,
        audit_report_review_path=review_json_path,
        citation_mode="auto",
        references_file=None,
        compile_pdf=False,
        allow_source_only=False,
        emit_paper_zip=True,
    )

    assert manifest["references"]["mode"] == "auto"
    assert manifest["references"]["count"] >= 3
    assert all(entry["source"] == "auto" for entry in manifest["references"]["entries"])
    assert (run_dir / "paper" / "references.bib").exists()
    assert (run_dir / "paper_bundle.zip").exists()


def test_auto_citation_mode_fails_when_required_queries_cannot_be_resolved(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    artifact_dir, review_json_path = prepare_reviewed_run(run_dir)

    monkeypatch.setattr(manuscript_module, "search_for_papers", lambda *_args, **_kwargs: [])

    with pytest.raises(ManuscriptGenerationError):
        build_audit_manuscript_bundle(
            run_dir=run_dir,
            artifact_dir=artifact_dir,
            audit_report_review_path=review_json_path,
            citation_mode="auto",
            references_file=None,
            compile_pdf=False,
            allow_source_only=False,
            emit_paper_zip=True,
        )


def test_invalid_audit_artifacts_block_paper_generation(tmp_path):
    run_dir = tmp_path / "run"
    artifact_dir, review_json_path = prepare_reviewed_run(run_dir)
    references_path = tmp_path / "references.bib"
    write_references_bib(references_path)
    (artifact_dir / "evidence" / "exact_duplicate_pairs.parquet").unlink()

    with pytest.raises(AuditArtifactError):
        build_audit_manuscript_bundle(
            run_dir=run_dir,
            artifact_dir=artifact_dir,
            audit_report_review_path=review_json_path,
            citation_mode="provided",
            references_file=references_path,
            compile_pdf=False,
            allow_source_only=False,
            emit_paper_zip=True,
        )

    assert not (run_dir / "paper_bundle.zip").exists()


def test_failed_report_review_blocks_paper_zip_generation(tmp_path):
    run_dir = tmp_path / "run"
    artifact_dir = write_valid_audit_bundle(run_dir)
    references_path = tmp_path / "references.bib"
    write_references_bib(references_path)
    failed_review_path = run_dir / "audit_report_review.json"
    failed_review_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "issues": [
                    {
                        "code": "overclaiming_certainty",
                        "severity": "critical",
                        "message": "Unsupported certainty claim remained unresolved.",
                        "fixable": False,
                    }
                ],
            },
            indent=2,
        )
    )

    with pytest.raises(AuditArtifactError):
        build_audit_manuscript_bundle(
            run_dir=run_dir,
            artifact_dir=artifact_dir,
            audit_report_review_path=failed_review_path,
            citation_mode="provided",
            references_file=references_path,
            compile_pdf=False,
            allow_source_only=False,
            emit_paper_zip=True,
        )

    assert not (run_dir / "paper_bundle.zip").exists()


def test_missing_required_citations_fail_the_manuscript_stage(tmp_path):
    run_dir = tmp_path / "run"
    artifact_dir, review_json_path = prepare_reviewed_run(run_dir)
    empty_references_path = tmp_path / "empty_references.bib"
    empty_references_path.write_text("")

    with pytest.raises(ManuscriptGenerationError):
        build_audit_manuscript_bundle(
            run_dir=run_dir,
            artifact_dir=artifact_dir,
            audit_report_review_path=review_json_path,
            citation_mode="provided",
            references_file=empty_references_path,
            compile_pdf=False,
            allow_source_only=False,
            emit_paper_zip=True,
        )

    manifest_path = run_dir / "paper" / "paper_manifest.json"
    assert not manifest_path.exists()
