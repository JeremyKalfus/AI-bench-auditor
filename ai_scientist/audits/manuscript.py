from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ai_scientist.tools.semantic_scholar import search_for_papers

from .artifacts import AuditArtifactBundle, AuditArtifactError, load_validated_audit_bundle
from .report_review import ensure_review_passes


class ManuscriptGenerationError(RuntimeError):
    """Raised when the audit-native paper bundle cannot be generated honestly."""


@dataclass(frozen=True)
class ReferenceEntry:
    key: str
    title: str
    raw_bibtex: str
    source: str
    query: str | None = None


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "item"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _maybe_relpath(path: Path, start: Path) -> str:
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return str(path)


def _parse_bibtex_entries(raw_text: str, source: str) -> list[ReferenceEntry]:
    entries: list[ReferenceEntry] = []
    for match in re.finditer(r"@\w+\s*\{\s*([^,]+),", raw_text):
        start = match.start()
        brace_depth = 0
        end = start
        in_entry = False
        for index in range(start, len(raw_text)):
            char = raw_text[index]
            if char == "{":
                brace_depth += 1
                in_entry = True
            elif char == "}":
                brace_depth -= 1
                if in_entry and brace_depth == 0:
                    end = index + 1
                    break
        if not end:
            continue
        entry_text = raw_text[start:end].strip()
        key = match.group(1).strip()
        title_match = re.search(r"title\s*=\s*[{|\"](.+?)[}|\"]\s*,", entry_text, re.I | re.S)
        title = title_match.group(1).replace("\n", " ").strip() if title_match else key
        entries.append(ReferenceEntry(key=key, title=title, raw_bibtex=entry_text, source=source))
    return entries


def _normalize_query_terms(query: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) > 2]


def _select_reference_for_query(query: str, papers: list[dict[str, Any]]) -> ReferenceEntry | None:
    if not papers:
        return None

    query_terms = _normalize_query_terms(query)
    best_score = -1.0
    best_paper: dict[str, Any] | None = None
    for paper in papers:
        title = str(paper.get("title") or "")
        title_terms = set(_normalize_query_terms(title))
        overlap = len(title_terms.intersection(query_terms))
        if query_terms:
            score = overlap / len(set(query_terms))
        else:
            score = 0.0
        score += min(float(paper.get("citationCount") or 0), 5000.0) / 50000.0
        if query.lower() in title.lower():
            score += 1.0
        if score > best_score:
            best_score = score
            best_paper = paper

    if best_paper is None:
        return None

    if query_terms:
        title_lower = str(best_paper.get("title") or "").lower()
        if query.lower() not in title_lower and best_score < 0.34:
            return None

    bibtex = ((best_paper.get("citationStyles") or {}).get("bibtex") or "").strip()
    if not bibtex:
        return None

    entries = _parse_bibtex_entries(bibtex, source="auto")
    if not entries:
        return None
    entry = entries[0]
    return ReferenceEntry(
        key=entry.key,
        title=str(best_paper.get("title") or entry.title),
        raw_bibtex=entry.raw_bibtex,
        source="auto",
        query=query,
    )


def _load_reference_entries(
    *,
    citation_mode: str,
    references_file: str | Path | None,
    review: dict[str, Any],
) -> list[ReferenceEntry]:
    if citation_mode == "off":
        raise ManuscriptGenerationError(
            "citation-mode=off is incompatible with audit manuscript generation because the paper requires honest related-work citations."
        )

    entries: list[ReferenceEntry] = []
    seen_keys: set[str] = set()
    seen_titles: set[str] = set()

    def add_entry(entry: ReferenceEntry) -> None:
        title_key = entry.title.strip().lower()
        if entry.key in seen_keys or title_key in seen_titles:
            return
        seen_keys.add(entry.key)
        seen_titles.add(title_key)
        entries.append(entry)

    if references_file is not None:
        raw_text = Path(references_file).read_text()
        for entry in _parse_bibtex_entries(raw_text, source="provided"):
            add_entry(entry)

    if citation_mode == "provided":
        if len(entries) < 2:
            raise ManuscriptGenerationError(
                "Provided references did not contain enough valid BibTeX entries to support the manuscript."
            )
        return entries

    unresolved_required: list[str] = []
    for need in review["citation_needs"]:
        papers = search_for_papers(need["query"], result_limit=5) or []
        selected = _select_reference_for_query(need["query"], papers)
        if selected is None:
            if need["required"]:
                unresolved_required.append(need["query"])
            continue
        add_entry(selected)

    if unresolved_required:
        raise ManuscriptGenerationError(
            "Could not resolve required citations honestly for the following queries: "
            + ", ".join(unresolved_required)
        )

    if len(entries) < 2:
        raise ManuscriptGenerationError(
            "Automatic citation gathering did not resolve enough unique references for a related-work section."
        )

    return entries


def _save_references(entries: list[ReferenceEntry], output_path: Path) -> None:
    _write_text(output_path, "\n\n".join(entry.raw_bibtex.strip() for entry in entries) + "\n")


def _make_table_tex(
    *,
    caption: str,
    label: str,
    columns: list[str],
    rows: list[list[Any]],
) -> str:
    column_spec = "l" * len(columns)
    header = " & ".join(_latex_escape(column) for column in columns) + r" \\"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{_latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(item) for item in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def _format_float(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if math.isfinite(value):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _generate_tables(bundle: AuditArtifactBundle, tables_dir: Path) -> list[dict[str, str]]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, str]] = []

    split_rows = []
    for split in bundle.split_manifest["splits"]:
        split_rows.append(
            [
                split["name"],
                split["record_count"],
                ", ".join(split["file_paths"]),
                (
                    split.get("group_key_summary", {}).get("unique_group_count")
                    if split.get("group_key_summary")
                    else "n/a"
                ),
            ]
        )
    split_table = tables_dir / "split_summary_table.tex"
    _write_text(
        split_table,
        _make_table_tex(
            caption="Benchmark split summary derived from split_manifest.json.",
            label="tab:split_summary",
            columns=["Split", "Rows", "Files", "Unique groups"],
            rows=split_rows,
        ),
    )
    outputs.append({"name": "split_summary_table", "path": split_table.name})

    detector_rows = [
        [
            detector["name"],
            detector["version"],
            detector["status"],
            detector["finding_count"],
        ]
        for detector in bundle.audit_results["detectors_run"]
    ]
    detector_table = tables_dir / "detector_summary_table.tex"
    _write_text(
        detector_table,
        _make_table_tex(
            caption="Detector coverage summary derived from audit_results.json.",
            label="tab:detector_summary",
            columns=["Detector", "Version", "Status", "Findings"],
            rows=detector_rows,
        ),
    )
    outputs.append({"name": "detector_summary_table", "path": detector_table.name})

    findings_rows = []
    if bundle.findings.empty:
        findings_rows.append(["No findings", "n/a", "n/a", "n/a", "n/a"])
    else:
        findings = bundle.findings.copy().sort_values(
            ["severity", "confidence", "detector_name"],
            ascending=[True, False, True],
        )
        for _, row in findings.head(10).iterrows():
            findings_rows.append(
                [
                    row["finding_id"],
                    row["detector_name"],
                    row["severity"],
                    _format_float(float(row["confidence"])),
                    row["remediation_status"],
                ]
            )
    findings_table = tables_dir / "findings_table.tex"
    _write_text(
        findings_table,
        _make_table_tex(
            caption="Representative findings derived from the audit findings artifact.",
            label="tab:findings",
            columns=["Finding", "Detector", "Severity", "Confidence", "Status"],
            rows=findings_rows,
        ),
    )
    outputs.append({"name": "findings_table", "path": findings_table.name})

    remediation_rows: list[list[Any]] = []
    if bundle.metrics_before_after is not None:
        for delta in bundle.metrics_before_after["deltas"]:
            remediation_rows.append(
                [
                    delta["metric_name"],
                    delta["split"],
                    _format_float(delta["baseline_value"]),
                    _format_float(delta["remediated_value"]),
                    _format_float(delta["delta"]),
                ]
            )
    else:
        remediation_rows.append(
            [
                "not_applicable",
                "n/a",
                "n/a",
                "n/a",
                "No metrics_before_after.json artifact was required for this clean audit.",
            ]
        )
    remediation_table = tables_dir / "remediation_comparison_table.tex"
    _write_text(
        remediation_table,
        _make_table_tex(
            caption="Before/after remediation summary derived from metrics_before_after.json when available.",
            label="tab:remediation",
            columns=["Metric", "Split", "Baseline", "Remediated", "Delta/Note"],
            rows=remediation_rows,
        ),
    )
    outputs.append({"name": "remediation_comparison_table", "path": remediation_table.name})
    return outputs


def _save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _generate_figures(bundle: AuditArtifactBundle, figures_dir: Path) -> list[dict[str, str]]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, str]] = []

    split_counts = pd.DataFrame(bundle.split_manifest["splits"])[["name", "record_count"]]
    split_plot = figures_dir / "split_record_counts.png"
    plt.figure(figsize=(6, 3.5))
    plt.bar(split_counts["name"], split_counts["record_count"], color="#4C78A8")
    plt.xlabel("Split")
    plt.ylabel("Records")
    plt.title("Benchmark Split Record Counts")
    _save_plot(split_plot)
    outputs.append({"name": "split_record_counts", "path": split_plot.name})

    detector_counts = pd.DataFrame(bundle.audit_results["detectors_run"])[
        ["name", "finding_count"]
    ]
    detector_plot = figures_dir / "detector_findings_counts.png"
    plt.figure(figsize=(6, 3.5))
    plt.bar(detector_counts["name"], detector_counts["finding_count"], color="#F58518")
    plt.xlabel("Detector")
    plt.ylabel("Findings")
    plt.xticks(rotation=20, ha="right")
    plt.title("Detector Finding Counts")
    _save_plot(detector_plot)
    outputs.append({"name": "detector_findings_counts", "path": detector_plot.name})

    if bundle.metrics_before_after is not None:
        delta_frame = pd.DataFrame(bundle.metrics_before_after["deltas"])
        delta_plot = figures_dir / "remediation_delta_plot.png"
        plt.figure(figsize=(6, 3.5))
        x_labels = [
            f"{row.metric_name} ({row.split})" for row in delta_frame.itertuples(index=False)
        ]
        plt.bar(x_labels, delta_frame["delta"], color="#54A24B")
        plt.xlabel("Metric")
        plt.ylabel("Delta")
        plt.xticks(rotation=20, ha="right")
        plt.title("Remediation Metric Deltas")
        _save_plot(delta_plot)
        outputs.append({"name": "remediation_delta_plot", "path": delta_plot.name})

    return outputs


def _generate_evidence_map(bundle: AuditArtifactBundle, run_dir: Path, appendix_dir: Path) -> Path:
    appendix_dir.mkdir(parents=True, exist_ok=True)
    evidence_rows = [
        [
            "Benchmark split inventory",
            _maybe_relpath(bundle.split_manifest_path, run_dir),
            "Split names, file inventory, and row counts.",
        ],
        [
            "Detector coverage",
            _maybe_relpath(bundle.audit_results_path, run_dir),
            "Detector versions, statuses, and finding counts.",
        ],
        [
            "Primary findings",
            _maybe_relpath(bundle.findings_path, run_dir),
            "Finding severity, confidence, evidence pointer, and remediation status.",
        ],
        [
            "Confidence and limitations",
            _maybe_relpath(bundle.audit_results_path, run_dir),
            "Confidence score and reviewer-facing notes.",
        ],
    ]
    if bundle.metrics_before_after_path is not None:
        evidence_rows.append(
            [
                "Remediation deltas",
                _maybe_relpath(bundle.metrics_before_after_path, run_dir),
                "Before/after comparison metrics used in remediation discussion.",
            ]
        )

    appendix_path = appendix_dir / "evidence_map.tex"
    _write_text(
        appendix_path,
        _make_table_tex(
            caption="Evidence map linking manuscript claims to concrete run artifacts.",
            label="tab:evidence_map",
            columns=["Claim", "Artifact path", "Support"],
            rows=evidence_rows,
        ),
    )
    return appendix_path


def _compile_pdf(paper_dir: Path, tex_name: str) -> dict[str, Any]:
    available = shutil.which("pdflatex") is not None and shutil.which("bibtex") is not None
    result = {
        "attempted": False,
        "available": available,
        "succeeded": False,
        "pdf_path": str((paper_dir / "paper.pdf").resolve()),
        "log_path": str((paper_dir / "paper_build.log").resolve()),
        "error": None,
    }
    if not available:
        result["error"] = "LaTeX toolchain unavailable."
        return result

    commands = [
        ["pdflatex", "-interaction=nonstopmode", tex_name],
        ["bibtex", Path(tex_name).stem],
        ["pdflatex", "-interaction=nonstopmode", tex_name],
        ["pdflatex", "-interaction=nonstopmode", tex_name],
    ]
    log_lines = []
    result["attempted"] = True
    for command in commands:
        proc = subprocess.run(
            command,
            cwd=paper_dir,
            text=True,
            capture_output=True,
        )
        log_lines.append(f"$ {' '.join(command)}\n")
        log_lines.append(proc.stdout)
        log_lines.append(proc.stderr)
        if proc.returncode != 0:
            result["error"] = f"Command failed with exit code {proc.returncode}: {' '.join(command)}"
            break

    (paper_dir / "paper_build.log").write_text("\n".join(log_lines))
    built_pdf = paper_dir / Path(tex_name).with_suffix(".pdf")
    if result["error"] is None and built_pdf.exists():
        if built_pdf.name != "paper.pdf":
            shutil.move(str(built_pdf), str(paper_dir / "paper.pdf"))
        result["succeeded"] = True
    elif result["error"] is None:
        result["error"] = "LaTeX commands completed without producing a PDF."
    return result


def _render_related_work(entries: list[ReferenceEntry]) -> str:
    keys = ", ".join(entry.key for entry in entries[: min(4, len(entries))])
    titles = "; ".join(_latex_escape(entry.title) for entry in entries[: min(3, len(entries))])
    return (
        "This audit keeps related-work claims narrow: we cite external sources only for benchmark "
        f"background and leakage-audit context, including {titles} \\citep{{{keys}}}. "
        "These references provide context, while every substantive finding in this paper is supported "
        "by deterministic artifacts from the run itself."
    )


def _render_paper_tex(
    *,
    bundle: AuditArtifactBundle,
    run_dir: Path,
    review: dict[str, Any],
    references: list[ReferenceEntry],
    figures: list[dict[str, str]],
    tables: list[dict[str, str]],
) -> str:
    benchmark = bundle.audit_results["benchmark_summary"]
    findings_summary = bundle.audit_results["findings_summary"]
    confidence = bundle.audit_results["confidence"]
    title = (
        f"Benchmark Leakage Audit of {benchmark['benchmark_name']} "
        f"on {benchmark['dataset_name']}"
    )

    related_work = _render_related_work(references)
    detector_names = ", ".join(
        _latex_escape(detector["name"]) for detector in bundle.audit_results["detectors_run"]
    )
    split_names = ", ".join(_latex_escape(name) for name in benchmark["split_names"])
    findings_sentence = (
        f"The audit recorded {findings_summary['total_findings']} finding(s), "
        f"with {findings_summary['open_findings']} still open at the end of the run."
    )
    if bundle.metrics_before_after is not None:
        remediation_sentence = (
            "Remediation analysis is included and summarized in "
            r"Table~\ref{tab:remediation} and Figure~\ref{fig:remediation_delta_plot}."
        )
    else:
        remediation_sentence = (
            "No remediation delta artifact was required for this clean audit, so the manuscript records that state explicitly."
        )

    figure_block = [
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=0.75\linewidth]{figures/split_record_counts.png}",
        r"\caption{Split record counts derived directly from split_manifest.json.}",
        r"\label{fig:split_record_counts}",
        r"\end{figure}",
        "",
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=0.75\linewidth]{figures/detector_findings_counts.png}",
        r"\caption{Detector-level finding counts derived directly from audit_results.json.}",
        r"\label{fig:detector_findings_counts}",
        r"\end{figure}",
    ]
    if any(item["name"] == "remediation_delta_plot" for item in figures):
        figure_block.extend(
            [
                "",
                r"\begin{figure}[t]",
                r"\centering",
                r"\includegraphics[width=0.75\linewidth]{figures/remediation_delta_plot.png}",
                r"\caption{Metric deltas derived directly from metrics_before_after.json.}",
                r"\label{fig:remediation_delta_plot}",
                r"\end{figure}",
            ]
        )

    return (
        r"\documentclass[11pt]{article}" "\n"
        r"\usepackage[margin=1in]{geometry}" "\n"
        r"\usepackage{booktabs}" "\n"
        r"\usepackage{graphicx}" "\n"
        r"\usepackage{hyperref}" "\n"
        r"\usepackage{natbib}" "\n"
        r"\usepackage{longtable}" "\n"
        r"\title{" + _latex_escape(title) + "}\n"
        r"\author{AI Scientist v2 Audit Workflow}" "\n"
        r"\date{}" "\n"
        r"\begin{document}" "\n"
        r"\maketitle" "\n"
        r"\begin{abstract}" "\n"
        + _latex_escape(
            f"This paper reports a deterministic benchmark leakage audit of {benchmark['benchmark_name']} "
            f"using the dataset {benchmark['dataset_name']}. The audit evaluated splits {benchmark['split_names']} "
            f"with detector coverage from {detector_names}. {findings_sentence} "
            f"Audit confidence was {confidence['overall']:.2f} with evidence coverage {confidence['evidence_coverage']:.2f}. "
            "Every empirical claim in the manuscript is grounded in structured run artifacts, not free-form narrative."
        )
        + "\n" + r"\end{abstract}" "\n"
        r"\section{Introduction}" "\n"
        + _latex_escape(
            f"This run studies whether the benchmark {benchmark['benchmark_name']} exhibits leakage or contamination across "
            f"its declared splits. The audit follows a deterministic artifact-first workflow so the manuscript can be traced "
            "back to structured JSON, CSV/Parquet, and evidence files."
        )
        + "\n"
        r"\section{Benchmark and Protocol Description}" "\n"
        + _latex_escape(
            f"The benchmark summary is derived from split_manifest.json and dataset_card.md. The audit evaluated splits "
            f"{split_names} with a total record count of {benchmark['record_count']} and dataset fingerprint "
            f"{bundle.audit_results['provenance']['dataset_fingerprint']}."
        )
        + "\n"
        r"\input{tables/split_summary_table.tex}" "\n"
        r"\section{Audit Methodology}" "\n"
        + _latex_escape(
            "The workflow kept the existing four-stage audit scaffold intact: reproduce the benchmark protocol, run leakage "
            "detectors, confirm findings with remediation or falsification, and finish with robustness plus synthesis."
        )
        + "\n"
        r"\section{Detector Suite}" "\n"
        + _latex_escape(
            f"The detector suite recorded in audit_results.json covered: {detector_names}. Detector versions and observed "
            "finding counts are summarized below."
        )
        + "\n"
        r"\input{tables/detector_summary_table.tex}" "\n"
        + "\n".join(figure_block)
        + "\n"
        r"\section{Findings}" "\n"
        + _latex_escape(findings_sentence)
        + " "
        + _latex_escape(
            f"The findings table is derived directly from {bundle.findings_path.name} and keeps only artifact-backed fields."
        )
        + "\n"
        r"\input{tables/findings_table.tex}" "\n"
        r"\section{Remediation Experiments}" "\n"
        + _latex_escape(remediation_sentence)
        + "\n"
        r"\input{tables/remediation_comparison_table.tex}" "\n"
        r"\section{Limitations}" "\n"
        + _latex_escape(
            f"Limitations are reported conservatively. Confidence notes from audit_results.json state: {confidence['notes']}"
        )
        + "\n"
        r"\section{Related Work}" "\n"
        + related_work
        + "\n"
        r"\section{Conclusion}" "\n"
        + _latex_escape(
            "The manuscript is intentionally constrained: it summarizes only the benchmark context, detector coverage, "
            "findings, remediation results, and limitations that can be verified directly from run artifacts."
        )
        + "\n"
        r"\appendix" "\n"
        r"\section{Evidence Map}" "\n"
        + _latex_escape(
            "The appendix maps major claims in the manuscript back to concrete run outputs so a human reviewer can audit the paper bundle."
        )
        + "\n"
        r"\input{appendix/evidence_map.tex}" "\n"
        r"\bibliographystyle{plainnat}" "\n"
        r"\bibliography{references}" "\n"
        r"\end{document}" "\n"
    )


def build_audit_manuscript_bundle(
    *,
    run_dir: str | Path,
    artifact_dir: str | Path,
    audit_report_review_path: str | Path,
    citation_mode: str,
    references_file: str | Path | None,
    compile_pdf: bool,
    allow_source_only: bool,
    emit_paper_zip: bool,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    artifact_dir = Path(artifact_dir)
    audit_report_review_path = Path(audit_report_review_path)

    bundle = load_validated_audit_bundle(artifact_dir)
    review = json.loads(audit_report_review_path.read_text())
    ensure_review_passes(review)

    paper_dir = run_dir / "paper"
    figures_dir = paper_dir / "figures"
    tables_dir = paper_dir / "tables"
    appendix_dir = paper_dir / "appendix"
    paper_dir.mkdir(parents=True, exist_ok=True)

    references = _load_reference_entries(
        citation_mode=citation_mode,
        references_file=references_file,
        review=review,
    )
    _save_references(references, paper_dir / "references.bib")

    tables = _generate_tables(bundle, tables_dir)
    figures = _generate_figures(bundle, figures_dir)
    evidence_map_path = _generate_evidence_map(bundle, run_dir, appendix_dir)

    paper_tex = _render_paper_tex(
        bundle=bundle,
        run_dir=run_dir,
        review=review,
        references=references,
        figures=figures,
        tables=tables,
    )
    _write_text(paper_dir / "paper.tex", paper_tex)

    pdf_compilation = {
        "requested": compile_pdf,
        "attempted": False,
        "available": False,
        "succeeded": False,
        "pdf_path": str((paper_dir / "paper.pdf").resolve()),
        "log_path": str((paper_dir / "paper_build.log").resolve()),
        "error": None,
    }
    if compile_pdf:
        pdf_compilation = _compile_pdf(paper_dir, "paper.tex")
        pdf_compilation["requested"] = True
    else:
        pdf_compilation["requested"] = False

    manifest = {
        "contract_version": 1,
        "paper_dir": str(paper_dir.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "audit_report_review_path": str(audit_report_review_path.resolve()),
        "tables": tables,
        "figures": figures,
        "appendix_files": [{"name": "evidence_map", "path": evidence_map_path.name}],
        "references": {
            "mode": citation_mode,
            "count": len(references),
            "entries": [
                {
                    "key": reference.key,
                    "title": reference.title,
                    "source": reference.source,
                    "query": reference.query,
                }
                for reference in references
            ],
        },
        "pdf_compilation": pdf_compilation,
        "source_only_fallback_used": (
            bool(compile_pdf)
            and not pdf_compilation["succeeded"]
            and bool(allow_source_only)
        ),
        "zip_bundle": {
            "requested": emit_paper_zip,
            "emitted": False,
            "path": None,
        },
    }

    if compile_pdf and not pdf_compilation["succeeded"] and not allow_source_only:
        (paper_dir / "paper_manifest.json").write_text(json.dumps(manifest, indent=2))
        raise ManuscriptGenerationError(
            "Paper PDF compilation was requested but did not succeed, and allow-source-only mode is disabled: "
            + str(pdf_compilation["error"])
        )

    bundle_zip_path = run_dir / "paper_bundle.zip"
    if emit_paper_zip:
        archive_base = bundle_zip_path.with_suffix("")
        created = shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=run_dir,
            base_dir="paper",
        )
        bundle_zip_path = Path(created)
        manifest["zip_bundle"]["emitted"] = True
        manifest["zip_bundle"]["path"] = str(bundle_zip_path.resolve())

    (paper_dir / "paper_manifest.json").write_text(json.dumps(manifest, indent=2))

    if emit_paper_zip and not bundle_zip_path.exists():
        raise ManuscriptGenerationError("paper_bundle.zip was requested but was not created.")

    return manifest
