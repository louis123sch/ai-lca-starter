from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

from .audited_extraction import extract_inventory_from_documents_audited
from .benchmark import (
    BENCHMARK_EXTRA_INSTRUCTIONS,
    evaluate_extraction,
    format_report,
    load_expected,
    report_to_dict,
)


def run_multimodal_benchmark(
    source_paths: list[Path],
    expected_path: Path,
    *,
    runs: int,
    model: str | None,
    output_dir: Path,
    max_visual_assets: int = 24,
):
    """Run the benchmark through the audited multimodal document path."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Configure it before running the live benchmark.")

    documents = [(path.name, path.read_bytes()) for path in source_paths]
    expected = load_expected(expected_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    for n in range(1, runs + 1):
        extraction = extract_inventory_from_documents_audited(
            documents,
            model=model,
            extra_instructions=BENCHMARK_EXTRA_INSTRUCTIONS,
            max_visual_assets=max_visual_assets,
        )
        (output_dir / f"extraction_run_{n:02d}.json").write_text(
            extraction.model_dump_json(indent=2)
        )
        report = evaluate_extraction(extraction, expected)
        (output_dir / f"report_run_{n:02d}.json").write_text(
            json.dumps(report_to_dict(report), indent=2) + "\n"
        )
        reports.append(report)
        print(f"\n=== Run {n}/{runs} ===\n{format_report(report)}")

    summary = {
        "benchmark_id": expected.get("benchmark_id"),
        "runs": runs,
        "model": model or os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        "multimodal": True,
        "max_visual_assets": max_visual_assets,
        "mean_overall_score": statistics.fmean(r.overall_score for r in reports),
        "min_overall_score": min(r.overall_score for r in reports),
        "mean_process_recall": statistics.fmean(r.process_recall for r in reports),
        "mean_process_precision": statistics.fmean(r.process_precision for r in reports),
        "mean_flow_recall": statistics.fmean(r.flow_recall for r in reports),
        "mean_flow_precision": statistics.fmean(r.flow_precision for r in reports),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== Aggregate ===\n" + json.dumps(summary, indent=2))
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal paper-grounded AI-LCA benchmark")
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--source", required=True, nargs="+", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model")
    parser.add_argument("--max-visual-assets", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_runs/multimodal"))
    args = parser.parse_args()
    run_multimodal_benchmark(
        args.source,
        args.expected,
        runs=args.runs,
        model=args.model,
        output_dir=args.output_dir,
        max_visual_assets=args.max_visual_assets,
    )


if __name__ == "__main__":
    main()
