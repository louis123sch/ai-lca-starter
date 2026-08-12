from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from ai_lca.audited_extraction import extract_inventory_from_documents_audited
from ai_lca.benchmark import BENCHMARK_EXTRA_INSTRUCTIONS, evaluate_extraction, load_expected, report_to_dict
from ai_lca.llm import extract_inventory_from_documents


CASES = {
    "gerloff": {
        "expected": Path("benchmarks/gerloff_2021/expected.json"),
        "sources": [
            Path("benchmarks/gerloff_2021/source_main_excerpt.txt"),
            Path("benchmarks/gerloff_2021/source_supplement_machine_readable.txt"),
            Path("benchmarks/gerloff_2021/source_supplement_workbook.txt"),
        ],
    },
    "hermesmann": {
        "expected": Path("benchmarks/hermesmann_2022/expected.json"),
        "sources": [
            Path("benchmarks/hermesmann_2022/source_main_excerpt.txt"),
            Path("benchmarks/hermesmann_2022/source_supplement_excerpt.txt"),
        ],
    },
    "yang": {
        "expected": Path("benchmarks/yang_2024/expected.json"),
        "sources": [Path("benchmarks/yang_2024/source_excerpt.txt")],
    },
}


def _ensure_gerloff_workbook() -> None:
    out = Path("benchmarks/gerloff_2021/source_supplement_workbook.txt")
    if out.exists():
        return
    src = Path("benchmarks/gerloff_2021/source_supplement_workbook.txt.gz")
    out.write_bytes(gzip.decompress(src.read_bytes()))


def _run(case: str, audited: bool, out_dir: Path, model: str) -> dict:
    if case == "gerloff":
        _ensure_gerloff_workbook()
    spec = CASES[case]
    docs = [(p.name, p.read_bytes()) for p in spec["sources"]]
    fn = extract_inventory_from_documents_audited if audited else extract_inventory_from_documents
    extraction = fn(
        docs,
        model=model,
        extra_instructions=BENCHMARK_EXTRA_INSTRUCTIONS,
        max_visual_assets=0,
    )
    expected = load_expected(spec["expected"])
    report = evaluate_extraction(extraction, expected)
    label = "audited" if audited else "baseline"
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / f"{label}_extraction.json").write_text(extraction.model_dump_json(indent=2) + "\n")
    payload = report_to_dict(report)
    (case_dir / f"{label}_report.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _delta(before: dict, after: dict) -> dict:
    keys = [
        "overall_score",
        "process_recall",
        "process_precision",
        "flow_recall",
        "flow_precision",
        "amount_accuracy",
        "unit_accuracy",
        "direction_accuracy",
    ]
    return {k: float(after[k]) - float(before[k]) for k in keys}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["target", "canary"], required=True)
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--output-dir", type=Path, default=Path("document_reader_iteration"))
    args = p.parse_args()

    cases = ["gerloff"] if args.mode == "target" else ["hermesmann", "yang"]
    result = {"mode": args.mode, "model": args.model, "cases": {}}
    for case in cases:
        before = _run(case, False, args.output_dir, args.model)
        after = _run(case, True, args.output_dir, args.model)
        result["cases"][case] = {"baseline": before, "audited": after, "delta": _delta(before, after)}

    (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
