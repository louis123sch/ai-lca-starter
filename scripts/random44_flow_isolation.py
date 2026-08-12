from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_lca.benchmark import evaluate_extraction, report_to_dict
from ai_lca.llm import FLOW_SYSTEM_PROMPT, _client
from ai_lca.models import FlowExtraction, ForegroundStructure, InventoryExtraction

from random44_document_reader_iteration import (
    TARGET_DOI,
    TARGET_SLUG,
    _build_document,
    _candidate_extract_docx_text,
    _gold_from_source,
)


def _locked_structure(extraction: InventoryExtraction) -> ForegroundStructure:
    return ForegroundStructure(
        process_name=extraction.process_name,
        functional_unit=extraction.functional_unit,
        source_summary=extraction.source_summary,
        study_context=extraction.study_context,
        assumptions_or_warnings=extraction.assumptions_or_warnings,
        candidate_activities=extraction.candidate_activities,
        processes=extraction.processes,
    )


def _flow_reread(text: str, baseline: InventoryExtraction, *, model: str) -> InventoryExtraction:
    structure = _locked_structure(baseline)
    user_prompt = (
        "Extract foreground flows using ONLY the locked process structure below. "
        "Do not add, split, merge, or rename processes.\n\n"
        f"LOCKED PROCESS STRUCTURE:\n{structure.model_dump_json(indent=2)}\n\n"
        f"SOURCE MATERIAL:\n{text}"
    )
    completion = _client().beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": FLOW_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=FlowExtraction,
    )
    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"Model refused flow reread: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("The model returned no parsed foreground flows")
    warnings = list(dict.fromkeys(structure.assumptions_or_warnings + message.parsed.assumptions_or_warnings))
    return baseline.model_copy(update={"flows": message.parsed.flows, "assumptions_or_warnings": warnings})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("random44_flow_isolation"))
    parser.add_argument("--model", default="gpt-5-mini")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = args.artifact_root / "corpus" / TARGET_SLUG / "source" / "article.xml"
    previous_extraction = args.previous_root / "baseline_extraction.json"
    previous_report = args.previous_root / "baseline_report.json"
    if not source.exists() or not previous_extraction.exists() or not previous_report.exists():
        raise FileNotFoundError("Required frozen source or previous baseline evidence is missing")

    xml_bytes = source.read_bytes()
    docx_bytes = _build_document(xml_bytes)
    detailed_flow_text = _candidate_extract_docx_text(docx_bytes)
    baseline = InventoryExtraction.model_validate_json(previous_extraction.read_text())
    baseline_report = json.loads(previous_report.read_text())
    expected = _gold_from_source(xml_bytes)

    candidate = _flow_reread(detailed_flow_text, baseline, model=args.model)
    candidate_report = report_to_dict(evaluate_extraction(candidate, expected))
    (args.output_dir / "candidate_locked_structure_extraction.json").write_text(candidate.model_dump_json(indent=2) + "\n")
    (args.output_dir / "candidate_locked_structure_report.json").write_text(json.dumps(candidate_report, indent=2) + "\n")

    keys = ["overall_score", "process_recall", "process_precision", "flow_recall", "flow_precision", "amount_accuracy", "unit_accuracy", "direction_accuracy"]
    delta = {key: float(candidate_report[key]) - float(baseline_report[key]) for key in keys}
    accepted = (
        candidate_report["flow_recall"] > baseline_report["flow_recall"]
        and candidate_report["flow_precision"] >= baseline_report["flow_precision"] - 0.02
        and candidate_report["amount_accuracy"] >= baseline_report["amount_accuracy"] - 0.05
        and candidate_report["process_recall"] == baseline_report["process_recall"]
        and candidate_report["process_precision"] == baseline_report["process_precision"]
    )
    result = {
        "doi": TARGET_DOI,
        "purpose": "Isolate the same multiline-table hypothesis to the flow-reading pass while freezing the baseline foreground structure.",
        "baseline": baseline_report,
        "candidate_locked_structure": candidate_report,
        "delta": delta,
        "accepted": accepted,
        "model_calls_expected": 1,
        "structure_locked_from_previous_baseline": True,
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not accepted:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
