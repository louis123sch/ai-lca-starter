from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from .documents import combine_document_evidence
from .llm import FLOW_SYSTEM_PROMPT, _client, transcribe_visual_evidence
from .models import FlowExtraction


PROCESS_NAMES = {
    "aec": "Alkaline electrolysis",
    "pemec": "Polymer electrolyte membrane electrolysis",
    "soec": "Solid oxide electrolysis cell",
}


def _norm(value: str | None) -> str:
    if value is None:
        return ""
    value = value.casefold().replace("³", "3")
    value = re.sub(r"[^a-z0-9.]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _amount_equal(expected: str, actual: float | None) -> bool:
    if actual is None:
        return False
    try:
        return abs(float(expected) - float(actual)) <= max(1e-9, abs(float(expected)) * 1e-6)
    except ValueError:
        return False


def load_expected(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def score_flows(flows: FlowExtraction, expected: list[dict[str, str]]) -> dict:
    unmatched_actual = list(flows.flows)
    matched: list[dict] = []
    missing: list[dict] = []

    for row in expected:
        found_index = None
        for i, flow in enumerate(unmatched_actual):
            if _norm(flow.process_id) != _norm(row["process_key"]):
                continue
            if _norm(flow.name) != _norm(row["name"]):
                continue
            if _norm(flow.unit) != _norm(row["unit"]):
                continue
            if not _amount_equal(row["amount"], flow.amount):
                continue
            found_index = i
            break
        payload = {
            "process_key": row["process_key"],
            "name": row["name"],
            "amount": row["amount"],
            "unit": row["unit"],
        }
        if found_index is None:
            missing.append(payload)
        else:
            flow = unmatched_actual.pop(found_index)
            matched.append(payload | {"actual": flow.model_dump(mode="json")})

    expected_keys = {(_norm(r["process_key"]), _norm(r["name"])) for r in expected}
    unsupported = [
        flow.model_dump(mode="json")
        for flow in unmatched_actual
        if (_norm(flow.process_id), _norm(flow.name)) not in expected_keys
    ]
    total = len(expected)
    recall = len(matched) / total if total else 1.0
    precision_denominator = len(matched) + len(unsupported)
    precision = len(matched) / precision_denominator if precision_denominator else 1.0
    return {
        "expected_rows": total,
        "matched_rows": len(matched),
        "missing_rows": len(missing),
        "unsupported_rows": len(unsupported),
        "recall": recall,
        "precision": precision,
        "matched": matched,
        "missing": missing,
        "unsupported": unsupported,
        "all_extracted_flows": [f.model_dump(mode="json") for f in flows.flows],
        "warnings": flows.assumptions_or_warnings,
    }


def extract_visual_flows(transcription: str, *, model: str | None = None) -> FlowExtraction:
    locked = {
        "processes": [
            {
                "process_id": key,
                "name": name,
                "role": "assessed_product_system",
            }
            for key, name in PROCESS_NAMES.items()
        ]
    }
    user_prompt = (
        "Extract foreground flows using ONLY the locked process structure below. "
        "This is a focused regression of the generic flow-extraction stage. "
        "Every explicit row in every supplied inventory table must be represented once. "
        "Do not add, split, merge, or rename processes.\n\n"
        f"LOCKED PROCESS STRUCTURE:\n{json.dumps(locked, indent=2)}\n\n"
        "SOURCE MATERIAL:\n"
        f"[BEGIN TRANSCRIBED VISUAL EVIDENCE]\n{transcription}\n[END TRANSCRIBED VISUAL EVIDENCE]"
    )
    chosen_model = model or "gpt-5-mini"
    completion = _client().beta.chat.completions.parse(
        model=chosen_model,
        messages=[
            {"role": "system", "content": FLOW_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=FlowExtraction,
    )
    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"Model refused flow extraction: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("The model returned no parsed foreground flows")
    return message.parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast regression from visual inventory evidence to locked foreground flows."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-visual-assets", type=int, default=24)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    _, assets, ingestion_warnings = combine_document_evidence(
        [(args.source.name, args.source.read_bytes())],
        max_visual_assets=args.max_visual_assets,
    )
    transcription, vision_warnings = transcribe_visual_evidence(assets, model=args.model)
    extraction = extract_visual_flows(transcription, model=args.model)
    report = score_flows(extraction, load_expected(args.expected))
    report["source"] = str(args.source)
    report["visual_assets"] = len(assets)
    report["ingestion_warnings"] = ingestion_warnings
    report["vision_warnings"] = vision_warnings
    report["transcription"] = transcription

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    if report["recall"] < args.min_recall or report["precision"] < args.min_precision:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
