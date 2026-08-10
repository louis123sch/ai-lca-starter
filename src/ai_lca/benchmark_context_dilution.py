from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark_visual_evidence import load_expected
from .benchmark_visual_to_flows import PROCESS_NAMES, score_flows
from .documents import combine_document_evidence
from .llm import FLOW_SYSTEM_PROMPT, _client, transcribe_visual_evidence
from .models import FlowExtraction


def extract_with_full_context(text_context: str, transcription: str, *, model: str | None = None) -> FlowExtraction:
    locked = {
        "processes": [
            {"process_id": key, "name": name, "role": "assessed_product_system"}
            for key, name in PROCESS_NAMES.items()
        ]
    }
    user_prompt = (
        "Extract foreground flows using ONLY the locked process structure below. "
        "Do not add, split, merge, or rename processes. Preserve every explicit inventory-list row.\n\n"
        f"LOCKED PROCESS STRUCTURE:\n{json.dumps(locked, indent=2)}\n\n"
        f"SOURCE MATERIAL:\n{text_context}\n\n"
        f"[BEGIN TRANSCRIBED VISUAL EVIDENCE]\n{transcription}\n[END TRANSCRIBED VISUAL EVIDENCE]"
    )
    completion = _client().beta.chat.completions.parse(
        model=model or "gpt-5-mini",
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
    parser = argparse.ArgumentParser(description="Test whether full-paper context dilutes visual inventory-row recall.")
    parser.add_argument("--visual-source", type=Path, required=True)
    parser.add_argument("--text-source", type=Path, action="append", default=[])
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    _, assets, ingestion_warnings = combine_document_evidence(
        [(args.visual_source.name, args.visual_source.read_bytes())], max_visual_assets=8
    )
    transcription, vision_warnings = transcribe_visual_evidence(assets, model=args.model)
    text_context = "\n\n".join(
        f"[DOCUMENT: {path.name}]\n{path.read_text(encoding='utf-8')}" for path in args.text_source
    )
    extraction = extract_with_full_context(text_context, transcription, model=args.model)
    expected = [
        {"process_key": row.process_key, "name": row.name, "amount": row.amount, "unit": row.unit}
        for row in load_expected(args.expected)
    ]
    report = score_flows(extraction, expected)
    report.update(
        {
            "text_sources": [str(p) for p in args.text_source],
            "visual_source": str(args.visual_source),
            "ingestion_warnings": ingestion_warnings,
            "vision_warnings": vision_warnings,
            "context_characters": len(text_context),
        }
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if report["recall"] < args.min_recall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
