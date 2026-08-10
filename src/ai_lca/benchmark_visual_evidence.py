from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .documents import combine_document_evidence
from .llm import transcribe_visual_evidence


@dataclass(frozen=True)
class ExpectedVisualRow:
    process_key: str
    name: str
    amount: str
    unit: str


def _norm(value: str) -> str:
    value = value.casefold().replace("³", "3")
    value = re.sub(r"[^a-z0-9.]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _amount_variants(value: str) -> set[str]:
    raw = value.strip()
    if not raw:
        return set()
    variants = {_norm(raw)}
    try:
        number = float(raw)
    except ValueError:
        return variants
    variants.add(_norm(f"{number:g}"))
    variants.add(_norm(f"{number:.1f}"))
    return {v for v in variants if v}


def score_transcription(text: str, rows: list[ExpectedVisualRow]) -> dict:
    lines = [_norm(line) for line in text.splitlines() if _norm(line)]
    matched: list[dict] = []
    missing: list[dict] = []

    for row in rows:
        name = _norm(row.name)
        unit = _norm(row.unit)
        amounts = _amount_variants(row.amount)
        found_line = None
        for line in lines:
            if name not in line:
                continue
            if unit and unit not in line:
                continue
            if amounts and not any(amount in line for amount in amounts):
                continue
            found_line = line
            break
        payload = {
            "process_key": row.process_key,
            "name": row.name,
            "amount": row.amount,
            "unit": row.unit,
        }
        if found_line is None:
            missing.append(payload)
        else:
            matched.append(payload | {"matched_line": found_line})

    total = len(rows)
    recall = len(matched) / total if total else 1.0
    return {
        "expected_rows": total,
        "matched_rows": len(matched),
        "missing_rows": len(missing),
        "recall": recall,
        "matched": matched,
        "missing": missing,
    }


def load_expected(path: Path) -> list[ExpectedVisualRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            ExpectedVisualRow(
                process_key=row["process_key"],
                name=row["name"],
                amount=row["amount"],
                unit=row["unit"],
            )
            for row in csv.DictReader(handle)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast visual-evidence regression: ingest only visual source evidence, transcribe it, "
            "and report exactly which expected inventory rows were recovered."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-visual-assets", type=int, default=24)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    source = args.source
    _, assets, ingestion_warnings = combine_document_evidence(
        [(source.name, source.read_bytes())],
        max_visual_assets=args.max_visual_assets,
    )
    transcription, vision_warnings = transcribe_visual_evidence(assets, model=args.model)
    report = score_transcription(transcription, load_expected(args.expected))
    report["source"] = str(source)
    report["visual_assets"] = len(assets)
    report["ingestion_warnings"] = ingestion_warnings
    report["vision_warnings"] = vision_warnings
    report["transcription"] = transcription

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    if report["recall"] < args.min_recall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
