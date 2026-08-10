from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .benchmark_visual_to_flows import PROCESS_NAMES, score_flows
from .llm import FLOW_SYSTEM_PROMPT, _client
from .models import FlowExtraction


def load_unquantified_expected(path: Path) -> list[dict[str, str]]:
    """Return explicit inventory-list rows whose source does not state a quantity."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if not (row.get("amount") or "").strip()]


def extract_list_flows(text: str, *, model: str | None = None) -> FlowExtraction:
    locked = {
        "processes": [
            {"process_id": key, "name": name, "role": "assessed_product_system"}
            for key, name in PROCESS_NAMES.items()
        ]
    }
    user_prompt = (
        "Extract ONLY explicit modeled foreground inventory-list items from the source below. "
        "This focused pass exists to prevent long-document context from causing explicit numbered, bulleted, "
        "or tabulated inventory rows to be silently dropped. Preserve every listed item once, attach it to the "
        "appropriate locked process, use amount=null when the list states no amount, and do not infer background "
        "dataset detail or add descriptive prose. Do not redesign the process structure.\n\n"
        f"LOCKED PROCESS STRUCTURE:\n{json.dumps(locked, indent=2)}\n\n"
        f"SOURCE MATERIAL:\n{text}"
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
        raise RuntimeError(f"Model refused list-flow extraction: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("The model returned no parsed list flows")
    return message.parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast regression for explicit machine-readable inventory lists without full-paper context."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    text = args.source.read_text(encoding="utf-8")
    extraction = extract_list_flows(text, model=args.model)
    report = score_flows(extraction, load_unquantified_expected(args.expected))
    report["source"] = str(args.source)
    report["source_characters"] = len(text)

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    if report["recall"] < args.min_recall or report["precision"] < args.min_precision:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
