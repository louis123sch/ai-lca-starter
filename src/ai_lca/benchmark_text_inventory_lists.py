from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from .benchmark_visual_to_flows import PROCESS_NAMES
from .llm import FLOW_SYSTEM_PROMPT, _client
from .models import FlowExtraction


def _norm(value: str | None) -> str:
    if value is None:
        return ""
    value = value.casefold().replace("&", " and ").replace("³", "3")
    value = value.replace("preheating", "pre heater")
    value = re.sub(r"\b(components?|pieces?|units?)\b", " ", value)
    value = re.sub(r"[^a-z0-9.]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _tokens(value: str | None) -> set[str]:
    tokens = set(_norm(value).split())
    singular = set()
    for token in tokens:
        singular.add(token[:-1] if len(token) > 4 and token.endswith("s") else token)
    return singular - {"and", "for", "with"}


def _name_matches(expected: str, actual: str) -> bool:
    if _norm(expected) == _norm(actual):
        return True
    exp = _tokens(expected)
    act = _tokens(actual)
    if not exp or not act:
        return False
    # Benchmark matching should accept printed source qualifiers/counts and word-order changes,
    # but still require the expected concept tokens to be substantially present.
    overlap = len(exp & act)
    return overlap / len(exp) >= 0.8 and overlap / min(len(exp), len(act)) >= 0.8


def load_unquantified_expected(path: Path) -> list[dict[str, str]]:
    """Return explicit inventory-list rows whose source does not state a quantity."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if not (row.get("amount") or "").strip()]


def score_unquantified_flows(extraction: FlowExtraction, expected: list[dict[str, str]]) -> dict:
    """Score only the unquantified inventory-list slice targeted by this focused regression.

    The focused pass may legitimately recover quantified foreground flows that happen to be
    present in the same source excerpt. Those rows are useful extraction output, but they are
    outside this benchmark slice rather than unsupported hallucinations. Precision is therefore
    computed against unmatched *unquantified* rows only; quantified rows remain visible in the
    report as out-of-scope extractions so the benchmark does not hide them.
    """
    unmatched = list(extraction.flows)
    matched: list[dict] = []
    missing: list[dict] = []
    for row in expected:
        found = None
        for index, flow in enumerate(unmatched):
            if _norm(flow.process_id) != _norm(row["process_key"]):
                continue
            aliases = [row["name"]] + [x for x in (row.get("aliases") or "").split("|") if x]
            if not any(_name_matches(alias, flow.name) for alias in aliases):
                continue
            if flow.amount is not None:
                continue
            found = index
            break
        payload = {"process_key": row["process_key"], "name": row["name"]}
        if found is None:
            missing.append(payload)
        else:
            flow = unmatched.pop(found)
            matched.append(payload | {"actual": flow.model_dump(mode="json")})

    unsupported_flows = [flow for flow in unmatched if flow.amount is None]
    quantified_out_of_scope = [flow for flow in unmatched if flow.amount is not None]
    unsupported = [flow.model_dump(mode="json") for flow in unsupported_flows]
    total = len(expected)
    recall = len(matched) / total if total else 1.0
    denominator = len(matched) + len(unsupported)
    precision = len(matched) / denominator if denominator else 1.0
    return {
        "expected_rows": total,
        "matched_rows": len(matched),
        "missing_rows": len(missing),
        "unsupported_rows": len(unsupported),
        "quantified_out_of_scope_rows": len(quantified_out_of_scope),
        "recall": recall,
        "precision": precision,
        "matched": matched,
        "missing": missing,
        "unsupported": unsupported,
        "quantified_out_of_scope": [flow.model_dump(mode="json") for flow in quantified_out_of_scope],
        "all_extracted_flows": [flow.model_dump(mode="json") for flow in extraction.flows],
        "warnings": extraction.assumptions_or_warnings,
    }


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
    report = score_unquantified_flows(extraction, load_unquantified_expected(args.expected))
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
