from __future__ import annotations

import re
from collections.abc import Iterable

from .documents import combine_document_evidence
from .llm import FLOW_SYSTEM_PROMPT, _client, extract_inventory_from_documents, transcribe_visual_evidence
from .models import FlowExtraction, ForegroundStructure, InventoryExtraction, InventoryFlow


FOCUSED_LIST_PROMPT = """This is a focused COMPLETENESS RECOVERY pass over a bounded source-evidence chunk.
The foreground process structure is already locked and MUST NOT be changed.

Extract only explicit modeled foreground inventory items that are visibly/listed in this chunk, especially:
- numbered or bulleted component lists;
- rows in LCI, component, BoP, stack, material, input/output, or inventory tables;
- transcribed visual-table rows;
- explicit supplementary inventory lists.

Rules:
1. Recover EVERY distinct source-supported listed item in the chunk; do not stop at a representative subset.
2. Preserve the source's printed flow/component name and word order. Parenthetical counts or qualifiers may remain in the name when printed.
3. Attach each item to exactly one supplied locked process ID. Never create or rename a process.
4. Use amount=null when no quantity is stated. Never turn a component count into a material mass or other unsupported quantity.
5. Do not extract descriptive technology prose, generic operating conditions, literature-only examples, or background dataset detail unless it is explicitly part of a modeled inventory list/table.
6. Do not infer ecoinvent mappings, voltage levels, markets, geographies, production routes, or missing quantities.
7. Keep evidence text short and source-faithful. Preserve provenance markers when present.
8. If the chunk contains no explicit modeled inventory list/table rows, return an empty flow list.
9. Do not emit both an intermediate calculation/plant-total row and its explicitly reported final functional-unit-normalised row as separate flows when the source makes clear they represent the same inventory contribution. Prefer the final modeled basis; retain both only when the source explicitly models them as distinct exchanges.
10. Column-unit suffixes, row numbers, counts, or explanatory qualifiers are source evidence, not reasons to duplicate an inventory item already recovered from the same process and direction.
"""


_UNIT_SUFFIX_RE = re.compile(
    r"\(\s*(?:l|ml|kg|g|mg|t|kwh|mwh|wh|mj|gj|j|m3|m2|m|kw|mw|w|bar|pa)\s*\)\s*$",
    re.IGNORECASE,
)


def _normalise_name(value: str) -> str:
    value = value.casefold().replace("&", " and ").replace("³", "3")
    # Row numbers and column-unit suffixes are presentation metadata, not flow identity.
    value = re.sub(r"^\s*\d+[\s.)-]+(?=\S)", "", value)
    value = _UNIT_SUFFIX_RE.sub(" ", value)
    # Counts and explanatory parentheticals are useful evidence but should not create duplicate identities.
    value = re.sub(r"\((?:\s*\d+\s*(?:x|×)?\s*|including[^)]*|rectifier[^)]*|de-oxo[^)]*)\)", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _manufacturing_energy_identity(flow: InventoryFlow) -> str | None:
    """Collapse alternate labels for the same explicitly identified manufacturing-energy contribution."""
    unit = (flow.unit or "").casefold()
    if not any(token in unit for token in ("wh", "joule", "kj", "mj", "gj")):
        return None
    evidence = (flow.evidence.evidence_text or "").casefold()
    combined = f"{flow.name.casefold()} {evidence}"
    if "manufacturing energy" not in combined:
        return None

    residual = _normalise_name(flow.name)
    process_token = _normalise_name(flow.process_id)
    if process_token:
        residual = re.sub(rf"\b{re.escape(process_token)}\b", " ", residual)
    residual = re.sub(r"\badditional manufacturing energy\b", " ", residual)
    residual = re.sub(r"\bmanufacturing energy\b", " ", residual)
    residual = re.sub(r"\s+", " ", residual).strip()
    if "stack" in residual:
        return "manufacturing energy stack"
    if re.search(r"\b(?:bop|balance of plant)\b", residual):
        return "manufacturing energy bop"
    return None


def _flow_identity(flow: InventoryFlow) -> tuple[str, str, str]:
    semantic = _manufacturing_energy_identity(flow)
    return flow.process_id.casefold().strip(), semantic or _normalise_name(flow.name), flow.direction


def _flow_quality(flow: InventoryFlow) -> tuple[int, int, int, int]:
    """Prefer quantified, specific-basis and better-provenanced duplicates without inventing reconciliation."""
    unit = (flow.unit or "").casefold()
    specific_basis = int("/" in unit or " per " in f" {unit} ")
    return (
        int(flow.amount is not None),
        specific_basis,
        int(bool(flow.unit)),
        len((flow.evidence.evidence_text or "").strip()),
    )


def merge_supported_flows(primary: Iterable[InventoryFlow], recovered: Iterable[InventoryFlow]) -> list[InventoryFlow]:
    """Merge evidence-backed flow candidates deterministically, preserving the strongest duplicate."""
    merged: list[InventoryFlow] = []
    positions: dict[tuple[str, str, str], int] = {}
    for flow in [*primary, *recovered]:
        key = _flow_identity(flow)
        index = positions.get(key)
        if index is None:
            positions[key] = len(merged)
            merged.append(flow)
            continue
        if _flow_quality(flow) > _flow_quality(merged[index]):
            merged[index] = flow
    return merged


def _bounded_chunks(text: str, *, max_chars: int = 9000) -> list[str]:
    """Split evidence at provenance boundaries so explicit lists are not diluted by whole-paper context."""
    text = (text or "").strip()
    if not text:
        return []

    # Keep document/page/table/visual markers at the start of their following block.
    parts = re.split(r"(?=\n?\[(?:DOCUMENT:|PAGE\s+\d+|TABLE\s+\d+|VISUAL EVIDENCE:))", text)
    parts = [part.strip() for part in parts if part.strip()]
    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(part) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(part), max_chars):
                chunks.append(part[start : start + max_chars])
            continue
        candidate = part if not current else f"{current}\n\n{part}"
        if len(candidate) > max_chars:
            chunks.append(current)
            current = part
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _looks_inventory_dense(chunk: str) -> bool:
    text = chunk.casefold()
    signals = (
        "inventory",
        "lci",
        "component",
        "bop",
        "stack",
        "input",
        "output",
        "material",
        "[table",
        "visual evidence",
    )
    numbered_rows = len(re.findall(r"(?m)^\s*\d+[\s.)-]+\S", chunk))
    pipe_rows = len(re.findall(r"(?m)^.*\|.*\|.*$", chunk))
    return numbered_rows >= 3 or pipe_rows >= 2 or (any(signal in text for signal in signals) and (numbered_rows or pipe_rows))


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


def _recover_chunk_flows(
    chunk: str,
    structure: ForegroundStructure,
    *,
    model: str,
    api_key: str | None,
) -> FlowExtraction:
    user_prompt = (
        f"LOCKED PROCESS STRUCTURE:\n{structure.model_dump_json(indent=2)}\n\n"
        f"SOURCE EVIDENCE CHUNK:\n{chunk}"
    )
    completion = _client(api_key).beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": FLOW_SYSTEM_PROMPT + "\n\n" + FOCUSED_LIST_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=FlowExtraction,
    )
    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"Model refused focused inventory recovery: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("The model returned no parsed focused inventory recovery")
    return message.parsed


def extract_inventory_from_documents_resilient(
    documents: list[tuple[str, bytes]],
    *,
    model: str | None = None,
    api_key: str | None = None,
    extra_instructions: str = "",
    max_visual_assets: int = 24,
) -> InventoryExtraction:
    """Run the normal extractor, then recover explicit list/table rows in bounded evidence chunks.

    The initial extraction remains authoritative for process structure and general interpretation.
    The recovery pass can only add/strengthen evidence-backed flows attached to those locked processes.
    """
    chosen_model = model or __import__("os").getenv("OPENAI_MODEL", "gpt-5-mini")
    base = extract_inventory_from_documents(
        documents,
        model=chosen_model,
        api_key=api_key,
        extra_instructions=extra_instructions,
        max_visual_assets=max_visual_assets,
    )
    structure = _locked_structure(base)

    native_text, assets, ingestion_warnings = combine_document_evidence(
        documents,
        max_visual_assets=max_visual_assets,
    )
    visual_text, vision_warnings = transcribe_visual_evidence(
        assets,
        model=model,
        api_key=api_key,
    )

    chunks = [chunk for chunk in _bounded_chunks(native_text) if _looks_inventory_dense(chunk)]
    chunks.extend(chunk for chunk in _bounded_chunks(visual_text) if _looks_inventory_dense(chunk))

    recovered: list[InventoryFlow] = []
    recovery_warnings: list[str] = []
    for chunk in chunks:
        try:
            result = _recover_chunk_flows(
                chunk,
                structure,
                model=chosen_model,
                api_key=api_key,
            )
        except Exception as exc:
            recovery_warnings.append(f"Focused inventory recovery skipped one evidence chunk: {exc}")
            continue
        recovered.extend(result.flows)
        recovery_warnings.extend(result.assumptions_or_warnings)

    merged = merge_supported_flows(base.flows, recovered)
    warnings = list(
        dict.fromkeys(
            base.assumptions_or_warnings
            + ingestion_warnings
            + vision_warnings
            + recovery_warnings
        )
    )
    return base.model_copy(update={"flows": merged, "assumptions_or_warnings": warnings})
