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
11. Calculation coefficients, conversion factors, lifetime production totals used only for normalisation, and cells explicitly marked as absent (for example '-') are not inventory exchanges. Do not emit them as flows unless the source explicitly models them as an exchange.
12. When a row labels one exchange as `component (material for component)`, treat that as one material exchange; do not emit separate parent and parenthetical variants unless the source explicitly models both.
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
    value = re.sub(r"\((?:\s*\d+\s*(?:x|×)?\s*|(?:including|incl\.?)[^)]*|rectifier[^)]*|de-oxo[^)]*)\)", " ", value)
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


def _embedded_material_identity(flow: InventoryFlow) -> str | None:
    """Treat `component (material for component)` as the material exchange, not two identities.

    The rule is deliberately structural and source-agnostic: it activates only when a trailing
    parenthetical contains a `... for ...` material-style label whose target overlaps the
    outer component label. This collapses presentation variants while retaining distinct
    materials for the same component.
    """
    match = re.search(r"\(([^()]*)\)\s*$", flow.name or "")
    if not match:
        return None
    inner = _normalise_name(match.group(1))
    outer = _normalise_name((flow.name or "")[: match.start()])
    if " for " not in f" {inner} " or not outer:
        return None
    _, target = inner.split(" for ", 1)
    outer_tokens = {token for token in outer.split() if len(token) >= 4}
    target_tokens = {token for token in target.split() if len(token) >= 4}
    if not (outer_tokens & target_tokens):
        return None
    return inner


def _is_non_exchange_calculation_metadata(flow: InventoryFlow) -> bool:
    """Reject rows that explicitly describe calculation metadata rather than LCI exchanges.

    This is deliberately narrow: source-backed unquantified components remain valid flows.
    Only explicit factor/normalisation/absence signals are excluded.
    """
    name = (flow.name or "").casefold().strip()
    evidence = (flow.evidence.evidence_text or "").casefold().strip()

    if "additional manufacturing energy factor" in name:
        return True

    if re.search(r"\bproduced amount of\b.*\b(?:in|over)\s+\d+(?:\.\d+)?\s*years?\b", name):
        return True

    if flow.amount is None:
        explicit_absence = (
            re.search(r"=\s*-\s*(?:\)|$)", evidence)
            or re.search(r"\|\s*-\s*(?:\||$)", evidence)
            or re.search(r"\t-\s*(?:\t|$)", evidence)
        )
        if explicit_absence:
            return True

    return False


def _flow_identity(flow: InventoryFlow) -> tuple[str, str, str]:
    semantic = _manufacturing_energy_identity(flow) or _embedded_material_identity(flow)
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
        if _is_non_exchange_calculation_metadata(flow):
            continue
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


def _canonical_locked_process_id(process_id: str, structure: ForegroundStructure) -> str | None:
    """Resolve a flow attachment to the locked graph without creating subprocesses.

    Structured-output models can occasionally return an internal candidate ID even though
    the prompt supplies only locked process IDs. Internal stages are bookkeeping children of
    a retained process, so their flows belong on the nearest locked ancestor. Candidates with
    no locked ancestry are not valid flow attachments and are rejected.
    """
    locked_ids = {process.process_id for process in structure.processes}
    if process_id in locked_ids:
        return process_id

    candidates = {candidate.candidate_id: candidate for candidate in structure.candidate_activities}
    current = candidates.get(process_id)
    seen: set[str] = set()
    while current is not None and current.candidate_id not in seen:
        seen.add(current.candidate_id)
        parent_id = current.parent_candidate_id
        if parent_id in locked_ids:
            return parent_id
        current = candidates.get(parent_id) if parent_id else None
    return None


def _enforce_locked_flow_attachments(
    flows: Iterable[InventoryFlow], structure: ForegroundStructure
) -> list[InventoryFlow]:
    """Keep flows on the locked foreground graph and collapse internal-stage attachments upward."""
    enforced: list[InventoryFlow] = []
    for flow in flows:
        process_id = _canonical_locked_process_id(flow.process_id, structure)
        if process_id is None:
            continue
        if process_id != flow.process_id:
            flow = flow.model_copy(update={"process_id": process_id})
        enforced.append(flow)
    return enforced


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

    base_flows = _enforce_locked_flow_attachments(base.flows, structure)
    recovered_flows = _enforce_locked_flow_attachments(recovered, structure)
    merged = merge_supported_flows(base_flows, recovered_flows)
    warnings = list(
        dict.fromkeys(
            base.assumptions_or_warnings
            + ingestion_warnings
            + vision_warnings
            + recovery_warnings
        )
    )
    return base.model_copy(update={"flows": merged, "assumptions_or_warnings": warnings})
