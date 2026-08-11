from __future__ import annotations

import re

from openai import OpenAI

from .models import FlowExtraction, ForegroundStructure, InventoryFlow


COMPLETENESS_AUDIT_SYSTEM_PROMPT = """You are a conservative completeness auditor for foreground life-cycle inventory (LCI) extraction.
The foreground process structure is already locked and an initial flow extraction already exists. Your only task is to identify source-supported foreground flows that are genuinely missing from that initial extraction.

Rules:
1. Never add, split, merge, rename, or reinterpret locked processes.
2. Return ONLY missing flows. Do not repeat a flow already represented in the initial extraction, even if wording differs slightly.
3. Audit explicit modeled inventory evidence exhaustively: LCI table rows, component/BOM lists, numbered or bulleted inventory lists, stack/BoP groups, supplementary inventory tables, and transcribed visual tables.
4. Do not treat descriptive technology prose, background-dataset explanations, operating conditions, options, or examples as missing foreground inventory.
5. Preserve each missing item's printed source name and word order. Do not canonicalise to an ecoinvent-style name.
6. Never invent amounts, units, materials, voltage levels, markets, routes, geographies, stages, or provenance.
7. If an explicit modeled inventory item has no stated amount, use amount=null.
8. Every returned flow must include a short evidence_text copied from the supplied source material that directly supports that item.
9. linked_process_id is only for an explicit exchange between two already locked foreground process IDs; otherwise null.
10. Precision is more important than speculative completion: if it is unclear whether an item is modeled foreground inventory, omit it and record the ambiguity as a warning.
11. Preserve the source's inventory granularity. A machine-readable background-dataset breakdown beneath an already modeled equipment/component/application does NOT automatically create additional foreground flows. Do not expand an equipment or BoP assembly into every material, fabrication, utility, or treatment dataset used to model it unless the source independently presents those rows as explicit foreground inventory items at that same modeled level.
12. When a machine-readable row contains both a background dataset field and a component/application field, use the component/application field to determine what foreground item the row belongs to. The dataset field is supporting composition/mapping evidence, not by itself proof that its dataset name should become a foreground flow.
13. Return a material/service row from such a detailed export only when the surrounding source evidence clearly treats that material/service as an explicit modeled inventory item rather than merely a background decomposition of a broader component. If the broader component is the modeled inventory item, audit that component instead and do not emit its internal dataset rows.
14. This completeness audit supplements missing modeled INPUTS from inventory lists. Do not add reference-product outputs, coproduct outputs, internal foreground-unit references, or background transport-dataset labels merely because they appear in an operation/workbook sheet; those remain the responsibility of the initial extraction unless they are explicitly part of the locked foreground inventory definition.
"""


def _normalise(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().casefold())


def _flow_key(flow: InventoryFlow) -> tuple[str, str, str]:
    return (_normalise(flow.process_id), _normalise(flow.name), _normalise(flow.direction))


def _evidence_supported(flow: InventoryFlow, source_text: str) -> bool:
    evidence = _normalise(flow.evidence.evidence_text)
    source = _normalise(source_text)
    return bool(evidence) and evidence in source


def _safe_audit_input(flow: InventoryFlow) -> bool:
    """Keep the audit additive but prevent it from manufacturing other exchange classes."""
    return flow.direction == "input" and flow.linked_process_id is None


def merge_missing_flows(
    initial: list[InventoryFlow],
    audited: list[InventoryFlow],
    *,
    source_text: str,
    allowed_process_ids: set[str],
) -> list[InventoryFlow]:
    """Conservatively merge audited flows, rejecting unsupported evidence and duplicates."""
    merged = list(initial)
    seen = {_flow_key(flow) for flow in initial}
    allowed = {_normalise(process_id) for process_id in allowed_process_ids}

    for flow in audited:
        if _normalise(flow.process_id) not in allowed:
            continue
        if not _safe_audit_input(flow):
            continue
        if not _evidence_supported(flow, source_text):
            continue
        key = _flow_key(flow)
        if key in seen:
            continue
        merged.append(flow)
        seen.add(key)
    return merged


def audit_missing_flows(
    source_text: str,
    structure: ForegroundStructure,
    initial: FlowExtraction,
    *,
    client: OpenAI,
    model: str,
) -> FlowExtraction:
    """Run an independent missing-flow audit against a locked foreground structure."""
    user_prompt = (
        "Audit the initial extraction against the complete source material and return only genuinely missing "
        "source-supported foreground flows.\n\n"
        f"LOCKED PROCESS STRUCTURE:\n{structure.model_dump_json(indent=2)}\n\n"
        f"INITIAL FLOW EXTRACTION:\n{initial.model_dump_json(indent=2)}\n\n"
        f"SOURCE MATERIAL:\n{source_text}"
    )
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": COMPLETENESS_AUDIT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=FlowExtraction,
    )
    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"Model refused flow completeness audit: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("The model returned no parsed flow completeness audit")
    return message.parsed
