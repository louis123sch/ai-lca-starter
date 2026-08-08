from __future__ import annotations

import json
import os
from openai import OpenAI

from .models import InventoryExtraction, ProcessMap
from .validation import normalise_process_ids, validate_inventory_against_process_map


PROCESS_MAP_SYSTEM_PROMPT = """You are assisting with life-cycle assessment (LCA) model reconstruction from source documents.
Your first job is PROCESS-STRUCTURE DISCOVERY, not inventory extraction and not engineering decomposition.

A foreground process means a process that the SOURCE LCA itself treats as a distinct modelled unit.
An engineering operation or physical step is not automatically a separate foreground process.

Strict rules:
1. Create a foreground process only when the source provides direct evidence that the LCA models it separately.
2. Good evidence includes a dedicated inventory table, a separate process box in an LCA/system-boundary figure, a separately quantified mass/energy balance, an explicit intermediate linkage between modelled processes, or explicit text clearly defining a separate LCA process.
3. Do NOT create separate foreground processes merely because the technology description mentions reactor heating, compression, separation, purification, pumping, plasma generation, electricity supply, natural-gas supply, or similar engineering operations.
4. If one inventory table represents the whole technology/pathway, create ONE foreground process and record internal engineering steps under operations.
5. Group related processes under the technology/pathway they belong to (for example, all methane-pyrolysis processes together).
6. Every foreground process must include direct source evidence. Do not use general engineering knowledge as evidence.
7. Do not infer an electricity voltage level, grid market, fuel route, geography, or technology variant unless the source supports it.
8. Capture overall and process-specific geography/time context when the source states or clearly defines it; otherwise leave null and warn.
9. Missing a process is preferable to inventing one. Record ambiguity in assumptions_or_warnings for human review.
10. The result is a proposed process map for human review, not an approved Brightway model.
"""


INVENTORY_SYSTEM_PROMPT = """You are assisting with life-cycle inventory (LCI) construction under an already defined ProcessMap.
Your job is extraction, not invention.

Rules:
1. Extract flows ONLY for the foreground process IDs supplied in the approved process map.
2. Never create another process or process ID. Described operations are context inside a process, not separate processes.
3. Extract only foreground flows supported by the supplied source text.
4. Never invent an amount, unit, material, voltage level, process, page, table, functional unit, geography, or market.
5. If the source says only 'electricity', the flow name must remain 'electricity'. Do not turn it into low-, medium-, or high-voltage electricity unless the source explicitly says so.
6. A technical operation is not itself an exchange. Only extract material/energy/transport/output/emission flows actually stated or clearly tabulated for the modelled process.
7. If a flow is mentioned but no amount is given, amount must be null.
8. Keep construction/capital inputs distinct from operational inputs where the source permits.
9. Keep outputs/co-products distinct from inputs.
10. Preserve the stated basis (per kg product, per year, per plant, etc.). Do not silently convert bases.
11. Include a short evidence_text snippet for every flow.
12. Use [PAGE N] markers to populate page only when available.
13. Record ambiguity, missing denominators, allocation issues, unclear units, or possible double counting in assumptions_or_warnings.
14. Do not repeat the same exchange because it appears in multiple places in the document.
15. The result is a proposal for human review, not an approved LCA model.
"""


def _client(api_key: str | None = None) -> OpenAI:
    return OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))


def _chosen_model(model: str | None = None) -> str:
    return model or os.getenv("OPENAI_MODEL", "gpt-5-mini")


def extract_process_map_from_text(
    text: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    extra_instructions: str = "",
) -> ProcessMap:
    """Discover the evidence-backed foreground-process hierarchy before extracting flows."""
    text = (text or "").strip()
    if not text:
        raise ValueError("No source text supplied")

    user_prompt = (
        "Reconstruct the foreground process structure represented by the source LCA. "
        "Separate actual foreground processes from engineering operations described inside them.\n\n"
    )
    if extra_instructions.strip():
        user_prompt += f"Study-specific instructions:\n{extra_instructions.strip()}\n\n"
    user_prompt += f"SOURCE MATERIAL:\n{text}"

    completion = _client(api_key).beta.chat.completions.parse(
        model=_chosen_model(model),
        messages=[
            {"role": "system", "content": PROCESS_MAP_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=ProcessMap,
    )

    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"Model refused process-map extraction: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("The model returned no parsed process map")

    return normalise_process_ids(message.parsed)


def extract_inventory_from_text(
    text: str,
    *,
    process_map: ProcessMap,
    approved_process_ids: list[str] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    extra_instructions: str = "",
) -> InventoryExtraction:
    """Extract an auditable inventory constrained to already identified foreground processes."""
    text = (text or "").strip()
    if not text:
        raise ValueError("No source text supplied")

    approved = set(approved_process_ids or [])
    selected_groups = []
    for group in process_map.technology_groups:
        processes = [
            process.model_dump()
            for process in group.processes
            if not approved or process.process_id in approved
        ]
        if processes:
            selected_groups.append({"name": group.name, "processes": processes})

    if not selected_groups:
        raise ValueError("No foreground processes are selected for inventory extraction")

    approved_payload = {
        "functional_unit": process_map.functional_unit,
        "system_boundary": process_map.system_boundary,
        "geographic_context": process_map.geographic_context,
        "temporal_context": process_map.temporal_context,
        "technology_groups": selected_groups,
    }

    user_prompt = (
        "Extract foreground inventory flows for ONLY the approved foreground processes below. "
        "Operations listed inside a process are descriptive context and must not become new processes or extra ecoinvent searches.\n\n"
        f"APPROVED PROCESS MAP:\n{json.dumps(approved_payload, indent=2, ensure_ascii=False)}\n\n"
    )
    if extra_instructions.strip():
        user_prompt += f"Study-specific instructions:\n{extra_instructions.strip()}\n\n"
    user_prompt += f"SOURCE MATERIAL:\n{text}"

    completion = _client(api_key).beta.chat.completions.parse(
        model=_chosen_model(model),
        messages=[
            {"role": "system", "content": INVENTORY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=InventoryExtraction,
    )

    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"Model refused inventory extraction: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("The model returned no parsed structured inventory")

    return validate_inventory_against_process_map(message.parsed, process_map, text)
