from __future__ import annotations

import json
import os
from openai import OpenAI

from .models import InventoryExtraction, ProcessMap
from .validation import normalise_process_ids, validate_inventory_against_process_map


PROCESS_MAP_SYSTEM_PROMPT = """You are assisting with life-cycle assessment (LCA) model reconstruction from an EVIDENCE CORPUS that may contain several uploaded documents.
Your first job is PROCESS-STRUCTURE DISCOVERY across the corpus, not inventory extraction and not engineering decomposition.

Treat all uploaded documents as one evidence base. A process may be supported jointly by several documents. The main paper, supplementary information, appendices, technical reports, datasheets, and notes are not separate modelling worlds unless the evidence explicitly says they describe different systems or scenarios.

A foreground process means a process that the combined evidence supports as a distinct modelled LCA unit. An engineering operation or physical step is not automatically a separate foreground process.

Strict rules:
1. Build one coherent process map from ALL supplied documents together.
2. Merge descriptions from different documents when they refer to the same foreground process. Do not create duplicate processes merely because the same process appears in several files.
3. Create a foreground process only when the evidence corpus provides direct support that the LCA models it separately.
4. Good evidence includes a dedicated inventory table, a separate process box in an LCA/system-boundary figure, a separately quantified mass/energy balance, an explicit intermediate linkage between modelled processes, or explicit text clearly defining a separate LCA process.
5. Do NOT create separate foreground processes merely because the technology description mentions reactor heating, compression, separation, purification, pumping, plasma generation, electricity supply, natural-gas supply, or similar engineering operations.
6. If the combined evidence represents one inventory for the whole technology/pathway, create ONE foreground process and record internal engineering steps under operations even if those steps are described across several documents.
7. Group related processes under the technology/pathway they belong to (for example, all methane-pyrolysis processes together).
8. Every foreground process must include direct evidence. Add evidence from every relevant document where useful. Populate source_document from the nearest [DOCUMENT ...] marker.
9. Do not use general engineering knowledge as evidence and do not infer an electricity voltage level, grid market, fuel route, geography, or technology variant unless the corpus supports it.
10. Capture overall and process-specific geography/time context when supported anywhere in the corpus; otherwise leave null and warn.
11. If two documents conflict about process structure, quantity basis, geography, or scenario, do not silently choose. Record the conflict in assumptions_or_warnings and keep distinct scenarios separate only when the source evidence explicitly treats them as distinct.
12. Missing a process is preferable to inventing one.
13. The result is a proposed process map for human review, not an approved Brightway model.
"""


INVENTORY_SYSTEM_PROMPT = """You are assisting with life-cycle inventory (LCI) construction from an EVIDENCE CORPUS under an already defined ProcessMap.
Your job is evidence synthesis and extraction, not invention.

Treat all uploaded documents as one evidence base. A single foreground flow may be supported by several documents. Combine complementary evidence for the same flow instead of duplicating the flow once per document.

Rules:
1. Extract flows ONLY for the foreground process IDs supplied in the approved process map.
2. Never create another process or process ID. Described operations are context inside a process, not separate processes.
3. Build each process inventory from ALL relevant evidence across the document corpus.
4. If one document gives the process identity and another provides its quantities, combine them when the evidence clearly refers to the same process/scenario.
5. Do not duplicate a flow because it appears in multiple documents. Return one flow with multiple evidence records.
6. If documents provide conflicting amounts, units, bases, scenarios, or system boundaries for what appears to be the same flow, do not average or arbitrarily choose. Record the conflict in assumptions_or_warnings; keep separate rows only when the documents clearly define distinct scenarios/bases.
7. Extract only foreground flows supported by the supplied evidence corpus.
8. Never invent an amount, unit, material, voltage level, process, page, table, functional unit, geography, or market.
9. If the evidence says only 'electricity', the flow name must remain 'electricity'. Do not turn it into low-, medium-, or high-voltage electricity unless explicitly stated in the corpus.
10. A technical operation is not itself an exchange. Only extract material/energy/transport/output/emission flows actually stated or clearly tabulated for the modelled process.
11. If a flow is mentioned but no amount is given, amount must be null.
12. Keep construction/capital inputs distinct from operational inputs where the evidence permits.
13. Keep outputs/co-products distinct from inputs.
14. Preserve the stated basis (per kg product, per year, per plant, etc.). Do not silently convert bases.
15. Include evidence records for each flow. Populate source_document from [DOCUMENT ...] markers and page/table/section where available.
16. The result is a proposal for human review, not an approved LCA model.
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
    """Discover one evidence-backed foreground-process hierarchy across the full corpus."""
    text = (text or "").strip()
    if not text:
        raise ValueError("No source text supplied")

    user_prompt = (
        "Reconstruct one coherent foreground process structure from the complete evidence corpus below. "
        "Use evidence across documents jointly, merge repeated descriptions of the same modelled process, and separate actual foreground processes from engineering operations described inside them.\n\n"
    )
    if extra_instructions.strip():
        user_prompt += f"Study-specific instructions:\n{extra_instructions.strip()}\n\n"
    user_prompt += f"EVIDENCE CORPUS:\n{text}"

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
    """Extract one auditable inventory across all evidence, constrained to approved processes."""
    text = (text or "").strip()
    if not text:
        raise ValueError("No source text supplied")

    approved = set(approved_process_ids) if approved_process_ids is not None else None
    selected_groups = []
    selected_ids: set[str] = set()
    for group in process_map.technology_groups:
        processes = [
            process.model_dump()
            for process in group.processes
            if approved is None or process.process_id in approved
        ]
        if processes:
            selected_groups.append({"name": group.name, "processes": processes})
            selected_ids.update(process["process_id"] for process in processes)

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
        "Build the foreground inventory for ONLY the approved foreground processes below using the entire evidence corpus. "
        "Combine complementary evidence across documents into the same process and flow. Operations listed inside a process are descriptive context and must not become new processes or extra ecoinvent searches.\n\n"
        f"APPROVED PROCESS MAP:\n{json.dumps(approved_payload, indent=2, ensure_ascii=False)}\n\n"
    )
    if extra_instructions.strip():
        user_prompt += f"Study-specific instructions:\n{extra_instructions.strip()}\n\n"
    user_prompt += f"EVIDENCE CORPUS:\n{text}"

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

    return validate_inventory_against_process_map(
        message.parsed,
        process_map,
        text,
        allowed_process_ids=selected_ids,
    )
