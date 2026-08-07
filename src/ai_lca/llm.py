from __future__ import annotations

import os
from openai import OpenAI

from .models import (
    DocumentUnderstanding,
    InventoryExtraction,
    InventoryItemsResult,
)


UNDERSTANDING_PROMPT = """You are an expert life-cycle assessment modeller familiar with ecoinvent v3.
Read the supplied technical source as a whole before proposing any inventory items.
This first pass is ONLY for understanding the technology and foreground process context.

Important ecoinvent concepts to keep in mind:
- ecoinvent datasets represent activities; foreground inputs normally consume intermediate products/services supplied by background activities.
- Market activities conventionally start with 'market for ...' and represent regional consumption mixes.
- Treatment activities conventionally start with 'treatment of ...'.
- Geography is a separate dataset attribute and should not be embedded into the canonical product name.
- A specific known supplier/technology can justify a direct transforming-activity link instead of a generic market.

Rules:
1. Identify the technology/system actually described by the source.
2. Propose only foreground process/stage names that are explicitly described or unambiguously supported by the source. Do not add standard process stages merely because they are common in the technology.
3. If the source clearly links an input/output to a particular stage (e.g. electricity to CO2 capture), preserve that stage relationship for the next pass.
4. Identify explicit geography/provenance information separately from process names.
5. Do not extract a bill of materials or list of numeric inventory items in this pass.
6. Return concise modelling observations, not private chain-of-thought.
"""


EXTRACTION_PROMPT = """You are an expert LCA foreground modeller familiar with ecoinvent v3.
You have already read and interpreted the source once. Now use the supplied document understanding plus the source evidence to build a selective draft foreground inventory.

Your main task is NOT to copy every quantitative phrase. Your main task is to identify the things that most plausibly become foreground exchanges linked to ecoinvent background activities, while retaining genuinely useful model parameters separately.

For each item:
- Canonicalise the underlying product/service. Example: 'natural gas - SMR + CCS 90%' becomes the technosphere concept 'natural gas'; SMR/CCS is foreground context and 90% capture is a parameter if supported.
- Assign parent_process when the source clearly links the item to one of the proposed foreground processes/stages.
- Keep geography separate in geography_hint. Do not put GB, RER, Norway, etc. into the canonical name unless it is genuinely part of the product identity.
- For a plausible technosphere background link, set search_worthy=true and give a concise ecoinvent_search_term representing the product/service, not an invented exact dataset name.
- Suggest ecoinvent_activity_type_hint only as a semantic hint: market, transforming, treatment, transport, service, construction, operation, or unknown.
- Generic purchased/traded commodities normally suggest a market; explicitly specified production technologies/suppliers may suggest transforming; wastes needing disposal suggest treatment.
- Put explicit supply technology/provenance such as offshore wind or North Sea production into supplier_technology_hint rather than contaminating the canonical product name.

Be selective:
- Good background-link candidates include materials, fuels, electricity/heat, water, chemicals, transport services, treatment services, infrastructure materials/equipment, and other intermediate products/services actually consumed by the foreground system.
- Do NOT make scenario labels, technology names, capture percentages, plant capacity, efficiency, lifetime, operating hours, load factor, yield, or similar modelling descriptors into ecoinvent-searchable flows.
- Retain such values as parameters only when they are useful for scaling, allocation, amortisation, mass/energy balance, emissions, or scenario definition.
- Direct elementary emissions/resources are biosphere_flow and are not sent to the ecoinvent technosphere matcher.
- The modelled output/co-product is reference_product and is not sent to the ecoinvent technosphere matcher.

Evidence and auditability:
1. Never invent an amount, unit, process relationship, geography, supplier technology, page, table, or functional unit.
2. If an exchange is explicitly present but has no amount, amount may be null.
3. Preserve the stated basis; do not silently convert denominators.
4. Include a short verbatim evidence_text for every item.
5. Use [PAGE N] and [TABLE N] markers only when actually available.
6. interpretation_reason must be a short reviewable justification, e.g. 'Natural gas is a consumed feedstock; SMR + CCS describes the foreground process and capture scenario.' Do not provide hidden chain-of-thought.
7. The output is a proposal for human review, not an approved LCA model or assertion that an ecoinvent dataset exists.
"""


def _client(api_key: str | None = None) -> OpenAI:
    return OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))


def understand_document(
    text: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    extra_instructions: str = "",
    source_document: str | None = None,
) -> DocumentUnderstanding:
    """First AI pass: understand technology and foreground process context."""
    text = (text or "").strip()
    if not text:
        raise ValueError("No source text supplied")

    chosen_model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
    user_prompt = "Interpret this source for foreground LCA modelling before extracting inventory.\n\n"
    if source_document:
        user_prompt += f"Source filename: {source_document}\n"
    if extra_instructions.strip():
        user_prompt += f"Study-specific instructions:\n{extra_instructions.strip()}\n\n"
    user_prompt += f"SOURCE MATERIAL:\n{text}"

    completion = _client(api_key).beta.chat.completions.parse(
        model=chosen_model,
        messages=[
            {"role": "system", "content": UNDERSTANDING_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=DocumentUnderstanding,
    )

    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"Model refused document interpretation: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("The model returned no parsed document understanding")

    understanding = message.parsed
    if source_document:
        for process in understanding.foreground_processes:
            process.source_document = source_document
    return understanding


def extract_inventory_from_text(
    text: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    extra_instructions: str = "",
    source_document: str | None = None,
) -> InventoryExtraction:
    """Two-pass AI interpretation: understand the process, then extract a draft foreground."""
    text = (text or "").strip()
    if not text:
        raise ValueError("No source text supplied")

    chosen_model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
    understanding = understand_document(
        text,
        model=chosen_model,
        api_key=api_key,
        extra_instructions=extra_instructions,
        source_document=source_document,
    )

    user_prompt = (
        "Build the selective draft foreground inventory from this source using the prior document understanding below.\n\n"
        f"DOCUMENT UNDERSTANDING:\n{understanding.model_dump_json(indent=2)}\n\n"
    )
    if source_document:
        user_prompt += f"Source filename: {source_document}\n"
    if extra_instructions.strip():
        user_prompt += f"Study-specific instructions:\n{extra_instructions.strip()}\n\n"
    user_prompt += f"SOURCE MATERIAL:\n{text}"

    completion = _client(api_key).beta.chat.completions.parse(
        model=chosen_model,
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=InventoryItemsResult,
    )

    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"Model refused foreground extraction: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("The model returned no parsed foreground inventory")

    items = message.parsed

    # Deterministic routing invariants. The LLM proposes semantics; code decides what can reach search.
    for flow in items.flows:
        if source_document:
            flow.evidence.source_document = source_document

        if flow.item_type != "technosphere_flow":
            flow.search_worthy = False
            flow.ecoinvent_search_term = None
            flow.ecoinvent_activity_type_hint = None
            flow.supplier_technology_hint = None
        elif flow.search_worthy and not (flow.ecoinvent_search_term or "").strip():
            flow.ecoinvent_search_term = flow.name.strip()

    return InventoryExtraction(
        process_name=items.process_name or understanding.technology_name,
        technology_name=understanding.technology_name,
        functional_unit=items.functional_unit,
        system_description=understanding.system_description,
        foreground_processes=understanding.foreground_processes,
        source_summary=(
            f"Two-pass foreground interpretation of {source_document}"
            if source_document
            else "Two-pass foreground interpretation of supplied source material"
        ),
        assumptions_or_warnings=[
            *understanding.interpretation_notes,
            *items.assumptions_or_warnings,
        ],
        flows=items.flows,
    )
