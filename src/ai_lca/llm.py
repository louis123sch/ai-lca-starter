from __future__ import annotations

import os
from openai import OpenAI

from .models import InventoryExtraction


SYSTEM_PROMPT = """You are assisting with life-cycle inventory (LCI) construction.
Your job is extraction and classification, not invention.

Rules:
1. Extract only LCA-relevant information supported by the supplied source text.
2. Never invent an amount, unit, material, process, page, table, parameter, source document, or functional unit.
3. If an item is mentioned but no amount is given, amount must be null.
4. Keep construction/capital inputs distinct from operational inputs where the source permits.
5. Keep outputs/co-products distinct from inputs.
6. Preserve the stated basis (per kg product, per year, per plant, etc.). Do not silently convert bases.
7. Include a short evidence_text snippet for every extracted item.
8. Use [PAGE N] markers to populate page only when available. Use [TABLE N] markers to populate table when available.
9. Record ambiguity, missing denominators, allocation issues, unclear units, conflicting values, or possible double counting in assumptions_or_warnings.
10. Classify every item using exactly one item_type:
    - technosphere_flow: purchased/consumed materials, fuels, electricity, transport, services, or infrastructure exchanges that may map to a background activity.
    - biosphere_flow: direct elementary flows such as emissions to air/water/soil or resource extraction.
    - parameter: model/scaling values such as plant lifetime, operating hours per year, capacity, efficiency, yield, load factor, degradation rate, or utilisation. Parameters must NOT be treated as ecoinvent-searchable flows.
    - reference_product: the modelled product/output or co-product used to define the foreground activity or functional unit.
11. Plant lifetime, operating hours, capacity, efficiency, yield, and load factor are parameters even when they have units.
12. The result is a proposal for human review, not an approved LCA model.
"""


def extract_inventory_from_text(
    text: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    extra_instructions: str = "",
    source_document: str | None = None,
) -> InventoryExtraction:
    """Extract one auditable proposed inventory using OpenAI Structured Outputs."""
    text = (text or "").strip()
    if not text:
        raise ValueError("No source text supplied")

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    chosen_model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")

    user_prompt = (
        "Extract and classify the LCA-relevant information from this source. "
        "Treat values as unapproved until human review.\n\n"
    )
    if source_document:
        user_prompt += f"Source filename: {source_document}\n"
        user_prompt += (
            "The application will enforce this filename on all extracted evidence after parsing.\n\n"
        )
    if extra_instructions.strip():
        user_prompt += f"Study-specific instructions:\n{extra_instructions.strip()}\n\n"
    user_prompt += f"SOURCE MATERIAL:\n{text}"

    completion = client.beta.chat.completions.parse(
        model=chosen_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=InventoryExtraction,
    )

    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"Model refused extraction: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("The model returned no parsed structured inventory")

    parsed = message.parsed
    if source_document:
        for flow in parsed.flows:
            flow.evidence.source_document = source_document
    return parsed
