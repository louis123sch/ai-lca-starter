from __future__ import annotations

import os
from openai import OpenAI

from .models import InventoryExtraction


SYSTEM_PROMPT = """You are assisting with life-cycle inventory (LCI) construction.
Your job is extraction, not invention.

Rules:
1. Extract only foreground flows supported by the supplied source text.
2. Never invent an amount, unit, material, process, page, table, or functional unit.
3. If a flow is mentioned but no amount is given, amount must be null.
4. Keep construction/capital inputs distinct from operational inputs where the source permits.
5. Keep outputs/co-products distinct from inputs.
6. Preserve the stated basis (per kg product, per year, per plant, etc.). Do not silently convert bases.
7. Include a short evidence_text snippet for every flow.
8. Use [PAGE N] markers to populate page only when available.
9. Record ambiguity, missing denominators, allocation issues, unclear units, or possible double counting in assumptions_or_warnings.
10. The result is a proposal for human review, not an approved LCA model.
"""


def extract_inventory_from_text(
    text: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    extra_instructions: str = "",
) -> InventoryExtraction:
    """Extract an auditable proposed inventory using OpenAI Structured Outputs.

    The OpenAI Python SDK's Pydantic parser converts the model output directly into
    ``InventoryExtraction`` and raises if the result does not satisfy the schema.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("No source text supplied")

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    chosen_model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")

    user_prompt = (
        "Extract the foreground LCI information from the source below. "
        "Treat values as unapproved until human review.\n\n"
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

    return message.parsed
