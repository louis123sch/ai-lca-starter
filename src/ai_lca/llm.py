from __future__ import annotations

import os

from openai import OpenAI

from .models import FlowExtraction, ForegroundStructure, InventoryExtraction


STRUCTURE_SYSTEM_PROMPT = """You are assisting with life-cycle inventory (LCI) construction from a supplied paper or technical document.
This first pass is PROCESS INTERPRETATION, not inventory completion. Your job is to identify the foreground process structure the source actually represents.

Rules:
1. Identify only foreground processes/subprocesses supported by the supplied source. Do not invent a process because an input exists.
2. Distinguish the LCA MODEL from descriptive technology-review prose. Identify product systems/configurations actually assessed in the LCA, not every technology, catalyst, solvent, reactor option, or unit operation discussed in the review.
3. Prefer the smallest process hierarchy justified by the modeled LCI. A process-flow diagram can show internal unit operations without those being separate foreground LCA processes. Treat LCI tables, goal/scope, system-boundary figures, and explicit model-configuration statements as stronger evidence than generic process-description prose.
4. Create a subprocess only when the LCA source itself models it separately with its own inventory basis or separately quantified inputs/outputs.
5. Do not turn background supplies such as electricity, natural gas, water, steel, transport, or chemicals into foreground subprocesses unless the source explicitly models them as such.
6. Do not infer voltage level, production technology, market dataset, geography code, or other ecoinvent detail from generic flow names.
7. Extract the primary/reference operational geography when explicit. If the paper additionally assesses country or regional scenarios, record them in additional_geographies rather than replacing the reference geography.
8. If geography cannot be supported, use not_identified. Never fill it from a user preference list.
9. Keep construction/capital, operation, maintenance, transport, and end-of-life separate only when the modeled LCI actually distinguishes them.
10. Include short evidence snippets for every process and for study context where possible. Populate document from [DOCUMENT: ...], and page/paragraph/table only from explicit source markers.
11. Record ambiguity and possible double counting in assumptions_or_warnings.
12. This is a proposal for human review, not an approved LCA model.
"""


FLOW_SYSTEM_PROMPT = """You are extracting foreground LCI flows from a supplied paper or technical document.
A first-pass foreground process structure has already been identified. You MUST fill that structure; you may not redesign it.

Rules:
1. Every flow must be attached to one of the supplied process IDs. Never create a new process ID or implicit subprocess.
2. Extract only flows that the source indicates are part of the MODELED foreground LCI. Do not extract catalysts, solvents, materials, operating conditions, or technology options merely because they appear in review/process-description prose. Prefer explicit LCI tables and inventory-analysis sections.
3. Never invent an amount, unit, material, process, functional unit, voltage level, market type, production route, document, page, table, or paragraph.
4. If the paper says only 'electricity', extract 'electricity'. Do NOT turn it into 'medium-voltage electricity', 'market for electricity', or another ecoinvent-style dataset name unless the source states that detail.
5. If a flow is mentioned but no amount is given, amount must be null.
6. Preserve the stated basis (per kg product, per year, per plant, etc.). Do not silently convert bases.
7. Keep outputs/co-products distinct from inputs and elementary emissions.
8. Do not duplicate the same flow merely because it appears in prose and a table; use the clearest supporting evidence and warn about conflicting values.
9. Include a short evidence_text snippet for every flow. Populate document from [DOCUMENT: ...], and page/paragraph/table only from explicit source markers.
10. Record ambiguity, missing denominators, allocation issues, unclear units, or possible double counting in assumptions_or_warnings.
"""


def _client(api_key: str | None = None) -> OpenAI:
    return OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))


def identify_foreground_structure(
    text: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    extra_instructions: str = "",
) -> ForegroundStructure:
    """First pass: identify the evidence-backed process hierarchy and study context."""
    text = (text or "").strip()
    if not text:
        raise ValueError("No source text supplied")

    chosen_model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
    user_prompt = "Identify the foreground process structure and study context from the source below.\n\n"
    if extra_instructions.strip():
        user_prompt += f"Study-specific instructions:\n{extra_instructions.strip()}\n\n"
    user_prompt += f"SOURCE MATERIAL:\n{text}"

    completion = _client(api_key).beta.chat.completions.parse(
        model=chosen_model,
        messages=[
            {"role": "system", "content": STRUCTURE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=ForegroundStructure,
    )
    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"Model refused structure extraction: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("The model returned no parsed foreground structure")
    if not message.parsed.processes:
        raise RuntimeError("No evidence-backed foreground process was identified")
    return message.parsed


def extract_inventory_from_text(
    text: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    extra_instructions: str = "",
) -> InventoryExtraction:
    """Two-pass, schema-first foreground extraction using OpenAI Structured Outputs.

    Pass 1 identifies the actual process hierarchy and study context. Pass 2 extracts
    evidence-backed flows but is constrained to those process IDs. This prevents a
    flow-extraction pass from silently creating extra subprocesses.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("No source text supplied")

    chosen_model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
    structure = identify_foreground_structure(
        text,
        model=chosen_model,
        api_key=api_key,
        extra_instructions=extra_instructions,
    )

    user_prompt = (
        "Extract foreground flows using ONLY the process structure below. "
        "Do not add, split, merge, or rename processes.\n\n"
        f"LOCKED PROCESS STRUCTURE:\n{structure.model_dump_json(indent=2)}\n\n"
    )
    if extra_instructions.strip():
        user_prompt += f"Study-specific instructions:\n{extra_instructions.strip()}\n\n"
    user_prompt += f"SOURCE MATERIAL:\n{text}"

    completion = _client(api_key).beta.chat.completions.parse(
        model=chosen_model,
        messages=[
            {"role": "system", "content": FLOW_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=FlowExtraction,
    )
    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"Model refused flow extraction: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("The model returned no parsed foreground flows")

    warnings = list(dict.fromkeys(structure.assumptions_or_warnings + message.parsed.assumptions_or_warnings))
    return InventoryExtraction(
        process_name=structure.process_name,
        functional_unit=structure.functional_unit,
        source_summary=structure.source_summary,
        study_context=structure.study_context,
        assumptions_or_warnings=warnings,
        processes=structure.processes,
        flows=message.parsed.flows,
    )
