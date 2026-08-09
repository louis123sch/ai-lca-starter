from __future__ import annotations

import base64
import os
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from .documents import VisualAsset, combine_document_evidence
from .models import FlowExtraction, ForegroundStructure, InventoryExtraction


STRUCTURE_SYSTEM_PROMPT = """You are assisting with life-cycle inventory (LCI) construction from a supplied paper or technical document.
This first pass is PROCESS INTERPRETATION, not inventory completion. Your job is to identify the foreground process structure the source actually represents.

Rules:
1. Identify only foreground processes/subprocesses supported by the supplied source. Do not invent a process because an input exists.
2. Distinguish the LCA MODEL from descriptive technology-review prose. Identify product systems/configurations actually assessed in the LCA, not every technology, catalyst, solvent, reactor option, or unit operation discussed in the review.
3. Prefer the smallest process hierarchy justified by the modeled LCI. A process-flow diagram can show internal unit operations without those being separate foreground LCA processes. Treat LCI tables, goal/scope, system-boundary figures, and explicit model-configuration statements as stronger evidence than generic process-description prose.
4. Create a subprocess only when the LCA source itself models it as a separately reusable/interconnected foreground activity with its own reference product or an explicitly quantified exchange to another foreground activity. A separately tabulated inventory category alone is not enough.
5. Do NOT split a product system merely because its inventory is reported in sections such as stack, balance of plant (BoP), construction/capital, manufacturing, operation, maintenance, transport, replacement, or end-of-life. When those sections are component/life-cycle inventories that are aggregated to assess one technology or product system, attach their flows to that product system unless the source explicitly models exchanges between separate foreground activities.
6. Conversely, preserve genuinely separate modeled configurations or pathways when the study reports them as distinct alternatives, even if they share technology or upstream inputs. Do not collapse distinct assessed configurations into a parent technology solely because they are related.
7. Do not turn background supplies such as electricity, natural gas, water, steel, transport, or chemicals into foreground subprocesses unless the source explicitly models them as such.
8. Do not infer voltage level, production technology, market dataset, geography code, or other ecoinvent detail from generic flow names.
9. Extract the primary/reference operational geography when explicit. If the paper additionally assesses country or regional scenarios, record them in additional_geographies rather than replacing the reference geography.
10. If geography cannot be supported, use not_identified. Never fill it from a user preference list.
11. Before finalising the hierarchy, perform a boundary check: for every proposed child process ask whether the paper quantifies an exchange/reference flow connecting it to another foreground process. If not, and it is only an inventory grouping or life-cycle stage of the assessed product system, merge it into the parent product system. Also check that distinct assessed alternatives have not been accidentally nested under one another.
12. Include short evidence snippets for every process and for study context where possible. Populate document from [DOCUMENT: ...], and page/paragraph/table only from explicit source markers.
13. Record ambiguity and possible double counting in assumptions_or_warnings.
14. This is a proposal for human review, not an approved LCA model.
15. Naming and identifiers: emit concise, machine-friendly process identifiers and short display names. If the study uses acronyms or short labels (e.g. technology acronyms), derive a short lowercase id from them (for example, when the paper repeatedly uses an acronym like "AEC", use "aec" as the id). Avoid embedding capacity, basis text, or parenthetical qualifiers (for example "1 MW" or "per 1 kg H2") inside the primary process name; record such metadata (capacity, basis) in the functional_unit or study_context fields. Use these short ids consistently and include them in every evidence snippet so downstream extraction and matching can rely on stable keys.
"""


FLOW_SYSTEM_PROMPT = """You are extracting foreground LCI flows from a supplied paper or technical document.
A first-pass foreground process structure has already been identified. You MUST fill that structure; you may not redesign it.

Rules:
1. Every flow must be attached to one of the supplied process IDs. Never create a new process ID or implicit subprocess. Use the process ids exactly as provided in the LOCKED PROCESS STRUCTURE (they are expected to be short, stable ids derived from the study's own labels).
2. Extract only flows that the source indicates are part of the MODELED foreground LCI. Do not extract catalysts, solvents, materials, operating conditions, or technology options merely because they appear in review/process-description prose. Prefer explicit LCI tables and inventory-analysis sections.
2a. When the source presents explicit component lists, BoP/stack tables, or "Table of components" for a modelled product system, treat each listed component as a foreground input flow to that assessed product system (or to the specified subcomponent if the locked structure includes a distinct subprocess for stack/BoP). Extract the component names and any provided amounts/units exactly as given; do not reclassify such tabulated components as background subprocesses.

IMPORTANT CLARIFICATION (applies to main text, supplement, and visual evidence):
- When component lists or tabulated components appear anywhere in the provided SOURCE MATERIAL (including main paper tables, supplementary machine-readable tables, or transcribed visual evidence blocks), treat them as explicit inventory items to be attached as foreground inputs to the modelled product system. Do not omit tabulated components simply because they appear in a supplement or a figure rather than the main prose.
- Preserve the component names and any available amounts/units exactly as printed; if no amount is given, set amount=null and include an evidence snippet pointing to the table/figure.

Protected invariants (must be respected verbatim in extractions):
- explicit component lists
- foreground input flow
- do not reclassify such tabulated components as background subprocesses

3. Never invent an amount, unit, material, process, functional unit, voltage level, market type, production route, document, page, table, or paragraph.
4. If the paper says only 'electricity', extract 'electricity'. Do NOT turn it into 'medium-voltage electricity', 'market for electricity', or another ecoinvent-style dataset name unless the source states that detail.
5. If a flow is mentioned but no amount is given, amount must be null.
6. Preserve the stated basis (per kg product, per year, per plant, etc.). Do not silently convert bases.
7. Keep outputs/co-products distinct from inputs and elementary emissions.
8. Do not duplicate the same flow merely because it appears in prose and a table; use the clearest supporting evidence and warn about conflicting values.
9. Component groups and life-cycle stages (for example stack, BoP, construction, manufacturing, operation, maintenance, transport, replacement, and end-of-life) do not require new process IDs. If the locked structure represents the assessed product system as one process, attach source-supported flows from those inventory sections to that process while preserving their stated basis and evidence.
10. Treat [VISUAL EVIDENCE: ...] blocks exactly like source evidence: use only what is visibly transcribed there and preserve the asset/document provenance in evidence_text/notes. Do not infer values that the visual-evidence stage did not transcribe.
11. Include a short evidence_text snippet for every flow. Populate document from [DOCUMENT: ...], and page/paragraph/table only from explicit source markers.
12. Record ambiguity, missing denominators, allocation issues, unclear units, or possible double counting in assumptions_or_warnings.

SUPPLEMENT AND LIST HANDLING (clarification to reduce missed tabulated items):
- If the SOURCE MATERIAL indicates that component names are presented in tables, numbered lists, or labelled groups (for example "BoP components", "Stack components", "Table X - ... components" or similarly titled lists), explicitly parse those lists as inventory entries. This includes numbered or bullet lists in the main text, labelled groups inside figure/table captions, and machine-readable supplementary blocks provided alongside the paper.
- For each listed component, create a foreground input flow attached to the appropriate process id in the LOCKED PROCESS STRUCTURE. If the locked structure distinguishes stack vs BoP subprocesses, attach to the specified subprocess; otherwise attach to the parent process. Preserve the component name exactly as printed.
- If the supplementary material contains amounts or units corresponding to a listed component, extract and attach them. If no numeric amount/unit is present in any provided material, set amount=null and include an evidence snippet pointing to the source table/paragraph/figure.
- Do not discard items because they appear only in supplementary text, figure captions, or transcribed visual evidence; they are valid sources for explicit component inventories.

MANDATORY LIST EXTRACTION (added emphasis):
- When numbered, bulleted, or table-form component lists appear anywhere in the provided source (main text, supplement, or transcribed visuals), you MUST extract every distinct listed item as a foreground input flow. Extract verbatim names and attach an evidence snippet for each occurrence. If the same exact component name appears in multiple places, you may merge into one flow but include all supporting evidence locations in the evidence text or warnings. It is better to include a listed component with amount=null than to omit it entirely.
"""


VISUAL_SYSTEM_PROMPT = """You are a document-evidence transcription stage for life-cycle assessment documents.
Your task is to read supplied figures/pages and transcribe visible evidence faithfully BEFORE any LCA interpretation occurs.

Rules:
1. Extract visible tables, labels, quantities, units, component/material names, process/configuration names, system-boundary labels, and other evidence that could matter to an LCI reconstruction.
2. Do not decide the foreground process hierarchy and do not map anything to ecoinvent. Transcribe what is visibly present.
3. Never invent obscured/missing values. If a row/value is unreadable, say so in warnings instead of guessing.
4. Preserve row associations: a quantity must remain attached to the material/component it visibly belongs to.
5. Preserve ecoinvent dataset names if they are visibly printed, but label them as background mapping text rather than silently converting them into foreground names.
6. Mark decorative photographs/logos/irrelevant figures as not relevant.
7. Return exactly one result for every labelled asset supplied by the user.
"""


class VisualEvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    relevant_to_lca: bool
    evidence_type: Literal["table", "diagram", "chart", "page", "figure", "other"] = "figure"
    evidence_text: str = Field(description="Faithful visible transcription; use compact table-like lines where appropriate.")
    warnings: list[str] = Field(default_factory=list)


class VisualEvidenceBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[VisualEvidenceItem] = Field(default_factory=list)


def _client(api_key: str | None = None) -> OpenAI:
    return OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))


def _visual_data_url(asset: VisualAsset) -> str:
    encoded = base64.b64encode(asset.data).decode("ascii")
    return f"data:{asset.mime_type};base64,{encoded}"


def transcribe_visual_evidence(
    assets: list[VisualAsset],
    *,
    model: str | None = None,
    api_key: str | None = None,
    batch_size: int = 4,
) -> tuple[str, list[str]]:
    """Convert visual document evidence into provenance-tagged machine-readable text."""
    if not assets:
        return "", []
    chosen_model = model or os.getenv("OPENAI_VISION_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5-mini")
    blocks: list[str] = []
    warnings: list[str] = []
    client = _client(api_key)

    for start in range(0, len(assets), max(1, batch_size)):
        batch = assets[start : start + max(1, batch_size)]
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "Transcribe the labelled document assets below. Return one item per asset_id. "
                    "Nearby context is supplied only to orient the image; visible image content is the authority."
                ),
            }
        ]
        for asset in batch:
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"ASSET_ID: {asset.asset_id}\nDOCUMENT: {asset.document}\n"
                        f"SOURCE_TYPE: {asset.source_type}\nNEARBY_CONTEXT: {asset.context or 'None'}"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _visual_data_url(asset), "detail": "high"},
                }
            )

        completion = client.beta.chat.completions.parse(
            model=chosen_model,
            messages=[
                {"role": "system", "content": VISUAL_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format=VisualEvidenceBatch,
        )
        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            warnings.append(f"Vision model refused a visual evidence batch: {message.refusal}")
            continue
        if message.parsed is None:
            warnings.append("Vision model returned no parsed visual evidence for a batch.")
            continue
        by_id = {item.asset_id: item for item in message.parsed.items}
        for asset in batch:
            item = by_id.get(asset.asset_id)
            if item is None:
                warnings.append(f"No visual transcription returned for {asset.document}/{asset.asset_id}.")
                continue
            # surface any warnings from the visual stage
            warnings.extend(f"{asset.document}/{asset.asset_id}: {w}" for w in item.warnings)
            # NOTE: include any non-empty transcriptions even if the vision model flagged them as not relevant.
            # Previously we skipped items marked not relevant; that could drop legitimate tabulated component lists
            # when the vision model mis-labelled relevance. To be conservative and recover explicit component
            # inventories, include any non-empty evidence_text and log a warning when relevance is False.
            if not item.evidence_text.strip():
                continue
            if not item.relevant_to_lca:
                warnings.append(f"{asset.document}/{asset.asset_id}: visual evidence marked not relevant but included for LCA extraction")
            blocks.append(
                f"[VISUAL EVIDENCE: {asset.document} | {asset.asset_id} | {item.evidence_type}]\n"
                f"Nearby context: {asset.context or 'None'}\n"
                f"{item.evidence_text.strip()}"
            )
    return "\n\n".join(blocks), warnings


def augment_text_with_visual_evidence(
    text: str,
    assets: list[VisualAsset],
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[str, list[str]]:
    visual_text, warnings = transcribe_visual_evidence(assets, model=model, api_key=api_key)
    if not visual_text:
        return text, warnings
    return f"{text.rstrip()}\n\n[BEGIN TRANSCRIBED VISUAL EVIDENCE]\n{visual_text}\n[END TRANSCRIBED VISUAL EVIDENCE]", warnings


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
    """Two-pass, schema-first foreground extraction using OpenAI Structured Outputs."""
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


def extract_inventory_from_documents(
    documents: list[tuple[str, bytes]],
    *,
    model: str | None = None,
    api_key: str | None = None,
    extra_instructions: str = "",
    max_visual_assets: int = 24,
) -> InventoryExtraction:
    """Multimodal document path: native text + visual evidence -> existing two-pass LCA reasoning."""
    text, assets, ingestion_warnings = combine_document_evidence(
        documents,
        max_visual_assets=max_visual_assets,
    )
    enriched_text, vision_warnings = augment_text_with_visual_evidence(
        text,
        assets,
        model=model,
        api_key=api_key,
    )
    extraction = extract_inventory_from_text(
        enriched_text,
        model=model,
        api_key=api_key,
        extra_instructions=extra_instructions,
    )
    warnings = list(
        dict.fromkeys(
            extraction.assumptions_or_warnings + ingestion_warnings + vision_warnings
        )
    )
    return extraction.model_copy(update={"assumptions_or_warnings": warnings})