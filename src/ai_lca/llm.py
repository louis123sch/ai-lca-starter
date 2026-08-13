from __future__ import annotations

import base64
import os
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from .documents import VisualAsset, combine_document_evidence
from .models import FlowExtraction, ForegroundInterpretation, ForegroundStructure, InventoryExtraction
from .runtime import build_extraction_provenance
from .structure import lock_foreground_interpretation


STRUCTURE_SYSTEM_PROMPT = """You are assisting with life-cycle inventory (LCI) construction from a supplied paper or technical document.
This first pass is PROCESS INTERPRETATION, not inventory completion.

Your task has two conceptual stages inside one structured response:
A. Identify process-like entities that a reader might plausibly mistake for foreground processes.
B. Classify every such candidate by its role in the actual LCA model. A deterministic downstream step will only promote candidates classified as assessed_product_system or interconnected_foreground_process.

Use exactly these role meanings:
- assessed_product_system: a product system, pathway, configuration, or alternative that is itself assessed/reported by the LCA on the study functional-unit basis.
- interconnected_foreground_process: a separately modelled foreground activity with its own reference product or explicit quantified foreground exchange to another retained activity, or one that the source explicitly identifies as a distinct foreground unit process with activity-owned quantitative LCI rows.
- internal_stage: a unit operation, life-cycle stage, inventory section, or internal bookkeeping stage belonging to a retained product system rather than an independently modelled activity.
- shared_supporting_activity: a foreground-looking service/infrastructure/activity shared by assessed alternatives but not itself an independently assessed product system and not evidenced as a separately interconnected foreground process.
- background_supply: electricity, fuels, materials, transport, chemicals, water, waste treatment or other supplied activities that belong as inventory exchanges/background data unless the source explicitly models them as foreground.
- descriptive_only: technology/process discussion that is not part of the modelled LCA inventory or assessed alternatives.

Rules:
1. Classify from the LCA MODEL, not from descriptive technology-review prose. Goal/scope, system-boundary figures, LCI tables, explicit scenario definitions and quantified foreground exchanges outweigh generic process descriptions.
2. Prefer the smallest foreground graph justified by source evidence. A process-flow diagram can show several unit operations while the LCA still models one product system.
3. A table section, component list, stack/BoP section, construction/manufacturing/operation/maintenance/transport/replacement/end-of-life section, or shared functional-unit basis does not by itself establish a separate foreground process.
4. Preserve genuinely separate assessed alternatives/configurations even when they share technologies or upstream inputs.
5. A candidate should be interconnected_foreground_process only when the source supports it as a distinct foreground activity. For an activity-owned LCI block to establish this role, the source must explicitly identify the candidate as a distinct foreground unit process or activity, assign quantitative inventory rows (inputs and/or outputs) to it, and distinguish that ownership from the aggregate inventory of the containing product system. Merely partitioning one aggregate inventory by life-cycle stage is not enough; neither is a component or equipment group, process-diagram box, background database activity, or impact/result column, even when quantities are shown. A quantified named intermediate passed between activity-owned blocks is strong evidence of separate foreground ownership. A terminal activity need not feed a downstream foreground process, but it qualifies from its own inventory block only when the source explicitly models it quantitatively as a foreground activity. Do not manufacture a reference product to justify promotion. When separately owned unit-process blocks together describe one foreground chain, do not additionally classify the chain's title or functional-unit product label as assessed_product_system; it is a container unless the source presents it as a separately compared alternative or configuration with a non-duplicative inventory block.
6. Shared support activities should remain shared_supporting_activity unless the source explicitly gives them the independent foreground-activity status described above.
7. Do not turn background supplies such as electricity, natural gas, water, steel, transport, or chemicals into foreground processes unless the source explicitly models them as such.
8. Do not infer voltage level, production technology, market dataset, geography code, reference product, unit, or other ecoinvent detail from generic wording.
9. Extract the primary/reference operational geography when explicit. If additional country or regional scenarios are assessed, record them in additional_geographies rather than replacing the reference geography.
10. If geography cannot be supported, use not_identified. Never fill it from a user preference list.
11. Use concise source-derived candidate identifiers. If the study uses stable acronyms/short labels, use them consistently. Do not embed capacities, functional-unit text, or explanatory parentheticals in IDs/names.
12. Include short evidence snippets and a brief evidence-grounded rationale for every candidate. Populate document/page/paragraph/table only from explicit provenance markers.
13. Record ambiguity or possible double counting in assumptions_or_warnings.
14. This is a proposal for human review, not an approved LCA model.
"""


FLOW_SYSTEM_PROMPT = """You are extracting foreground LCI flows from a supplied paper or technical document.
A first-pass foreground process structure has already been identified and locked. You MUST fill that structure; you may not redesign it.

Rules:
1. Every flow must be attached to one supplied process ID. Never create a new process ID or implicit subprocess.
2. Extract only flows that the source indicates are part of the MODELED foreground LCI. Do not extract catalysts, solvents, materials, operating conditions, or technology options merely because they appear in descriptive prose. Prefer explicit LCI tables and inventory-analysis sections.
3. Explicit component lists, BoP/stack tables, component tables, numbered inventory lists, supplementary tables, and transcribed visual tables are valid source evidence. Attach their listed components as foreground input flows to the appropriate locked process. Do not promote components into subprocesses.
4. Preserve printed component/flow names and word order in `name` rather than canonicalising them into ecoinvent-style wording. If multiple source locations use variants, prefer one source-supported name and record ambiguity in notes/warnings. Use `background_process_hint` (rule 16) for an explicit technical identifier; never fold that wording into `name` itself.
5. Never invent an amount, unit, material, process, functional unit, voltage level, market type, production route, document, page, table, or paragraph.
6. If the paper says only 'electricity', extract 'electricity'. Do not turn it into 'medium-voltage electricity', 'market for electricity', or another background dataset name unless the source states that detail.
7. If a flow is mentioned but no amount is given, amount must be null.
8. Preserve the stated basis (per kg product, per year, per plant, etc.). Do not silently convert bases.
9. Keep outputs/co-products distinct from inputs and elementary emissions.
10. Do not duplicate the same flow merely because it appears in prose and a table; use the clearest supporting evidence and warn about conflicting values.
11. Internal stages/component groups do not require new process IDs. If the locked structure has one process, attach source-supported flows from its internal inventory sections to that process.
12. linked_process_id is ONLY for an explicit foreground-to-foreground exchange between two IDs already present in the locked structure. Set it null for ordinary background inputs, emissions, and outputs. Never infer a link merely because two processes use the same material or service.
13. Treat [VISUAL EVIDENCE: ...] blocks exactly like source evidence: use only visibly transcribed content and preserve provenance.
14. Include a short evidence_text snippet for every flow. Populate document/page/paragraph/table only from explicit source markers.
15. Record ambiguity, missing denominators, allocation issues, unclear units, or possible double counting in assumptions_or_warnings.
16. A supplied document may include a separate technical process-database export alongside the main paper text — for example an LCA-software "upstream tree"/contribution list, process network table, or similar SimaPro/OpenLCA/GaBi-style output — giving a verbatim background-database identifier for the same material, energy, transport or service, typically formatted like 'activity name | reference product | system model, type - location'. When such an identifier is explicitly printed and clearly corresponds to the flow you are extracting, copy it EXACTLY into background_process_hint. Never construct, canonicalise, paraphrase, translate units within, or guess this identifier. Leave background_process_hint null when no explicit technical identifier is printed for that exact flow, or when the correspondence between the plain-language flow and the technical export entry is ambiguous; note real ambiguity in assumptions_or_warnings instead of guessing.

Protected extraction invariants:
- explicit component lists are valid inventory evidence;
- listed components remain foreground input flows, not background subprocesses;
- do not reclassify such tabulated components as background subprocesses;
- it is better to retain an explicit listed inventory item with amount=null than to invent or omit a quantity.

EXHAUSTIVE INVENTORY-LIST COMPLETENESS CHECK:
- Before finalising the flow list, scan the entire supplied source again specifically for explicit inventory lists: table rows, numbered or bulleted component lists, BoP/stack component groups, supplementary inventory tables, and transcribed visual tables.
- For each locked process, compare every distinct listed inventory item against the flows you are about to return. If a listed modeled item is missing, add it before finalising.
- This completeness check applies equally to the main paper, supplementary material, and transcribed visual evidence. Do not stop after extracting a representative subset from a long list.
- When an explicit list contains many items, extract EVERY distinct source-supported item belonging to the modeled foreground inventory, even if no amount is stated. Use amount=null when the source gives no amount.
- Preserve each listed item's printed name and word order. Do not replace it with a synonym or a more convenient canonical name.
- Do not use this completeness check to add descriptive prose, background dataset detail, or unsupported items. It is only an exhaustive audit of explicit source-supported inventory lists against the already locked foreground processes.
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
            warnings.extend(f"{asset.document}/{asset.asset_id}: {w}" for w in item.warnings)
            if not item.evidence_text.strip():
                continue
            if not item.relevant_to_lca:
                warnings.append(
                    f"{asset.document}/{asset.asset_id}: visual evidence marked not relevant but included for LCA extraction"
                )
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
    """Classify process-like source entities, then deterministically lock the foreground graph."""
    text = (text or "").strip()
    if not text:
        raise ValueError("No source text supplied")

    chosen_model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
    user_prompt = (
        "Identify and classify process-like entities, study context, and source-supported reference products/units. "
        "Do not directly decide downstream Brightway mappings.\n\n"
    )
    if extra_instructions.strip():
        user_prompt += f"Study-specific instructions:\n{extra_instructions.strip()}\n\n"
    user_prompt += f"SOURCE MATERIAL:\n{text}"

    completion = _client(api_key).beta.chat.completions.parse(
        model=chosen_model,
        messages=[
            {"role": "system", "content": STRUCTURE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=ForegroundInterpretation,
    )
    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"Model refused structure extraction: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("The model returned no parsed foreground interpretation")
    if not message.parsed.candidates:
        raise RuntimeError("No source-supported process-like candidate was identified")
    return lock_foreground_interpretation(message.parsed)


def extract_inventory_from_text(
    text: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    extra_instructions: str = "",
    source_mode: Literal["text", "documents"] = "text",
) -> InventoryExtraction:
    """Role-classified, two-pass foreground extraction using OpenAI Structured Outputs."""
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
        "Extract foreground flows using ONLY the locked process structure below. "
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

    warnings = list(
        dict.fromkeys(
            structure.assumptions_or_warnings + message.parsed.assumptions_or_warnings
        )
    )
    return InventoryExtraction(
        process_name=structure.process_name,
        functional_unit=structure.functional_unit,
        source_summary=structure.source_summary,
        study_context=structure.study_context,
        assumptions_or_warnings=warnings,
        candidate_activities=structure.candidate_activities,
        processes=structure.processes,
        flows=message.parsed.flows,
        provenance=build_extraction_provenance(model=chosen_model, source_mode=source_mode),
    )


def extract_inventory_from_documents(
    documents: list[tuple[str, bytes]],
    *,
    model: str | None = None,
    api_key: str | None = None,
    extra_instructions: str = "",
    max_visual_assets: int = 24,
) -> InventoryExtraction:
    """Multimodal path: native text + visual transcription -> role classification -> locked flows."""
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
        source_mode="documents",
    )
    warnings = list(
        dict.fromkeys(
            extraction.assumptions_or_warnings + ingestion_warnings + vision_warnings
        )
    )
    return extraction.model_copy(update={"assumptions_or_warnings": warnings})