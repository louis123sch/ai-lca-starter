from __future__ import annotations

from .documents import combine_document_evidence
from .flow_audit import audit_missing_flows, merge_missing_flows
from .llm import _client, augment_text_with_visual_evidence, extract_inventory_from_text
from .models import FlowExtraction, ForegroundStructure, InventoryExtraction


def extract_inventory_from_documents_audited(
    documents: list[tuple[str, bytes]],
    *,
    model: str | None = None,
    api_key: str | None = None,
    extra_instructions: str = "",
    max_visual_assets: int = 24,
) -> InventoryExtraction:
    """Document extraction with an independent conservative missing-flow audit."""
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
    structure = ForegroundStructure(
        process_name=extraction.process_name,
        functional_unit=extraction.functional_unit,
        source_summary=extraction.source_summary,
        study_context=extraction.study_context,
        assumptions_or_warnings=extraction.assumptions_or_warnings,
        candidate_activities=extraction.candidate_activities,
        processes=extraction.processes,
    )
    initial = FlowExtraction(
        assumptions_or_warnings=extraction.assumptions_or_warnings,
        flows=extraction.flows,
    )
    chosen_model = model or (extraction.provenance.model if extraction.provenance else "gpt-5-mini")
    audited = audit_missing_flows(
        enriched_text,
        structure,
        initial,
        client=_client(api_key),
        model=chosen_model,
    )
    flows = merge_missing_flows(
        extraction.flows,
        audited.flows,
        source_text=enriched_text,
        allowed_process_ids={process.process_id for process in extraction.processes},
    )
    warnings = list(
        dict.fromkeys(
            extraction.assumptions_or_warnings
            + audited.assumptions_or_warnings
            + ingestion_warnings
            + vision_warnings
        )
    )
    return extraction.model_copy(
        update={
            "flows": flows,
            "assumptions_or_warnings": warnings,
        }
    )
