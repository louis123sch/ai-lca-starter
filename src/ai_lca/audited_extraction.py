from __future__ import annotations

from .documents import combine_document_evidence
from .flow_audit import audit_missing_flows, merge_missing_flows
from .llm import _client, augment_text_with_visual_evidence, extract_inventory_from_text
from .models import FlowExtraction, ForegroundStructure, InventoryExtraction


def _structure_for_process(structure: ForegroundStructure, process_id: str) -> ForegroundStructure:
    """Create a single-process audit view without changing the locked foreground graph."""
    process = next(process for process in structure.processes if process.process_id == process_id)
    relevant_candidates = [
        candidate
        for candidate in structure.candidate_activities
        if candidate.candidate_id == process_id or candidate.parent_candidate_id == process_id
    ]
    return structure.model_copy(
        update={
            "processes": [process],
            "candidate_activities": relevant_candidates,
        }
    )


def extract_inventory_from_documents_audited(
    documents: list[tuple[str, bytes]],
    *,
    model: str | None = None,
    api_key: str | None = None,
    extra_instructions: str = "",
    max_visual_assets: int = 24,
) -> InventoryExtraction:
    """Document extraction with independent conservative per-process missing-flow audits."""
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
    chosen_model = model or (extraction.provenance.model if extraction.provenance else "gpt-5-mini")
    client = _client(api_key)
    flows = list(extraction.flows)
    audit_warnings: list[str] = []

    # Audit one locked foreground process at a time. Long multi-process inventories can cause
    # a single completion to return a representative subset even when all source evidence is
    # available. Partitioning by the already-locked process boundary reduces completion load
    # without changing process interpretation, evidence rules, or merge acceptance criteria.
    for process in extraction.processes:
        process_structure = _structure_for_process(structure, process.process_id)
        process_initial = FlowExtraction(
            assumptions_or_warnings=extraction.assumptions_or_warnings,
            flows=[flow for flow in flows if flow.process_id == process.process_id],
        )
        audited = audit_missing_flows(
            enriched_text,
            process_structure,
            process_initial,
            client=client,
            model=chosen_model,
        )
        flows = merge_missing_flows(
            flows,
            audited.flows,
            source_text=enriched_text,
            allowed_process_ids={process.process_id},
        )
        audit_warnings.extend(audited.assumptions_or_warnings)

    warnings = list(
        dict.fromkeys(
            extraction.assumptions_or_warnings
            + audit_warnings
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
