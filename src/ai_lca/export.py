from __future__ import annotations

import json
import pandas as pd
from .models import InventoryExtraction, ProcessMap


def _evidence_summary(evidence_list) -> tuple[str, str, str, str]:
    documents = []
    pages = []
    tables = []
    snippets = []
    for evidence in evidence_list or []:
        if evidence.source_document and evidence.source_document not in documents:
            documents.append(evidence.source_document)
        if evidence.page is not None:
            page_text = str(evidence.page)
            if page_text not in pages:
                pages.append(page_text)
        if evidence.table and evidence.table not in tables:
            tables.append(evidence.table)
        if evidence.evidence_text and evidence.evidence_text not in snippets:
            snippets.append(evidence.evidence_text)
    return (
        "; ".join(documents),
        "; ".join(pages),
        "; ".join(tables),
        " || ".join(snippets),
    )


def process_map_to_dataframe(process_map: ProcessMap) -> pd.DataFrame:
    rows = []
    for group in process_map.technology_groups:
        for process in group.processes:
            documents, pages, tables, snippets = _evidence_summary(process.evidence)
            rows.append(
                {
                    "include": True,
                    "process_id": process.process_id,
                    "technology_group": group.name,
                    "process_name": process.name,
                    "geographic_context": process.geographic_context,
                    "temporal_context": process.temporal_context,
                    "evidence_type": process.evidence_type,
                    "confidence": process.confidence,
                    "reason_for_separate_process": process.reason_for_separate_process,
                    "operations_not_separate_processes": "; ".join(op.name for op in process.operations),
                    "source_documents": documents,
                    "pages": pages,
                    "tables": tables,
                    "evidence_text": snippets,
                }
            )
    return pd.DataFrame(rows)


def extraction_to_dataframe(extraction: InventoryExtraction) -> pd.DataFrame:
    rows = []
    for i, flow in enumerate(extraction.flows):
        technosphere_eligible = (
            flow.exchange_type == "technosphere"
            and flow.direction == "input"
            and flow.amount is not None
        )
        biosphere_eligible = flow.exchange_type == "biosphere" and flow.amount is not None
        eligible = technosphere_eligible or biosphere_eligible
        match_target = (
            "technosphere" if technosphere_eligible else "biosphere" if biosphere_eligible else "none"
        )

        documents, pages, tables, snippets = _evidence_summary(flow.evidence)
        mapping_documents, mapping_pages, mapping_tables, mapping_snippets = _evidence_summary(
            flow.background_mapping_evidence
        )
        rows.append(
            {
                "include": eligible,
                "background_match_eligible": eligible,
                "match_target": match_target,
                "flow_id": i,
                "technology_group": flow.technology_group,
                "process_id": flow.process_id,
                "process_name": flow.process_name,
                "name": flow.name,
                "source_label": flow.source_label,
                "amount": flow.amount,
                "unit": flow.unit,
                "direction": flow.direction,
                "exchange_type": flow.exchange_type,
                "flow_kind": flow.flow_kind,
                "operation_context": flow.operation_context,
                "component_or_stage": flow.component_or_stage,
                "basis": flow.basis,
                "exchange_geography_hint": flow.exchange_geography_hint,
                "supplier_technology_hint": flow.supplier_technology_hint,
                "interpretation_reason": flow.interpretation_reason,
                "ecoinvent_search_term": flow.ecoinvent_search_term or (flow.name if flow.exchange_type == "technosphere" else None),
                "ecoinvent_activity_hint": flow.ecoinvent_activity_hint,
                "ecoinvent_location_hint": flow.ecoinvent_location_hint,
                "background_mapping_relation": flow.background_mapping_relation,
                "background_mapping_rationale": flow.background_mapping_rationale,
                "biosphere_search_term": flow.biosphere_search_term or (flow.name if flow.exchange_type == "biosphere" else None),
                "biosphere_compartment_hint": flow.biosphere_compartment_hint,
                "source_documents": documents,
                "pages": pages,
                "tables": tables,
                "evidence_text": snippets,
                "mapping_source_documents": mapping_documents,
                "mapping_pages": mapping_pages,
                "mapping_tables": mapping_tables,
                "mapping_evidence_text": mapping_snippets,
                "notes": flow.notes,
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_json(df: pd.DataFrame) -> str:
    return json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False, default=str)
