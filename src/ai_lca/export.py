from __future__ import annotations

import json

import pandas as pd

from .models import InventoryExtraction


def extraction_to_dataframe(extraction: InventoryExtraction) -> pd.DataFrame:
    rows = []
    process_names = {p.process_id: p.name for p in extraction.processes}
    for i, flow in enumerate(extraction.flows):
        rows.append(
            {
                "include": True,
                "flow_id": i,
                "process_id": flow.process_id,
                "process_name": process_names.get(flow.process_id, ""),
                "name": flow.name,
                "amount": flow.amount,
                "unit": flow.unit,
                "direction": flow.direction,
                "linked_process_id": flow.linked_process_id,
                "component_or_stage": flow.component_or_stage,
                "basis": flow.basis,
                "document": flow.evidence.document,
                "page": flow.evidence.page,
                "paragraph": flow.evidence.paragraph,
                "table": flow.evidence.table,
                "evidence_text": flow.evidence.evidence_text,
                "notes": flow.notes,
            }
        )
    return pd.DataFrame(rows)


def process_structure_to_dataframe(extraction: InventoryExtraction) -> pd.DataFrame:
    rows = []
    for process in extraction.processes:
        evidence = process.evidence[0].evidence_text if process.evidence else ""
        rows.append(
            {
                "include": True,
                "process_id": process.process_id,
                "process": process.name,
                "merge_into": "",
                "parent": process.parent_process_id or "",
                "role": process.role,
                "reference_product": process.reference_product or "",
                "reference_unit": process.reference_unit or "",
                "classification_rationale": process.classification_rationale or "",
                "stage": process.stage,
                "evidence": evidence,
            }
        )
    return pd.DataFrame(rows)


def candidate_structure_to_dataframe(extraction: InventoryExtraction) -> pd.DataFrame:
    rows = []
    locked = {process.process_id for process in extraction.processes}
    for candidate in extraction.candidate_activities:
        evidence = candidate.evidence[0].evidence_text if candidate.evidence else ""
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "name": candidate.name,
                "role": candidate.role,
                "locked_as_process": candidate.candidate_id in locked,
                "parent_candidate_id": candidate.parent_candidate_id or "",
                "reference_product": candidate.reference_product or "",
                "reference_unit": candidate.reference_unit or "",
                "rationale": candidate.rationale,
                "evidence": evidence,
            }
        )
    return pd.DataFrame(rows)


def _records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def dataframe_to_json(df: pd.DataFrame) -> str:
    return json.dumps(_records(df), indent=2, ensure_ascii=False)


def review_bundle_to_json(
    extraction: InventoryExtraction,
    inventory_df: pd.DataFrame,
    mapping_df: pd.DataFrame | None = None,
) -> str:
    """Export source-grounded extraction, human-reviewed rows and mappings together."""
    payload = {
        "extraction": extraction.model_dump(mode="json"),
        "reviewed_inventory": _records(inventory_df),
        "selected_mappings": _records(mapping_df),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
