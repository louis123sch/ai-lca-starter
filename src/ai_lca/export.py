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
                "process_id": process.process_id,
                "process": process.name,
                "parent": process.parent_process_id or "",
                "stage": process.stage,
                "evidence": evidence,
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_json(df: pd.DataFrame) -> str:
    return json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False, default=str)
