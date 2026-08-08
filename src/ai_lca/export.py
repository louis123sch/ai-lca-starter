from __future__ import annotations

import json
import pandas as pd
from .models import InventoryExtraction, ProcessMap


def process_map_to_dataframe(process_map: ProcessMap) -> pd.DataFrame:
    rows = []
    for group in process_map.technology_groups:
        for process in group.processes:
            first_evidence = process.evidence[0] if process.evidence else None
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
                    "page": first_evidence.page if first_evidence else None,
                    "table": first_evidence.table if first_evidence else None,
                    "evidence_text": first_evidence.evidence_text if first_evidence else None,
                }
            )
    return pd.DataFrame(rows)


def extraction_to_dataframe(extraction: InventoryExtraction) -> pd.DataFrame:
    rows = []
    for i, flow in enumerate(extraction.flows):
        eligible = flow.direction == "input" and flow.amount is not None
        rows.append(
            {
                "include": eligible,
                "background_match_eligible": eligible,
                "flow_id": i,
                "technology_group": flow.technology_group,
                "process_id": flow.process_id,
                "process_name": flow.process_name,
                "name": flow.name,
                "amount": flow.amount,
                "unit": flow.unit,
                "direction": flow.direction,
                "flow_kind": flow.flow_kind,
                "operation_context": flow.operation_context,
                "component_or_stage": flow.component_or_stage,
                "basis": flow.basis,
                "page": flow.evidence.page,
                "table": flow.evidence.table,
                "evidence_text": flow.evidence.evidence_text,
                "notes": flow.notes,
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_json(df: pd.DataFrame) -> str:
    return json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False, default=str)
