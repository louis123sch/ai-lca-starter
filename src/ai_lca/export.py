from __future__ import annotations

import json
import pandas as pd
from .models import InventoryExtraction


def extraction_to_dataframe(extraction: InventoryExtraction) -> pd.DataFrame:
    rows = []
    for i, flow in enumerate(extraction.flows):
        rows.append(
            {
                "include": True,
                "flow_id": i,
                "name": flow.name,
                "amount": flow.amount,
                "unit": flow.unit,
                "direction": flow.direction,
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
