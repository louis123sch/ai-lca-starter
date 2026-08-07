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
                "source_label": flow.source_label,
                "item_type": flow.item_type,
                "parent_process": flow.parent_process or "",
                "amount": flow.amount,
                "unit": flow.unit,
                "direction": flow.direction,
                "basis": flow.basis,
                "search_worthy": flow.search_worthy,
                "ecoinvent_search_term": flow.ecoinvent_search_term,
                "ecoinvent_activity_type_hint": flow.ecoinvent_activity_type_hint or "unknown",
                "geography_hint": flow.geography_hint,
                "supplier_technology_hint": flow.supplier_technology_hint,
                "interpretation_reason": flow.interpretation_reason,
                "source_document": flow.evidence.source_document,
                "page": flow.evidence.page,
                "table": flow.evidence.table,
                "evidence_text": flow.evidence.evidence_text,
                "notes": flow.notes,
            }
        )
    return pd.DataFrame(rows)


def searchable_exchanges(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministically select only approved technosphere concepts for ecoinvent search."""
    if df.empty:
        return df.copy()
    return df[
        (df["include"] == True)  # noqa: E712
        & (df["item_type"] == "technosphere_flow")
        & (df["search_worthy"] == True)  # noqa: E712
        & df["ecoinvent_search_term"].fillna("").astype(str).str.strip().ne("")
    ].copy()


def dataframe_to_json(df: pd.DataFrame) -> str:
    return json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False, default=str)
