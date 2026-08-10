import json

import pandas as pd

from ai_lca.export import ensure_flow_ids, review_bundle_to_json
from ai_lca.models import ForegroundProcess, InventoryExtraction


def test_review_bundle_preserves_original_ai_proposal_and_reviewed_state():
    original = InventoryExtraction(
        process_name="Synthetic system",
        source_summary="Original proposal",
        processes=[
            ForegroundProcess(process_id="p1", name="Original process"),
            ForegroundProcess(process_id="p2", name="Second process"),
        ],
    )
    reviewed = original.model_copy(
        update={
            "processes": [
                original.processes[0].model_copy(update={"name": "Human-reviewed process"})
            ]
        }
    )
    inventory = pd.DataFrame(
        [{"include": True, "flow_id": 0, "process_id": "p1", "name": "electricity"}]
    )

    payload = json.loads(
        review_bundle_to_json(
            reviewed,
            inventory,
            pd.DataFrame(),
            original_extraction=original,
        )
    )

    assert [p["process_id"] for p in payload["original_ai_extraction"]["processes"]] == ["p1", "p2"]
    assert payload["original_ai_extraction"]["processes"][0]["name"] == "Original process"
    assert [p["process_id"] for p in payload["reviewed_extraction"]["processes"]] == ["p1"]
    assert payload["reviewed_extraction"]["processes"][0]["name"] == "Human-reviewed process"


def test_ensure_flow_ids_fills_missing_and_duplicate_editor_ids():
    df = pd.DataFrame(
        [
            {"flow_id": 0, "name": "existing"},
            {"flow_id": 0, "name": "duplicate"},
            {"flow_id": None, "name": "human-added"},
            {"flow_id": 7, "name": "existing-seven"},
        ]
    )

    result = ensure_flow_ids(df)

    assert result["flow_id"].tolist() == [0, 1, 2, 7]
    assert len(set(result["flow_id"])) == len(result)
