import json

import pandas as pd

from ai_lca.export import ensure_flow_ids, normalize_inventory_review, review_bundle_to_json
from ai_lca.models import (
    ForegroundProcess,
    InventoryExtraction,
    InventoryFlow,
    SourceEvidence,
)


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


def test_normalize_inventory_review_marks_edits_and_user_added_rows():
    original = InventoryExtraction(
        process_name="Synthetic system",
        source_summary="Source-grounded proposal",
        processes=[
            ForegroundProcess(process_id="p1", name="Main process"),
            ForegroundProcess(process_id="p2", name="Second process"),
        ],
        flows=[
            InventoryFlow(
                process_id="p1",
                name="electricity",
                amount=4.0,
                unit="kWh",
                direction="input",
                evidence=SourceEvidence(evidence_text="4 kWh electricity"),
            ),
            InventoryFlow(
                process_id="p2",
                name="water",
                amount=2.0,
                unit="kg",
                direction="input",
                evidence=SourceEvidence(evidence_text="2 kg water"),
            ),
        ],
    )
    reviewed_structure = original.model_copy(
        update={
            "processes": [
                original.processes[0].model_copy(update={"name": "Reviewed main"}),
                original.processes[1],
            ]
        }
    )
    df = pd.DataFrame(
        [
            {
                "include": True,
                "flow_id": 0,
                "process_id": "p1",
                "name": "electricity",
                "amount": 5.0,
                "unit": "kWh",
                "direction": "input",
                "linked_process_id": None,
                "component_or_stage": None,
                "basis": None,
                "notes": None,
                "evidence_text": "4 kWh electricity",
            },
            {
                "include": True,
                "flow_id": 1,
                "process_id": "p2",
                "name": "water",
                "amount": 2.0,
                "unit": "kg",
                "direction": "input",
                "linked_process_id": None,
                "component_or_stage": None,
                "basis": None,
                "notes": None,
                "evidence_text": "2 kg water",
            },
            {
                "include": True,
                "flow_id": None,
                "process_id": "p1",
                "name": "manual correction flow",
                "amount": 1.0,
                "unit": "kg",
                "direction": "input",
                "linked_process_id": None,
                "component_or_stage": None,
                "basis": None,
                "notes": "Added by reviewer",
                "evidence_text": "",
            },
        ]
    )

    result = normalize_inventory_review(
        df,
        extraction=reviewed_structure,
        original_extraction=original,
    )

    assert result["flow_id"].tolist() == [0, 1, 2]
    assert result["review_status"].tolist() == ["human_edited", "ai_proposed", "user_added"]
    assert result.loc[0, "process_name"] == "Reviewed main"
    assert result.loc[2, "process_name"] == "Reviewed main"
    # Evidence remains source text; status is derived from reviewed modelling fields, not evidence edits.
    assert result.loc[0, "evidence_text"] == "4 kWh electricity"
