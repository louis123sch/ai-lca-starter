import pandas as pd

from ai_lca.export import extraction_to_dataframe
from ai_lca.models import ForegroundProcess, InventoryExtraction, InventoryFlow, SourceEvidence
from ai_lca.review import (
    apply_process_review,
    process_review_id_map,
    remap_inventory_dataframe,
)


def build_extraction():
    return InventoryExtraction(
        process_name="Synthetic system",
        source_summary="Synthetic review test",
        processes=[
            ForegroundProcess(
                process_id="p1",
                name="Main process",
                reference_product="product",
                reference_unit="kg",
            ),
            ForegroundProcess(
                process_id="p2",
                name="Internal-looking process",
                reference_product="intermediate",
                reference_unit="kg",
            ),
            ForegroundProcess(
                process_id="p3",
                name="Separate process",
                reference_product="other",
                reference_unit="kg",
            ),
        ],
        flows=[
            InventoryFlow(
                process_id="p1",
                name="electricity",
                amount=1,
                unit="kWh",
                direction="input",
                evidence=SourceEvidence(evidence_text="1 kWh electricity"),
            ),
            InventoryFlow(
                process_id="p2",
                name="water",
                amount=2,
                unit="kg",
                direction="input",
                evidence=SourceEvidence(evidence_text="2 kg water"),
            ),
            InventoryFlow(
                process_id="p3",
                name="intermediate",
                amount=0.5,
                unit="kg",
                direction="input",
                linked_process_id="p2",
                evidence=SourceEvidence(evidence_text="0.5 kg intermediate from p2"),
            ),
        ],
    )


def test_process_review_merges_and_reassigns_flows():
    extraction = build_extraction()
    review = pd.DataFrame(
        [
            {"include": True, "process_id": "p1", "process": "Renamed main", "merge_into": "", "parent": "", "reference_product": "product", "reference_unit": "kg"},
            {"include": False, "process_id": "p2", "process": "Internal-looking process", "merge_into": "p1", "parent": "", "reference_product": "intermediate", "reference_unit": "kg"},
            {"include": True, "process_id": "p3", "process": "Separate process", "merge_into": "", "parent": "", "reference_product": "other", "reference_unit": "kg"},
        ]
    )

    reviewed = apply_process_review(extraction, review)

    assert {process.process_id for process in reviewed.processes} == {"p1", "p3"}
    assert next(process for process in reviewed.processes if process.process_id == "p1").name == "Renamed main"
    assert [flow.process_id for flow in reviewed.flows] == ["p1", "p1", "p3"]
    assert reviewed.flows[2].linked_process_id == "p1"


def test_process_review_remove_without_merge_drops_attached_flows():
    extraction = build_extraction()
    review = pd.DataFrame(
        [
            {"include": True, "process_id": "p1", "process": "Main process", "merge_into": "", "parent": "", "reference_product": "product", "reference_unit": "kg"},
            {"include": True, "process_id": "p2", "process": "Internal-looking process", "merge_into": "", "parent": "", "reference_product": "intermediate", "reference_unit": "kg"},
            {"include": False, "process_id": "p3", "process": "Separate process", "merge_into": "", "parent": "", "reference_product": "other", "reference_unit": "kg"},
        ]
    )

    reviewed = apply_process_review(extraction, review)
    assert {process.process_id for process in reviewed.processes} == {"p1", "p2"}
    assert [flow.name for flow in reviewed.flows] == ["electricity", "water"]


def test_dataframe_remap_preserves_original_flow_ids_after_process_removal():
    extraction = build_extraction()
    original_df = extraction_to_dataframe(extraction)
    review = pd.DataFrame(
        [
            {"include": True, "process_id": "p1", "process": "Main process", "merge_into": "", "parent": "", "reference_product": "product", "reference_unit": "kg"},
            {"include": False, "process_id": "p2", "process": "Internal-looking process", "merge_into": "", "parent": "", "reference_product": "intermediate", "reference_unit": "kg"},
            {"include": True, "process_id": "p3", "process": "Separate process", "merge_into": "", "parent": "", "reference_product": "other", "reference_unit": "kg"},
        ]
    )

    reviewed = apply_process_review(extraction, review)
    id_map = process_review_id_map(extraction, review)
    names = {process.process_id: process.name for process in reviewed.processes}
    remapped = remap_inventory_dataframe(
        original_df,
        process_id_map=id_map,
        process_names=names,
    )

    assert remapped["flow_id"].tolist() == [0, 2]
    assert remapped["name"].tolist() == ["electricity", "intermediate"]
    assert remapped.loc[1, "linked_process_id"] == ""


def test_dataframe_remap_preserves_flow_ids_and_links_after_merge():
    extraction = build_extraction()
    original_df = extraction_to_dataframe(extraction)
    review = pd.DataFrame(
        [
            {"include": True, "process_id": "p1", "process": "Renamed main", "merge_into": "", "parent": "", "reference_product": "product", "reference_unit": "kg"},
            {"include": False, "process_id": "p2", "process": "Internal-looking process", "merge_into": "p1", "parent": "", "reference_product": "intermediate", "reference_unit": "kg"},
            {"include": True, "process_id": "p3", "process": "Separate process", "merge_into": "", "parent": "", "reference_product": "other", "reference_unit": "kg"},
        ]
    )

    reviewed = apply_process_review(extraction, review)
    id_map = process_review_id_map(extraction, review)
    names = {process.process_id: process.name for process in reviewed.processes}
    remapped = remap_inventory_dataframe(
        original_df,
        process_id_map=id_map,
        process_names=names,
    )

    assert remapped["flow_id"].tolist() == [0, 1, 2]
    assert remapped["process_id"].tolist() == ["p1", "p1", "p3"]
    assert remapped.loc[2, "linked_process_id"] == "p1"
    assert remapped.loc[0, "process_name"] == "Renamed main"
