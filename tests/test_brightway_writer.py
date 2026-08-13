import pandas as pd

from ai_lca.brightway_writer import build_write_plan
from ai_lca.models import ForegroundProcess, InventoryExtraction


def extraction():
    return InventoryExtraction(
        process_name="Synthetic",
        functional_unit="1 kg product",
        source_summary="Synthetic writer test",
        processes=[
            ForegroundProcess(
                process_id="p1",
                name="Main process",
                reference_product="product",
                reference_unit="kg",
            ),
            ForegroundProcess(
                process_id="p2",
                name="Intermediate process",
                reference_product="intermediate",
                reference_unit="kg",
                role="interconnected_foreground_process",
            ),
        ],
    )


def test_write_plan_accepts_mapped_input_emission_and_foreground_link():
    inv = pd.DataFrame(
        [
            {"include": True, "flow_id": 0, "process_id": "p1", "name": "electricity", "amount": 4.0, "unit": "kWh", "direction": "input", "linked_process_id": ""},
            {"include": True, "flow_id": 1, "process_id": "p1", "name": "carbon dioxide", "amount": 0.2, "unit": "kg", "direction": "emission", "linked_process_id": ""},
            {"include": True, "flow_id": 2, "process_id": "p1", "name": "intermediate", "amount": 0.5, "unit": "kg", "direction": "input", "linked_process_id": "p2"},
        ]
    )
    mappings = pd.DataFrame(
        [
            {"flow_id": 0, "database": "ecoinvent", "code": "electricity-code", "name": "electricity", "unit": "kilowatt hour"},
            {"flow_id": 1, "database": "biosphere3", "code": "co2-code", "name": "Carbon dioxide", "unit": "kilogram"},
        ]
    )

    plan = build_write_plan(extraction(), inv, mappings)
    assert plan.ready is True
    assert [exchange["exchange_type"] for exchange in plan.exchanges] == [
        "technosphere_background",
        "biosphere",
        "technosphere_foreground",
    ]


def test_write_plan_blocks_missing_amount_and_unmapped_output():
    inv = pd.DataFrame(
        [
            {"include": True, "flow_id": 0, "process_id": "p1", "name": "water", "amount": None, "unit": "kg", "direction": "input", "linked_process_id": ""},
            {"include": True, "flow_id": 1, "process_id": "p1", "name": "co-product", "amount": 0.2, "unit": "kg", "direction": "output", "linked_process_id": ""},
        ]
    )
    plan = build_write_plan(extraction(), inv, pd.DataFrame())
    assert plan.ready is False
    assert any("no reviewed numeric amount" in blocker for blocker in plan.blockers)
    assert any("no selected Brightway mapping" in blocker for blocker in plan.blockers)


def test_write_plan_accepts_mapped_output_as_avoided_burden_credit():
    inv = pd.DataFrame(
        [
            {"include": True, "flow_id": 0, "process_id": "p1", "name": "scrap steel", "amount": -880.0, "unit": "kg", "direction": "output", "linked_process_id": ""},
        ]
    )
    mappings = pd.DataFrame(
        [
            {"flow_id": 0, "database": "ecoinvent", "code": "steel-market-code", "name": "market for steel", "unit": "kilogram"},
        ]
    )
    plan = build_write_plan(extraction(), inv, mappings)
    assert plan.ready is True
    assert plan.exchanges == [
        {
            "flow_id": 0,
            "process_id": "p1",
            "flow_name": "scrap steel",
            "amount": -880.0,
            "unit": "kg",
            "exchange_type": "technosphere_background",
            "target_database": "ecoinvent",
            "target_code": "steel-market-code",
            "target_name": "market for steel",
        }
    ]


def test_write_plan_blocks_unit_mismatch_without_conversion():
    inv = pd.DataFrame(
        [
            {"include": True, "flow_id": 0, "process_id": "p1", "name": "heat", "amount": 5.0, "unit": "MJ", "direction": "input", "linked_process_id": ""},
        ]
    )
    mappings = pd.DataFrame(
        [
            {"flow_id": 0, "database": "ecoinvent", "code": "heat-code", "name": "heat", "unit": "kWh"},
        ]
    )
    plan = build_write_plan(extraction(), inv, mappings)
    assert plan.ready is False
    assert any("no automatic conversion" in blocker for blocker in plan.blockers)


def test_write_plan_blocks_process_ids_that_collapse_to_same_brightway_code():
    ambiguous = InventoryExtraction(
        process_name="Synthetic",
        source_summary="Code collision test",
        processes=[
            ForegroundProcess(
                process_id="route A",
                name="Route A",
                reference_product="product A",
                reference_unit="kg",
            ),
            ForegroundProcess(
                process_id="route-A",
                name="Route B",
                reference_product="product B",
                reference_unit="kg",
            ),
        ],
    )
    plan = build_write_plan(ambiguous, pd.DataFrame())
    assert plan.ready is False
    assert any("same Brightway code" in blocker for blocker in plan.blockers)
