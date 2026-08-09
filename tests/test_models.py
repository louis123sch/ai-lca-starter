import pytest
from pydantic import ValidationError

from ai_lca.models import (
    ForegroundProcess,
    InventoryExtraction,
    InventoryFlow,
    SourceEvidence,
)


def test_inventory_model_accepts_flow_linked_to_process():
    result = InventoryExtraction(
        process_name="Hydrogen production",
        functional_unit="1 kg H2",
        source_summary="Test source",
        processes=[ForegroundProcess(process_id="P1", name="Electrolysis", stage="operation")],
        flows=[
            InventoryFlow(
                process_id="P1",
                name="electricity",
                amount=52.0,
                unit="kWh",
                direction="input",
                basis="per kg H2",
                evidence=SourceEvidence(page=3, evidence_text="Electricity use is 52 kWh/kg H2."),
            )
        ],
    )
    assert result.flows[0].amount == 52.0
    assert result.flows[0].process_id == "P1"


def test_inventory_rejects_flow_for_invented_process():
    with pytest.raises(ValidationError):
        InventoryExtraction(
            source_summary="Test source",
            processes=[ForegroundProcess(process_id="P1", name="Pyrolysis")],
            flows=[
                InventoryFlow(
                    process_id="P2",
                    name="electricity",
                    evidence=SourceEvidence(evidence_text="Electricity is consumed."),
                )
            ],
        )
