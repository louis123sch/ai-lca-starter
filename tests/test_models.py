from ai_lca.models import InventoryExtraction, InventoryFlow, SourceEvidence


def test_inventory_model_accepts_auditable_flow():
    result = InventoryExtraction(
        process_name="Hydrogen production",
        functional_unit="1 kg H2",
        source_summary="Test source",
        flows=[
            InventoryFlow(
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
