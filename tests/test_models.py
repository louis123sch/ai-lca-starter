from ai_lca.models import InventoryExtraction, InventoryFlow, SourceEvidence


def test_inventory_model_accepts_auditable_flow():
    result = InventoryExtraction(
        process_name="Hydrogen production",
        functional_unit="1 kg H2",
        source_summary="Test source",
        flows=[
            InventoryFlow(
                name="electricity",
                item_type="technosphere_flow",
                amount=52.0,
                unit="kWh",
                direction="input",
                basis="per kg H2",
                evidence=SourceEvidence(
                    source_document="electrolyser.pdf",
                    page=3,
                    evidence_text="Electricity use is 52 kWh/kg H2.",
                ),
            )
        ],
    )
    assert result.flows[0].amount == 52.0
    assert result.flows[0].item_type == "technosphere_flow"
    assert result.flows[0].evidence.source_document == "electrolyser.pdf"


def test_inventory_model_classifies_parameter_without_background_mapping():
    result = InventoryExtraction(
        source_summary="Test source",
        flows=[
            InventoryFlow(
                name="plant lifetime",
                item_type="parameter",
                amount=20,
                unit="year",
                direction="unknown",
                evidence=SourceEvidence(
                    source_document="datasheet.docx",
                    evidence_text="The plant lifetime is 20 years.",
                ),
            )
        ],
    )
    assert result.flows[0].item_type == "parameter"
