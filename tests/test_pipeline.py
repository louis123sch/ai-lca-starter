from ai_lca.models import InventoryExtraction, InventoryFlow, SourceEvidence
from ai_lca.pipeline import merge_inventory_extractions


def _extraction(name: str, amount: float) -> InventoryExtraction:
    return InventoryExtraction(
        process_name="Hydrogen production",
        functional_unit="1 kg H2",
        source_summary=f"Source {name}",
        flows=[
            InventoryFlow(
                name="plant lifetime",
                item_type="parameter",
                amount=amount,
                unit="year",
                direction="unknown",
                evidence=SourceEvidence(
                    source_document="wrong-name.docx",
                    evidence_text=f"Plant lifetime is {amount} years.",
                ),
            )
        ],
    )


def test_merge_preserves_each_source_and_does_not_deduplicate():
    merged = merge_inventory_extractions(
        [
            ("source-a.docx", _extraction("a", 20)),
            ("source-b.pdf", _extraction("b", 25)),
        ]
    )

    assert len(merged.flows) == 2
    assert merged.flows[0].evidence.source_document == "source-a.docx"
    assert merged.flows[1].evidence.source_document == "source-b.pdf"
    assert merged.flows[0].amount == 20
    assert merged.flows[1].amount == 25
