from ai_lca.models import (
    ForegroundProcessProposal,
    InventoryExtraction,
    InventoryFlow,
    OperatingContextProposal,
    SourceEvidence,
)
from ai_lca.export import extraction_to_dataframe, searchable_exchanges


def test_inventory_model_accepts_econivent_linkable_exchange_with_process_context():
    result = InventoryExtraction(
        process_name="Hydrogen production, SMR + CCS",
        technology_name="SMR + CCS",
        functional_unit="1 kg H2",
        source_summary="Test source",
        foreground_processes=[
            ForegroundProcessProposal(
                name="SMR",
                role="Steam methane reforming stage",
                source_document="smr.pdf",
                evidence_text="Hydrogen is produced by steam methane reforming.",
            )
        ],
        operating_contexts=[
            OperatingContextProposal(
                intended_geography="United Kingdom",
                ecoinvent_location_hint="GB",
                geography_basis="explicit",
                operating_setting="UK industrial hydrogen production plant",
                source_document="smr.pdf",
                evidence_text="The hydrogen production plant is located in the United Kingdom.",
            )
        ],
        flows=[
            InventoryFlow(
                name="natural gas",
                source_label="Natural gas - SMR + CCS 90%",
                item_type="technosphere_flow",
                parent_process="SMR",
                amount=3.5,
                unit="kg",
                direction="input",
                basis="per kg H2",
                search_worthy=True,
                ecoinvent_search_term="natural gas",
                ecoinvent_activity_type_hint="market",
                geography_hint="NO",
                supplier_technology_hint="Norwegian natural gas production",
                interpretation_reason=(
                    "Natural gas is a consumed feedstock; SMR + CCS and capture rate describe foreground context."
                ),
                evidence=SourceEvidence(
                    source_document="smr.pdf",
                    page=3,
                    evidence_text="Natural gas is supplied from Norway to the UK SMR unit.",
                ),
            )
        ],
    )

    flow = result.flows[0]
    assert flow.name == "natural gas"
    assert flow.parent_process == "SMR"
    assert flow.ecoinvent_search_term == "natural gas"
    assert flow.geography_hint == "NO"
    assert flow.search_worthy is True
    assert result.operating_contexts[0].ecoinvent_location_hint == "GB"
    assert result.operating_contexts[0].intended_geography == "United Kingdom"


def test_operating_context_can_explicitly_remain_unspecified():
    result = InventoryExtraction(
        source_summary="Generic technology paper",
        operating_contexts=[
            OperatingContextProposal(
                geography_basis="not_specified",
                intended_geography=None,
                ecoinvent_location_hint=None,
                operating_setting="Generic technology model",
                note="The paper does not define a deployment country.",
            )
        ],
    )

    context = result.operating_contexts[0]
    assert context.geography_basis == "not_specified"
    assert context.ecoinvent_location_hint is None


def test_parameter_cannot_enter_deterministic_ecoinvent_search_selection():
    extraction = InventoryExtraction(
        source_summary="Test source",
        flows=[
            InventoryFlow(
                name="natural gas",
                item_type="technosphere_flow",
                parent_process="SMR",
                search_worthy=True,
                ecoinvent_search_term="natural gas",
                interpretation_reason="Consumed feedstock.",
                evidence=SourceEvidence(
                    source_document="smr.pdf",
                    evidence_text="Natural gas is used as feedstock.",
                ),
            ),
            InventoryFlow(
                name="CO2 capture rate",
                source_label="90% CCS",
                item_type="parameter",
                parent_process="CO2 capture",
                amount=90,
                unit="%",
                search_worthy=True,  # Even a mistaken UI toggle must not be sufficient.
                ecoinvent_search_term="CCS 90%",
                interpretation_reason="Capture efficiency is a model parameter.",
                evidence=SourceEvidence(
                    source_document="smr.pdf",
                    evidence_text="A capture rate of 90% is assumed.",
                ),
            ),
        ],
    )

    df = extraction_to_dataframe(extraction)
    searchable = searchable_exchanges(df)

    assert len(searchable) == 1
    assert searchable.iloc[0]["name"] == "natural gas"
    assert searchable.iloc[0]["ecoinvent_search_term"] == "natural gas"
