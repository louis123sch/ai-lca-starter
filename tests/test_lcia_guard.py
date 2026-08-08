from ai_lca.models import ForegroundProcess, InventoryExtraction, InventoryFlow, ProcessMap, SourceEvidence, TechnologyGroup
from ai_lca.validation import normalise_process_ids, validate_inventory_against_process_map


def test_co2_equivalent_result_is_not_kept_as_biosphere_emission():
    process_map = normalise_process_ids(
        ProcessMap(
            source_summary="test",
            technology_groups=[
                TechnologyGroup(
                    name="SMR",
                    processes=[
                        ForegroundProcess(
                            name="Steam methane reforming",
                            evidence_type="inventory_table",
                            reason_for_separate_process="LCI table",
                            evidence=[SourceEvidence(evidence_text="SMR inventory")],
                        )
                    ],
                )
            ],
        )
    )
    extraction = InventoryExtraction(
        source_summary="test",
        flows=[
            InventoryFlow(
                process_id="p001",
                technology_group="SMR",
                process_name="Steam methane reforming",
                name="carbon dioxide",
                amount=9.5,
                unit="kg CO2-eq/kg H2",
                direction="emission",
                exchange_type="biosphere",
                flow_kind="emission",
                evidence=[SourceEvidence(evidence_text="GWP result: 9.5 kg CO2-eq/kg H2")],
            )
        ],
    )

    result = validate_inventory_against_process_map(extraction, process_map, "")
    assert result.flows == []
    assert any("LCIA-equivalent indicator" in warning for warning in result.assumptions_or_warnings)
