from ai_lca.models import ForegroundProcess, InventoryExtraction, InventoryFlow, ProcessMap, SourceEvidence, TechnologyGroup
from ai_lca.validation import normalise_process_ids, validate_inventory_against_process_map


def _process_map() -> ProcessMap:
    return normalise_process_ids(
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
                            evidence=[SourceEvidence(table="Table 2", evidence_text="LCI data for SMR")],
                        )
                    ],
                )
            ],
        )
    )


def test_generic_steel_does_not_inherit_specific_steel_dataset():
    process_map = _process_map()
    extraction = InventoryExtraction(
        source_summary="paper plus supplement",
        flows=[
            InventoryFlow(
                process_id="p001",
                technology_group="SMR",
                process_name="Steam methane reforming",
                name="steel",
                amount=5.06e-3,
                unit="kg",
                direction="input",
                flow_kind="material",
                component_or_stage="plant construction",
                ecoinvent_search_term="steel",
                ecoinvent_activity_hint="Market for steel, low-alloyed",
                ecoinvent_location_hint="GLO",
                evidence=[SourceEvidence(table="Table 2", evidence_text="Steel kg 5.06e-03")],
                background_mapping_evidence=[
                    SourceEvidence(table="Table S1", evidence_text="Low alloyed steel | Market for steel, low-alloyed | GLO | kg")
                ],
            )
        ],
    )

    result = validate_inventory_against_process_map(extraction, process_map, "")
    flow = result.flows[0]
    assert flow.name == "steel"
    assert flow.ecoinvent_search_term == "steel"
    assert flow.ecoinvent_activity_hint is None
    assert flow.ecoinvent_location_hint is None
    assert any("over-specific steel mapping" in warning for warning in result.assumptions_or_warnings)
