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


def test_generic_steel_can_keep_an_explicit_proxy_mapping():
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
                ecoinvent_search_term="steel",
                ecoinvent_activity_hint="Market for steel, low-alloyed",
                ecoinvent_location_hint="GLO",
                background_mapping_relation="proxy",
                background_mapping_rationale="A separate source explicitly states that low-alloyed steel is used as the proxy for this generic steel input.",
                evidence=[SourceEvidence(table="Table 2", evidence_text="Steel kg 5.06e-03")],
                background_mapping_evidence=[
                    SourceEvidence(table="Mapping note", evidence_text="Generic steel represented by market for steel, low-alloyed")
                ],
            )
        ],
    )

    result = validate_inventory_against_process_map(extraction, process_map, "")
    assert result.flows[0].ecoinvent_activity_hint == "Market for steel, low-alloyed"
    assert result.flows[0].background_mapping_relation == "proxy"


def test_context_suffix_is_not_part_of_search_concept():
    process_map = _process_map()
    extraction = InventoryExtraction(
        source_summary="test",
        flows=[
            InventoryFlow(
                process_id="p001",
                technology_group="SMR",
                process_name="Steam methane reforming",
                name="concrete (plant construction)",
                amount=6.6e-6,
                unit="m3",
                direction="input",
                flow_kind="material",
                evidence=[SourceEvidence(table="Table 2", evidence_text="Concrete m3 6.60e-06")],
            )
        ],
    )

    result = validate_inventory_against_process_map(extraction, process_map, "")
    flow = result.flows[0]
    assert flow.name == "concrete"
    assert flow.ecoinvent_search_term == "concrete"
    assert "plant construction" in (flow.component_or_stage or "")


def test_steam_turbine_can_use_gas_turbine_as_proxy_without_renaming_foreground():
    process_map = _process_map()
    extraction = InventoryExtraction(
        source_summary="paper plus supplement",
        flows=[
            InventoryFlow(
                process_id="p001",
                technology_group="SMR",
                process_name="Steam methane reforming",
                name="steam turbine",
                amount=7.9e-10,
                unit="unit",
                direction="input",
                flow_kind="other",
                component_or_stage="plant construction",
                ecoinvent_search_term="steam turbine",
                ecoinvent_activity_hint="Market for gas turbine, 10MW electrical",
                ecoinvent_location_hint="GLO",
                evidence=[SourceEvidence(table="Table 2", evidence_text="Steam turbine Unit 7.90e-10")],
                background_mapping_evidence=[
                    SourceEvidence(table="Table S1", evidence_text="Gas turbine | Market for gas turbine, 10MW electrical | GLO | unit")
                ],
            )
        ],
    )

    result = validate_inventory_against_process_map(extraction, process_map, "")
    flow = result.flows[0]
    assert flow.name == "steam turbine"
    assert flow.ecoinvent_activity_hint == "Market for gas turbine, 10MW electrical"
    assert flow.ecoinvent_location_hint == "GLO"
    assert flow.background_mapping_relation == "proxy"
    assert "proxy" in (flow.background_mapping_rationale or "").lower()
