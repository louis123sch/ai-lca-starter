from ai_lca.export import extraction_to_dataframe
from ai_lca.models import (
    ForegroundProcess,
    InventoryExtraction,
    InventoryFlow,
    ProcessMap,
    SourceEvidence,
    TechnologyGroup,
)
from ai_lca.selection import recommended_candidate_index
from ai_lca.validation import normalise_process_ids, validate_inventory_against_process_map


def _process_map() -> ProcessMap:
    return normalise_process_ids(
        ProcessMap(
            source_summary="test corpus",
            geographic_context="Germany",
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


def test_direct_emission_is_biosphere_and_search_eligible():
    process_map = _process_map()
    extraction = InventoryExtraction(
        source_summary="test",
        flows=[
            InventoryFlow(
                process_id="p001",
                technology_group="SMR",
                process_name="Steam methane reforming",
                name="carbon dioxide",
                amount=9.0,
                unit="kg",
                direction="emission",
                flow_kind="emission",
                ecoinvent_search_term="carbon dioxide",
                evidence=[
                    SourceEvidence(
                        table="Table 2",
                        evidence_text="Direct CO2 emission | 9.0 kg",
                    )
                ],
            )
        ],
    )

    result = validate_inventory_against_process_map(extraction, process_map, "")
    flow = result.flows[0]
    assert flow.exchange_type == "biosphere"
    assert flow.biosphere_search_term == "carbon dioxide"
    assert flow.ecoinvent_search_term is None

    df = extraction_to_dataframe(result)
    assert bool(df.iloc[0]["background_match_eligible"]) is True
    assert df.iloc[0]["match_target"] == "biosphere"


def test_source_mapping_table_recovers_steam_turbine_proxy():
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
                exchange_type="technosphere",
                flow_kind="other",
                component_or_stage="plant construction",
                evidence=[
                    SourceEvidence(
                        source_document="paper.pdf",
                        table="Table 2",
                        evidence_text="Steam turbine | Unit | 7.90e-10",
                    )
                ],
            )
        ],
    )
    corpus = """[DOCUMENT supplement.docx]
[TABLE 1]
Gas turbine | Market for gas turbine, 10MW electrical | GLO | unit
[END DOCUMENT supplement.docx]
"""

    result = validate_inventory_against_process_map(extraction, process_map, corpus)
    flow = result.flows[0]
    assert flow.name == "steam turbine"
    assert flow.ecoinvent_activity_hint == "Market for gas turbine, 10MW electrical"
    assert flow.ecoinvent_location_hint == "GLO"
    assert flow.background_mapping_relation == "proxy"
    assert flow.background_mapping_evidence


def test_ambiguous_search_result_defaults_to_no_selection():
    candidates = [
        {
            "name": "market for steel, chromium steel 18/8, hot rolled",
            "match_score": 71.8,
            "match_reasons": "activity name contains query; keyword overlap: steel",
        },
        {
            "name": "market for steel, low-alloyed",
            "match_score": 68.0,
            "match_reasons": "activity name contains query; keyword overlap: steel",
        },
    ]
    index, reason = recommended_candidate_index(candidates, target="technosphere")
    assert index is None
    assert "not unambiguous" in reason.lower()


def test_source_exact_mapping_preselects_named_candidate_even_if_not_first():
    candidates = [
        {
            "name": "market for concrete, normal strength",
            "match_score": 80.0,
            "match_reasons": "activity name contains query",
        },
        {
            "name": "Market for concrete, normal",
            "match_score": 76.0,
            "match_reasons": "activity name contains query",
        },
    ]
    index, reason = recommended_candidate_index(
        candidates,
        source_activity_hint="Market for concrete, normal",
        mapping_relation="exact",
        target="technosphere",
    )
    assert index == 1
    assert "exact source mapping" in reason.lower()


def test_source_proxy_mapping_ignores_cosmetic_spacing():
    candidates = [
        {
            "name": "market for gas turbine, 10 MW electrical",
            "match_score": 82.0,
            "match_reasons": "activity name contains query",
        }
    ]
    index, reason = recommended_candidate_index(
        candidates,
        source_activity_hint="Market for gas turbine, 10MW electrical",
        mapping_relation="proxy",
        target="technosphere",
    )
    assert index == 0
    assert "proxy" in reason.lower()


def test_uncertain_source_mapping_is_never_auto_approved():
    candidates = [
        {
            "name": "market for water, decarbonised",
            "match_score": 110.0,
            "match_reasons": "activity name exactly matches query; reference product exactly matches query",
        }
    ]
    index, reason = recommended_candidate_index(
        candidates,
        source_activity_hint="market for water, decarbonised",
        mapping_relation="uncertain",
        target="technosphere",
    )
    assert index is None
    assert "uncertain" in reason.lower()


def test_high_confidence_biosphere_match_can_be_preselected():
    candidates = [
        {
            "name": "Carbon dioxide, fossil",
            "match_score": 112.0,
            "match_reasons": "biosphere name contains query; unit matches biosphere flow; compartment matches source hint",
        },
        {
            "name": "Carbon dioxide, non-fossil",
            "match_score": 90.0,
            "match_reasons": "biosphere name contains query; unit matches biosphere flow",
        },
    ]
    index, reason = recommended_candidate_index(candidates, target="biosphere")
    assert index == 0
    assert "biosphere match is strong" in reason.lower()
