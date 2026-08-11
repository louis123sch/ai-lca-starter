from ai_lca.controlled_accuracy import final_flow_risk


def assignment(disposition="modeled_inventory"):
    return {"candidate_id": "x", "disposition": disposition, "process_ids": ["p"]}


def test_final_midpoint_result_flow_is_flagged():
    candidate = {
        "evidence_type": "table_row",
        "evidence_text": "70,000 | 0.092 | 0.267 | 0.137",
        "context": "caption=Total environmental impact of the scenarios to principal midpoint categories and sensitivity of environmental impact results",
        "table": "Table 5",
    }
    flow = {"name": "Climate change", "amount": 0.267, "unit": "kg CO2 eq"}
    assert final_flow_risk(flow, candidate, assignment()) is True


def test_gwp_result_flow_is_flagged():
    candidate = {
        "evidence_type": "table_row",
        "evidence_text": "LC3 | UHMWPE | Carbon | 2 | TR-SHCC | 912.62 | 15.51",
        "context": "caption=Design configurations in ascending order of GWP per m3 of composite",
        "table": "Table 4",
    }
    flow = {"name": "GWP/1 m3", "amount": 912.62, "unit": "kg CO2 eq"}
    assert final_flow_risk(flow, candidate, assignment()) is True


def test_direct_inventory_co2_emission_is_not_flagged():
    candidate = {
        "evidence_type": "table_row",
        "evidence_text": "Carbon dioxide emission | 0.42 kg CO2 | output",
        "context": "caption=Life cycle inventory inputs and outputs",
        "table": "LCI",
    }
    flow = {"name": "Carbon dioxide emission", "amount": 0.42, "unit": "kg CO2"}
    assert final_flow_risk(flow, candidate, assignment()) is False


def test_normal_input_in_result_discussion_is_not_flagged():
    candidate = {
        "evidence_type": "table_row",
        "evidence_text": "Grid electricity | 2.0 kWh",
        "context": "caption=Environmental impact sensitivity scenarios",
        "table": "Table 8",
    }
    flow = {"name": "Electricity, high voltage", "amount": 2.0, "unit": "kWh"}
    assert final_flow_risk(flow, candidate, assignment()) is False


def test_non_modeled_candidate_is_not_flagged():
    candidate = {
        "evidence_type": "table_row",
        "evidence_text": "GWP total | 0.2",
        "context": "caption=Life cycle impact assessment results",
        "table": "Table 11",
    }
    flow = {"name": "GWP total", "amount": 0.2, "unit": "kg CO2-eq"}
    assert final_flow_risk(flow, candidate, assignment("not_inventory")) is False
