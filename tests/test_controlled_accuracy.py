from ai_lca.controlled_accuracy import candidate_risks, final_flow_risk


def a(disposition="modeled_inventory"):
    return {"candidate_id": "x", "disposition": disposition, "process_ids": ["p"]}


def test_lcia_result_table_is_flagged():
    candidate = {
        "evidence_type": "table_row",
        "evidence_text": "GWP (kg CO2-eq) | 0.61 | 0.59",
        "context": "caption=Absolute environmental impact data by impact category",
        "table": "T5",
    }
    assert "MODELED_LCIA_RESULT_TABLE_RISK" in candidate_risks(candidate, a())


def test_method_comparison_table_is_flagged():
    candidate = {
        "evidence_type": "table_row",
        "evidence_text": "System expansion / substitution | allocation alternative",
        "context": "caption=Advantages and limitations of each attribution method",
        "table": "T1",
    }
    assert "MODELED_METHOD_TABLE_RISK" in candidate_risks(candidate, a())


def test_direct_inventory_co2_emission_is_not_flagged():
    candidate = {
        "evidence_type": "table_row",
        "evidence_text": "Carbon dioxide emission | 0.42 kg CO2 | output",
        "context": "caption=Life cycle inventory inputs and outputs",
        "table": "LCI",
    }
    assert candidate_risks(candidate, a()) == []
    flow = {"name": "Carbon dioxide emission", "amount": 0.42, "unit": "kg CO2"}
    assert final_flow_risk(flow, candidate, a()) is False


def test_final_midpoint_result_flow_is_flagged():
    candidate = {
        "evidence_type": "table_row",
        "evidence_text": "70,000 | 0.092 | 0.267 | 0.137",
        "context": "caption=Total environmental impact of the scenarios to principal midpoint categories and sensitivity of environmental impact results",
        "table": "Table 5",
    }
    flow = {"name": "Climate change", "amount": 0.267, "unit": "kg CO2 eq"}
    assert final_flow_risk(flow, candidate, a()) is True


def test_normal_input_in_result_discussion_is_not_a_final_lcia_flow():
    candidate = {
        "evidence_type": "table_row",
        "evidence_text": "Grid electricity | 2.0 kWh",
        "context": "caption=Environmental impact sensitivity scenarios",
        "table": "Table 8",
    }
    flow = {"name": "Electricity, high voltage", "amount": 2.0, "unit": "kWh"}
    assert final_flow_risk(flow, candidate, a()) is False
