from ai_lca.controlled_accuracy import candidate_risks


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
