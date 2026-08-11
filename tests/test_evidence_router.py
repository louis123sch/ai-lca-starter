from ai_lca.evidence_router import EvidenceRelevanceRouter


def test_router_separates_inventory_like_from_impact_like_evidence():
    router = EvidenceRelevanceRouter(high_threshold=0.60, low_threshold=0.40).fit([
        ("table_row inventory electricity input 52 kWh natural gas 1.2 kg", True),
        ("table_row material input nickel 2.8 kg water 4.0 kg", True),
        ("caption life cycle inventory diesel 0.5 kg transport 12 tkm", True),
        ("caption ReCiPe impact categories climate change kg CO2 eq", False),
        ("LCIA result global warming potential 4.3 kg CO2-eq", False),
        ("impact category acidification eutrophication toxicity result", False),
    ])
    inventory = router.route_candidate({
        "candidate_id": "a",
        "evidence_type": "table_row",
        "context": "caption=life cycle inventory inputs",
        "evidence_text": "Electricity | 45.2 kWh",
    })
    impact = router.route_candidate({
        "candidate_id": "b",
        "evidence_type": "table_row",
        "context": "caption=ReCiPe impact results",
        "evidence_text": "Climate change | kg CO2 eq | 4.3",
    })
    assert inventory.probability_lci > impact.probability_lci


def test_router_requires_both_training_classes():
    router = EvidenceRelevanceRouter()
    try:
        router.fit([("electricity 10 kWh", True)])
    except ValueError as exc:
        assert "positive and negative" in str(exc)
    else:
        raise AssertionError("expected ValueError")
