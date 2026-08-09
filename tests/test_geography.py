from ai_lca.geography import ecoinvent_location_hints


def test_geography_hints_are_derived_from_paper_context():
    assert ecoinvent_location_hints("The plant is operated in Germany") == ["DE"]
    assert ecoinvent_location_hints("UK / Great Britain scenario") == ["GB"]


def test_unknown_geography_does_not_force_a_location():
    assert ecoinvent_location_hints(None) == []
    assert ecoinvent_location_hints("Location not specified") == []
