from ai_lca.geography import ecoinvent_location_hints, parse_flow_location_hint


def test_geography_hints_are_derived_from_paper_context():
    assert ecoinvent_location_hints("The plant is operated in Germany") == ["DE"]
    assert ecoinvent_location_hints("UK / Great Britain scenario") == ["GB"]


def test_unknown_geography_does_not_force_a_location():
    assert ecoinvent_location_hints(None) == []
    assert ecoinvent_location_hints("Location not specified") == []


def test_flow_location_hint_detects_trailing_ecoinvent_style_code():
    assert parse_flow_location_hint("Electricity, medium voltage, US-SERC") == ["US-SERC"]
    assert parse_flow_location_hint("Electricity, medium voltage, US-WECC") == ["US-WECC"]
    assert parse_flow_location_hint("market for electricity, high voltage, GLO") == ["GLO"]
    assert parse_flow_location_hint("Some transport, RoW") == ["RoW"]


def test_flow_location_hint_ignores_non_code_trailing_text():
    assert parse_flow_location_hint("Insulation, rigid PUR (polyurethane) foam") == []
    assert parse_flow_location_hint("Transport, freight, lorry 16-32 t, EURO 6") == []
    assert parse_flow_location_hint("Tap water, municipal") == []
    assert parse_flow_location_hint(None) == []
