from ai_lca.brightway_search import _activity_type, _normalise_location, _query_variants


def test_market_hint_adds_econivent_style_variant_without_replacing_product_concept():
    assert _query_variants("natural gas", "market") == [
        "natural gas",
        "market for natural gas",
    ]


def test_treatment_hint_uses_treatment_of_naming_convention():
    assert _query_variants("wastewater", "treatment") == [
        "wastewater",
        "treatment of wastewater",
    ]


def test_geography_is_normalised_separately_from_query():
    assert _normalise_location("UK") == "GB"
    assert _normalise_location("Norway") == "NO"


def test_activity_type_is_inferred_from_real_candidate_name():
    assert _activity_type("market for natural gas, high pressure") == "market"
    assert _activity_type("treatment of wastewater, average") == "treatment"
