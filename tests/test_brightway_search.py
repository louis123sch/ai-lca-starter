from ai_lca.brightway_search import _query_variants, location_preferences_from_context


def test_country_name_is_normalised_to_ecoinvent_style_code():
    preferences = location_preferences_from_context("Germany")
    assert "DE" in preferences


def test_country_is_found_inside_long_evidence_context():
    preferences = location_preferences_from_context(
        "Germany primary; country-specific results for multiple European countries reported"
    )
    assert "DE" in preferences
    assert "RER" in preferences


def test_uk_alias_is_normalised_to_gb():
    preferences = location_preferences_from_context("UK")
    assert "GB" in preferences


def test_regional_context_keeps_ecoinvent_region_hint():
    preferences = location_preferences_from_context("Europe")
    assert "RER" in preferences


def test_missing_geography_has_no_preference():
    assert location_preferences_from_context(None) == []


def test_market_activity_hint_adds_market_for_query_variant():
    variants = _query_variants("natural gas", "market")
    assert variants[0] == "natural gas"
    assert "market for natural gas" in variants


def test_existing_market_query_is_not_repeated():
    variants = _query_variants("market for natural gas", "market")
    assert variants == ["market for natural gas"]
