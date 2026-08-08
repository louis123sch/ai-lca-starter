from ai_lca.brightway_search import _query_variants, location_preferences_from_context


def test_country_name_is_normalised_to_ecoinvent_style_code():
    preferences = location_preferences_from_context("Germany")
    assert "DE" in preferences


def test_uk_alias_is_normalised_to_gb():
    preferences = location_preferences_from_context("UK")
    assert "GB" in preferences


def test_regional_context_keeps_ecoinvent_region_hint():
    preferences = location_preferences_from_context("Europe")
    assert "RER" in preferences


def test_missing_geography_has_no_preference():
    assert location_preferences_from_context(None) == []


def test_longer_geography_phrase_extracts_country_and_region():
    preferences = location_preferences_from_context(
        "Germany primary; country-specific results for multiple European countries reported"
    )
    assert "DE" in preferences
    assert "RER" in preferences


def test_market_hint_adds_market_query_variant():
    variants = _query_variants("natural gas", "market")
    assert variants == ["natural gas", "market for natural gas"]


def test_exact_market_activity_query_is_not_prefixed_twice():
    variants = _query_variants("Market for concrete, normal", "market")
    assert variants == ["Market for concrete, normal"]


def test_supplier_technology_hint_adds_retrieval_variant_without_changing_base_query():
    variants = _query_variants("electricity", "market", "offshore wind")
    assert variants == ["electricity", "market for electricity", "electricity offshore wind"]
