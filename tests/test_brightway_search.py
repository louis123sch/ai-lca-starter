from ai_lca.brightway_search import location_preferences_from_context


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
