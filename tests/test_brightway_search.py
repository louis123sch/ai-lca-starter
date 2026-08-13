from ai_lca.brightway_search import normalize_search_query, split_off_location


def test_strips_cutoff_qualifier_before_location():
    assert (
        normalize_search_query(
            "market for electricity, medium voltage | electricity, medium voltage | Cutoff, U - US-SERC"
        )
        == "market for electricity, medium voltage | electricity, medium voltage | US-SERC"
    )


def test_strips_apos_and_consequential_qualifiers():
    assert normalize_search_query("Concrete, 25-30MPa, APOS, U - RoW") == "Concrete, 25-30MPa, RoW"
    assert normalize_search_query("Steel, low-alloyed, Consequential, S - GLO") == "Steel, low-alloyed, GLO"


def test_leaves_plain_queries_unchanged():
    assert normalize_search_query("Electricity, medium voltage, US-SERC") == "Electricity, medium voltage, US-SERC"
    assert normalize_search_query("Concrete") == "Concrete"
    assert normalize_search_query("") == ""
    assert normalize_search_query(None) == ""


def test_split_off_location_removes_hard_filter_text():
    query, location = split_off_location(
        "market for concrete, 30-32MPa | concrete, 30-32MPa | RoW"
    )
    assert query == "market for concrete, 30-32MPa | concrete, 30-32MPa"
    assert location == "RoW"


def test_split_off_location_handles_plain_comma_wording():
    query, location = split_off_location("Electricity, medium voltage, US-SERC")
    assert query == "Electricity, medium voltage"
    assert location == "US-SERC"


def test_split_off_location_leaves_query_without_a_code_unchanged():
    query, location = split_off_location("Concrete")
    assert query == "Concrete"
    assert location is None
