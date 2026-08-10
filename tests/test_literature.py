from ai_lca.literature import LiteratureRecord, record_from_crossref, rough_lca_relevance


def test_record_from_crossref_normalizes_metadata():
    item = {
        "DOI": "10.1007/S11367-026-00000-X",
        "title": ["Life cycle assessment of a test system"],
        "container-title": ["The International Journal of Life Cycle Assessment"],
        "published-online": {"date-parts": [[2026, 5, 1]]},
        "publisher": "Springer Science and Business Media LLC",
        "type": "journal-article",
        "URL": "https://doi.org/10.1007/s11367-026-00000-x",
        "abstract": "An LCA case study with a life cycle inventory and functional unit.",
        "license": [{"URL": "https://example.test/license"}],
        "link": [{"URL": "https://example.test/fulltext.xml", "content-type": "application/xml"}],
    }
    record = record_from_crossref(item)
    assert record.doi == "10.1007/s11367-026-00000-x"
    assert record.published_year == 2026
    assert record.has_abstract
    assert record.has_tdm_link
    assert rough_lca_relevance(record) >= 8


def test_relevance_downranks_corrections():
    research = LiteratureRecord(
        doi="10.test/research",
        title="Life cycle assessment and inventory of hydrogen production",
        published_year=2026,
        journal="Test",
        abstract="Functional unit and system boundary are reported.",
        url=None,
        publisher=None,
        type="journal-article",
        license_urls=(),
        tdm_links=(),
    )
    correction = LiteratureRecord(
        doi="10.test/correction",
        title="Correction to life cycle assessment",
        published_year=2026,
        journal="Test",
        abstract=None,
        url=None,
        publisher=None,
        type="journal-article",
        license_urls=(),
        tdm_links=(),
    )
    assert rough_lca_relevance(research) > rough_lca_relevance(correction)
