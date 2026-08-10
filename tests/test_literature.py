from urllib.parse import parse_qs, urlparse

from ai_lca.literature import LiteratureRecord, record_from_crossref, rough_lca_relevance
from ai_lca.literature_acquire import springer_query_url


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


def test_springer_oa_and_tdm_urls_are_doi_scoped():
    doi = "10.1007/s11367-026-01234-5"
    oa = urlparse(springer_query_url(doi, "secret", tdm=False))
    tdm = urlparse(springer_query_url(doi, "secret", tdm=True))
    assert oa.path == "/openaccess/jats"
    assert tdm.path == "/xmldata/jats"
    assert parse_qs(oa.query)["q"] == [f"doi:{doi}"]
    assert parse_qs(tdm.query)["q"] == [f"doi:{doi}"]
