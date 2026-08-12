from ai_lca.jats import parse_jats_bytes


def test_jats_parser_enumerates_table_and_section_candidates():
    xml = b"""<?xml version='1.0'?>
    <article>
      <front>
        <article-meta>
          <article-id pub-id-type="doi">10.1000/test</article-id>
          <title-group><article-title>Life cycle assessment of test widget</article-title></title-group>
          <abstract><p>A cradle-to-gate LCA study.</p></abstract>
        </article-meta>
      </front>
      <body>
        <sec><title>Life cycle inventory</title>
          <p>The functional unit is 1 kg of widget.</p>
          <list><list-item><p>Electricity use was 2.5 kWh per kg.</p></list-item></list>
          <table-wrap>
            <label>Table 2</label>
            <caption><p>Foreground inventory</p></caption>
            <table>
              <thead><tr><th>Input</th><th>Amount</th></tr></thead>
              <tbody>
                <tr><td>Steel</td><td>4.2 kg</td></tr>
                <tr><td>Water</td><td>3.0 kg</td></tr>
              </tbody>
            </table>
          </table-wrap>
        </sec>
      </body>
    </article>"""
    doc = parse_jats_bytes(xml, expected_doi="10.1000/test")
    assert doc.doi == "10.1000/test"
    assert "test widget" in doc.title
    assert len(doc.tables) == 1
    evidence = [c.evidence_text for c in doc.inventory_candidates]
    assert any("Steel" in row and "4.2 kg" in row for row in evidence)
    assert any("Water" in row and "3.0 kg" in row for row in evidence)
    assert any("Electricity use" in row for row in evidence)
    assert "[TABLE: Table 2" in doc.structure_text()
