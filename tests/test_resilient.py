from ai_lca.models import InventoryFlow, SourceEvidence
from ai_lca.resilient import _bounded_chunks, _looks_inventory_dense, merge_supported_flows


def flow(name: str, *, amount=None, unit=None, evidence="listed"):
    return InventoryFlow(
        process_id="P1",
        name=name,
        amount=amount,
        unit=unit,
        direction="input",
        evidence=SourceEvidence(evidence_text=evidence),
    )


def test_merge_recovery_adds_missing_source_supported_flow():
    merged = merge_supported_flows([flow("electricity", amount=50, unit="kWh")], [flow("steel frame")])
    assert [x.name for x in merged] == ["electricity", "steel frame"]


def test_merge_prefers_quantified_duplicate_without_duplicating():
    merged = merge_supported_flows(
        [flow("container steel")],
        [flow("container steel", amount=100, unit="kg", evidence="container steel | 100 | kg")],
    )
    assert len(merged) == 1
    assert merged[0].amount == 100
    assert merged[0].unit == "kg"


def test_merge_treats_parenthetical_counts_as_same_identity():
    merged = merge_supported_flows([flow("heat exchanger")], [flow("Heat exchanger (11)")])
    assert len(merged) == 1


def test_inventory_dense_detection_requires_list_or_table_structure():
    assert _looks_inventory_dense("Table 2\nMaterial | Amount | Unit\nsteel | 2 | kg")
    assert _looks_inventory_dense("BoP components:\n1 Pump\n2 Tank\n3 Heat exchanger")
    assert not _looks_inventory_dense("This section discusses component durability in general terms.")


def test_bounded_chunks_preserve_provenance_sections():
    text = "[PAGE 1]\nintro\n\n[PAGE 2]\n" + ("x" * 200)
    chunks = _bounded_chunks(text, max_chars=80)
    assert chunks
    assert any("[PAGE 1]" in chunk for chunk in chunks)
    assert any("[PAGE 2]" in chunk for chunk in chunks)
    assert all(len(chunk) <= 80 or chunk.startswith("[PAGE") for chunk in chunks)
