from ai_lca.models import InventoryFlow, SourceEvidence
from ai_lca.resilient import (
    _bounded_chunks,
    _canonical_locked_process_id,
    _enforce_locked_flow_attachments,
    _looks_inventory_dense,
    merge_supported_flows,
)
from types import SimpleNamespace


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


def test_merge_excludes_explicit_calculation_factors_not_inventory_exchanges():
    merged = merge_supported_flows(
        [flow("steel frame", amount=100, unit="kg")],
        [flow("steel manufacturing (additional manufacturing energy factor)", amount=5.3, unit="kWh/kg")],
    )
    assert [x.name for x in merged] == ["steel frame"]


def test_merge_excludes_lifetime_normalisation_totals_not_inventory_exchanges():
    merged = merge_supported_flows(
        [flow("electricity", amount=50, unit="kWh")],
        [flow("Produced amount of hydrogen in 20 years", amount=3_000_000, unit="kg")],
    )
    assert [x.name for x in merged] == ["electricity"]


def test_merge_excludes_explicit_dash_absence_but_keeps_unquantified_component():
    merged = merge_supported_flows(
        [],
        [
            flow("heat", amount=None, unit="kWh", evidence="Heat (kWh) = -"),
            flow("heat exchanger", amount=None, unit=None, evidence="BoP components: 9 Heat exchanger"),
        ],
    )
    assert [x.name for x in merged] == ["heat exchanger"]


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


def test_merge_collapses_component_material_parenthetical_variant():
    merged = merge_supported_flows(
        [flow("bipolar plate (titanium for bipolar plate)", amount=528, unit="kg")],
        [flow("titanium for bipolar plate", amount=528, unit="kg")],
    )
    assert len(merged) == 1


def test_merge_keeps_distinct_materials_for_same_component():
    merged = merge_supported_flows(
        [flow("electrode frame (chromium steel for electrode frame)", amount=10, unit="kg")],
        [flow("electrode frame (nickel for electrode frame)", amount=2, unit="kg")],
    )
    assert len(merged) == 2


def test_merge_treats_incl_parenthetical_as_explanatory_duplicate():
    merged = merge_supported_flows(
        [flow("diaphragm compressor", amount=100, unit="kg")],
        [flow("diaphragm compressor (incl. frequency converter)", amount=100, unit="kg")],
    )
    assert len(merged) == 1


def _structure_for_attachment_tests():
    return SimpleNamespace(
        processes=[SimpleNamespace(process_id="P1"), SimpleNamespace(process_id="P2")],
        candidate_activities=[
            SimpleNamespace(candidate_id="P1", parent_candidate_id=None),
            SimpleNamespace(candidate_id="P1_bop", parent_candidate_id="P1"),
            SimpleNamespace(candidate_id="P1_stack", parent_candidate_id="P1"),
            SimpleNamespace(candidate_id="shared_transport", parent_candidate_id=None),
        ],
    )


def test_internal_stage_flow_attachment_resolves_to_locked_parent():
    structure = _structure_for_attachment_tests()
    assert _canonical_locked_process_id("P1_bop", structure) == "P1"
    fixed = _enforce_locked_flow_attachments(
        [flow("pump").model_copy(update={"process_id": "P1_bop"})], structure
    )
    assert len(fixed) == 1
    assert fixed[0].process_id == "P1"


def test_non_locked_candidate_without_locked_ancestry_is_rejected():
    structure = _structure_for_attachment_tests()
    fixed = _enforce_locked_flow_attachments(
        [flow("transport").model_copy(update={"process_id": "shared_transport"})], structure
    )
    assert fixed == []


def test_locked_flow_attachment_is_unchanged():
    structure = _structure_for_attachment_tests()
    fixed = _enforce_locked_flow_attachments([flow("electricity")], structure)
    assert len(fixed) == 1
    assert fixed[0].process_id == "P1"
