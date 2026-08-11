from ai_lca.evidence_router import (
    build_structure_evidence,
    partition_inventory_candidates,
    route_inventory_candidate,
)
from ai_lca.jats import InventoryCandidate, JATSDocument


def candidate(text: str, *, context: str = "caption=inventory", evidence_type: str = "table_row") -> InventoryCandidate:
    return InventoryCandidate(
        candidate_id="cand_test",
        source_location="table:T1:row:1",
        evidence_text=text,
        context=context,
        evidence_type=evidence_type,
        table="T1",
    )


def test_foreground_inventory_is_retained_for_reasoning():
    item = candidate("Electricity input | 52.4 kWh per kg H2")
    route = route_inventory_candidate(item)
    assert route.label == "foreground_lci"
    assert route.safe_to_exclude_from_inventory_reasoning is False


def test_strong_lcia_result_can_be_safely_excluded():
    item = candidate(
        "Global warming potential | 4.2 kg CO2-eq",
        context="caption=Life cycle impact assessment results; impact category",
    )
    route = route_inventory_candidate(item)
    assert route.label == "lcia_result"
    assert route.safe_to_exclude_from_inventory_reasoning is True


def test_co2_emission_is_not_mistaken_for_lcia_result():
    item = candidate(
        "Carbon dioxide emission | 1.8 kg",
        context="caption=Life cycle inventory outputs",
    )
    route = route_inventory_candidate(item)
    assert route.safe_to_exclude_from_inventory_reasoning is False


def test_uncertain_numeric_parameter_is_retained():
    item = candidate("Reactor temperature | 850 °C", context="caption=Operating parameters")
    retained, excluded, _ = partition_inventory_candidates([item])
    assert retained == [item]
    assert excluded == []


def test_structure_evidence_prefers_scope_and_inventory_context():
    doc = JATSDocument(
        doi="10.0000/example",
        title="Hydrogen production LCA",
        abstract="A comparative life cycle assessment of two hydrogen pathways.",
        sections=[
            ("Introduction", "Long background discussion without modelling details." * 40),
            ("Goal and scope", "The functional unit is 1 kg hydrogen. The system boundary is cradle-to-gate."),
            ("Process description", "Hydrogen production uses electricity and water in the foreground process."),
            ("Discussion", "The results are compared with previous literature." * 40),
        ],
        tables=[
            ("T1", "Life cycle inventory inputs", ["Electricity | 52 kWh", "Water | 10 kg"]),
            ("T2", "Life cycle impact assessment results", ["GWP | 4.2 kg CO2-eq"]),
        ],
        inventory_candidates=[],
    )
    pack = build_structure_evidence(doc, max_chars=5000)
    assert "functional unit is 1 kg hydrogen" in pack.text
    assert "Life cycle inventory inputs" in pack.text
    assert len(pack.text) <= 5000
