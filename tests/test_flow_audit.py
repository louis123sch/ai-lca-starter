from ai_lca.flow_audit import merge_missing_flows
from ai_lca.models import InventoryFlow, SourceEvidence


def _flow(name: str, evidence_text: str, process_id: str = "P1") -> InventoryFlow:
    return InventoryFlow(
        process_id=process_id,
        name=name,
        amount=None,
        unit=None,
        direction="input",
        linked_process_id=None,
        component_or_stage=None,
        basis=None,
        notes=None,
        evidence=SourceEvidence(evidence_text=evidence_text),
    )


def test_merge_adds_source_supported_missing_flow():
    source = "Inventory table: steel frame; cooling water; nickel mesh."
    initial = [_flow("steel frame", "steel frame")]
    audited = [_flow("cooling water", "cooling water")]

    merged = merge_missing_flows(
        initial,
        audited,
        source_text=source,
        allowed_process_ids={"P1"},
    )

    assert [flow.name for flow in merged] == ["steel frame", "cooling water"]


def test_merge_rejects_unsupported_and_unknown_process_flows():
    source = "Inventory table: steel frame."
    initial = [_flow("steel frame", "steel frame")]
    audited = [
        _flow("medium voltage electricity", "medium voltage electricity"),
        _flow("steel frame", "steel frame", process_id="P2"),
    ]

    merged = merge_missing_flows(
        initial,
        audited,
        source_text=source,
        allowed_process_ids={"P1"},
    )

    assert merged == initial


def test_merge_deduplicates_exact_normalised_flow_key():
    source = "Inventory table: steel frame."
    initial = [_flow("Steel Frame", "steel frame")]
    audited = [_flow("  steel   frame ", "steel frame")]

    merged = merge_missing_flows(
        initial,
        audited,
        source_text=source,
        allowed_process_ids={"P1"},
    )

    assert merged == initial
