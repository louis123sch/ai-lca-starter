from ai_lca.benchmark_text_inventory_lists import score_unquantified_flows
from ai_lca.models import FlowExtraction, InventoryFlow, SourceEvidence


def _flow(name: str, *, amount=None, unit=None) -> InventoryFlow:
    return InventoryFlow(
        process_id="p1",
        name=name,
        amount=amount,
        unit=unit,
        direction="input",
        evidence=SourceEvidence(evidence_text=name),
    )


def test_quantified_rows_are_out_of_scope_not_false_positives():
    extraction = FlowExtraction(
        flows=[
            _flow("listed component"),
            _flow("explicit electricity demand", amount=12.5, unit="kWh"),
        ]
    )
    expected = [{"process_key": "p1", "name": "listed component", "aliases": ""}]

    report = score_unquantified_flows(extraction, expected)

    assert report["recall"] == 1.0
    assert report["precision"] == 1.0
    assert report["unsupported_rows"] == 0
    assert report["quantified_out_of_scope_rows"] == 1


def test_unmatched_unquantified_rows_still_reduce_precision():
    extraction = FlowExtraction(
        flows=[
            _flow("listed component"),
            _flow("invented unquantified component"),
        ]
    )
    expected = [{"process_key": "p1", "name": "listed component", "aliases": ""}]

    report = score_unquantified_flows(extraction, expected)

    assert report["recall"] == 1.0
    assert report["precision"] == 0.5
    assert report["unsupported_rows"] == 1
