import json
from pathlib import Path

from ai_lca.benchmark import evaluate_extraction, load_expected
from ai_lca.llm import FLOW_SYSTEM_PROMPT
from ai_lca.models import (
    ForegroundProcess,
    InventoryExtraction,
    InventoryFlow,
    SourceEvidence,
    StudyContext,
)


EXPECTED_PATH = Path(__file__).parents[1] / "benchmarks" / "hermesmann_2022" / "expected.json"


def _expected():
    return load_expected(EXPECTED_PATH)


def _perfect_extraction():
    expected = _expected()
    processes = []
    process_id_by_key = {}
    for i, process in enumerate(expected["processes"], start=1):
        process_id = f"P{i}"
        process_id_by_key[process["key"]] = process_id
        processes.append(
            ForegroundProcess(
                process_id=process_id,
                name=process["name"],
                stage="operation",
                evidence=[SourceEvidence(document="paper.pdf", page=10, evidence_text="Modeled configuration")],
            )
        )
    flows = []
    for flow in expected["flows"]:
        flows.append(
            InventoryFlow(
                process_id=process_id_by_key[flow["process_key"]],
                name=flow["name"],
                amount=flow["amount"],
                unit=flow["unit"],
                direction=flow["direction"],
                basis="per 1 kg H2",
                evidence=SourceEvidence(document="paper.pdf", page=11, evidence_text="LCI table value"),
            )
        )
    return InventoryExtraction(
        process_name="Hydrogen production technologies",
        functional_unit="1 kg H2 at 30 bar at the production site",
        source_summary="Hermesmann and Müller hydrogen production LCA",
        study_context=StudyContext(
            operational_geography="Germany",
            geography_basis="explicit",
            additional_geographies=["United Kingdom", "France"],
            system_boundary="cradle-to-gate",
        ),
        processes=processes,
        flows=flows,
    )


def test_hermesmann_benchmark_scores_perfect_fixture():
    report = evaluate_extraction(_perfect_extraction(), _expected())
    assert report.process_recall == 1.0
    assert report.process_precision == 1.0
    assert report.flow_recall == 1.0
    assert report.flow_precision == 1.0
    assert report.amount_accuracy == 1.0
    assert report.unit_accuracy == 1.0
    assert report.direction_accuracy == 1.0
    assert report.forbidden_processes == []
    assert report.forbidden_foreground_names == []
    assert report.overall_score > 0.99


def test_benchmark_catches_overdecomposition_and_dataset_name_leakage():
    extraction = _perfect_extraction()
    extraction.processes.append(
        ForegroundProcess(
            process_id="P10",
            name="Electricity supply",
            stage="operation",
            evidence=[SourceEvidence(evidence_text="Not actually a separate foreground process")],
        )
    )
    extraction.flows.append(
        InventoryFlow(
            process_id="P10",
            name="market for electricity, high voltage",
            amount=1.0,
            unit="kWh",
            direction="input",
            evidence=SourceEvidence(evidence_text="Background dataset leaked into foreground"),
        )
    )
    report = evaluate_extraction(extraction, _expected())
    assert "Electricity supply" in report.forbidden_processes
    assert "market for electricity, high voltage" in report.forbidden_foreground_names
    assert report.process_precision < 1.0
    assert report.flow_precision < 1.0


def test_flow_prompt_keeps_explicit_component_tables_as_inventory_flows():
    prompt = FLOW_SYSTEM_PROMPT.casefold()
    assert "explicit component lists" in prompt
    assert "foreground input flow" in prompt
    assert "do not reclassify such tabulated components as background subprocesses" in prompt


def test_published_result_comparison_is_zero_for_paper_values():
    from ai_lca.benchmark import compare_published_gwi

    expected = _expected()
    comparison = compare_published_gwi(
        expected["published_gwi_reference_case_without_byproducts"],
        expected,
    )
    assert comparison["matched_results"] == 9
    assert comparison["mean_absolute_percent_error"] == 0.0



def test_process_matching_prefers_more_specific_substring_candidate():
    from types import SimpleNamespace

    from ai_lca.benchmark import _score_name, _unique_match

    expected = [
        {"name": "Transport to waste treatment", "aliases": ["Transport to waste treatment"]},
        {"name": "Waste treatment", "aliases": ["Waste treatment"]},
    ]
    actual = [
        SimpleNamespace(name="Transport to waste treatment (C2)"),
        SimpleNamespace(name="Waste treatment (C3)"),
    ]
    matches, missing, extra = _unique_match(
        expected,
        actual,
        lambda e, a: _score_name(a.name, e["aliases"]),
        0.60,
    )
    assert matches == {0: 0, 1: 1}
    assert missing == []
    assert extra == []


def test_expected_child_process_is_not_reported_as_unexpected_child():
    extraction = _perfect_extraction()
    extraction.processes[0].parent_process_id = "assessed_product_system"
    report = evaluate_extraction(extraction, _expected())
    assert not any("unexpected child process" in name for name in report.forbidden_processes)
