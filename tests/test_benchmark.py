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


def test_prompts_distinguish_owned_unit_process_inventories_from_nested_stages():
    from ai_lca.llm import STRUCTURE_SYSTEM_PROMPT

    flow_prompt = FLOW_SYSTEM_PROMPT.casefold()
    assert "explicit component lists" in flow_prompt
    assert "foreground input flow" in flow_prompt
    assert "do not reclassify such tabulated components as background subprocesses" in flow_prompt

    structure_prompt = STRUCTURE_SYSTEM_PROMPT.casefold()
    assert "activity-owned quantitative lci rows" in structure_prompt
    assert "merely partitioning one aggregate inventory by life-cycle stage is not enough" in structure_prompt
    assert "terminal activity need not feed a downstream foreground process" in structure_prompt
    assert "do not additionally classify the chain's title" in structure_prompt


def test_published_result_comparison_is_zero_for_paper_values():
    from ai_lca.benchmark import compare_published_gwi

    expected = _expected()
    comparison = compare_published_gwi(
        expected["published_gwi_reference_case_without_byproducts"],
        expected,
    )
    assert comparison["matched_results"] == 9
    assert comparison["mean_absolute_percent_error"] == 0.0
