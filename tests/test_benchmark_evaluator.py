from ai_lca.benchmark import _score_name, evaluate_extraction
from ai_lca.models import ForegroundProcess, InventoryExtraction, StudyContext


def _extraction(processes):
    return InventoryExtraction(
        process_name="test",
        functional_unit="1 kg H2",
        source_summary="test",
        study_context=StudyContext(
            operational_geography="Germany",
            geography_basis="explicit",
            system_boundary="cradle to grave",
        ),
        processes=processes,
        flows=[],
    )


def _expected():
    return {
        "benchmark_id": "evaluator_regression",
        "context": {
            "functional_unit_terms": ["1 kg h2"],
            "system_boundary": "cradle to grave",
            "reference_geography": "Germany",
        },
        "processes": [
            {
                "key": "pemec",
                "name": "Polymer electrolyte membrane electrolysis",
                "aliases": ["PEMEC", "polymer electrolyte membrane electrolysis"],
            }
        ],
        "flows": [],
        "forbidden_process_terms": ["membrane", "compressor"],
    }


def test_forbidden_term_inside_matched_gold_process_is_not_overdecomposition():
    report = evaluate_extraction(
        _extraction(
            [
                ForegroundProcess(
                    process_id="pemec",
                    name="Polymer electrolyte membrane electrolysis (PEMEC)",
                )
            ]
        ),
        _expected(),
    )

    assert report.matched_processes == 1
    assert report.forbidden_processes == []


def test_forbidden_term_still_flags_unmatched_extra_process():
    report = evaluate_extraction(
        _extraction(
            [
                ForegroundProcess(
                    process_id="pemec",
                    name="Polymer electrolyte membrane electrolysis (PEMEC)",
                ),
                ForegroundProcess(
                    process_id="compressor",
                    name="Hydrogen compressor",
                ),
            ]
        ),
        _expected(),
    )

    assert report.matched_processes == 1
    assert report.unexpected_processes == ["Hydrogen compressor"]
    assert report.forbidden_processes == ["Hydrogen compressor"]


def test_expected_child_process_is_not_forbidden_merely_for_having_parent():
    report = evaluate_extraction(
        _extraction(
            [
                ForegroundProcess(
                    process_id="parent",
                    name="Hydrogen system",
                ),
                ForegroundProcess(
                    process_id="pemec",
                    name="PEMEC",
                    parent_process_id="parent",
                ),
            ]
        ),
        _expected(),
    )

    assert report.matched_processes == 1
    assert "PEMEC (unexpected child process)" not in report.forbidden_processes


def test_unmatched_child_process_is_still_forbidden():
    report = evaluate_extraction(
        _extraction(
            [
                ForegroundProcess(
                    process_id="pemec",
                    name="PEMEC",
                ),
                ForegroundProcess(
                    process_id="compressor",
                    name="Hydrogen compressor",
                    parent_process_id="pemec",
                ),
            ]
        ),
        _expected(),
    )

    assert report.matched_processes == 1
    assert "Hydrogen compressor (unexpected child process)" in report.forbidden_processes


def test_generic_modelling_qualifiers_do_not_create_false_negative():
    assert _score_name(
        "MPW to methanol (foreground product system)",
        ["MPW-methanol", "mixed plastic waste methanol"],
    ) >= 0.60
    assert _score_name(
        "MPW to hydrogen (cradle-to-gate LCA foreground process)",
        ["MPW-hydrogen", "mixed plastic waste hydrogen"],
    ) >= 0.60


def test_name_core_preserves_technology_token_order():
    forward = _score_name("methanol to hydrogen foreground process", ["methanol-hydrogen"])
    reversed_direction = _score_name("hydrogen to methanol foreground process", ["methanol-hydrogen"])
    assert forward > reversed_direction
