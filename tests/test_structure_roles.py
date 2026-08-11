from ai_lca.models import ForegroundInterpretation, ProcessCandidate, SourceEvidence
from ai_lca.structure import lock_foreground_interpretation


def candidate(candidate_id: str, name: str, role: str, parent: str | None = None):
    return ProcessCandidate(
        candidate_id=candidate_id,
        name=name,
        role=role,
        parent_candidate_id=parent,
        rationale=f"Synthetic evidence classifies {name} as {role}.",
        evidence=[SourceEvidence(evidence_text=f"Synthetic source: {name}")],
    )


def interpretation(candidates):
    return ForegroundInterpretation(
        process_name="Synthetic product",
        functional_unit="1 kg product",
        source_summary="Synthetic structural test",
        candidates=candidates,
    )


def test_one_product_system_with_many_unit_operations_locks_one_process():
    result = lock_foreground_interpretation(
        interpretation(
            [
                candidate("product", "Product system", "assessed_product_system"),
                candidate("prep", "Feed preparation", "internal_stage", "product"),
                candidate("reactor", "Reaction", "internal_stage", "product"),
                candidate("separation", "Separation", "internal_stage", "product"),
                candidate("compression", "Compression", "internal_stage", "product"),
                candidate("storage", "Storage", "internal_stage", "product"),
            ]
        )
    )
    assert [process.process_id for process in result.processes] == ["product"]


def test_assessed_alternatives_do_not_promote_shared_support_activity():
    result = lock_foreground_interpretation(
        interpretation(
            [
                candidate("route_a", "Route A", "assessed_product_system"),
                candidate("route_b", "Route B", "assessed_product_system"),
                candidate("shared_storage", "Shared storage", "shared_supporting_activity"),
            ]
        )
    )
    assert {process.process_id for process in result.processes} == {"route_a", "route_b"}


def test_parented_assessed_product_system_is_candidate_only():
    result = lock_foreground_interpretation(
        interpretation(
            [
                candidate("route", "Route", "assessed_product_system"),
                candidate("route_scenario", "Route under scenario", "assessed_product_system", "route"),
            ]
        )
    )
    assert [process.process_id for process in result.processes] == ["route"]
    assert any(
        "Nested foreground activities require the interconnected_foreground_process role" in warning
        for warning in result.assumptions_or_warnings
    )


def test_explicit_interconnected_foreground_process_is_retained():
    result = lock_foreground_interpretation(
        interpretation(
            [
                candidate("main", "Main product system", "assessed_product_system"),
                ProcessCandidate(
                    candidate_id="intermediate",
                    name="Intermediate production",
                    role="interconnected_foreground_process",
                    parent_candidate_id="main",
                    reference_product="intermediate",
                    reference_unit="kg",
                    rationale="The source gives a separate reference product and quantified exchange.",
                    evidence=[SourceEvidence(evidence_text="1.2 kg intermediate is supplied to the main process.")],
                ),
            ]
        )
    )
    assert [process.process_id for process in result.processes] == ["main", "intermediate"]
    assert result.processes[1].parent_process_id == "main"
    assert result.processes[1].reference_product == "intermediate"


def test_background_and_descriptive_entities_never_lock_as_processes():
    result = lock_foreground_interpretation(
        interpretation(
            [
                candidate("main", "Main product system", "assessed_product_system"),
                candidate("electricity", "Electricity", "background_supply"),
                candidate("alt_catalyst", "Alternative catalyst", "descriptive_only"),
            ]
        )
    )
    assert [process.process_id for process in result.processes] == ["main"]


def test_parent_pointing_to_excluded_candidate_is_removed_conservatively():
    result = lock_foreground_interpretation(
        interpretation(
            [
                candidate("main", "Main system", "assessed_product_system", "support"),
                candidate("support", "Shared support", "shared_supporting_activity"),
            ]
        )
    )
    # A parented assessed system is now conservatively retained as a candidate only.
    # With no lockable process remaining, the structural guard rejects the interpretation.
    try:
        lock_foreground_interpretation(
            interpretation(
                [
                    candidate("main", "Main system", "assessed_product_system", "support"),
                    candidate("support", "Shared support", "shared_supporting_activity"),
                ]
            )
        )
    except RuntimeError as exc:
        assert "No candidate was classified" in str(exc)
    else:
        raise AssertionError("Expected an interpretation with no lockable foreground process to fail")
