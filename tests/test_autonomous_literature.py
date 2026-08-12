from pathlib import Path

from ai_lca.autonomous_literature import (
    ASSIGN_SYSTEM_PROMPT,
    Budget,
    CandidateAssignment,
    CandidateAssignmentBatch,
    CandidateFlow,
    ProcessCandidateExtraction,
    RunConfig,
    StateStore,
    _job_key,
    _validate_assignments,
    _validate_process_extraction,
)
from ai_lca.jats import InventoryCandidate


def _candidate(cid: str, text: str = "Steel | 4.2 kg") -> InventoryCandidate:
    return InventoryCandidate(
        candidate_id=cid,
        source_location="table:T1:row:2",
        evidence_text=text,
        context="caption=Foreground inventory; headers=Input | Amount",
        evidence_type="table_row",
        table="T1",
    )


def test_state_store_completed_job_is_resumable(tmp_path: Path):
    store = StateStore(tmp_path)
    result = tmp_path / "result.json"
    result.write_text("{}")
    key = _job_key("10.test/x", "abc", "screen", process_id=None, model="gpt-5-nano", prompt_version="v1")
    store.start_job(key, "10.test/x", "screen", model="gpt-5-nano")
    store.complete_job(key, result)
    assert store.completed_job_result(key) == result


def test_assignment_validation_marks_missing_candidates():
    candidates = [_candidate("a"), _candidate("b", "Water | 3 kg")]
    batch = CandidateAssignmentBatch(assignments=[
        CandidateAssignment(
            candidate_id="a",
            disposition="modeled_inventory",
            process_ids=["P1"],
            rationale="inventory row",
        )
    ])
    accepted, missing = _validate_assignments(candidates, batch, {"P1"})
    assert len(accepted) == 1
    assert missing == ["b"]


def test_assignment_prompt_excludes_lcia_results_without_blocking_direct_emissions():
    prompt = ASSIGN_SYSTEM_PROMPT.casefold()
    assert "lcia midpoint or endpoint result rows" in prompt
    assert "must be not_inventory" in prompt
    assert "direct elementary co2 emission" in prompt
    assert "copy only supplied candidate_ids exactly" in prompt
    assert "never invent wildcard or placeholder ids" in prompt


def test_process_gate_detects_omitted_candidate():
    assigned = [_candidate("a"), _candidate("b", "Water | 3 kg")]
    extraction = ProcessCandidateExtraction(
        process_id="P1",
        flows=[
            CandidateFlow(
                candidate_id="a",
                name="Steel",
                amount=4.2,
                unit="kg",
                direction="input",
                evidence_text="Steel | 4.2 kg",
            )
        ],
    )
    cleaned, missing, failures = _validate_process_extraction("P1", assigned, extraction, {"P1"})
    assert len(cleaned.flows) == 1
    assert missing == ["b"]
    assert failures


def test_budget_enforces_call_cap(tmp_path: Path):
    store = StateStore(tmp_path)
    config = RunConfig(state_dir=tmp_path, max_total_calls=1, max_calls_per_paper=1)
    budget = Budget(config, store)
    budget.reserve_call("10.test/x")
    try:
        budget.reserve_call("10.test/x")
    except Exception as exc:
        assert "MAX_TOTAL_API_CALLS" in str(exc) or "MAX_CALLS_PER_PAPER" in str(exc)
    else:
        raise AssertionError("expected budget cap")
