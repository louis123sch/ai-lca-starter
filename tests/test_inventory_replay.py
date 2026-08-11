from __future__ import annotations

from ai_lca.inventory_replay import compare


def test_compare_accepts_real_improvement_without_regression() -> None:
    before = {
        "a": {
            "status": "UNRESOLVED_INVENTORY",
            "ambiguous_or_missing_candidate_count": 3,
            "flow_count": 4,
            "candidate_coverage": 0.5,
        },
        "b": {
            "status": "COMPLETE",
            "ambiguous_or_missing_candidate_count": 0,
            "flow_count": 5,
            "candidate_coverage": 1.0,
        },
    }
    results = [
        {
            "doi": "a",
            "status": "UNRESOLVED_INVENTORY",
            "ambiguous_or_missing_candidate_count": 1,
            "flow_count": 6,
        },
        {
            "doi": "b",
            "status": "COMPLETE",
            "ambiguous_or_missing_candidate_count": 0,
            "flow_count": 5,
        },
    ]
    report = compare(before, results)
    assert report["pass_gate"] is True
    assert not report["regressions"]


def test_compare_blocks_resolved_control_regression() -> None:
    before = {
        "control": {
            "status": "COMPLETE",
            "ambiguous_or_missing_candidate_count": 0,
            "flow_count": 5,
            "candidate_coverage": 1.0,
        }
    }
    results = [
        {
            "doi": "control",
            "status": "UNRESOLVED_INVENTORY",
            "ambiguous_or_missing_candidate_count": 1,
            "flow_count": 5,
        }
    ]
    report = compare(before, results)
    assert report["pass_gate"] is False
    assert report["regressions"][0]["reason"] == "resolved control regressed"
