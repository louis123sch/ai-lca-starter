from __future__ import annotations

import json
from pathlib import Path

from ai_lca.corpus_diagnostics import analyze


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_diagnostics_rank_recurring_failures_and_keep_resolved_control(tmp_path: Path) -> None:
    state = tmp_path / "state"
    manifest = tmp_path / "papers.json"
    papers = {
        "baseline_id": "test",
        "papers": [
            {
                "doi": "10.test/fail",
                "title": "fail",
                "status": "UNRESOLVED_INVENTORY",
                "paper_dir": "literature_state/corpus/fail",
            },
            {
                "doi": "10.test/pass",
                "title": "pass",
                "status": "COMPLETE",
                "paper_dir": "literature_state/corpus/pass",
            },
        ],
    }
    _write(manifest, papers)

    fail = state / "corpus" / "fail" / "extraction"
    _write(
        fail / "qc.json",
        {
            "process_count": 2,
            "candidate_count": 4,
            "modeled_candidate_count": 2,
            "candidate_coverage": 0.5,
            "ambiguous_or_missing_candidate_count": 1,
            "flow_count": 0,
            "amount_coverage": 0.0,
            "unit_coverage": 0.0,
            "process_failures": {"p1": ["1 candidate(s) remain ambiguous"]},
        },
    )
    _write(
        fail / "inventory_candidates.json",
        [
            {"candidate_id": "a", "evidence_type": "table_row"},
            {"candidate_id": "b", "evidence_type": "table_row"},
            {"candidate_id": "c", "evidence_type": "table_row"},
            {"candidate_id": "d", "evidence_type": "section_statement"},
        ],
    )
    _write(
        fail / "assignments.json",
        {"assignments": [{"candidate_id": "a", "process_ids": ["p1", "p2"]}]},
    )

    passed = state / "corpus" / "pass" / "extraction"
    _write(
        passed / "qc.json",
        {
            "process_count": 1,
            "candidate_count": 2,
            "modeled_candidate_count": 1,
            "candidate_coverage": 1.0,
            "ambiguous_or_missing_candidate_count": 0,
            "flow_count": 1,
            "amount_coverage": 1.0,
            "unit_coverage": 1.0,
            "process_failures": {},
        },
    )
    _write(passed / "inventory_candidates.json", [])
    _write(passed / "assignments.json", {"assignments": []})

    report = analyze(state, manifest)
    assert report["paper_count"] == 2
    assert report["status_counts"] == {"UNRESOLVED_INVENTORY": 1, "COMPLETE": 1}
    failing = next(p for p in report["papers"] if p["doi"] == "10.test/fail")
    assert "TABLE_HEAVY_UNRESOLVED" in failing["failure_classes"]
    assert "CANDIDATE_AMBIGUITY" in failing["failure_classes"]
    assert "ZERO_FLOW_EXTRACTION" in failing["failure_classes"]
    assert "10.test/pass" in report["recommended_canary_dois"]
