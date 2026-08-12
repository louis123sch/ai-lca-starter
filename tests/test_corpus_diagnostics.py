from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ai_lca.corpus_diagnostics import analyze, load_baseline_papers


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


def test_exact_baseline_is_loaded_from_sqlite_and_count_checked(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    conn = sqlite3.connect(state / "phase1.sqlite3")
    conn.execute(
        "CREATE TABLE papers (doi TEXT, title TEXT, status TEXT, source_hash TEXT, paper_dir TEXT, last_error TEXT)"
    )
    rows = [
        ("10.test/u", "unresolved", "UNRESOLVED_INVENTORY", "u", "literature_state/corpus/u", None),
        ("10.test/c", "complete", "COMPLETE", "c", "literature_state/corpus/c", None),
        ("10.test/reject", "reject", "SCREEN_REJECTED", "r", "literature_state/corpus/r", None),
    ]
    conn.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()

    selection = tmp_path / "selection.json"
    _write(
        selection,
        {
            "baseline_id": "exact-test",
            "sqlite_file": "phase1.sqlite3",
            "selection_sql": "SELECT doi,title,status,source_hash,paper_dir,last_error FROM papers WHERE status IN ('COMPLETE','UNRESOLVED_INVENTORY') ORDER BY status, doi",
            "expected_status_counts": {"UNRESOLVED_INVENTORY": 1, "COMPLETE": 1},
            "expected_total": 2,
        },
    )
    baseline_id, papers = load_baseline_papers(state, selection)
    assert baseline_id == "exact-test"
    assert {p["doi"] for p in papers} == {"10.test/u", "10.test/c"}


def test_exact_baseline_rejects_count_drift(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    conn = sqlite3.connect(state / "phase1.sqlite3")
    conn.execute(
        "CREATE TABLE papers (doi TEXT, title TEXT, status TEXT, source_hash TEXT, paper_dir TEXT, last_error TEXT)"
    )
    conn.execute(
        "INSERT INTO papers VALUES (?,?,?,?,?,?)",
        ("10.test/u", "unresolved", "UNRESOLVED_INVENTORY", "u", "literature_state/corpus/u", None),
    )
    conn.commit()
    conn.close()
    selection = tmp_path / "selection.json"
    _write(
        selection,
        {
            "baseline_id": "exact-test",
            "sqlite_file": "phase1.sqlite3",
            "expected_status_counts": {"UNRESOLVED_INVENTORY": 1, "COMPLETE": 1},
            "expected_total": 2,
        },
    )
    with pytest.raises(ValueError, match="count drift"):
        load_baseline_papers(state, selection)


def test_zero_modeled_candidates_not_mislabelled_incomplete_review(tmp_path: Path) -> None:
    state = tmp_path / "state"
    manifest = tmp_path / "papers.json"
    _write(
        manifest,
        {
            "baseline_id": "test",
            "papers": [
                {
                    "doi": "10.test/zero",
                    "title": "zero",
                    "status": "UNRESOLVED_INVENTORY",
                    "paper_dir": "literature_state/corpus/zero",
                }
            ],
        },
    )
    extraction = state / "corpus" / "zero" / "extraction"
    _write(
        extraction / "qc.json",
        {
            "process_count": 1,
            "candidate_count": 3,
            "modeled_candidate_count": 0,
            "candidate_coverage": 0.0,
            "ambiguous_or_missing_candidate_count": 0,
            "flow_count": 0,
            "process_failures": {},
        },
    )
    _write(extraction / "inventory_candidates.json", [])
    _write(extraction / "assignments.json", {"assignments": []})
    report = analyze(state, manifest)
    row = report["papers"][0]
    assert "INCOMPLETE_CANDIDATE_REVIEW" not in row["failure_classes"]
    assert "ZERO_FLOW_EXTRACTION" in row["failure_classes"]
