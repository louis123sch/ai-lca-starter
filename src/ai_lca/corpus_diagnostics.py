from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BASELINE_MANIFEST = Path("benchmarks/corpus_baseline_v1_2026-08-11/papers.json")


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _paper_root(state_dir: Path, paper: dict[str, Any]) -> Path:
    raw = Path(str(paper.get("paper_dir") or ""))
    parts = list(raw.parts)
    if parts and parts[0] == "literature_state":
        raw = Path(*parts[1:])
    return state_dir / raw


def _failure_classes(
    qc: dict[str, Any],
    candidates: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
) -> set[str]:
    classes: set[str] = set()
    unresolved = int(qc.get("ambiguous_or_missing_candidate_count") or 0)
    process_failures = qc.get("process_failures") or {}
    failure_text = " ".join(" ".join(v or []) for v in process_failures.values()).casefold()
    candidate_count = int(qc.get("candidate_count") or len(candidates))
    modeled_count = int(qc.get("modeled_candidate_count") or 0)
    flow_count = int(qc.get("flow_count") or 0)
    coverage = float(qc.get("candidate_coverage") or 0.0)
    amount_cov = float(qc.get("amount_coverage") or 0.0)
    unit_cov = float(qc.get("unit_coverage") or 0.0)
    process_count = int(qc.get("process_count") or 0)

    table_rows = sum(1 for c in candidates if c.get("evidence_type") == "table_row")
    table_share = table_rows / len(candidates) if candidates else 0.0
    multi_assigned = sum(1 for a in assignments if len(set(a.get("process_ids") or [])) > 1)
    duplicate_process_refs = sum(
        1
        for a in assignments
        if len(a.get("process_ids") or []) != len(set(a.get("process_ids") or []))
    )

    if unresolved:
        classes.add("CANDIDATE_AMBIGUITY")
    if "no candidates assigned" in failure_text:
        classes.add("PROCESS_WITHOUT_ASSIGNED_CANDIDATES")
    if "remain ambiguous" in failure_text or "not reviewed" in failure_text:
        classes.add("FLOW_REVIEW_AMBIGUITY")
    if modeled_count and flow_count / max(1, modeled_count) < 0.5:
        classes.add("SPARSE_FLOW_EXTRACTION")
    if coverage < 0.95:
        classes.add("INCOMPLETE_CANDIDATE_REVIEW")
    if table_share >= 0.5 and unresolved:
        classes.add("TABLE_HEAVY_UNRESOLVED")
    if candidate_count >= 60 and unresolved:
        classes.add("LARGE_CANDIDATE_SET")
    if process_count >= 6 and unresolved:
        classes.add("MULTI_PROCESS_COMPLEXITY")
    if multi_assigned:
        classes.add("CROSS_PROCESS_ASSIGNMENT")
    if duplicate_process_refs:
        classes.add("DUPLICATE_PROCESS_ASSIGNMENT")
    if flow_count and amount_cov < 0.8:
        classes.add("LOW_AMOUNT_COVERAGE")
    if flow_count and unit_cov < 0.8:
        classes.add("LOW_UNIT_COVERAGE")
    if not flow_count:
        classes.add("ZERO_FLOW_EXTRACTION")
    if not classes and qc:
        classes.add("OTHER_QC_FAILURE")
    return classes


def analyze(
    state_dir: Path,
    manifest_path: Path = BASELINE_MANIFEST,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path, {}) or {}
    papers = manifest.get("papers") or []
    rows: list[dict[str, Any]] = []
    class_papers: dict[str, list[str]] = defaultdict(list)

    for paper in papers:
        root = _paper_root(state_dir, paper)
        extraction = root / "extraction"
        qc = _read_json(extraction / "qc.json", {}) or {}
        candidates = _read_json(extraction / "inventory_candidates.json", []) or []
        assignment_payload = _read_json(extraction / "assignments.json", {}) or {}
        assignments = assignment_payload.get("assignments") or []
        classes = (
            _failure_classes(qc, candidates, assignments)
            if paper.get("status") == "UNRESOLVED_INVENTORY"
            else set()
        )
        evidence_counts = Counter(c.get("evidence_type") or "unknown" for c in candidates)
        multi = sum(1 for a in assignments if len(set(a.get("process_ids") or [])) > 1)
        duplicates = sum(
            1
            for a in assignments
            if len(a.get("process_ids") or []) != len(set(a.get("process_ids") or []))
        )
        row = {
            "doi": paper.get("doi"),
            "title": paper.get("title"),
            "baseline_status": paper.get("status"),
            "process_count": int(qc.get("process_count") or 0),
            "candidate_count": int(qc.get("candidate_count") or len(candidates)),
            "modeled_candidate_count": int(qc.get("modeled_candidate_count") or 0),
            "candidate_coverage": float(qc.get("candidate_coverage") or 0.0),
            "ambiguous_or_missing_candidate_count": int(
                qc.get("ambiguous_or_missing_candidate_count") or 0
            ),
            "flow_count": int(qc.get("flow_count") or 0),
            "amount_coverage": float(qc.get("amount_coverage") or 0.0),
            "unit_coverage": float(qc.get("unit_coverage") or 0.0),
            "evidence_type_counts": dict(evidence_counts),
            "multi_process_assignment_count": multi,
            "duplicate_process_reference_count": duplicates,
            "failure_classes": sorted(classes),
        }
        rows.append(row)
        for cls in classes:
            class_papers[cls].append(str(paper.get("doi")))

    # Lower relative cost means the failure is more attractive to solve first.
    cost_weight = {
        "DUPLICATE_PROCESS_ASSIGNMENT": 0.25,
        "PROCESS_WITHOUT_ASSIGNED_CANDIDATES": 0.5,
        "INCOMPLETE_CANDIDATE_REVIEW": 0.5,
        "TABLE_HEAVY_UNRESOLVED": 0.75,
        "CANDIDATE_AMBIGUITY": 1.0,
        "FLOW_REVIEW_AMBIGUITY": 1.0,
        "SPARSE_FLOW_EXTRACTION": 1.0,
        "CROSS_PROCESS_ASSIGNMENT": 1.25,
        "LARGE_CANDIDATE_SET": 1.5,
        "MULTI_PROCESS_COMPLEXITY": 1.5,
        "LOW_AMOUNT_COVERAGE": 1.0,
        "LOW_UNIT_COVERAGE": 1.0,
        "ZERO_FLOW_EXTRACTION": 1.0,
        "OTHER_QC_FAILURE": 2.0,
    }
    ranked = []
    for cls, dois in class_papers.items():
        cost = cost_weight.get(cls, 1.5)
        ranked.append(
            {
                "failure_class": cls,
                "affected_papers": len(set(dois)),
                "estimated_relative_repair_cost": cost,
                "priority_score": round(len(set(dois)) / cost, 3),
                "dois": sorted(set(dois)),
            }
        )
    ranked.sort(
        key=lambda x: (-x["priority_score"], -x["affected_papers"], x["failure_class"])
    )

    unresolved_rows = [r for r in rows if r["baseline_status"] == "UNRESOLVED_INVENTORY"]
    complete_rows = [r for r in rows if r["baseline_status"] == "COMPLETE"]
    canary: list[str] = []
    by_doi = {r["doi"]: r for r in rows}

    for group in ranked[:6]:
        representatives = [by_doi[d] for d in group["dois"] if d in by_doi]
        representatives.sort(
            key=lambda r: (
                -r["ambiguous_or_missing_candidate_count"],
                -r["candidate_count"],
                r["doi"],
            )
        )
        if representatives:
            doi = representatives[0]["doi"]
            if doi not in canary:
                canary.append(doi)

    complete_rows.sort(key=lambda r: (-r["candidate_count"], -r["process_count"], r["doi"]))
    for row in complete_rows:
        complete_in_canary = sum(
            1 for doi in canary if by_doi[doi]["baseline_status"] == "COMPLETE"
        )
        if complete_in_canary >= 2:
            break
        if row["doi"] not in canary:
            canary.append(row["doi"])

    unresolved_rows.sort(
        key=lambda r: (
            -r["ambiguous_or_missing_candidate_count"],
            -r["candidate_count"],
            r["doi"],
        )
    )
    for row in unresolved_rows:
        if len(canary) >= 8:
            break
        if row["doi"] not in canary:
            canary.append(row["doi"])

    return {
        "baseline_id": manifest.get("baseline_id"),
        "state_dir": str(state_dir),
        "paper_count": len(rows),
        "status_counts": dict(Counter(r["baseline_status"] for r in rows)),
        "failure_class_ranking": ranked,
        "recommended_canary_dois": canary[:10],
        "papers": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-token diagnostics for the frozen AI-LCA development corpus."
    )
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=BASELINE_MANIFEST)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/corpus_diagnostics.json")
    )
    args = parser.parse_args()
    report = analyze(args.state_dir, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = {
        key: report[key]
        for key in (
            "baseline_id",
            "paper_count",
            "status_counts",
            "failure_class_ranking",
            "recommended_canary_dois",
        )
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
