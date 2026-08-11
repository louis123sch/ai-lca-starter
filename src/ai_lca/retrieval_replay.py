from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .autonomous_literature import (
    ASSIGN_SYSTEM_PROMPT,
    ApiRunner,
    Budget,
    CandidateAssignment,
    CandidateAssignmentBatch,
    RunConfig,
    StateStore,
    _validate_assignments,
    _write_json,
)
from .corpus_diagnostics import BASELINE_MANIFEST, load_baseline_papers
from .evidence_router import (
    build_structure_evidence,
    normalised_contains,
    partition_inventory_candidates,
    route_inventory_candidates,
    routed_candidate_payload,
)
from .inventory_replay import (
    TargetedReplayProcessor,
    _paper_dir,
    _read,
    _snapshot_qc,
    compare,
    select_dois,
)
from .jats import InventoryCandidate, parse_jats_file


def _collect_evidence_texts(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "evidence_text" and isinstance(item, str) and item.strip():
                found.append(item.strip())
            else:
                found.extend(_collect_evidence_texts(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_evidence_texts(item))
    return found


def audit_router(state_dir: Path, manifest: Path, dois: list[str]) -> dict[str, Any]:
    """Measure retrieval safety against the frozen baseline before API calls."""

    _, papers = load_baseline_papers(state_dir, manifest)
    by_doi = {paper["doi"]: paper for paper in papers}
    per_paper: list[dict[str, Any]] = []
    total_modeled = 0
    total_excluded = 0
    total_excluded_modeled = 0
    total_structure_evidence = 0
    total_structure_evidence_retrieved = 0

    for doi in dois:
        paper = by_doi.get(doi)
        if not paper:
            continue
        paper_dir = _paper_dir(state_dir, paper)
        raw_candidates = _read(paper_dir / "extraction" / "inventory_candidates.json", []) or []
        candidates = [InventoryCandidate(**item) for item in raw_candidates]
        retained, excluded, routes = partition_inventory_candidates(candidates)
        excluded_ids = {candidate.candidate_id for candidate in excluded}

        raw_assignments = _read(paper_dir / "extraction" / "assignments.json", {}) or {}
        baseline = CandidateAssignmentBatch.model_validate(raw_assignments)
        modeled_ids = {
            assignment.candidate_id
            for assignment in baseline.assignments
            if assignment.disposition == "modeled_inventory"
        }
        lost = sorted(modeled_ids & excluded_ids)

        structure_total = 0
        structure_hit = 0
        source_path = paper_dir / "source" / "article.xml"
        structure_path = paper_dir / "extraction" / "structure.json"
        if source_path.exists() and structure_path.exists():
            try:
                doc = parse_jats_file(source_path, expected_doi=doi)
                pack = build_structure_evidence(doc)
                baseline_structure = _read(structure_path, {}) or {}
                evidence_texts = list(dict.fromkeys(_collect_evidence_texts(baseline_structure)))
                structure_total = len(evidence_texts)
                structure_hit = sum(normalised_contains(pack.text, text) for text in evidence_texts)
            except Exception:
                # Structure retrieval recall is diagnostic only in this first replay;
                # inventory safety remains the hard gate.
                structure_total = 0
                structure_hit = 0

        total_modeled += len(modeled_ids)
        total_excluded += len(excluded)
        total_excluded_modeled += len(lost)
        total_structure_evidence += structure_total
        total_structure_evidence_retrieved += structure_hit
        per_paper.append(
            {
                "doi": doi,
                "candidate_count": len(candidates),
                "retained_for_reasoning": len(retained),
                "safe_excluded_from_reasoning": len(excluded),
                "baseline_modeled_candidate_count": len(modeled_ids),
                "excluded_baseline_modeled_candidate_ids": lost,
                "structure_evidence_count": structure_total,
                "structure_evidence_retrieved": structure_hit,
                "route_counts": {
                    label: sum(route.label == label for route in routes)
                    for label in ("foreground_lci", "structure", "modelling_assumption", "lcia_result", "uncertain")
                },
            }
        )

    inventory_recall = 1.0 if total_modeled == 0 else (total_modeled - total_excluded_modeled) / total_modeled
    structure_recall = (
        1.0
        if total_structure_evidence == 0
        else total_structure_evidence_retrieved / total_structure_evidence
    )
    return {
        "paper_count": len(per_paper),
        "baseline_modeled_candidate_count": total_modeled,
        "safe_excluded_candidate_count": total_excluded,
        "excluded_baseline_modeled_candidate_count": total_excluded_modeled,
        "inventory_recall_against_baseline": inventory_recall,
        "structure_evidence_recall_against_baseline": structure_recall,
        "inventory_safety_pass": total_excluded_modeled == 0,
        "per_paper": per_paper,
    }


class RoutedTargetedReplayProcessor(TargetedReplayProcessor):
    """Replay only unresolved baseline evidence after conservative routing."""

    def _assign(self, doi, source_hash, structure, candidates, paper_dir):  # noqa: ANN001
        path = paper_dir / "extraction" / "assignments.json"
        payload = _read(path, {}) or {}
        baseline = CandidateAssignmentBatch.model_validate(payload)
        allowed = {process.process_id for process in structure.processes}
        accepted, missing = _validate_assignments(candidates, baseline, allowed)
        unresolved_ids = set(missing) | {
            assignment.candidate_id
            for assignment in accepted
            if assignment.disposition == "ambiguous"
        }
        if not unresolved_ids:
            return accepted, missing

        unresolved = [candidate for candidate in candidates if candidate.candidate_id in unresolved_ids]
        retained, excluded, routes = partition_inventory_candidates(unresolved)
        route_by_id = {route.candidate_id: route for route in routes}
        _write_json(
            paper_dir / "extraction" / "retrieval" / "replay_candidate_routes.json",
            {
                "unresolved_candidate_count": len(unresolved),
                "retained_for_reasoning": len(retained),
                "safe_excluded_from_reasoning": len(excluded),
                "routes": [route.as_dict() for route in routes],
            },
        )

        deterministic = [
            CandidateAssignment(
                candidate_id=candidate.candidate_id,
                disposition="not_inventory",
                process_ids=[],
                rationale=(
                    "High-confidence retrieval router identified LCIA/result evidence with no competing inventory "
                    "signal; source candidate remains preserved. "
                    + "; ".join(route_by_id[candidate.candidate_id].reasons)
                ),
            )
            for candidate in excluded
        ]

        repair_assignments = []
        if retained:
            prompt = (
                "LOCKED PROCESSES:\n"
                + json.dumps(
                    [
                        {"process_id": process.process_id, "name": process.name, "stage": process.stage}
                        for process in structure.processes
                    ],
                    ensure_ascii=False,
                )
                + "\n\nREVIEW ONLY THESE PREVIOUSLY UNRESOLVED, ROUTED CANDIDATES:\n"
                + json.dumps(
                    [routed_candidate_payload(candidate, route_by_id[candidate.candidate_id]) for candidate in retained],
                    ensure_ascii=False,
                )
                + "\n\nThe route is advisory. Override it if the source evidence requires it."
            )
            repair = self.api.parse(
                doi=doi,
                stage="candidate_assignment_retrieval_replay",
                model=self.config.screen_model,
                reasoning_effort=self.config.screen_reasoning,
                system_prompt=ASSIGN_SYSTEM_PROMPT,
                user_prompt=prompt,
                response_format=CandidateAssignmentBatch,
            )
            repair_assignments = repair.assignments

        kept = [assignment for assignment in accepted if assignment.candidate_id not in unresolved_ids]
        return _validate_assignments(
            candidates,
            CandidateAssignmentBatch(assignments=kept + deterministic + repair_assignments),
            allowed,
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    dois = select_dois(
        state_dir=args.state_dir,
        manifest=args.manifest,
        subset=args.subset,
        cohort=args.cohort,
    )
    before = _snapshot_qc(args.state_dir, args.manifest, dois)
    audit = audit_router(args.state_dir, args.manifest, dois)
    if not audit["inventory_safety_pass"]:
        report = {
            "subset": args.subset,
            "cohort": args.cohort,
            "dois": dois,
            "before": before,
            "router_audit": audit,
            "results": [],
            "comparison": {"improved_papers": [], "regressions": [{"reason": "router safety audit failed"}], "unchanged_papers": [], "pass_gate": False},
            "usage": {"calls_this_run": 0, "tokens_this_run": 0, "estimated_cost_this_run_usd": 0.0},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return report

    config = RunConfig(
        state_dir=args.state_dir,
        screen_model=args.screen_model,
        core_model=args.core_model,
        max_concurrent_requests=args.max_concurrent_requests,
        max_process_workers=args.max_process_workers,
        max_paper_workers=1,
        max_total_calls=args.max_total_calls,
        max_calls_per_paper=args.max_calls_per_paper,
        max_total_tokens=args.max_total_tokens,
        max_estimated_cost_usd=args.max_estimated_cost_usd,
        max_repair_calls_per_process=1,
    )
    store = StateStore(args.state_dir)
    budget = Budget(config, store)
    processor = RoutedTargetedReplayProcessor(
        config,
        store,
        ApiRunner(config, store, budget),
        os.environ.get("SPRINGER_API_KEY", ""),
    )
    results = [processor.process(doi) for doi in dois]
    comparison = compare(before, results)
    report = {
        "subset": args.subset,
        "cohort": args.cohort,
        "dois": dois,
        "before": before,
        "router_audit": audit,
        "results": results,
        "comparison": comparison,
        "usage": budget.summary(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "subset": args.subset,
                "paper_count": len(dois),
                "router_audit": {key: value for key, value in audit.items() if key != "per_paper"},
                "comparison": comparison,
                "usage": budget.summary(),
            },
            indent=2,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval-before-reasoning replay on the frozen corpus.")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=BASELINE_MANIFEST)
    parser.add_argument("--subset", choices=["canary", "cohort", "full"], default="canary")
    parser.add_argument("--cohort")
    parser.add_argument("--screen-model", default=os.getenv("OPENAI_SCREEN_MODEL", "gpt-5-nano"))
    parser.add_argument("--core-model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    parser.add_argument("--max-concurrent-requests", type=int, default=3)
    parser.add_argument("--max-process-workers", type=int, default=3)
    parser.add_argument("--max-total-calls", type=int, default=40)
    parser.add_argument("--max-calls-per-paper", type=int, default=8)
    parser.add_argument("--max-total-tokens", type=int, default=750_000)
    parser.add_argument("--max-estimated-cost-usd", type=float, default=0.75)
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval_replay.json"))
    args = parser.parse_args()
    report = run(args)
    if not report["router_audit"]["inventory_safety_pass"] or not report["comparison"]["pass_gate"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
