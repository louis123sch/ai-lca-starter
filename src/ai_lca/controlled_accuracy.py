from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .autonomous_literature import (
    ASSIGN_SYSTEM_PROMPT,
    ApiRunner,
    Budget,
    CandidateAssignmentBatch,
    RunConfig,
    StateStore,
    _candidate_payload,
    _load_model,
)
from .corpus_diagnostics import BASELINE_MANIFEST, load_baseline_papers
from .inventory_replay import _paper_dir, _read
from .jats import InventoryCandidate
from .models import ForegroundStructure

SELECTION = Path("benchmarks/accuracy_iteration_v1/selection.json")

_RESULT = [re.compile(x, re.I) for x in (
    r"\bgwp\b", r"global warming", r"impact categor", r"environmental impact",
    r"life cycle impact", r"\blcia\b", r"acidification", r"eutrophication",
    r"toxicity", r"ozone depletion", r"photochemical", r"characteri[sz]",
    r"kg\s*(?:co2|co₂)[- ]?(?:eq|equiv)",
)]
_METHOD = [re.compile(x, re.I) for x in (
    r"\ballocation\b", r"system expansion", r"\bsubstitution\b",
    r"attribution method", r"normalization", r"weighting",
)]
_INVENTORY = [re.compile(x, re.I) for x in (
    r"\binventory\b", r"\blci\b", r"\binput(?:s)?\b", r"\boutput(?:s)?\b",
    r"consumption", r"demand", r"feedstock", r"raw material", r"electricity",
    r"natural gas", r"water", r"steam", r"diesel", r"fuel", r"steel",
    r"alumin", r"nickel", r"copper", r"cement", r"concrete", r"transport",
    r"freight", r"\bwaste\b", r"\bemission(?:s)?\b",
)]
_RESULT_CONTEXT = re.compile(
    r"impact result|total environmental impact|impact categor|life cycle impact assessment|GWP per",
    re.I,
)
_METHOD_CONTEXT = [re.compile(x, re.I) for x in (
    r"attribution method", r"allocation method", r"advantages and limitations",
    r"multifunctionality",
)]
_IMPACT_FLOW_NAME = re.compile(
    r"\b(?:gwp|climate change|human toxicity|acidification|eutrophication|ozone depletion|"
    r"photochemical|metal depletion|water depletion|fossil depletion|ecotoxicity|"
    r"resource depletion|impact score|environmental impact)\b",
    re.I,
)

ADDENDUM = """
Controlled accuracy experiment. The supplied rows were previously called modeled
inventory and then survived into the final foreground inventory, but they come from
explicit LCIA/result tables. Re-review ONLY the supplied rows.

LCIA result rows such as GWP, climate-change score, human toxicity, depletion,
acidification, eutrophication, characterized impact scores, or other midpoint/impact
results are NOT LCI exchanges. They must be not_inventory with process_ids=[]. Do not
apply this rule to a direct source-supported elementary emission in an actual LCI
input/output table. Do not alter the locked graph or invent any evidence, quantity,
unit, process, or dataset. Return exactly one assignment for every supplied candidate.
"""


def _n(patterns: list[re.Pattern[str]], text: str) -> int:
    return sum(bool(p.search(text or "")) for p in patterns)


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def candidate_risks(candidate: dict[str, Any], assignment: dict[str, Any]) -> list[str]:
    """Intermediate high-confidence diagnostics; never changes extraction."""
    if assignment.get("disposition") != "modeled_inventory" or candidate.get("evidence_type") != "table_row":
        return []
    evidence = str(candidate.get("evidence_text") or "")
    context = " ".join(str(candidate.get(k) or "") for k in ("context", "table", "source_location"))
    text = evidence + "\n" + context
    result, method, inv = _n(_RESULT, text), _n(_METHOD, text), _n(_INVENTORY, text)
    risks: list[str] = []
    if result >= 2 and _RESULT_CONTEXT.search(context) and inv <= 1:
        risks.append("MODELED_LCIA_RESULT_TABLE_RISK")
    if method >= 2 and (_n(_METHOD_CONTEXT, context) >= 1 or inv == 0):
        risks.append("MODELED_METHOD_TABLE_RISK")
    return risks


def final_flow_risk(
    flow: dict[str, Any],
    candidate: dict[str, Any],
    assignment: dict[str, Any],
) -> bool:
    """True only for an explicit impact-result table row that became a final LCI flow."""
    if assignment.get("disposition") != "modeled_inventory":
        return False
    if candidate.get("evidence_type") != "table_row":
        return False
    if not _IMPACT_FLOW_NAME.search(str(flow.get("name") or "")):
        return False
    return bool(_RESULT_CONTEXT.search(str(candidate.get("context") or "")))


def _root(state: Path, paper: dict[str, Any]) -> Path:
    return _paper_dir(state, paper) / "extraction"


def _baseline_data(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidates = _read(root / "inventory_candidates.json", []) or []
    assignments = (_read(root / "assignments.json", {}) or {}).get("assignments") or []
    inventory = _read(root / "inventory.json", {}) or {}
    return candidates, assignments, inventory


def final_lcia_targets(root: Path) -> tuple[list[str], list[dict[str, Any]]]:
    candidates, assignments, inventory = _baseline_data(root)
    amap = {str(a.get("candidate_id")): a for a in assignments if a.get("candidate_id")}
    lookup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        lookup[(_norm(candidate.get("evidence_text")), _norm(candidate.get("table")))].append(candidate)

    target_ids: set[str] = set()
    examples: list[dict[str, Any]] = []
    for flow in inventory.get("flows", []) or []:
        evidence = flow.get("evidence") or {}
        key = (_norm(evidence.get("evidence_text")), _norm(evidence.get("table")))
        for candidate in lookup.get(key, []):
            cid = str(candidate.get("candidate_id") or "")
            if cid and final_flow_risk(flow, candidate, amap.get(cid, {})):
                target_ids.add(cid)
                examples.append({
                    "candidate_id": cid,
                    "flow_name": flow.get("name"),
                    "amount": flow.get("amount"),
                    "unit": flow.get("unit"),
                    "table": candidate.get("table"),
                    "source_location": candidate.get("source_location"),
                })
    return sorted(target_ids), examples


def diagnose(state: Path, manifest: Path = BASELINE_MANIFEST) -> dict[str, Any]:
    baseline_id, papers = load_baseline_papers(state, manifest)
    counts: Counter[str] = Counter()
    affected: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    for paper in papers:
        root = _root(state, paper)
        qc = _read(root / "qc.json", {}) or {}
        candidates, assignments, _ = _baseline_data(root)
        cmap = {str(c.get("candidate_id")): c for c in candidates if c.get("candidate_id")}
        signals: Counter[str] = Counter()

        target_ids, final_examples = final_lcia_targets(root)
        if target_ids:
            signals["FINAL_LCIA_RESULT_AS_INVENTORY_FLOW"] += len(target_ids)
            examples["FINAL_LCIA_RESULT_AS_INVENTORY_FLOW"].extend(final_examples[:10])

        for assignment in assignments:
            for signal in candidate_risks(cmap.get(str(assignment.get("candidate_id")), {}), assignment):
                signals[signal] += 1
                if len(examples[signal]) < 10:
                    candidate = cmap[str(assignment["candidate_id"])]
                    examples[signal].append({
                        "candidate_id": assignment["candidate_id"],
                        "table": candidate.get("table"),
                        "source_location": candidate.get("source_location"),
                        "snippet": " ".join(str(candidate.get("evidence_text") or "").split())[:400],
                    })

        unresolved = int(qc.get("ambiguous_or_missing_candidate_count") or 0)
        flows = int(qc.get("flow_count") or 0)
        if unresolved:
            signals["UNRESOLVED_CANDIDATES"] += unresolved
        if not flows:
            signals["ZERO_FLOW_EXTRACTION"] += 1
        if float(qc.get("candidate_coverage") or 0) < .95:
            signals["LOW_CANDIDATE_COVERAGE"] += 1
        if flows and float(qc.get("amount_coverage") or 0) < .8:
            signals["LOW_AMOUNT_COVERAGE"] += 1
        if flows and float(qc.get("unit_coverage") or 0) < .8:
            signals["LOW_UNIT_COVERAGE"] += 1

        doi = str(paper.get("doi") or "")
        for signal, number in signals.items():
            counts[signal] += number
            affected[signal].add(doi)
        rows.append({
            "doi": doi,
            "title": paper.get("title"),
            "baseline_status": paper.get("status"),
            "signals": dict(signals),
        })

    confidence = {
        "FINAL_LCIA_RESULT_AS_INVENTORY_FLOW": 1.0,
        "MODELED_LCIA_RESULT_TABLE_RISK": 1.0,
        "MODELED_METHOD_TABLE_RISK": 1.0,
        "UNRESOLVED_CANDIDATES": .75,
        "ZERO_FLOW_EXTRACTION": .85,
        "LOW_CANDIDATE_COVERAGE": .8,
        "LOW_AMOUNT_COVERAGE": .6,
        "LOW_UNIT_COVERAGE": .6,
    }
    ranked = []
    for signal, dois in affected.items():
        severity = 3.0 if signal == "FINAL_LCIA_RESULT_AS_INVENTORY_FLOW" else (2.0 if signal.startswith("MODELED_") else 1.0)
        ranked.append({
            "failure_signal": signal,
            "affected_papers": len(dois),
            "signal_count": counts[signal],
            "priority_score": round(len(dois) * confidence.get(signal, .5) * severity, 3),
            "dois": sorted(dois),
            "examples": examples.get(signal, [])[:20],
        })
    ranked.sort(key=lambda x: (-x["priority_score"], -x["affected_papers"], x["failure_signal"]))

    final_dois = sorted(affected.get("FINAL_LCIA_RESULT_AS_INVENTORY_FLOW", set()))
    return {
        "baseline_id": baseline_id,
        "paper_count": len(rows),
        "status_counts": dict(Counter(str(p["status"]) for p in papers)),
        "failure_signal_ranking": ranked,
        "recommended_iteration_1_target": {
            "failure_class": "FINAL_LCIA_RESULT_AS_INVENTORY_FLOW",
            "mechanism": "Explicit LCIA result-table rows survive into the final foreground inventory as if they were LCI exchanges.",
            "affected_papers": len(final_dois),
            "dois": final_dois,
            "reason": "These are source-verifiable final-output errors, so they are a stronger accuracy target than intermediate ambiguity alone.",
        },
        "papers": rows,
    }


def _review_targets(
    *,
    doi: str,
    root: Path,
    api: ApiRunner,
    config: RunConfig,
) -> dict[str, Any]:
    candidates_raw, assignments_raw, inventory = _baseline_data(root)
    targets, target_examples = final_lcia_targets(root)
    baseline_map = {str(a["candidate_id"]): a for a in assignments_raw if a.get("candidate_id")}
    if not targets:
        return {
            "doi": doi,
            "target_count": 0,
            "baseline_target_flow_count": 0,
            "corrected_target_count": 0,
            "corrected_target_ids": [],
            "outside_target_changes": [],
            "response_assignments": [],
            "pass": True,
            "reasons": [],
            "control_only": True,
        }

    candidates = [InventoryCandidate(**item) for item in candidates_raw]
    cmap = {candidate.candidate_id: candidate for candidate in candidates}
    structure = _load_model(root / "structure.json", ForegroundStructure)
    subset = [cmap[cid] for cid in targets]
    prompt = (
        "LOCKED PROCESSES:\n"
        + json.dumps([
            {"process_id": process.process_id, "name": process.name, "stage": process.stage}
            for process in structure.processes
        ], ensure_ascii=False)
        + "\n\nREVIEW ONLY THESE FINAL-OUTPUT ERROR CANDIDATES:\n"
        + json.dumps(_candidate_payload(subset), ensure_ascii=False)
    )
    repair = api.parse(
        doi=doi,
        stage="candidate_assignment_accuracy_v1",
        model=config.screen_model,
        reasoning_effort=config.screen_reasoning,
        system_prompt=ASSIGN_SYSTEM_PROMPT + "\n\n" + ADDENDUM,
        user_prompt=prompt,
        response_format=CandidateAssignmentBatch,
    )

    response_rows = [assignment.model_dump() for assignment in repair.assignments]
    response_ids = [str(row.get("candidate_id") or "") for row in response_rows]
    response_map = {str(row["candidate_id"]): row for row in response_rows if row.get("candidate_id") in set(targets)}
    duplicates = sorted({cid for cid in response_ids if cid and response_ids.count(cid) > 1})
    missing = sorted(set(targets) - set(response_map))
    unexpected = sorted(set(response_ids) - set(targets))

    # Preserve every non-target baseline assignment exactly. Only target rows can be replaced.
    after_map = json.loads(json.dumps(baseline_map))
    for cid, row in response_map.items():
        after_map[cid] = row
    outside = sorted(
        cid for cid, old in baseline_map.items()
        if cid not in set(targets) and after_map.get(cid) != old
    )
    corrected = sorted(
        cid for cid in targets
        if after_map.get(cid, {}).get("disposition") == "not_inventory"
        and not (after_map.get(cid, {}).get("process_ids") or [])
    )
    ambiguous = sorted(
        cid for cid in targets
        if after_map.get(cid, {}).get("disposition") == "ambiguous"
    )

    reasons: list[str] = []
    if missing:
        reasons.append("model omitted target candidate(s)")
    if unexpected:
        reasons.append("model returned candidate(s) outside target set")
    if duplicates:
        reasons.append("model returned duplicate target assignment(s)")
    if len(corrected) != len(targets):
        reasons.append("not all final LCIA false-positive targets were corrected")
    if ambiguous:
        reasons.append("target candidate(s) became ambiguous")
    if outside:
        reasons.append("assignment outside target class changed")

    return {
        "doi": doi,
        "target_count": len(targets),
        "baseline_target_flow_count": len(target_examples),
        "target_examples": target_examples,
        "corrected_target_count": len(corrected),
        "corrected_target_ids": corrected,
        "projected_false_positive_flows_removed": len(target_examples) if len(corrected) == len(targets) else 0,
        "outside_target_changes": outside,
        "missing_response_ids": missing,
        "unexpected_response_ids": unexpected,
        "duplicate_response_ids": duplicates,
        "ambiguous_target_ids": ambiguous,
        "response_assignments": response_rows,
        "pass": not reasons,
        "reasons": reasons,
        "control_only": False,
    }


def run(
    state: Path,
    scope: str,
    manifest: Path,
    selection: Path,
    max_calls: int,
    max_tokens: int,
    max_cost: float,
) -> dict[str, Any]:
    selected = _read(selection, {}) or {}
    dois = [str(selected["single_doi"])] if scope == "single" else [str(x) for x in selected.get("canary_dois") or []]
    _, papers = load_baseline_papers(state, manifest)
    by_doi = {str(p["doi"]): p for p in papers}

    config = RunConfig(
        state_dir=state,
        screen_model=os.getenv("OPENAI_SCREEN_MODEL", "gpt-5-nano"),
        core_model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        max_concurrent_requests=1,
        max_process_workers=1,
        max_paper_workers=1,
        max_total_calls=max_calls,
        max_calls_per_paper=1,
        max_total_tokens=max_tokens,
        max_estimated_cost_usd=max_cost,
        max_repair_calls_per_process=0,
        infrastructure_retries=1,
    )
    store = StateStore(state)
    budget = Budget(config, store)
    api = ApiRunner(config, store, budget)

    paper_results = []
    for doi in dois:
        paper = by_doi.get(doi)
        if not paper:
            paper_results.append({"doi": doi, "pass": False, "reasons": ["DOI missing from frozen baseline"]})
            continue
        paper_results.append(_review_targets(doi=doi, root=_root(state, paper), api=api, config=config))

    target_papers = [paper for paper in paper_results if int(paper.get("target_count") or 0) > 0]
    reasons: list[str] = []
    if scope == "single" and not target_papers:
        reasons.append("single selected paper had no final LCIA target")
    if any(not paper.get("pass") for paper in paper_results):
        reasons.append("one or more selected papers failed the controlled gate")

    comparison = {
        "pass_gate": not reasons,
        "reasons": reasons,
        "papers": paper_results,
        "selected_paper_count": len(paper_results),
        "target_paper_count": len(target_papers),
        "target_total": sum(int(paper.get("target_count") or 0) for paper in paper_results),
        "corrected_total": sum(int(paper.get("corrected_target_count") or 0) for paper in paper_results),
        "projected_false_positive_flows_removed": sum(int(paper.get("projected_false_positive_flows_removed") or 0) for paper in paper_results),
    }
    return {
        "experiment": "accuracy_iteration_v1_final_lcia_result_flows",
        "scope": scope,
        "dois": dois,
        "comparison": comparison,
        "usage": budget.summary(),
        "limits": {
            "max_total_calls": max_calls,
            "max_calls_per_paper": 1,
            "max_total_tokens": max_tokens,
            "max_estimated_cost_usd": max_cost,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    diagnostic = sub.add_parser("diagnose")
    diagnostic.add_argument("--state-dir", type=Path, required=True)
    diagnostic.add_argument("--manifest", type=Path, default=BASELINE_MANIFEST)
    diagnostic.add_argument("--output", type=Path, required=True)
    experiment = sub.add_parser("run")
    experiment.add_argument("--state-dir", type=Path, required=True)
    experiment.add_argument("--manifest", type=Path, default=BASELINE_MANIFEST)
    experiment.add_argument("--selection", type=Path, default=SELECTION)
    experiment.add_argument("--scope", choices=["single", "canary"], default="single")
    experiment.add_argument("--max-total-calls", type=int, default=10)
    experiment.add_argument("--max-total-tokens", type=int, default=200000)
    experiment.add_argument("--max-estimated-cost-usd", type=float, default=.10)
    experiment.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "diagnose":
        report = diagnose(args.state_dir, args.manifest)
    else:
        report = run(
            args.state_dir,
            args.scope,
            args.manifest,
            args.selection,
            args.max_total_calls,
            args.max_total_tokens,
            args.max_estimated_cost_usd,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key not in {"papers", "failure_signal_ranking"}}, indent=2, ensure_ascii=False))
    if args.command == "run" and not report["comparison"]["pass_gate"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
