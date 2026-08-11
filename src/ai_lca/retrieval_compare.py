from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .corpus_diagnostics import BASELINE_MANIFEST, load_baseline_papers
from .inventory_replay import _paper_dir


BENCHMARK_INCOMPLETE_STATUSES = {"BUDGET_STOP", "INFRASTRUCTURE_FAILURE"}


def _read(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def _result_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["doi"]: row for row in report.get("results", []) if row.get("doi")}


def flow_candidate_ids(
    state_dir: Path,
    manifest: Path,
    dois: list[str],
) -> dict[str, set[str]]:
    """Return source candidate IDs represented by the final foreground inventory.

    Replay validation can clean process results in memory without rewriting a process
    JSON when no further model repair is required. Therefore candidate-level A/B
    comparison uses the final `inventory.json` written from those in-memory results
    and maps each flow's provenance back to deterministic candidate evidence. This
    avoids reading stale baseline process files.
    """

    _, papers = load_baseline_papers(state_dir, manifest)
    by_doi = {paper["doi"]: paper for paper in papers}
    result: dict[str, set[str]] = {}
    for doi in dois:
        paper = by_doi.get(doi)
        found: set[str] = set()
        if not paper:
            result[doi] = found
            continue

        paper_dir = _paper_dir(state_dir, paper)
        raw_candidates = _read(paper_dir / "extraction" / "inventory_candidates.json", []) or []
        candidate_lookup: dict[tuple[str, str], set[str]] = {}
        for candidate in raw_candidates:
            key = (
                _normalise(candidate.get("evidence_text")),
                _normalise(candidate.get("table")),
            )
            candidate_id = candidate.get("candidate_id")
            if candidate_id and key[0]:
                candidate_lookup.setdefault(key, set()).add(str(candidate_id))

        inventory = _read(paper_dir / "extraction" / "inventory.json", {}) or {}
        for flow in inventory.get("flows", []) or []:
            evidence = flow.get("evidence") or {}
            key = (
                _normalise(evidence.get("evidence_text")),
                _normalise(evidence.get("table")),
            )
            found.update(candidate_lookup.get(key, set()))
        result[doi] = found
    return result


def compare_reports(
    control: dict[str, Any],
    routed: dict[str, Any],
    *,
    control_flow_candidates: dict[str, set[str]] | None = None,
    routed_flow_candidates: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Compare retrieval against control without rewarding known false-positive flows.

    Raw flow count is deliberately not a monotonic quality metric. A retrieval run
    may correctly remove LCIA result rows that the old pipeline had turned into
    foreground flows. When candidate IDs are available, every lost flow candidate
    must be explicitly covered by the router's safe-exclusion set; loss of any other
    source-supported foreground-flow candidate is a hard regression.

    A/B arms that stop for budget or infrastructure reasons are *censored benchmark
    observations*, not extraction regressions. They make the benchmark incomplete and
    therefore cannot pass the gate, but they are kept separate from scientific
    regression reasons so the repair agent does not try to "fix" retrieval in response
    to an execution-budget artefact.
    """

    control_rows = _result_map(control)
    routed_rows = _result_map(routed)
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    unchanged: list[str] = []
    incomplete_pairs: list[dict[str, Any]] = []

    for doi in sorted(set(control_rows) | set(routed_rows)):
        old = control_rows.get(doi)
        new = routed_rows.get(doi)
        if old is None or new is None:
            incomplete_pairs.append(
                {
                    "doi": doi,
                    "reason": "missing control or routed result",
                    "control_status": None if old is None else old.get("status"),
                    "routed_status": None if new is None else new.get("status"),
                }
            )
            continue

        old_status = str(old.get("status") or "")
        new_status = str(new.get("status") or "")
        if old_status in BENCHMARK_INCOMPLETE_STATUSES or new_status in BENCHMARK_INCOMPLETE_STATUSES:
            incomplete_pairs.append(
                {
                    "doi": doi,
                    "reason": "benchmark arm did not complete",
                    "control_status": old_status,
                    "routed_status": new_status,
                    "control_error": old.get("error"),
                    "routed_error": new.get("error"),
                }
            )
            continue

        old_complete = old_status == "COMPLETE"
        new_complete = new_status == "COMPLETE"
        old_amb = int(old.get("ambiguous_or_missing_candidate_count") or 0)
        new_amb = int(new.get("ambiguous_or_missing_candidate_count") or 0)
        old_coverage = float(old.get("candidate_coverage") or 0.0)
        new_coverage = float(new.get("candidate_coverage") or 0.0)
        old_processes = int(old.get("process_count") or 0)
        new_processes = int(new.get("process_count") or 0)
        safe_excluded = set(new.get("retrieval_safe_excluded_candidate_ids") or [])

        paper_regressions: list[str] = []
        paper_improvements: list[str] = []

        if old_complete and not new_complete:
            paper_regressions.append("complete control regressed")
        if new_amb > old_amb:
            paper_regressions.append("candidate ambiguity increased")
        if new_coverage + 1e-12 < old_coverage:
            paper_regressions.append("candidate coverage decreased")
        if new_processes != old_processes:
            paper_regressions.append("locked process count changed")

        lost_flow_candidates: list[str] = []
        corrected_lcia_flow_candidates: list[str] = []
        added_flow_candidates: list[str] = []
        if control_flow_candidates is not None and routed_flow_candidates is not None:
            old_ids = set(control_flow_candidates.get(doi, set()))
            new_ids = set(routed_flow_candidates.get(doi, set()))
            lost = old_ids - new_ids
            corrected = lost & safe_excluded
            unsafe_lost = lost - safe_excluded
            added = new_ids - old_ids
            lost_flow_candidates = sorted(unsafe_lost)
            corrected_lcia_flow_candidates = sorted(corrected)
            added_flow_candidates = sorted(added)
            if unsafe_lost:
                paper_regressions.append("lost flow candidate(s) not covered by safe LCIA exclusion")
            if corrected:
                paper_improvements.append("removed explicit LCIA-result flow candidate(s)")
            if added:
                paper_improvements.append("added source-supported flow candidate(s)")

        if new_complete and not old_complete:
            paper_improvements.append("became complete")
        if new_amb < old_amb:
            paper_improvements.append("candidate ambiguity decreased")
        if new_coverage > old_coverage + 1e-12:
            paper_improvements.append("candidate coverage increased")
        corrected_baseline = list(new.get("retrieval_corrected_baseline_modeled_candidate_ids") or [])
        if corrected_baseline:
            paper_improvements.append("corrected baseline LCIA-as-inventory assignment(s)")

        if paper_regressions:
            regressions.append(
                {
                    "doi": doi,
                    "reasons": paper_regressions,
                    "lost_unprotected_flow_candidate_ids": lost_flow_candidates,
                    "corrected_lcia_flow_candidate_ids": corrected_lcia_flow_candidates,
                    "added_flow_candidate_ids": added_flow_candidates,
                    "control": old,
                    "routed": new,
                }
            )
        elif paper_improvements:
            improvements.append(
                {
                    "doi": doi,
                    "reasons": list(dict.fromkeys(paper_improvements)),
                    "corrected_lcia_flow_candidate_ids": corrected_lcia_flow_candidates,
                    "added_flow_candidate_ids": added_flow_candidates,
                    "control": old,
                    "routed": new,
                }
            )
        else:
            unchanged.append(doi)

    control_usage = control.get("usage", {}) or {}
    routed_usage = routed.get("usage", {}) or {}
    control_tokens = int(control_usage.get("tokens_this_run") or 0)
    routed_tokens = int(routed_usage.get("tokens_this_run") or 0)
    control_calls = int(control_usage.get("calls_this_run") or 0)
    routed_calls = int(routed_usage.get("calls_this_run") or 0)
    control_cost = float(control_usage.get("estimated_cost_this_run_usd") or 0.0)
    routed_cost = float(routed_usage.get("estimated_cost_this_run_usd") or 0.0)

    audit = routed.get("router_audit", {}) or {}
    safety_pass = bool(audit.get("inventory_safety_pass")) and int(audit.get("unsafe_exclusion_count") or 0) == 0
    efficiency_noninferior = routed_tokens <= control_tokens and routed_cost <= control_cost + 1e-9
    quality_improved = bool(improvements)
    benchmark_complete = not incomplete_pairs
    pass_gate = (
        safety_pass
        and benchmark_complete
        and not regressions
        and (quality_improved or efficiency_noninferior)
    )

    if not benchmark_complete:
        failure_mode = "benchmark_incomplete"
    elif regressions:
        failure_mode = "quality_regression"
    elif not safety_pass:
        failure_mode = "router_safety_failure"
    elif not (quality_improved or efficiency_noninferior):
        failure_mode = "no_quality_or_efficiency_gain"
    else:
        failure_mode = None

    return {
        "pass_gate": pass_gate,
        "failure_mode": failure_mode,
        "benchmark_complete": benchmark_complete,
        "benchmark_incomplete_pairs": incomplete_pairs,
        "router_safety_pass": safety_pass,
        "quality_improved": quality_improved,
        "efficiency_noninferior": efficiency_noninferior,
        "candidate_level_flow_comparison": control_flow_candidates is not None and routed_flow_candidates is not None,
        "improvements": improvements,
        "regressions": regressions,
        "unchanged": unchanged,
        "usage": {
            "control_calls": control_calls,
            "routed_calls": routed_calls,
            "call_change": routed_calls - control_calls,
            "control_tokens": control_tokens,
            "routed_tokens": routed_tokens,
            "token_change": routed_tokens - control_tokens,
            "control_estimated_cost_usd": control_cost,
            "routed_estimated_cost_usd": routed_cost,
            "estimated_cost_change_usd": routed_cost - control_cost,
        },
        "router_audit": {
            key: value
            for key, value in audit.items()
            if key != "per_paper"
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare control and retrieval frozen-corpus replays.")
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--routed", type=Path, required=True)
    parser.add_argument("--control-state", type=Path)
    parser.add_argument("--routed-state", type=Path)
    parser.add_argument("--manifest", type=Path, default=BASELINE_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    control = _read(args.control, {}) or {}
    routed = _read(args.routed, {}) or {}
    control_flows = routed_flows = None
    if args.control_state and args.routed_state:
        dois = sorted(set(control.get("dois", [])) | set(routed.get("dois", [])))
        control_flows = flow_candidate_ids(args.control_state, args.manifest, dois)
        routed_flows = flow_candidate_ids(args.routed_state, args.manifest, dois)

    result = compare_reports(
        control,
        routed,
        control_flow_candidates=control_flows,
        routed_flow_candidates=routed_flows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["pass_gate"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
