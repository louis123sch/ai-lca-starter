from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .corpus_diagnostics import BASELINE_MANIFEST, load_baseline_papers
from .inventory_replay import _paper_dir


def _read(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _result_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["doi"]: row for row in report.get("results", []) if row.get("doi")}


def flow_candidate_ids(
    state_dir: Path,
    manifest: Path,
    dois: list[str],
) -> dict[str, set[str]]:
    """Return candidate IDs that actually produced foreground flows in each state."""

    _, papers = load_baseline_papers(state_dir, manifest)
    by_doi = {paper["doi"]: paper for paper in papers}
    result: dict[str, set[str]] = {}
    for doi in dois:
        paper = by_doi.get(doi)
        found: set[str] = set()
        if paper:
            process_dir = _paper_dir(state_dir, paper) / "extraction" / "processes"
            for path in process_dir.glob("*.json") if process_dir.exists() else []:
                payload = _read(path, {}) or {}
                for flow in payload.get("flows", []) or []:
                    candidate_id = flow.get("candidate_id")
                    if candidate_id:
                        found.add(str(candidate_id))
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
    foreground flows. When state-level candidate IDs are available, every lost flow
    candidate must be explicitly covered by the router's safe-exclusion set; loss of
    any other flow candidate is a hard regression.
    """

    control_rows = _result_map(control)
    routed_rows = _result_map(routed)
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    unchanged: list[str] = []

    for doi in sorted(set(control_rows) | set(routed_rows)):
        old = control_rows.get(doi)
        new = routed_rows.get(doi)
        if old is None or new is None:
            regressions.append({"doi": doi, "reason": "missing control or routed result", "control": old, "routed": new})
            continue

        old_complete = old.get("status") == "COMPLETE"
        new_complete = new.get("status") == "COMPLETE"
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
    pass_gate = safety_pass and not regressions and (quality_improved or efficiency_noninferior)

    return {
        "pass_gate": pass_gate,
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
