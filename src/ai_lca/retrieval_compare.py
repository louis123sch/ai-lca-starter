from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _result_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["doi"]: row for row in report.get("results", []) if row.get("doi")}


def compare_reports(control: dict[str, Any], routed: dict[str, Any]) -> dict[str, Any]:
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
        old_flows = int(old.get("flow_count") or 0)
        new_flows = int(new.get("flow_count") or 0)

        if old_complete and not new_complete:
            regressions.append({"doi": doi, "reason": "complete control regressed", "control": old, "routed": new})
        elif new_amb > old_amb:
            regressions.append({"doi": doi, "reason": "candidate ambiguity increased", "control": old, "routed": new})
        elif new_flows < old_flows:
            regressions.append({"doi": doi, "reason": "flow count decreased", "control": old, "routed": new})
        elif new_complete and not old_complete:
            improvements.append({"doi": doi, "reason": "became complete", "control": old, "routed": new})
        elif new_amb < old_amb:
            improvements.append({"doi": doi, "reason": "candidate ambiguity decreased", "control": old, "routed": new})
        elif new_flows > old_flows:
            improvements.append({"doi": doi, "reason": "flow coverage increased", "control": old, "routed": new})
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
    safety_pass = bool(audit.get("inventory_safety_pass")) and int(audit.get("excluded_baseline_modeled_candidate_count") or 0) == 0
    token_change = routed_tokens - control_tokens
    cost_change = routed_cost - control_cost
    efficiency_noninferior = routed_tokens <= control_tokens and routed_cost <= control_cost + 1e-9
    quality_improved = bool(improvements)
    pass_gate = safety_pass and not regressions and (quality_improved or efficiency_noninferior)

    return {
        "pass_gate": pass_gate,
        "router_safety_pass": safety_pass,
        "quality_improved": quality_improved,
        "efficiency_noninferior": efficiency_noninferior,
        "improvements": improvements,
        "regressions": regressions,
        "unchanged": unchanged,
        "usage": {
            "control_calls": control_calls,
            "routed_calls": routed_calls,
            "call_change": routed_calls - control_calls,
            "control_tokens": control_tokens,
            "routed_tokens": routed_tokens,
            "token_change": token_change,
            "control_estimated_cost_usd": control_cost,
            "routed_estimated_cost_usd": routed_cost,
            "estimated_cost_change_usd": cost_change,
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = compare_reports(_read(args.control), _read(args.routed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["pass_gate"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
