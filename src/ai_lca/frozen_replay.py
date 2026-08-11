from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

from .autonomous_literature import (
    ApiRunner,
    Budget,
    CandidateAssignmentBatch,
    ForegroundStructure,
    RunConfig,
    StateStore,
    _load_model,
    _write_json,
)
from .corpus_diagnostics import BASELINE_MANIFEST
from .inventory_replay import (
    TargetedReplayProcessor,
    _snapshot_qc,
    compare,
    select_dois,
)
from .retrieval_replay import (
    RoutedTargetedReplayProcessor,
    _attach_route_summaries,
    audit_router,
)

ReplayMode = Literal["control", "routed"]


MIN_FAIR_AB_CALLS_PER_PAPER = 20


def fair_ab_calls_per_paper(requested: int, max_total_calls: int) -> int:
    """Give each A/B paper enough headroom to produce a comparable result.

    A per-paper cap that censors one arm creates a benchmark artefact rather than a
    meaningful quality comparison. The run-level call/token/cost caps remain hard,
    so raising this local ceiling cannot make an experiment unbounded.
    """

    return min(max_total_calls, max(requested, MIN_FAIR_AB_CALLS_PER_PAPER))


class FrozenStructureMixin:
    """Force replay to use the exact frozen foreground graph in both A/B arms."""

    def _structure(self, doi, source_hash, doc, paper_dir):  # noqa: ANN001
        path = paper_dir / "extraction" / "structure.json"
        if not path.exists():
            raise FileNotFoundError(f"Frozen replay requires existing structure: {path}")
        structure = _load_model(path, ForegroundStructure)
        self.store.record_processes(doi, structure)
        return structure


class FrozenControlProcessor(FrozenStructureMixin, TargetedReplayProcessor):
    def _assign(self, doi, source_hash, structure, candidates, paper_dir):  # noqa: ANN001
        assignments, missing = super()._assign(doi, source_hash, structure, candidates, paper_dir)
        _write_json(
            paper_dir / "extraction" / "replay_control_assignments.json",
            CandidateAssignmentBatch(assignments=assignments),
        )
        return assignments, missing


class FrozenRoutedProcessor(FrozenStructureMixin, RoutedTargetedReplayProcessor):
    def _assign(self, doi, source_hash, structure, candidates, paper_dir):  # noqa: ANN001
        assignments, missing = super()._assign(doi, source_hash, structure, candidates, paper_dir)
        _write_json(
            paper_dir / "extraction" / "retrieval" / "replay_assignments.json",
            CandidateAssignmentBatch(assignments=assignments),
        )
        return assignments, missing


def run(args: argparse.Namespace) -> dict[str, Any]:
    mode: ReplayMode = args.mode
    dois = select_dois(
        state_dir=args.state_dir,
        manifest=args.manifest,
        subset=args.subset,
        cohort=args.cohort,
    )
    before = _snapshot_qc(args.state_dir, args.manifest, dois)
    router_audit = None
    if mode == "routed":
        router_audit = audit_router(args.state_dir, args.manifest, dois)
        if not router_audit["inventory_safety_pass"]:
            report = {
                "mode": mode,
                "subset": args.subset,
                "cohort": args.cohort,
                "dois": dois,
                "before": before,
                "router_audit": router_audit,
                "results": [],
                "comparison": {
                    "improved_papers": [],
                    "regressions": [{"reason": "router structural safety audit failed"}],
                    "unchanged_papers": [],
                    "pass_gate": False,
                },
                "usage": {
                    "calls_this_run": 0,
                    "tokens_this_run": 0,
                    "estimated_cost_this_run_usd": 0.0,
                },
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return report

    effective_per_paper_cap = fair_ab_calls_per_paper(
        args.max_calls_per_paper,
        args.max_total_calls,
    )
    config = RunConfig(
        state_dir=args.state_dir,
        screen_model=args.screen_model,
        core_model=args.core_model,
        max_concurrent_requests=args.max_concurrent_requests,
        max_process_workers=args.max_process_workers,
        max_paper_workers=1,
        max_total_calls=args.max_total_calls,
        max_calls_per_paper=effective_per_paper_cap,
        max_total_tokens=args.max_total_tokens,
        max_estimated_cost_usd=args.max_estimated_cost_usd,
        max_repair_calls_per_process=1,
    )
    store = StateStore(args.state_dir)
    budget = Budget(config, store)
    processor_type = FrozenRoutedProcessor if mode == "routed" else FrozenControlProcessor
    processor = processor_type(
        config,
        store,
        ApiRunner(config, store, budget),
        os.environ.get("SPRINGER_API_KEY", ""),
    )
    results = [processor.process(doi) for doi in dois]
    if mode == "routed":
        _attach_route_summaries(args.state_dir, args.manifest, results)
    comparison = compare(before, results)
    report: dict[str, Any] = {
        "mode": mode,
        "subset": args.subset,
        "cohort": args.cohort,
        "dois": dois,
        "before": before,
        "results": results,
        "comparison": comparison,
        "usage": budget.summary(),
        "benchmark_limits": {
            "requested_max_calls_per_paper": args.max_calls_per_paper,
            "effective_max_calls_per_paper": effective_per_paper_cap,
            "max_total_calls": args.max_total_calls,
            "max_total_tokens": args.max_total_tokens,
            "max_estimated_cost_usd": args.max_estimated_cost_usd,
        },
    }
    if router_audit is not None:
        report["router_audit"] = router_audit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "mode": mode,
                "subset": args.subset,
                "paper_count": len(dois),
                "comparison": comparison,
                "usage": budget.summary(),
                "benchmark_limits": report["benchmark_limits"],
                "router_safety_pass": None if router_audit is None else router_audit["inventory_safety_pass"],
            },
            indent=2,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-structure control/retrieval corpus replay.")
    parser.add_argument("--mode", choices=["control", "routed"], required=True)
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
    parser.add_argument("--output", type=Path, default=Path("artifacts/frozen_replay.json"))
    args = parser.parse_args()
    report = run(args)
    if args.mode == "routed" and not report.get("router_audit", {}).get("inventory_safety_pass", False):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
