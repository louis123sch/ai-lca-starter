from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .autonomous_literature import (
    ASSIGN_SYSTEM_PROMPT,
    FLOW_CANDIDATE_SYSTEM_PROMPT,
    ApiRunner,
    Budget,
    CandidateAssignmentBatch,
    PaperProcessor,
    ProcessCandidateExtraction,
    RunConfig,
    StateStore,
    _candidate_payload,
    _load_model,
    _slug,
    _validate_assignments,
    _validate_process_extraction,
    _write_json,
)
from .corpus_diagnostics import BASELINE_MANIFEST, analyze

CANARY_MANIFEST = Path("benchmarks/corpus_development_v1/canary.json")


def _read(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _paper_dir(state_dir: Path, paper: dict[str, Any]) -> Path:
    raw = Path(str(paper.get("paper_dir") or ""))
    if raw.parts and raw.parts[0] == "literature_state":
        raw = Path(*raw.parts[1:])
    return state_dir / raw


def select_dois(
    *,
    state_dir: Path,
    manifest: Path,
    subset: str,
    cohort: str | None = None,
) -> list[str]:
    payload = _read(manifest, {}) or {}
    papers = payload.get("papers") or []
    if subset == "full":
        return [p["doi"] for p in papers]
    if subset == "canary":
        canary = _read(CANARY_MANIFEST, {}) or {}
        return list(canary.get("canary_dois") or [])
    if subset == "cohort":
        if not cohort:
            raise ValueError("--cohort is required for cohort replay")
        diagnostics = analyze(state_dir, manifest)
        for row in diagnostics["failure_class_ranking"]:
            if row["failure_class"] == cohort:
                return list(row["dois"])
        return []
    raise ValueError(f"unknown subset {subset}")


class TargetedReplayProcessor(PaperProcessor):
    """Reuse trusted baseline stages and spend model calls only on unresolved candidates."""

    def _assign(self, doi, source_hash, structure, candidates, paper_dir):  # noqa: ANN001
        path = paper_dir / "extraction" / "assignments.json"
        payload = _read(path, {}) or {}
        baseline = CandidateAssignmentBatch.model_validate(payload)
        allowed = {p.process_id for p in structure.processes}
        accepted, missing = _validate_assignments(candidates, baseline, allowed)
        unresolved_ids = set(missing) | {
            a.candidate_id for a in accepted if a.disposition == "ambiguous"
        }
        if not unresolved_ids:
            return accepted, missing

        subset = [c for c in candidates if c.candidate_id in unresolved_ids]
        prompt = (
            "LOCKED PROCESSES:\n"
            + json.dumps(
                [
                    {"process_id": p.process_id, "name": p.name, "stage": p.stage}
                    for p in structure.processes
                ],
                ensure_ascii=False,
            )
            + "\n\nREVIEW ONLY THESE PREVIOUSLY UNRESOLVED CANDIDATES:\n"
            + json.dumps(_candidate_payload(subset), ensure_ascii=False)
        )
        repair = self.api.parse(
            doi=doi,
            stage="candidate_assignment_replay",
            model=self.config.screen_model,
            reasoning_effort=self.config.screen_reasoning,
            system_prompt=ASSIGN_SYSTEM_PROMPT,
            user_prompt=prompt,
            response_format=CandidateAssignmentBatch,
        )
        kept = [a for a in accepted if a.candidate_id not in unresolved_ids]
        accepted, missing = _validate_assignments(
            candidates,
            CandidateAssignmentBatch(assignments=kept + repair.assignments),
            allowed,
        )
        return accepted, missing

    def _extract_process(self, doi, source_hash, process, allowed, assigned, paper_dir):  # noqa: ANN001
        path = paper_dir / "extraction" / "processes" / f"{_slug(process.process_id)}.json"
        if path.exists():
            baseline = _load_model(path, ProcessCandidateExtraction)
        else:
            baseline = ProcessCandidateExtraction(process_id=process.process_id)
        cleaned, missing, failures = _validate_process_extraction(
            process.process_id, assigned, baseline, allowed
        )
        unresolved = set(missing) | set(cleaned.ambiguous_candidate_ids)
        if not unresolved:
            return cleaned, failures

        subset = [c for c in assigned if c.candidate_id in unresolved]
        if not subset:
            return cleaned, failures
        repair = self.api.parse(
            doi=doi,
            stage="process_inventory_replay",
            process_id=process.process_id,
            model=self.config.core_model,
            reasoning_effort=self.config.flow_reasoning,
            system_prompt=FLOW_CANDIDATE_SYSTEM_PROMPT,
            user_prompt=(
                "LOCKED PROCESS:\n"
                + process.model_dump_json(indent=2)
                + "\n\nALL LOCKED IDS:\n"
                + json.dumps(sorted(allowed))
                + "\n\nReview ONLY these previously unresolved candidates. Preserve all already "
                "resolved baseline candidates outside this set:\n"
                + json.dumps(_candidate_payload(subset), ensure_ascii=False)
            ),
            response_format=ProcessCandidateExtraction,
        )
        kept_flows = [f for f in cleaned.flows if f.candidate_id not in unresolved]
        kept_noninv = [x for x in cleaned.non_inventory_candidate_ids if x not in unresolved]
        kept_ambiguous = [x for x in cleaned.ambiguous_candidate_ids if x not in unresolved]
        merged = ProcessCandidateExtraction(
            process_id=process.process_id,
            flows=kept_flows + repair.flows,
            non_inventory_candidate_ids=kept_noninv + repair.non_inventory_candidate_ids,
            ambiguous_candidate_ids=kept_ambiguous + repair.ambiguous_candidate_ids,
            warnings=cleaned.warnings + repair.warnings,
        )
        cleaned, _, failures = _validate_process_extraction(
            process.process_id, assigned, merged, allowed
        )
        _write_json(path, cleaned)
        return cleaned, failures


def _snapshot_qc(
    state_dir: Path, manifest: Path, dois: list[str]
) -> dict[str, dict[str, Any]]:
    payload = _read(manifest, {}) or {}
    by_doi = {p["doi"]: p for p in payload.get("papers") or []}
    result: dict[str, dict[str, Any]] = {}
    for doi in dois:
        paper = by_doi.get(doi)
        if not paper:
            continue
        qc = _read(_paper_dir(state_dir, paper) / "extraction" / "qc.json", {}) or {}
        result[doi] = {
            "status": paper.get("status"),
            "ambiguous_or_missing_candidate_count": int(
                qc.get("ambiguous_or_missing_candidate_count") or 0
            ),
            "flow_count": int(qc.get("flow_count") or 0),
            "candidate_coverage": float(qc.get("candidate_coverage") or 0.0),
        }
    return result


def compare(
    before: dict[str, dict[str, Any]], results: list[dict[str, Any]]
) -> dict[str, Any]:
    after = {r["doi"]: r for r in results if r.get("doi")}
    regressions = []
    improvements = []
    unchanged = []
    for doi, old in before.items():
        new = after.get(doi)
        if not new:
            continue
        old_amb = int(old.get("ambiguous_or_missing_candidate_count") or 0)
        new_amb = int(new.get("ambiguous_or_missing_candidate_count") or 0)
        old_flow = int(old.get("flow_count") or 0)
        new_flow = int(new.get("flow_count") or 0)
        if old.get("status") == "COMPLETE" and new.get("status") != "COMPLETE":
            regressions.append(
                {
                    "doi": doi,
                    "reason": "resolved control regressed",
                    "before": old,
                    "after": new,
                }
            )
        elif new_amb > old_amb or new_flow < old_flow:
            regressions.append(
                {
                    "doi": doi,
                    "reason": "ambiguity increased or flow count fell",
                    "before": old,
                    "after": new,
                }
            )
        elif new.get("status") == "COMPLETE" and old.get("status") != "COMPLETE":
            improvements.append(
                {
                    "doi": doi,
                    "reason": "became complete",
                    "before": old,
                    "after": new,
                }
            )
        elif new_amb < old_amb or new_flow > old_flow:
            improvements.append(
                {
                    "doi": doi,
                    "reason": "candidate ambiguity/flow coverage improved",
                    "before": old,
                    "after": new,
                }
            )
        else:
            unchanged.append(doi)
    return {
        "improved_papers": improvements,
        "regressions": regressions,
        "unchanged_papers": unchanged,
        "pass_gate": bool(improvements) and not regressions,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dois = select_dois(
        state_dir=args.state_dir,
        manifest=args.manifest,
        subset=args.subset,
        cohort=args.cohort,
    )
    before = _snapshot_qc(args.state_dir, args.manifest, dois)
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
    api = ApiRunner(config, store, budget)
    processor = TargetedReplayProcessor(
        config, store, api, os.environ.get("SPRINGER_API_KEY", "")
    )
    results = [processor.process(doi) for doi in dois]
    comparison = compare(before, results)
    report = {
        "subset": args.subset,
        "cohort": args.cohort,
        "dois": dois,
        "before": before,
        "results": results,
        "comparison": comparison,
        "usage": budget.summary(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "subset": args.subset,
                "paper_count": len(dois),
                "comparison": comparison,
                "usage": budget.summary(),
            },
            indent=2,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Token-efficient replay of only unresolved corpus evidence."
    )
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=BASELINE_MANIFEST)
    parser.add_argument(
        "--subset", choices=["canary", "cohort", "full"], default="canary"
    )
    parser.add_argument("--cohort")
    parser.add_argument(
        "--screen-model", default=os.getenv("OPENAI_SCREEN_MODEL", "gpt-5-nano")
    )
    parser.add_argument("--core-model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    parser.add_argument("--max-concurrent-requests", type=int, default=3)
    parser.add_argument("--max-process-workers", type=int, default=3)
    parser.add_argument("--max-total-calls", type=int, default=40)
    parser.add_argument("--max-calls-per-paper", type=int, default=8)
    parser.add_argument("--max-total-tokens", type=int, default=750_000)
    parser.add_argument("--max-estimated-cost-usd", type=float, default=0.75)
    parser.add_argument("--output", type=Path, default=Path("artifacts/replay.json"))
    args = parser.parse_args()
    report = run(args)
    if not report["comparison"]["pass_gate"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
