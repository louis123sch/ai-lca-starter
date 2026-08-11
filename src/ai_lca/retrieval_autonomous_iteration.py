from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .autonomous_code_iteration import (
    PatchProposal,
    _changed_paths,
    _normalise_unified_diff,
    _run,
    _validate_patch,
)
from .corpus_diagnostics import load_baseline_papers
from .inventory_replay import _paper_dir
from .llm import _client


FAILURE_CLASS = "RETRIEVAL_BEFORE_REASONING"
ALLOWED_FILES = [
    "src/ai_lca/evidence_router.py",
    "src/ai_lca/retrieval_processor.py",
    "src/ai_lca/retrieval_replay.py",
    "src/ai_lca/retrieval_compare.py",
    "src/ai_lca/frozen_replay.py",
    "tests/test_evidence_router.py",
    "tests/test_retrieval_compare.py",
    "tests/test_frozen_replay.py",
]


class RepairHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actionable: bool
    mechanism: str
    evidence: list[str] = Field(default_factory=list)
    target_files: list[str] = Field(default_factory=list)
    proposed_change: str


HYPOTHESIS_SYSTEM_PROMPT = """You diagnose a retrieval-before-reasoning layer for an evidence-grounded life-cycle inventory extractor.

Your job is to form ONE testable, general causal hypothesis from the supplied A/B diagnostics and candidate-level evidence packets. Do not propose paper-specific logic. Separate execution/benchmark censoring from genuine extraction regressions.

Scientific-integrity constraints are absolute:
- never use DOI/title/author identity or benchmark-specific exceptions;
- never weaken quality, provenance, coverage, ambiguity, foreground-structure, or router-safety gates;
- never invent LCI values, processes, quantities, units, geography, datasets, or evidence;
- deterministic source candidate enumeration remains the source of truth;
- hard exclusion remains limited to independently safe explicit LCIA/result evidence;
- structure retrieval remains disabled unless independent structure-evidence recall supports it.

If there is at least one genuine candidate-level regression with concrete evidence, prefer a small reversible hypothesis over saying no change is possible. Mark actionable=false only when the failure is benchmark/infrastructure censoring or every plausible change would violate the constraints."""


PATCH_SYSTEM_PROMPT = """You are a conservative software-repair agent improving a retrieval-before-reasoning layer for an evidence-grounded life-cycle inventory extractor.

Implement ONE small, general, reversible repair that tests the supplied causal hypothesis. The control and routed arms use the identical frozen foreground structure, so changes must improve evidence routing, candidate review, replay correctness, or candidate-level comparison without redesigning the foreground graph.

Scientific-integrity constraints are absolute:
- never add paper-specific logic, DOI/title/author checks, benchmark-specific exceptions, or hard-coded expected answers;
- never weaken quality gates, provenance checks, candidate coverage checks, ambiguity checks, or the router structural-safety invariant;
- never modify frozen corpus, benchmark, gold, manifest, workflow, or baseline data;
- never invent LCI values, processes, quantities, units, geography, datasets, or evidence;
- deterministic candidate enumeration remains the source of truth and every source candidate must remain auditable;
- hard exclusion is only acceptable for independently safe explicit LCIA/result evidence with no competing LCI or modelling-assumption cue;
- structure retrieval must remain disabled unless the supplied diagnostics demonstrate adequate structure-evidence recall;
- prefer deterministic/high-recall routing and small targeted fixes over larger prompts or broader model use;
- do not optimize merely for lower token use if extraction quality regresses.

Return ONLY a standard git unified diff in unified_diff touching ONLY allowed files, with tests for behavior changes. The unified_diff value MUST use conventional git patch headers exactly like `--- a/<path>` and `+++ b/<path>` followed by numbered `@@ -old,+new @@` hunks. Do NOT use Markdown fences, `*** Begin Patch`, `*** Update File`, `*** End Patch`, apply_patch syntax, prose, or any other custom patch format inside unified_diff. When genuine candidate-level regressions are present, do not return an empty diff merely because success is uncertain: implement the safest bounded experiment justified by the hypothesis. Empty diff is appropriate only for benchmark/infrastructure censoring or when all possible changes would violate scientific integrity."""


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _sanitize(value: Any) -> Any:
    """Remove paper identities while preserving diagnostic and evidence content."""
    if isinstance(value, dict):
        return {
            key: _sanitize(item)
            for key, item in value.items()
            if key not in {"doi", "title", "source_hash", "paper_dir"}
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _history(path: Path, max_entries: int) -> list[dict[str, Any]]:
    """Return only hypotheses that were actually corpus-tested.

    Proposal-format failures are not scientific negative results and must not poison
    the search memory as if a retrieval hypothesis had failed canary/full evaluation.
    """
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if row.get("failure_class") != FAILURE_CLASS:
            continue
        if row.get("rejected_at_gate") not in {"canary", "full"}:
            continue
        rows.append(_sanitize(row))
    return rows[-max_entries:]


def _balanced_code_context(max_chars: int) -> str:
    """Share context budget across all editable files rather than starving tail files."""
    existing = [Path(name) for name in ALLOWED_FILES if Path(name).exists()]
    if not existing or max_chars <= 0:
        return ""
    per_file = max(2_500, max_chars // len(existing))
    chunks: list[str] = []
    remaining = max_chars
    for path in existing:
        if remaining <= 0:
            break
        text = path.read_text(encoding="utf-8")
        quota = min(per_file, remaining)
        if len(text) <= quota:
            piece = text
        elif quota >= 1_000:
            head = int(quota * 0.72)
            tail = quota - head
            piece = text[:head] + "\n... [middle omitted] ...\n" + text[-tail:]
        else:
            piece = text[:quota]
        chunks.append(f"\n===== FILE: {path.as_posix()} =====\n{piece}")
        remaining -= len(piece)
    return "".join(chunks)


def _assignment_map(paper_dir: Path, *, routed: bool) -> dict[str, dict[str, Any]]:
    preferred = (
        paper_dir / "extraction" / "retrieval" / "replay_assignments.json"
        if routed
        else paper_dir / "extraction" / "replay_control_assignments.json"
    )
    payload = _read_json(preferred, None)
    if payload is None:
        payload = _read_json(paper_dir / "extraction" / "assignments.json", {}) or {}
    return {
        str(row.get("candidate_id")): row
        for row in (payload.get("assignments") or [])
        if row.get("candidate_id")
    }


def _route_map(paper_dir: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(
        paper_dir / "extraction" / "retrieval" / "replay_candidate_routes.json",
        {},
    ) or {}
    return {
        str(row.get("candidate_id")): row
        for row in (payload.get("routes") or [])
        if row.get("candidate_id")
    }


def _process_review_state(paper_dir: Path, candidate_id: str) -> list[dict[str, Any]]:
    process_dir = paper_dir / "extraction" / "processes"
    states: list[dict[str, Any]] = []
    if not process_dir.exists():
        return states
    for path in sorted(process_dir.glob("*.json")):
        payload = _read_json(path, {}) or {}
        process_id = payload.get("process_id") or path.stem
        for flow in payload.get("flows", []) or []:
            if str(flow.get("candidate_id")) == candidate_id:
                states.append({"process_id": process_id, "review_state": "flow"})
        if candidate_id in set(payload.get("non_inventory_candidate_ids") or []):
            states.append({"process_id": process_id, "review_state": "non_inventory"})
        if candidate_id in set(payload.get("ambiguous_candidate_ids") or []):
            states.append({"process_id": process_id, "review_state": "ambiguous"})
    return states[:12]


def _candidate_map(paper_dir: Path) -> dict[str, dict[str, Any]]:
    rows = _read_json(paper_dir / "extraction" / "inventory_candidates.json", []) or []
    return {
        str(row.get("candidate_id")): row
        for row in rows
        if row.get("candidate_id")
    }


def _truncate(value: Any, limit: int = 1_200) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) <= limit:
        return value
    return value[: limit - 20] + " ... [truncated]"


def _selected_comparison(raw_diagnostics: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    full = raw_diagnostics.get("current_full_comparison") or {}
    if full and (full.get("regressions") or full.get("benchmark_incomplete_pairs")):
        return "full", full
    return "canary", raw_diagnostics.get("current_canary_comparison") or {}


def _causal_packets(raw_diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    """Build paper-anonymous candidate-level failure packets from the replay states."""
    scope, comparison = _selected_comparison(raw_diagnostics)
    regressions = list(comparison.get("regressions") or [])
    if not regressions:
        return []

    if scope == "full":
        control_state = Path("current_control_full_state")
        routed_state = Path("current_routed_full_state")
    else:
        control_state = Path("current_control_canary_state")
        routed_state = Path("current_routed_canary_state")
    manifest = Path("artifacts/frozen_replay_manifest.json")
    if not control_state.exists() or not routed_state.exists() or not manifest.exists():
        return []

    try:
        _, papers = load_baseline_papers(control_state, manifest)
    except Exception:
        return []
    by_doi = {paper.get("doi"): paper for paper in papers}
    packets: list[dict[str, Any]] = []

    for index, regression in enumerate(regressions[:8], 1):
        doi = regression.get("doi")
        paper = by_doi.get(doi)
        if not doi or not paper:
            continue
        control_dir = _paper_dir(control_state, paper)
        routed_dir = _paper_dir(routed_state, paper)
        candidates = _candidate_map(routed_dir) or _candidate_map(control_dir)
        control_assignments = _assignment_map(control_dir, routed=False)
        routed_assignments = _assignment_map(routed_dir, routed=True)
        routes = _route_map(routed_dir)

        priority_ids = list(regression.get("lost_unprotected_flow_candidate_ids") or [])
        priority_ids += list(regression.get("added_flow_candidate_ids") or [])
        changed_assignment_ids = [
            candidate_id
            for candidate_id in sorted(set(control_assignments) | set(routed_assignments))
            if control_assignments.get(candidate_id) != routed_assignments.get(candidate_id)
        ]
        priority_ids += changed_assignment_ids
        priority_ids = list(dict.fromkeys(str(x) for x in priority_ids if x))[:20]

        candidate_packets: list[dict[str, Any]] = []
        for candidate_id in priority_ids:
            source = candidates.get(candidate_id, {})
            candidate_packets.append(
                {
                    "candidate_id": candidate_id,
                    "source": {
                        key: _truncate(source.get(key))
                        for key in (
                            "evidence_type",
                            "evidence_text",
                            "context",
                            "table",
                            "source_location",
                        )
                        if source.get(key) is not None
                    },
                    "route": routes.get(candidate_id),
                    "control_assignment": control_assignments.get(candidate_id),
                    "routed_assignment": routed_assignments.get(candidate_id),
                    "control_review_state": _process_review_state(control_dir, candidate_id),
                    "routed_review_state": _process_review_state(routed_dir, candidate_id),
                    "lost_unprotected_flow": candidate_id
                    in set(regression.get("lost_unprotected_flow_candidate_ids") or []),
                    "added_flow": candidate_id in set(regression.get("added_flow_candidate_ids") or []),
                }
            )

        control_result = regression.get("control") or {}
        routed_result = regression.get("routed") or {}
        packets.append(
            {
                "case": f"case_{index:02d}",
                "scope": scope,
                "regression_reasons": list(regression.get("reasons") or []),
                "control_metrics": {
                    key: control_result.get(key)
                    for key in (
                        "status",
                        "process_count",
                        "candidate_count",
                        "modeled_candidate_count",
                        "candidate_coverage",
                        "ambiguous_or_missing_candidate_count",
                        "flow_count",
                        "process_failures",
                    )
                    if key in control_result
                },
                "routed_metrics": {
                    key: routed_result.get(key)
                    for key in (
                        "status",
                        "process_count",
                        "candidate_count",
                        "modeled_candidate_count",
                        "candidate_coverage",
                        "ambiguous_or_missing_candidate_count",
                        "flow_count",
                        "process_failures",
                    )
                    if key in routed_result
                },
                "candidate_deltas": candidate_packets,
            }
        )
    return packets


def _diagnostic_prompt(
    diagnostics: dict[str, Any],
    history: list[dict[str, Any]],
    packets: list[dict[str, Any]],
    code_context: str,
) -> str:
    return (
        "AGGREGATE FROZEN-CORPUS DIAGNOSTICS:\n"
        + json.dumps(_sanitize(diagnostics), indent=2, ensure_ascii=False)
        + "\n\nCANDIDATE-LEVEL CAUSAL FAILURE PACKETS (paper identities removed):\n"
        + json.dumps(_sanitize(packets), indent=2, ensure_ascii=False)
        + "\n\nPRIOR CORPUS-TESTED REJECTED HYPOTHESES (do not repeat them):\n"
        + json.dumps(history, indent=2, ensure_ascii=False)
        + "\n\nALLOWED FILES:\n"
        + json.dumps(ALLOWED_FILES, indent=2)
        + "\n\nCURRENT CODE:\n"
        + code_context
    )


def propose_and_apply(args: argparse.Namespace) -> dict[str, Any]:
    raw_diagnostics = _read_json(args.diagnostics, {}) or {}
    diagnostics = _sanitize(raw_diagnostics)
    history = _history(args.history_file, args.max_history_entries)
    packets = _causal_packets(raw_diagnostics)
    code_context = _balanced_code_context(args.max_context_chars)
    base_prompt = _diagnostic_prompt(diagnostics, history, packets, code_context)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = _client()
    errors: list[str] = []
    allowed = set(ALLOWED_FILES)

    for attempt in range(1, args.max_proposal_attempts + 1):
        hypothesis_messages = [
            {"role": "system", "content": HYPOTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": base_prompt},
        ]
        if errors:
            hypothesis_messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous attempt failed before corpus testing for this reason: "
                        + errors[-1]
                        + "\nForm a materially different, testable causal hypothesis."
                    ),
                }
            )
        hypothesis_completion = client.beta.chat.completions.parse(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            messages=hypothesis_messages,
            response_format=RepairHypothesis,
        )
        hypothesis = hypothesis_completion.choices[0].message.parsed
        if hypothesis is None:
            errors.append("repair model returned no structured hypothesis")
            continue
        (args.output_dir / f"hypothesis_attempt_{attempt}.json").write_text(
            hypothesis.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        if set(hypothesis.target_files) - allowed:
            errors.append("hypothesis targeted disallowed file(s)")
            continue
        if not hypothesis.actionable:
            errors.append("hypothesis declared no safe actionable repair")
            continue

        patch_prompt = (
            base_prompt
            + "\n\nTESTABLE CAUSAL HYPOTHESIS TO IMPLEMENT:\n"
            + hypothesis.model_dump_json(indent=2)
            + "\n\nReturn the smallest reversible patch that tests this hypothesis."
        )
        patch_messages = [
            {"role": "system", "content": PATCH_SYSTEM_PROMPT},
            {"role": "user", "content": patch_prompt},
        ]
        patch_completion = client.beta.chat.completions.parse(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            messages=patch_messages,
            response_format=PatchProposal,
        )
        proposal = patch_completion.choices[0].message.parsed
        if proposal is None:
            errors.append("repair model returned no structured patch proposal")
            continue

        raw_path = args.output_dir / f"proposal_attempt_{attempt}.raw.patch"
        raw_path.write_text(proposal.unified_diff.rstrip() + "\n", encoding="utf-8")
        try:
            normalised_diff = _normalise_unified_diff(proposal.unified_diff)
            normalised = proposal.model_copy(update={"unified_diff": normalised_diff})
            if not _changed_paths(normalised_diff):
                raise ValueError("proposal contains no changed paths")
            _validate_patch(normalised, allowed)
            patch_path = args.output_dir / "proposal.patch"
            patch_path.write_text(normalised_diff, encoding="utf-8")
            check = _run(["git", "apply", "--check", str(patch_path)], check=False)
            if check.returncode:
                raise RuntimeError("git apply --check failed: " + check.stderr[-1600:])
            _run(["git", "apply", str(patch_path)])

            tests = _run(["python", "-m", "pytest", "-q"], check=False)
            (args.output_dir / "deterministic_tests.log").write_text(
                tests.stdout + "\n" + tests.stderr, encoding="utf-8"
            )
            if tests.returncode:
                _run(["git", "restore", "--", *ALLOWED_FILES], check=False)
                raise RuntimeError("deterministic tests failed; proposal reverted")

            result = {
                "failure_class": FAILURE_CLASS,
                "attempt": attempt,
                "summary": proposal.summary,
                "rationale": proposal.rationale,
                "hypothesis": hypothesis.model_dump(),
                "causal_packet_count": len(packets),
                "changed_paths": sorted(_changed_paths(normalised_diff)),
                "deterministic_tests_passed": True,
            }
            (args.output_dir / "iteration.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(result, indent=2))
            return result
        except Exception as exc:  # proposal validation failures are expected search outcomes
            _run(["git", "restore", "--", *ALLOWED_FILES], check=False)
            errors.append(str(exc))

    failure = {
        "failure_class": "RETRIEVAL_PROPOSAL_FAILURE",
        "proposal_failed": True,
        "summary": "No testable retrieval patch was produced",
        "rationale": (
            "Failure occurred before corpus testing. This is a proposal-generation failure, "
            "not evidence that a retrieval hypothesis regressed extraction quality."
        ),
        "causal_packet_count": len(packets),
        "errors": errors,
    }
    (args.output_dir / "iteration.json").write_text(
        json.dumps(failure, indent=2) + "\n", encoding="utf-8"
    )
    raise RuntimeError("No valid bounded retrieval repair proposal: " + " | ".join(errors[-3:]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic-driven bounded autonomous repair for retrieval-before-reasoning."
    )
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("OPENAI_REPAIR_MODEL", "gpt-5-mini"))
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-context-chars", type=int, default=120000)
    parser.add_argument("--max-proposal-attempts", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/retrieval_autonomous_iteration"))
    parser.add_argument(
        "--history-file",
        type=Path,
        default=Path(".autonomy/retrieval_rejection_history.jsonl"),
    )
    parser.add_argument("--max-history-entries", type=int, default=12)
    args = parser.parse_args()
    propose_and_apply(args)


if __name__ == "__main__":
    main()
