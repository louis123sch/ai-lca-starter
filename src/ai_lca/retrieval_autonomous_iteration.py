from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .autonomous_code_iteration import (
    PatchProposal,
    _changed_paths,
    _normalise_unified_diff,
    _run,
    _validate_patch,
)
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

SYSTEM_PROMPT = """You are a conservative software-repair agent improving a retrieval-before-reasoning layer for an evidence-grounded life-cycle inventory extractor.

Propose ONE small, general repair that addresses the aggregate frozen-corpus diagnostics supplied to you. The control and routed arms use the identical frozen foreground structure, so changes must improve evidence routing, candidate review, replay correctness, or candidate-level comparison without redesigning the foreground graph.

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

Return a small git unified diff touching ONLY allowed files, with tests for behavior changes. If the diagnostics do not justify a safe general change, return an empty diff rather than guessing."""


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _sanitize(value: Any) -> Any:
    """Remove paper identities while preserving aggregate failure evidence."""
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
        rows.append(_sanitize(row))
    return rows[-max_entries:]


def _code_context(max_chars: int) -> str:
    chunks: list[str] = []
    remaining = max_chars
    for name in ALLOWED_FILES:
        if remaining <= 0:
            break
        path = Path(name)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        piece = text[:remaining]
        chunks.append(f"\n===== FILE: {name} =====\n{piece}")
        remaining -= len(piece)
    return "".join(chunks)


def propose_and_apply(args: argparse.Namespace) -> dict[str, Any]:
    diagnostics = _sanitize(_read_json(args.diagnostics, {}) or {})
    history = _history(args.history_file, args.max_history_entries)
    prompt = (
        "AGGREGATE FROZEN-CORPUS DIAGNOSTICS:\n"
        + json.dumps(diagnostics, indent=2, ensure_ascii=False)
        + "\n\nPRIOR REJECTED RETRIEVAL EXPERIMENTS (do not repeat them):\n"
        + json.dumps(history, indent=2, ensure_ascii=False)
        + "\n\nALLOWED FILES:\n"
        + json.dumps(ALLOWED_FILES, indent=2)
        + "\n\nCURRENT CODE:\n"
        + _code_context(args.max_context_chars)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = _client()
    errors: list[str] = []
    allowed = set(ALLOWED_FILES)

    for attempt in range(1, args.max_proposal_attempts + 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if errors:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous proposal was rejected before corpus testing for this reason: "
                        + errors[-1]
                        + "\nReturn a materially corrected small patch, not the same malformed proposal."
                    ),
                }
            )
        completion = client.beta.chat.completions.parse(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            messages=messages,
            response_format=PatchProposal,
        )
        proposal = completion.choices[0].message.parsed
        if proposal is None:
            errors.append("repair model returned no structured proposal")
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
        "failure_class": FAILURE_CLASS,
        "proposal_failed": True,
        "errors": errors,
    }
    (args.output_dir / "iteration.json").write_text(
        json.dumps(failure, indent=2) + "\n", encoding="utf-8"
    )
    raise RuntimeError("No valid bounded retrieval repair proposal: " + " | ".join(errors[-3:]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bounded autonomous repair proposal for retrieval-before-reasoning."
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
