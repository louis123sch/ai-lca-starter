from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .llm import _client


class PatchProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    rationale: str
    files_touched: list[str] = Field(default_factory=list)
    unified_diff: str


TARGETS: dict[str, list[str]] = {
    "TABLE_HEAVY_UNRESOLVED": ["src/ai_lca/jats.py", "tests/test_jats.py"],
    "DUPLICATE_PROCESS_ASSIGNMENT": [
        "src/ai_lca/autonomous_literature.py",
        "tests/test_autonomous_literature.py",
    ],
    "PROCESS_WITHOUT_ASSIGNED_CANDIDATES": [
        "src/ai_lca/autonomous_literature.py",
        "src/ai_lca/inventory_replay.py",
        "tests/test_autonomous_literature.py",
        "tests/test_inventory_replay.py",
    ],
    "CANDIDATE_AMBIGUITY": [
        "src/ai_lca/autonomous_literature.py",
        "src/ai_lca/inventory_replay.py",
        "tests/test_autonomous_literature.py",
        "tests/test_inventory_replay.py",
    ],
    "CROSS_PROCESS_ASSIGNMENT": [
        "src/ai_lca/autonomous_literature.py",
        "src/ai_lca/inventory_replay.py",
        "tests/test_autonomous_literature.py",
        "tests/test_inventory_replay.py",
    ],
    "FLOW_REVIEW_AMBIGUITY": [
        "src/ai_lca/inventory_replay.py",
        "src/ai_lca/flow_audit.py",
        "tests/test_inventory_replay.py",
        "tests/test_flow_audit.py",
    ],
    "SPARSE_FLOW_EXTRACTION": [
        "src/ai_lca/inventory_replay.py",
        "src/ai_lca/flow_audit.py",
        "tests/test_inventory_replay.py",
        "tests/test_flow_audit.py",
    ],
    "INCOMPLETE_CANDIDATE_REVIEW": [
        "src/ai_lca/inventory_replay.py",
        "tests/test_inventory_replay.py",
    ],
}
DEFAULT_TARGETS = ["src/ai_lca/inventory_replay.py", "tests/test_inventory_replay.py"]
FORBIDDEN_PREFIXES = ("benchmarks/", ".github/", "docs/")
FORBIDDEN_PATCH_PATTERNS = (
    r"10\.1007/",
    r"HOLDOUT",
    r"gold[_ -]?standard",
    r"lower.{0,20}(threshold|target)",
    r"disable.{0,20}(check|validation|gate)",
)
SYSTEM_PROMPT = """You are a conservative software repair agent for an evidence-grounded life-cycle inventory extractor.
Propose ONE small, general architectural improvement for the supplied recurring failure class.
Scientific integrity constraints are absolute:
- never add paper-specific logic, DOI/title/author checks, benchmark-specific exceptions, or hard-coded expected answers;
- never weaken evaluation thresholds, validation gates, provenance rules, evidence requirements, or hallucination safeguards;
- never modify benchmark/gold/baseline data;
- never invent LCI values, datasets, geography, units, quantities, or processes;
- prefer deterministic parsing/validation and reuse of existing evidence over larger prompts;
- preserve the role-classified foreground architecture;
- optimize improvement per API token, not raw model usage.
Return ONLY a valid git unified diff in unified_diff. It must use standard headers exactly like `--- a/<path>` and `+++ b/<path>` followed by `@@` hunks. Do not use Markdown fences, `*** Begin Patch`, prose, or custom patch syntax inside unified_diff. Touch ONLY allowed files, include tests for behavior changes, and keep the patch small enough to review and revert.
A no-op or prose-only response is not an acceptable repair proposal. If you cannot justify a safe general change from the supplied aggregate diagnostics and code, return an empty diff; the controller will reject it safely rather than applying speculative code."""


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def _strip_markdown_fence(diff: str) -> str:
    text = diff.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```diff", "```patch"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _normalise_header_path(raw: str, prefix: str) -> str:
    value = raw.strip().split("\t", 1)[0].strip()
    if value == "/dev/null":
        return value
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    if value.startswith(("a/", "b/")):
        value = value[2:]
    return f"{prefix}/{value}"


def _normalise_unified_diff(diff: str) -> str:
    """Accept common model formatting variants but emit one git-applyable form."""
    text = _strip_markdown_fence(diff)
    lines = text.splitlines()
    # Ignore accidental prose before the first conventional patch header.
    first = next(
        (
            i
            for i, line in enumerate(lines)
            if line.startswith("diff --git ") or line.startswith("--- ")
        ),
        None,
    )
    if first is not None:
        lines = lines[first:]
    normalised: list[str] = []
    for line in lines:
        if line.startswith("--- "):
            normalised.append("--- " + _normalise_header_path(line[4:], "a"))
        elif line.startswith("+++ "):
            normalised.append("+++ " + _normalise_header_path(line[4:], "b"))
        else:
            normalised.append(line)
    return "\n".join(normalised).rstrip() + "\n"


def _changed_paths(diff: str) -> set[str]:
    paths: set[str] = set()
    text = _normalise_unified_diff(diff)
    for line in text.splitlines():
        if line.startswith("+++ "):
            value = line[4:].strip().split("\t", 1)[0].strip().strip('"')
            if value == "/dev/null":
                continue
            if value.startswith("b/"):
                value = value[2:]
            paths.add(value)
    if not paths:
        for line in text.splitlines():
            if not line.startswith("diff --git "):
                continue
            match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            if match:
                paths.add(match.group(2).strip().strip('"'))
    return paths


def _validate_patch(proposal: PatchProposal, allowed: set[str]) -> None:
    paths = _changed_paths(proposal.unified_diff)
    if not paths:
        raise ValueError("proposal contains no changed file paths")
    if paths - allowed:
        raise ValueError(f"proposal touched disallowed paths: {sorted(paths - allowed)}")
    metadata_paths = {
        p[2:] if p.startswith(("a/", "b/")) else p for p in proposal.files_touched
    }
    if metadata_paths - allowed:
        raise ValueError("proposal metadata contains disallowed paths")
    if any(path.startswith(FORBIDDEN_PREFIXES) for path in paths):
        raise ValueError("proposal attempted to change protected project data/infrastructure")
    for pattern in FORBIDDEN_PATCH_PATTERNS:
        if re.search(pattern, proposal.unified_diff, re.IGNORECASE | re.DOTALL):
            raise ValueError(f"proposal violates anti-overfitting policy: {pattern}")


def _context(files: list[str], max_chars: int) -> str:
    chunks = []
    remaining = max_chars
    for name in files:
        path = Path(name)
        if not path.exists() or remaining <= 0:
            continue
        text = path.read_text(encoding="utf-8")[:remaining]
        chunks.append(f"\n===== FILE: {name} =====\n{text}")
        remaining -= len(text)
    return "".join(chunks)


def _anonymized_failure_metrics(
    diagnostics: dict[str, Any], target: str, max_rows: int = 8
) -> list[dict[str, Any]]:
    """Return high-signal aggregate examples without paper identity or text."""
    keys = (
        "process_count",
        "candidate_count",
        "modeled_candidate_count",
        "candidate_coverage",
        "ambiguous_or_missing_candidate_count",
        "flow_count",
        "amount_coverage",
        "unit_coverage",
        "evidence_type_counts",
        "multi_process_assignment_count",
        "duplicate_process_reference_count",
    )
    rows = [
        row
        for row in (diagnostics.get("papers") or [])
        if target in (row.get("failure_classes") or [])
    ]
    rows.sort(
        key=lambda row: (
            -int(row.get("ambiguous_or_missing_candidate_count") or 0),
            -int(row.get("candidate_count") or 0),
        )
    )
    return [{key: row.get(key) for key in keys} for row in rows[:max_rows]]


def _write_proposal_attempt(
    output_dir: Path,
    proposal: PatchProposal,
    normalised_diff: str,
    attempt: int,
) -> None:
    (output_dir / f"proposal_attempt_{attempt}.json").write_text(
        proposal.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / f"proposal_attempt_{attempt}.raw.patch").write_text(
        proposal.unified_diff.rstrip() + "\n", encoding="utf-8"
    )
    (output_dir / f"proposal_attempt_{attempt}.patch").write_text(
        normalised_diff, encoding="utf-8"
    )


def propose_and_apply(args: argparse.Namespace) -> dict[str, Any]:
    diagnostics = _load(args.diagnostics)
    ranking = diagnostics.get("failure_class_ranking") or []
    if not ranking:
        raise RuntimeError("no diagnosable failure classes")
    target = args.failure_class or ranking[0]["failure_class"]
    allowed_files = TARGETS.get(target, DEFAULT_TARGETS)
    allowed = set(allowed_files)
    target_row = next((x for x in ranking if x["failure_class"] == target), ranking[0])
    aggregate_examples = _anonymized_failure_metrics(diagnostics, target)
    base_prompt = (
        f"FAILURE CLASS: {target}\n"
        f"AFFECTED PAPERS: {target_row.get('affected_papers')}\n"
        f"PRIORITY SCORE: {target_row.get('priority_score')}\n"
        "Do not use or encode individual paper identities. The counts and metrics below are diagnostic only.\n\n"
        "ANONYMIZED AFFECTED-PAPER METRICS:\n"
        f"{json.dumps(aggregate_examples, indent=2)}\n\n"
        f"ALLOWED FILES: {json.dumps(allowed_files)}\n"
        "Your unified_diff field must begin with `--- a/<path>` and `+++ b/<path>` and use standard `@@` hunks. Do not wrap it in Markdown.\n\n"
        "CURRENT CODE:\n"
        + _context(allowed_files, args.max_context_chars)
    )
    client = _client()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    accepted: PatchProposal | None = None
    accepted_diff = ""
    rejection_reasons: list[str] = []
    attempts_used = 0

    for attempt in range(1, args.max_proposal_attempts + 1):
        attempts_used = attempt
        feedback = ""
        if rejection_reasons:
            feedback = (
                "\n\nPREVIOUS PROPOSAL WAS REJECTED BY THE CONTROLLER:\n"
                f"{rejection_reasons[-1]}\n"
                "Return a corrected, small unified diff that satisfies the same scientific guardrails."
            )
        try:
            completion = client.beta.chat.completions.parse(
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": base_prompt + feedback},
                ],
                response_format=PatchProposal,
            )
            proposal = completion.choices[0].message.parsed
        except (TypeError, ValueError) as exc:
            rejection_reasons.append(
                f"attempt {attempt}: structured proposal parse failed: {exc}"
            )
            continue

        if proposal is None:
            rejection_reasons.append(
                f"attempt {attempt}: repair model returned no structured proposal"
            )
            continue

        normalised_diff = _normalise_unified_diff(proposal.unified_diff)
        normalised = proposal.model_copy(update={"unified_diff": normalised_diff})
        _write_proposal_attempt(args.output_dir, proposal, normalised_diff, attempt)

        try:
            _validate_patch(normalised, allowed)
        except ValueError as exc:
            rejection_reasons.append(f"attempt {attempt}: {exc}")
            continue

        patch_path = args.output_dir / f"proposal_attempt_{attempt}.patch"
        check = _run(["git", "apply", "--check", str(patch_path)], check=False)
        if check.returncode:
            rejection_reasons.append(
                "attempt "
                f"{attempt}: git apply --check failed: {(check.stderr or check.stdout)[-1200:]}"
            )
            continue

        accepted = normalised
        accepted_diff = normalised_diff
        (args.output_dir / "proposal.json").write_text(
            proposal.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "proposal.raw.patch").write_text(
            proposal.unified_diff.rstrip() + "\n", encoding="utf-8"
        )
        (args.output_dir / "proposal.patch").write_text(
            normalised_diff, encoding="utf-8"
        )
        _run(["git", "apply", str(patch_path)])
        break

    if accepted is None:
        result = {
            "failure_class": target,
            "applied": False,
            "proposal_attempts": attempts_used,
            "rejection_reasons": rejection_reasons,
            "allowed_files": allowed_files,
            "message": "No valid bounded repair was produced; no code was changed and corpus gates should be skipped for this cycle.",
        }
        (args.output_dir / "iteration.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return result

    tests = [
        "tests/test_jats.py",
        "tests/test_autonomous_literature.py",
        "tests/test_flow_audit.py",
        "tests/test_corpus_diagnostics.py",
        "tests/test_inventory_replay.py",
        "tests/test_autonomous_code_iteration.py",
    ]
    existing_tests = [p for p in tests if Path(p).exists()]
    test_run = _run(["python", "-m", "pytest", "-q", *existing_tests], check=False)
    (args.output_dir / "deterministic_tests.log").write_text(
        test_run.stdout + "\n" + test_run.stderr, encoding="utf-8"
    )
    if test_run.returncode:
        _run(["git", "reset", "--hard", "HEAD"])
        raise RuntimeError("deterministic tests failed; patch reverted")

    status = _run(["git", "diff", "--stat"]).stdout
    result = {
        "failure_class": target,
        "applied": True,
        "proposal_attempts": attempts_used,
        "rejection_reasons": rejection_reasons,
        "summary": accepted.summary,
        "rationale": accepted.rationale,
        "allowed_files": allowed_files,
        "changed_paths": sorted(_changed_paths(accepted_diff)),
        "diff_stat": status,
        "deterministic_tests_passed": True,
    }
    (args.output_dir / "iteration.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bounded autonomous code proposal for corpus failure classes."
    )
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--failure-class")
    parser.add_argument(
        "--model", default=os.getenv("OPENAI_REPAIR_MODEL", "gpt-5-mini")
    )
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-context-chars", type=int, default=90000)
    parser.add_argument("--max-proposal-attempts", type=int, default=3)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/autonomous_iteration")
    )
    args = parser.parse_args()
    propose_and_apply(args)


if __name__ == "__main__":
    main()
