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
Return a valid git unified diff that touches ONLY allowed files. Include tests for behavior changes. Keep the patch small enough to review and revert."""


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def _changed_paths(diff: str) -> set[str]:
    paths = set()
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            paths.add(line[6:].strip())
    return paths


def _validate_patch(proposal: PatchProposal, allowed: set[str]) -> None:
    paths = _changed_paths(proposal.unified_diff)
    if not paths:
        raise ValueError("proposal contains no changed file paths")
    if paths - allowed:
        raise ValueError(f"proposal touched disallowed paths: {sorted(paths - allowed)}")
    if set(proposal.files_touched) - allowed:
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


def propose_and_apply(args: argparse.Namespace) -> dict[str, Any]:
    diagnostics = _load(args.diagnostics)
    ranking = diagnostics.get("failure_class_ranking") or []
    if not ranking:
        raise RuntimeError("no diagnosable failure classes")
    target = args.failure_class or ranking[0]["failure_class"]
    allowed_files = TARGETS.get(target, DEFAULT_TARGETS)
    allowed = set(allowed_files)
    target_row = next(
        (x for x in ranking if x["failure_class"] == target), ranking[0]
    )
    prompt = (
        f"FAILURE CLASS: {target}\n"
        f"AFFECTED PAPERS: {target_row.get('affected_papers')}\n"
        f"PRIORITY SCORE: {target_row.get('priority_score')}\n"
        "Do not use or encode individual paper identities. The counts above are diagnostic only.\n\n"
        f"ALLOWED FILES: {json.dumps(allowed_files)}\n"
        "CURRENT CODE:\n"
        + _context(allowed_files, args.max_context_chars)
    )
    client = _client()
    completion = client.beta.chat.completions.parse(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format=PatchProposal,
    )
    proposal = completion.choices[0].message.parsed
    if proposal is None:
        raise RuntimeError("repair model returned no structured proposal")
    _validate_patch(proposal, allowed)

    patch_path = args.output_dir / "proposal.patch"
    meta_path = args.output_dir / "proposal.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(proposal.unified_diff.rstrip() + "\n", encoding="utf-8")
    meta_path.write_text(proposal.model_dump_json(indent=2) + "\n", encoding="utf-8")

    check = _run(["git", "apply", "--check", str(patch_path)], check=False)
    if check.returncode:
        raise RuntimeError("git apply --check failed: " + check.stderr[-2000:])
    _run(["git", "apply", str(patch_path)])

    tests = [
        "tests/test_jats.py",
        "tests/test_autonomous_literature.py",
        "tests/test_flow_audit.py",
        "tests/test_corpus_diagnostics.py",
        "tests/test_inventory_replay.py",
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
        "summary": proposal.summary,
        "rationale": proposal.rationale,
        "allowed_files": allowed_files,
        "changed_paths": sorted(_changed_paths(proposal.unified_diff)),
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
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/autonomous_iteration")
    )
    args = parser.parse_args()
    propose_and_apply(args)


if __name__ == "__main__":
    main()
