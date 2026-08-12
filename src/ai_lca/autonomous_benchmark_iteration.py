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


ALLOWED_FILES = [
    "src/ai_lca/llm.py",
    "src/ai_lca/structure.py",
    "src/ai_lca/models.py",
    "tests/test_benchmark.py",
    "tests/test_structure.py",
    "tests/test_llm.py",
]
FORBIDDEN_PREFIXES = (
    "benchmarks/",
    ".github/",
    ".autonomy/",
    "scripts/",
)
FORBIDDEN_PATCH_PATTERNS = (
    r"mycelium",
    r"hermesmann",
    r"terlouw",
    r"gonzales",
    r"calienes",
    r"\bafzal\b",
    r"\byang\b",
    r"\bgerloff\b",
    r"10\.\d{4,9}/",
    r"expected\.json",
    r"benchmark[_ -]?(id|specific)",
    r"lower.{0,30}(threshold|gate|minimum)",
    r"disable.{0,30}(check|validation|gate)",
    r"skip.{0,30}(benchmark|regression|gate)",
)
SYSTEM_PROMPT = """You are a conservative software-repair agent for an evidence-grounded life-cycle inventory extractor.

You are inside an autonomous regression-controlled development loop. Propose ONE small, general repair that improves the supplied extraction failure while preserving already-demonstrated behavior on other papers.

Hard constraints:
- Never add paper-, author-, DOI-, benchmark-, technology-instance-, process-name-, or expected-answer-specific branches or literals.
- Never modify benchmark/gold data, benchmark evaluator code, workflow code, thresholds, gates, baseline state, or regression policy.
- Never weaken evidence requirements, provenance rules, hallucination safeguards, or process-role distinctions.
- Never invent LCI values, quantities, units, datasets, geography, processes, or reference products.
- Prefer the smallest rule that is defensible from generic LCA document structure.
- Distinguish source-named quantitative foreground unit-process inventories from descriptive process diagrams, equipment/component lists, background database activities, and generic life-cycle headings.
- Terminal foreground processes may be real even when they do not feed a downstream foreground process, but only if the source explicitly models them quantitatively as foreground activities.
- A literature-derived or custom subsystem may be foreground when the study explicitly models it as part of the foreground LCI; do not treat all literature-derived or custom subsystems as foreground automatically.
- Include deterministic tests for behavior changes when practical.
- Return a standard git unified diff only in unified_diff. Do not use Markdown fences or custom patch syntax.

The patch will be rejected if any protected benchmark falls below its absolute quality floor or below the accepted baseline. Do not trade one benchmark for another.
"""


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
    text = _strip_markdown_fence(diff)
    lines = text.splitlines()
    first = next(
        (i for i, line in enumerate(lines) if line.startswith("diff --git ") or line.startswith("--- ")),
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
    for line in _normalise_unified_diff(diff).splitlines():
        if line.startswith("+++ "):
            value = line[4:].strip().split("\t", 1)[0].strip().strip('"')
            if value == "/dev/null":
                continue
            if value.startswith("b/"):
                value = value[2:]
            paths.add(value)
    return paths


def _validate_patch(proposal: PatchProposal) -> None:
    allowed = set(ALLOWED_FILES)
    paths = _changed_paths(proposal.unified_diff)
    if not paths:
        raise ValueError("proposal contains no changed file paths")
    if paths - allowed:
        raise ValueError(f"proposal touched disallowed paths: {sorted(paths - allowed)}")
    metadata_paths = {
        p[2:] if p.startswith(("a/", "b/")) else p for p in proposal.files_touched
    }
    if metadata_paths - allowed:
        raise ValueError(f"proposal metadata touched disallowed paths: {sorted(metadata_paths - allowed)}")
    if any(path.startswith(FORBIDDEN_PREFIXES) for path in paths):
        raise ValueError("proposal attempted to modify protected infrastructure/data")
    for pattern in FORBIDDEN_PATCH_PATTERNS:
        if re.search(pattern, proposal.unified_diff, re.IGNORECASE | re.DOTALL):
            raise ValueError(f"proposal violates anti-overfitting policy: {pattern}")


def _context(max_chars: int) -> str:
    chunks: list[str] = []
    remaining = max_chars
    for name in ALLOWED_FILES:
        path = Path(name)
        if not path.exists() or remaining <= 0:
            continue
        text = path.read_text(encoding="utf-8")
        text = text[:remaining]
        chunks.append(f"\n===== FILE: {name} =====\n{text}")
        remaining -= len(text)
    return "".join(chunks)


def propose_and_apply(
    diagnostics_path: Path,
    output_dir: Path,
    *,
    model: str,
    reasoning_effort: str,
    max_context_chars: int,
    max_attempts: int,
) -> dict[str, Any]:
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        retry_context = ""
        if errors:
            retry_context = (
                "\nPREVIOUS PROPOSAL REJECTION:\n"
                + errors[-1]
                + "\nReturn a different, valid, smaller general repair.\n"
            )
        prompt = (
            "DIAGNOSTICS FROM THE CURRENT ACCEPTED VERSION AND MOST RECENT REJECTIONS:\n"
            + json.dumps(diagnostics, indent=2)[:70000]
            + retry_context
            + "\nALLOWED FILES:\n"
            + json.dumps(ALLOWED_FILES)
            + "\nCURRENT CODE:\n"
            + _context(max_context_chars)
        )
        completion = _client().beta.chat.completions.parse(
            model=model,
            reasoning_effort=reasoning_effort,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=PatchProposal,
        )
        proposal = completion.choices[0].message.parsed
        if proposal is None:
            errors.append("repair model returned no structured proposal")
            continue

        attempt_dir = output_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "proposal.json").write_text(
            proposal.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        raw = proposal.unified_diff.rstrip() + "\n"
        (attempt_dir / "proposal.raw.patch").write_text(raw, encoding="utf-8")

        try:
            diff = _normalise_unified_diff(raw)
            normalised = proposal.model_copy(update={"unified_diff": diff})
            _validate_patch(normalised)
            patch_path = attempt_dir / "proposal.patch"
            patch_path.write_text(diff, encoding="utf-8")
            checked = _run(["git", "apply", "--check", str(patch_path)], check=False)
            if checked.returncode:
                raise RuntimeError("git apply --check failed: " + checked.stderr[-2500:])
            _run(["git", "apply", str(patch_path)])

            tests = _run(["python", "-m", "pytest", "-q"], check=False)
            (attempt_dir / "deterministic_tests.log").write_text(
                tests.stdout + "\n" + tests.stderr, encoding="utf-8"
            )
            if tests.returncode:
                _run(["git", "restore", "--", "src/ai_lca", "tests"], check=False)
                raise RuntimeError("deterministic tests failed; candidate patch reverted")

            result = {
                "applied": True,
                "attempt": attempt,
                "summary": proposal.summary,
                "rationale": proposal.rationale,
                "changed_paths": sorted(_changed_paths(diff)),
                "diff_stat": _run(["git", "diff", "--stat"], check=False).stdout,
                "deterministic_tests_passed": True,
            }
            (output_dir / "iteration.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(result, indent=2))
            return result
        except Exception as exc:
            _run(["git", "restore", "--", "src/ai_lca", "tests"], check=False)
            errors.append(str(exc))
            (attempt_dir / "rejection.txt").write_text(str(exc) + "\n", encoding="utf-8")

    result = {
        "applied": False,
        "errors": errors,
        "reason": "No valid bounded repair was produced within the proposal-attempt cap.",
    }
    (output_dir / "iteration.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Propose one bounded general extractor repair.")
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("OPENAI_REPAIR_MODEL", "gpt-5-mini"))
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-context-chars", type=int, default=90000)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    propose_and_apply(
        args.diagnostics,
        args.output_dir,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_context_chars=args.max_context_chars,
        max_attempts=args.max_attempts,
    )


if __name__ == "__main__":
    main()
