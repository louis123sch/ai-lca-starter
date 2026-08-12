from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .llm import _client


class TextReplacement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    old_text: str
    new_text: str


class RepairProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    rationale: str
    edits: list[TextReplacement] = Field(min_length=1, max_length=6)


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

EDIT FORMAT:
- Do NOT return a git diff or patch.
- Return exact text replacements only.
- Each edit must name one allowed file and provide an old_text block copied VERBATIM from CURRENT CODE plus the desired new_text replacement.
- old_text must be specific enough to occur exactly once in that file.
- Keep edits small. Prefer one or two focused replacements over rewriting whole files.

The repair will be rejected if any protected benchmark falls below its absolute quality floor or below the accepted baseline. Do not trade one benchmark for another.
"""


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def _changed_paths(proposal: RepairProposal) -> set[str]:
    return {edit.path for edit in proposal.edits}


def _validate_proposal(proposal: RepairProposal) -> None:
    allowed = set(ALLOWED_FILES)
    paths = _changed_paths(proposal)
    if not paths:
        raise ValueError("proposal contains no edits")
    if paths - allowed:
        raise ValueError(f"proposal touched disallowed paths: {sorted(paths - allowed)}")
    if any(path.startswith(FORBIDDEN_PREFIXES) for path in paths):
        raise ValueError("proposal attempted to modify protected infrastructure/data")

    proposed_text = "\n".join(edit.new_text for edit in proposal.edits)
    for pattern in FORBIDDEN_PATCH_PATTERNS:
        if re.search(pattern, proposed_text, re.IGNORECASE | re.DOTALL):
            raise ValueError(f"proposal violates anti-overfitting policy: {pattern}")

    for edit in proposal.edits:
        if not edit.old_text:
            raise ValueError(f"edit for {edit.path} has empty old_text")
        if edit.old_text == edit.new_text:
            raise ValueError(f"edit for {edit.path} makes no change")
        if len(edit.old_text) > 20000 or len(edit.new_text) > 30000:
            raise ValueError(f"edit for {edit.path} is too large for a bounded repair")


def _context(max_chars: int) -> str:
    chunks: list[str] = []
    remaining = max_chars
    for name in ALLOWED_FILES:
        path = Path(name)
        if not path.exists() or remaining <= 0:
            continue
        text = path.read_text(encoding="utf-8")
        if len(text) > remaining:
            text = text[:remaining]
        chunks.append(f"\n===== FILE: {name} =====\n{text}")
        remaining -= len(text)
    return "".join(chunks)


def _apply_exact_edits(proposal: RepairProposal) -> None:
    staged: dict[str, str] = {}
    for edit in proposal.edits:
        text = staged.get(edit.path)
        if text is None:
            text = Path(edit.path).read_text(encoding="utf-8")
        count = text.count(edit.old_text)
        if count != 1:
            raise RuntimeError(
                f"exact replacement failed for {edit.path}: old_text occurs {count} times; "
                "copy a unique old_text block verbatim from CURRENT CODE"
            )
        staged[edit.path] = text.replace(edit.old_text, edit.new_text, 1)

    for path, text in staged.items():
        Path(path).write_text(text, encoding="utf-8")


def _call_repair_model(
    *,
    model: str,
    reasoning_effort: str,
    prompt: str,
    api_attempts: int = 3,
) -> RepairProposal | None:
    last_error: Exception | None = None
    transient_names = {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }
    for api_attempt in range(1, api_attempts + 1):
        try:
            completion = _client().beta.chat.completions.parse(
                model=model,
                reasoning_effort=reasoning_effort,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=RepairProposal,
            )
            return completion.choices[0].message.parsed
        except Exception as exc:
            last_error = exc
            if exc.__class__.__name__ not in transient_names or api_attempt >= api_attempts:
                raise
            delay = 15 * (2 ** (api_attempt - 1))
            print(
                f"Transient repair-model failure on API attempt {api_attempt}/{api_attempts}: "
                f"{exc.__class__.__name__}: {exc}. Retrying in {delay}s."
            )
            time.sleep(delay)
    if last_error is not None:
        raise last_error
    return None


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
                + "\nKeep the same general objective, but return a valid smaller exact-text edit. "
                "Copy old_text verbatim from CURRENT CODE.\n"
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

        try:
            proposal = _call_repair_model(
                model=model,
                reasoning_effort=reasoning_effort,
                prompt=prompt,
                api_attempts=3,
            )
        except Exception as exc:
            errors.append(f"repair model call failed: {exc.__class__.__name__}: {exc}")
            attempt_dir = output_dir / f"attempt_{attempt:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            (attempt_dir / "rejection.txt").write_text(errors[-1] + "\n", encoding="utf-8")
            continue

        if proposal is None:
            errors.append("repair model returned no structured proposal")
            continue

        attempt_dir = output_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "proposal.json").write_text(
            proposal.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

        try:
            _validate_proposal(proposal)
            _apply_exact_edits(proposal)

            tests = _run(["python", "-m", "pytest", "-q"], check=False)
            (attempt_dir / "deterministic_tests.log").write_text(
                tests.stdout + "\n" + tests.stderr, encoding="utf-8"
            )
            if tests.returncode:
                _run(["git", "restore", "--", "src/ai_lca", "tests"], check=False)
                raise RuntimeError("deterministic tests failed; candidate edits reverted")

            diff = _run(["git", "diff", "--", "src/ai_lca", "tests"], check=False).stdout
            (attempt_dir / "applied.diff").write_text(diff, encoding="utf-8")
            result = {
                "applied": True,
                "attempt": attempt,
                "summary": proposal.summary,
                "rationale": proposal.rationale,
                "changed_paths": sorted(_changed_paths(proposal)),
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
