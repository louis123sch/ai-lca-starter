from __future__ import annotations

import argparse
import difflib
import json
import os
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field


TARGET = Path("src/ai_lca/llm.py")
BANNED_ADDED_FRAGMENTS = (
    "requests",
    "urllib",
    "socket",
    "os.system",
    "popen",
    "eval(",
    "exec(",
    "http://",
    "https://",
    "openai_api_key",
    "github_token",
    "gh_token",
)

# These are general extraction invariants already enforced by deterministic tests.
# Automatic prompt repair may strengthen them, but must not silently remove them.
REQUIRED_PROMPT_INVARIANTS = (
    "explicit component lists",
    "foreground input flow",
    "do not reclassify such tabulated components as background subprocesses",
)


class RepairProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    should_change: bool
    rationale: str
    replacement_content: str | None = Field(
        default=None,
        description="Complete replacement content for src/ai_lca/llm.py when should_change is true.",
    )


def _load_reports(root: Path) -> str:
    pieces: list[str] = []
    summary = root / "summary.json"
    if summary.exists():
        pieces.append("SUMMARY\n" + summary.read_text())
    for report in sorted(root.glob("report_run_*.json"))[:5]:
        pieces.append(f"{report.name}\n{report.read_text()}")
    return "\n\n".join(pieces)


def _validate_replacement(old: str, new: str) -> None:
    if not new.strip():
        raise ValueError("Repair agent returned empty replacement content")
    if "STRUCTURE_SYSTEM_PROMPT" not in new or "FLOW_SYSTEM_PROMPT" not in new:
        raise ValueError("Repair would remove required extraction architecture")

    lowered = new.casefold()
    missing_invariants = [
        invariant for invariant in REQUIRED_PROMPT_INVARIANTS if invariant not in lowered
    ]
    if missing_invariants:
        raise ValueError(
            "Repair would remove required prompt invariants: "
            + " | ".join(missing_invariants)
        )

    diff = difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="")
    added = [line[1:] for line in diff if line.startswith("+") and not line.startswith("+++")]
    suspicious = [
        line
        for line in added
        if any(fragment in line.casefold() for fragment in BANNED_ADDED_FRAGMENTS)
    ]
    if suspicious:
        raise ValueError("Repair contains disallowed security-sensitive additions: " + " | ".join(suspicious[:5]))


def propose_repair(
    benchmark_root: Path,
    *,
    model: str,
    source_paths: list[Path],
) -> RepairProposal:
    current = TARGET.read_text()
    reports = _load_reports(benchmark_root)
    source = "\n\n".join(f"SOURCE {p}:\n{p.read_text()}" for p in source_paths if p.exists())
    prompt = f"""A paper-grounded AI-LCA extraction benchmark failed.

You may make at most ONE narrow, generalisable change to src/ai_lca/llm.py. Do not hard-code this paper, process names, quantities, benchmark thresholds, expected values, or aliases. Do not modify the gold standard. Preserve the architecture: source evidence -> process structure -> locked flow extraction -> human review -> Brightway matching. Preserve anti-hallucination, foreground/background separation, and anti-over-decomposition constraints.

Prefer a prompt/reasoning improvement that would make sense for unseen LCA papers. If the failure is fundamentally document ingestion rather than LCA reasoning, set should_change=false; ingestion code must be fixed by a human-reviewed code change instead of pretending a prompt can see missing source evidence.

Do not add network libraries, shell/system execution, secret handling, filesystem writes, or external URLs. Preserve existing public functions and imports unless a small harmless import is essential.

BENCHMARK REPORTS:
{reports}

SOURCE FIXTURES (the model under test was allowed to see these):
{source}

CURRENT src/ai_lca/llm.py:
{current}
"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a conservative software repair agent for a scientific LCA extraction system. Make the smallest defensible generalisable change or decline to change.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=RepairProposal,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("Repair model returned no parsed proposal")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("OPENAI_REPAIR_MODEL", "gpt-5-mini"))
    parser.add_argument("--source", nargs="*", type=Path, default=[])
    parser.add_argument("--result", type=Path, default=Path("repair_result.json"))
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for repair-agent execution")

    old = TARGET.read_text()
    proposal = propose_repair(args.benchmark_root, model=args.model, source_paths=args.source)
    result = proposal.model_dump()

    if proposal.should_change:
        if proposal.replacement_content is None:
            raise ValueError("should_change=true but replacement_content is missing")
        _validate_replacement(old, proposal.replacement_content)
        if proposal.replacement_content != old:
            TARGET.write_text(proposal.replacement_content)
            result["changed"] = True
        else:
            result["changed"] = False
    else:
        result["changed"] = False

    args.result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"changed": result["changed"], "rationale": proposal.rationale}, indent=2))


if __name__ == "__main__":
    main()
