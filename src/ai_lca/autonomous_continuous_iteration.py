from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from . import autonomous_code_iteration as base


_HISTORY_KEYS = (
    "summary",
    "rationale",
    "rejected_at_gate",
    "regression_reasons",
)


def _load_rejection_history(
    path: Path, failure_class: str, max_entries: int = 12
) -> list[dict[str, Any]]:
    """Load only sanitized, same-class lessons from prior rejected experiments."""
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if record.get("failure_class") != failure_class:
            continue
        entries.append({key: record.get(key) for key in _HISTORY_KEYS})
    return entries[-max_entries:]


def _history_context(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return ""
    return (
        "\n===== PRIOR REJECTED EXPERIMENTS =====\n"
        "These are sanitized lessons from earlier corpus-gated attempts for the same "
        "failure class. Do not repeat an approach that failed for the same reason; "
        "find a materially different general repair. Paper identity is intentionally "
        "excluded.\n"
        + json.dumps(entries, indent=2)
        + "\n===== END PRIOR REJECTED EXPERIMENTS =====\n"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    diagnostics = base._load(args.diagnostics)
    ranking = diagnostics.get("failure_class_ranking") or []
    if not ranking:
        raise RuntimeError("no diagnosable failure classes")
    target = args.failure_class or ranking[0]["failure_class"]
    history = _load_rejection_history(
        args.history_file, target, max_entries=args.max_history_entries
    )
    extra_context = _history_context(history)

    original_context = base._context

    def context_with_history(files: list[str], max_chars: int) -> str:
        code_context = original_context(files, max_chars)
        return extra_context + code_context

    base._context = context_with_history
    try:
        return base.propose_and_apply(args)
    finally:
        base._context = original_context


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous autonomous repair proposal with rejection memory."
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
    parser.add_argument(
        "--history-file",
        type=Path,
        default=Path(".autonomy/inventory_rejection_history.jsonl"),
    )
    parser.add_argument("--max-history-entries", type=int, default=12)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
