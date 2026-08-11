from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .corpus_diagnostics import BASELINE_MANIFEST
from .inventory_replay import CANARY_MANIFEST, _paper_dir, _read


def build_review_queue(
    state_dir: Path,
    manifest: Path = BASELINE_MANIFEST,
    selection: Path = CANARY_MANIFEST,
    resolved_sample_per_paper: int = 8,
) -> dict[str, Any]:
    manifest_payload = _read(manifest, {}) or {}
    selected_payload = _read(selection, {}) or {}
    selected = set(selected_payload.get("canary_dois") or [])
    by_doi = {p["doi"]: p for p in manifest_payload.get("papers") or []}
    items: list[dict[str, Any]] = []

    for doi in selected:
        paper = by_doi.get(doi)
        if not paper:
            continue
        extraction = _paper_dir(state_dir, paper) / "extraction"
        candidates = _read(extraction / "inventory_candidates.json", []) or []
        candidate_map = {c["candidate_id"]: c for c in candidates}
        assignments = (_read(extraction / "assignments.json", {}) or {}).get(
            "assignments"
        ) or []
        assignment_map = {a["candidate_id"]: a for a in assignments}

        process_ambiguous: set[str] = set()
        process_dir = extraction / "processes"
        if process_dir.exists():
            for path in process_dir.glob("*.json"):
                payload = _read(path, {}) or {}
                process_ambiguous.update(payload.get("ambiguous_candidate_ids") or [])

        priority_ids: list[str] = []
        for candidate in candidates:
            cid = candidate["candidate_id"]
            assignment = assignment_map.get(cid)
            if (
                assignment is None
                or assignment.get("disposition") == "ambiguous"
                or cid in process_ambiguous
            ):
                priority_ids.append(cid)

        # Add a bounded resolved sample so precision can be checked as well as recall.
        if paper.get("status") == "COMPLETE":
            added = 0
            for assignment in assignments:
                if assignment.get("disposition") == "modeled_inventory":
                    cid = assignment["candidate_id"]
                    if cid not in priority_ids:
                        priority_ids.append(cid)
                        added += 1
                    if added >= resolved_sample_per_paper:
                        break

        for cid in priority_ids:
            candidate = candidate_map.get(cid)
            if not candidate:
                continue
            assignment = assignment_map.get(cid) or {}
            items.append(
                {
                    "review_id": f"{doi}|{cid}",
                    "doi": doi,
                    "title": paper.get("title"),
                    "baseline_status": paper.get("status"),
                    "candidate_id": cid,
                    "source_location": candidate.get("source_location"),
                    "evidence_type": candidate.get("evidence_type"),
                    "evidence_text": candidate.get("evidence_text"),
                    "context": candidate.get("context"),
                    "current_disposition": assignment.get("disposition"),
                    "current_process_ids": assignment.get("process_ids") or [],
                    "gold_disposition": None,
                    "gold_process_ids": [],
                    "gold_flow_name": None,
                    "gold_amount": None,
                    "gold_unit": None,
                    "gold_direction": None,
                    "review_notes": None,
                    "reviewed_by_human": False,
                }
            )

    return {
        "baseline_id": manifest_payload.get("baseline_id"),
        "selection": str(selection),
        "status": "PENDING_HUMAN_REVIEW",
        "review_item_count": len(items),
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a candidate-level human gold review queue without API calls."
    )
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=BASELINE_MANIFEST)
    parser.add_argument("--selection", type=Path, default=CANARY_MANIFEST)
    parser.add_argument("--resolved-sample-per-paper", type=int, default=8)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/gold_review_queue.json")
    )
    args = parser.parse_args()
    queue = build_review_queue(
        args.state_dir,
        args.manifest,
        args.selection,
        args.resolved_sample_per_paper,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(queue, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": queue["status"],
                "review_item_count": queue["review_item_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
