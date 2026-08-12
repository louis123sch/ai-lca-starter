from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./%-]{1,}|\d+(?:[.,]\d+)?")

Route = Literal["foreground_lci", "unlikely_lci", "uncertain"]


def _tokens(text: str) -> list[str]:
    return [m.group(0).casefold() for m in TOKEN_RE.finditer(text)]


def _candidate_text(candidate: dict) -> str:
    return " ".join(
        str(candidate.get(key) or "")
        for key in ("evidence_type", "table", "context", "evidence_text")
    )


@dataclass(frozen=True)
class RoutedEvidence:
    candidate_id: str
    probability_lci: float
    route: Route


class EvidenceRelevanceRouter:
    """Cheap corpus-trained router used before expensive LLM reasoning.

    This is intentionally a router, not an extractor. It never invents flows and
    never mutates candidates. Callers must retain the full candidate set for audit
    and completeness fallback.
    """

    def __init__(self, *, alpha: float = 1.0, high_threshold: float = 0.65, low_threshold: float = 0.20):
        if not 0 < low_threshold < high_threshold < 1:
            raise ValueError("thresholds must satisfy 0 < low < high < 1")
        self.alpha = float(alpha)
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        self.positive = Counter()
        self.negative = Counter()
        self.positive_docs = 0
        self.negative_docs = 0
        self.vocabulary: set[str] = set()

    def fit(self, examples: Iterable[tuple[str, bool]]) -> "EvidenceRelevanceRouter":
        for text, positive in examples:
            counts = Counter(_tokens(text))
            if not counts:
                continue
            target = self.positive if positive else self.negative
            target.update(counts)
            self.vocabulary.update(counts)
            if positive:
                self.positive_docs += 1
            else:
                self.negative_docs += 1
        if not self.positive_docs or not self.negative_docs:
            raise ValueError("router training requires positive and negative examples")
        return self

    def probability_lci(self, text: str) -> float:
        vocab = max(1, len(self.vocabulary))
        pos_total = sum(self.positive.values()) + self.alpha * vocab
        neg_total = sum(self.negative.values()) + self.alpha * vocab
        prior_pos = (self.positive_docs + self.alpha) / (self.positive_docs + self.negative_docs + 2 * self.alpha)
        prior_neg = 1.0 - prior_pos
        log_odds = math.log(prior_pos / prior_neg)
        for token, count in Counter(_tokens(text)).items():
            pos_p = (self.positive[token] + self.alpha) / pos_total
            neg_p = (self.negative[token] + self.alpha) / neg_total
            log_odds += min(count, 3) * math.log(pos_p / neg_p)
        log_odds = max(-30.0, min(30.0, log_odds))
        return 1.0 / (1.0 + math.exp(-log_odds))

    def route_candidate(self, candidate: dict) -> RoutedEvidence:
        p = self.probability_lci(_candidate_text(candidate))
        if p >= self.high_threshold:
            route: Route = "foreground_lci"
        elif p <= self.low_threshold:
            route = "unlikely_lci"
        else:
            route = "uncertain"
        return RoutedEvidence(str(candidate.get("candidate_id") or ""), p, route)

    def route(self, candidates: Iterable[dict]) -> list[RoutedEvidence]:
        return [self.route_candidate(candidate) for candidate in candidates]


def load_corpus_examples(state_dir: Path, *, excluded_paper: str | None = None) -> list[tuple[str, bool]]:
    examples: list[tuple[str, bool]] = []
    corpus = state_dir / "corpus"
    for paper_dir in sorted(corpus.iterdir() if corpus.exists() else []):
        if excluded_paper and paper_dir.name == excluded_paper:
            continue
        extraction = paper_dir / "extraction"
        candidates_path = extraction / "inventory_candidates.json"
        assignments_path = extraction / "assignments.json"
        if not candidates_path.exists() or not assignments_path.exists():
            continue
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
        assignments = json.loads(assignments_path.read_text(encoding="utf-8")).get("assignments", [])
        candidate_map = {str(c["candidate_id"]): c for c in candidates}
        for assignment in assignments:
            disposition = assignment.get("disposition")
            if disposition not in {"modeled_inventory", "not_inventory"}:
                continue
            candidate = candidate_map.get(str(assignment.get("candidate_id")))
            if candidate is not None:
                examples.append((_candidate_text(candidate), disposition == "modeled_inventory"))
    return examples


def evaluate_leave_one_paper_out(state_dir: Path, *, high_threshold: float = 0.65, low_threshold: float = 0.20) -> dict:
    totals = Counter()
    per_paper: list[dict] = []
    corpus = state_dir / "corpus"
    paper_dirs = [p for p in sorted(corpus.iterdir() if corpus.exists() else []) if (p / "extraction" / "assignments.json").exists()]
    for paper_dir in paper_dirs:
        train = load_corpus_examples(state_dir, excluded_paper=paper_dir.name)
        if not train:
            continue
        router = EvidenceRelevanceRouter(high_threshold=high_threshold, low_threshold=low_threshold).fit(train)
        candidates = json.loads((paper_dir / "extraction" / "inventory_candidates.json").read_text(encoding="utf-8"))
        assignments = json.loads((paper_dir / "extraction" / "assignments.json").read_text(encoding="utf-8")).get("assignments", [])
        cmap = {str(c["candidate_id"]): c for c in candidates}
        local = Counter()
        for assignment in assignments:
            disposition = assignment.get("disposition")
            if disposition not in {"modeled_inventory", "not_inventory"}:
                continue
            candidate = cmap.get(str(assignment.get("candidate_id")))
            if candidate is None:
                continue
            routed = router.route_candidate(candidate)
            is_positive = disposition == "modeled_inventory"
            local["labelled"] += 1
            local["positive" if is_positive else "negative"] += 1
            local[f"route_{routed.route}"] += 1
            if is_positive and routed.route == "unlikely_lci":
                local["false_negative_low"] += 1
            if is_positive and routed.route != "unlikely_lci":
                local["positive_retained"] += 1
            if not is_positive and routed.route == "unlikely_lci":
                local["negative_pruned"] += 1
        totals.update(local)
        per_paper.append({"paper": paper_dir.name, **dict(local)})

    positive = totals["positive"]
    negative = totals["negative"]
    labelled = totals["labelled"]
    return {
        "paper_count": len(per_paper),
        "labelled_candidates": labelled,
        "positive_candidates": positive,
        "negative_candidates": negative,
        "positive_recall_if_low_route_skipped": round(totals["positive_retained"] / positive, 6) if positive else None,
        "negative_prune_rate": round(totals["negative_pruned"] / negative, 6) if negative else None,
        "candidate_reduction_rate": round(totals["route_unlikely_lci"] / labelled, 6) if labelled else None,
        "false_negative_low": totals["false_negative_low"],
        "route_counts": {
            "foreground_lci": totals["route_foreground_lci"],
            "uncertain": totals["route_uncertain"],
            "unlikely_lci": totals["route_unlikely_lci"],
        },
        "thresholds": {"high": high_threshold, "low": low_threshold},
        "safety_rule": "Full deterministic candidate set remains stored; unlikely_lci is routing advice only until a recall gate is passed.",
        "per_paper": per_paper,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate the retrieval-before-reasoning LCI evidence router on a frozen corpus.")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--high-threshold", type=float, default=0.65)
    parser.add_argument("--low-threshold", type=float, default=0.20)
    args = parser.parse_args()
    report = evaluate_leave_one_paper_out(args.state_dir, high_threshold=args.high_threshold, low_threshold=args.low_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_paper"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
