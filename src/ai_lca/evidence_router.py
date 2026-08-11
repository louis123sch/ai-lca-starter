from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

from .jats import InventoryCandidate, JATSDocument

EvidenceLabel = Literal[
    "foreground_lci",
    "structure",
    "modelling_assumption",
    "lcia_result",
    "uncertain",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_/][a-z0-9]+)?", re.IGNORECASE)

_LCIA_PATTERNS = (
    re.compile(r"\bgwp\b", re.IGNORECASE),
    re.compile(r"global warming", re.IGNORECASE),
    re.compile(r"climate change", re.IGNORECASE),
    re.compile(r"impact categor", re.IGNORECASE),
    re.compile(r"acidification|eutrophication|photochemical|ozone depletion|human toxicity|ecotoxicity", re.IGNORECASE),
    re.compile(r"(?:kg|g|t)\s*(?:co2|co₂)[- ]?(?:eq|equiv)", re.IGNORECASE),
    re.compile(r"environmental impact|lcia|life cycle impact", re.IGNORECASE),
)

_LCI_PATTERNS = (
    re.compile(r"\binput(?:s)?\b|\boutput(?:s)?\b", re.IGNORECASE),
    re.compile(r"inventory|\blci\b|bill of materials|\bbom\b", re.IGNORECASE),
    re.compile(r"consumption|demand|feedstock|raw material", re.IGNORECASE),
    re.compile(r"electricity|natural gas|methane|hydrogen|water|steam|heat|diesel|fuel", re.IGNORECASE),
    re.compile(r"steel|aluminium|aluminum|nickel|copper|cement|concrete|glass|plastic|polymer|catalyst", re.IGNORECASE),
    re.compile(r"transport|freight|waste|emission|carbon dioxide|oxygen|nitrogen", re.IGNORECASE),
)

_STRUCTURE_PATTERNS = (
    re.compile(r"functional unit|reference flow|reference product", re.IGNORECASE),
    re.compile(r"system bound|goal and scope|goal & scope", re.IGNORECASE),
    re.compile(r"process|pathway|route|configuration|scenario|technology|foreground", re.IGNORECASE),
    re.compile(r"manufactur|production|construction|operation|use phase|end of life", re.IGNORECASE),
)

_ASSUMPTION_PATTERNS = (
    re.compile(r"allocation|cut[- ]?off|system expansion", re.IGNORECASE),
    re.compile(r"lifetime|service life|operating hours|capacity factor", re.IGNORECASE),
    re.compile(r"efficien|yield|conversion|recovery|loss(?:es)?", re.IGNORECASE),
    re.compile(r"recycl|reuse|replacement|utilisation|utilization", re.IGNORECASE),
)


@dataclass(frozen=True)
class EvidenceRoute:
    candidate_id: str
    label: EvidenceLabel
    confidence: float
    safe_to_exclude_from_inventory_reasoning: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "safe_to_exclude_from_inventory_reasoning": self.safe_to_exclude_from_inventory_reasoning,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class StructureEvidencePack:
    text: str
    selected_sections: tuple[str, ...]
    selected_tables: tuple[str, ...]
    omitted_section_count: int
    omitted_table_count: int
    char_count: int

    def manifest(self) -> dict:
        return {
            "selected_sections": list(self.selected_sections),
            "selected_tables": list(self.selected_tables),
            "omitted_section_count": self.omitted_section_count,
            "omitted_table_count": self.omitted_table_count,
            "char_count": self.char_count,
        }


def _count(patterns: Iterable[re.Pattern[str]], text: str) -> int:
    return sum(1 for pattern in patterns if pattern.search(text))


def _candidate_text(candidate: InventoryCandidate) -> str:
    return "\n".join(
        part
        for part in (candidate.evidence_text, candidate.context, candidate.table or "")
        if part
    )


def route_inventory_candidate(candidate: InventoryCandidate) -> EvidenceRoute:
    """Route evidence conservatively before expensive reasoning.

    Only very strong LCIA/result evidence is safe-excluded. All other labels are
    advisory and remain available to downstream reasoning. This intentionally
    optimizes recall over token reduction.
    """

    text = _candidate_text(candidate)
    lcia = _count(_LCIA_PATTERNS, text)
    lci = _count(_LCI_PATTERNS, text)
    structure = _count(_STRUCTURE_PATTERNS, text)
    assumption = _count(_ASSUMPTION_PATTERNS, text)
    reasons: list[str] = []

    if lcia:
        reasons.append(f"lcia_signals={lcia}")
    if lci:
        reasons.append(f"lci_signals={lci}")
    if structure:
        reasons.append(f"structure_signals={structure}")
    if assumption:
        reasons.append(f"assumption_signals={assumption}")

    # A CO2 emission can be an LCI flow, so LCIA is only auto-excluded when
    # impact/category semantics are strong and no inventory semantics compete.
    strong_lcia = lcia >= 2 and lci == 0
    if strong_lcia:
        confidence = min(0.99, 0.90 + 0.03 * (lcia - 2))
        return EvidenceRoute(candidate.candidate_id, "lcia_result", confidence, True, tuple(reasons))

    if lci >= 2 or (lci >= 1 and candidate.evidence_type == "table_row"):
        confidence = min(0.97, 0.72 + 0.07 * max(0, lci - 1))
        return EvidenceRoute(candidate.candidate_id, "foreground_lci", confidence, False, tuple(reasons))

    if assumption >= 2 and lci == 0:
        confidence = min(0.92, 0.72 + 0.06 * assumption)
        return EvidenceRoute(candidate.candidate_id, "modelling_assumption", confidence, False, tuple(reasons))

    if structure >= 2 and lci == 0:
        confidence = min(0.90, 0.70 + 0.05 * structure)
        return EvidenceRoute(candidate.candidate_id, "structure", confidence, False, tuple(reasons))

    return EvidenceRoute(candidate.candidate_id, "uncertain", 0.50, False, tuple(reasons or ["no decisive routing signal"]))


def route_inventory_candidates(candidates: Iterable[InventoryCandidate]) -> list[EvidenceRoute]:
    return [route_inventory_candidate(candidate) for candidate in candidates]


def partition_inventory_candidates(
    candidates: list[InventoryCandidate],
) -> tuple[list[InventoryCandidate], list[InventoryCandidate], list[EvidenceRoute]]:
    routes = route_inventory_candidates(candidates)
    route_by_id = {route.candidate_id: route for route in routes}
    retained: list[InventoryCandidate] = []
    excluded: list[InventoryCandidate] = []
    for candidate in candidates:
        route = route_by_id[candidate.candidate_id]
        if route.safe_to_exclude_from_inventory_reasoning:
            excluded.append(candidate)
        else:
            retained.append(candidate)
    return retained, excluded, routes


def routed_candidate_payload(candidate: InventoryCandidate, route: EvidenceRoute) -> dict:
    payload = candidate.as_dict()
    payload["evidence_route"] = route.label
    payload["route_confidence"] = round(route.confidence, 4)
    payload["route_is_advisory"] = True
    return payload


def _section_score(title: str, text: str) -> int:
    sample = f"{title}\n{text[:2400]}"
    return (
        4 * _count(_STRUCTURE_PATTERNS, sample)
        + 2 * _count(_ASSUMPTION_PATTERNS, sample)
        + 2 * _count(_LCI_PATTERNS, sample)
        - 2 * _count(_LCIA_PATTERNS, sample)
    )


def _table_score(caption: str, rows: list[str]) -> int:
    sample = f"{caption}\n" + "\n".join(rows[:12])
    return (
        3 * _count(_LCI_PATTERNS, sample)
        + 2 * _count(_STRUCTURE_PATTERNS, sample)
        + _count(_ASSUMPTION_PATTERNS, sample)
        - 2 * _count(_LCIA_PATTERNS, sample)
    )


def build_structure_evidence(doc: JATSDocument, max_chars: int = 50_000) -> StructureEvidencePack:
    """Build a bounded, provenance-preserving structure evidence pack.

    Title and abstract are always retained. Sections/tables are ranked, but the
    chosen evidence is restored to document order before rendering so the model
    still sees coherent source context. If the router cannot find enough clear
    evidence, it widens automatically rather than returning a tiny context.
    """

    header = [
        f"[DOCUMENT: JATS XML]\n[TITLE]\n{doc.title}",
        f"[ABSTRACT]\n{doc.abstract}",
    ]

    ranked_sections = sorted(
        ((idx, title, text, _section_score(title, text)) for idx, (title, text) in enumerate(doc.sections)),
        key=lambda item: (item[3], len(item[2])),
        reverse=True,
    )
    selected_section_ids = {idx for idx, _, _, score in ranked_sections if score >= 4}
    minimum_sections = min(8, len(ranked_sections))
    for idx, _, _, _ in ranked_sections[:minimum_sections]:
        selected_section_ids.add(idx)

    ranked_tables = sorted(
        ((idx, label, caption, rows, _table_score(caption, rows)) for idx, (label, caption, rows) in enumerate(doc.tables)),
        key=lambda item: (item[4], len(item[3])),
        reverse=True,
    )
    selected_table_ids = {idx for idx, _, _, _, score in ranked_tables if score >= 3}
    for idx, _, _, _, _ in ranked_tables[: min(6, len(ranked_tables))]:
        selected_table_ids.add(idx)

    parts = list(header)
    selected_sections: list[str] = []
    selected_tables: list[str] = []

    for idx, (title, text) in enumerate(doc.sections):
        if idx not in selected_section_ids:
            continue
        block = f"[SECTION: {title or 'untitled'}]\n{text}"
        if len("\n\n".join(parts + [block])) > max_chars:
            continue
        parts.append(block)
        selected_sections.append(title or f"section-{idx + 1}")

    for idx, (label, caption, rows) in enumerate(doc.tables):
        if idx not in selected_table_ids:
            continue
        row_text = "\n".join(f"[TABLE: {label} | ROW {row_idx}] {row}" for row_idx, row in enumerate(rows, 1))
        block = f"[TABLE: {label}]\nCAPTION: {caption}\n{row_text}"
        if len("\n\n".join(parts + [block])) > max_chars:
            # Keep the caption even when the full table will not fit.
            caption_only = f"[TABLE: {label}]\nCAPTION: {caption}"
            if len("\n\n".join(parts + [caption_only])) <= max_chars:
                parts.append(caption_only)
                selected_tables.append(label)
            continue
        parts.append(block)
        selected_tables.append(label)

    text = "\n\n".join(parts)[:max_chars]
    return StructureEvidencePack(
        text=text,
        selected_sections=tuple(selected_sections),
        selected_tables=tuple(selected_tables),
        omitted_section_count=max(0, len(doc.sections) - len(selected_sections)),
        omitted_table_count=max(0, len(doc.tables) - len(selected_tables)),
        char_count=len(text),
    )


def normalised_contains(haystack: str, needle: str) -> bool:
    def norm(value: str) -> str:
        return " ".join(_TOKEN_RE.findall(value.casefold()))

    target = norm(needle)
    return bool(target) and target in norm(haystack)
