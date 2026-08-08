from __future__ import annotations

import re


def _normalise_text(value: str | None) -> str:
    return " ".join((value or "").split()).casefold()


def _normalise_dataset_name(value: str | None) -> str:
    """Ignore case, punctuation, and cosmetic spacing in activity names."""
    return "".join(re.findall(r"[a-z0-9]+", (value or "").casefold()))


def recommended_candidate_index(
    candidates: list[dict],
    *,
    source_activity_hint: str | None = None,
    mapping_relation: str | None = None,
    target: str = "technosphere",
) -> tuple[int | None, str]:
    """Return a conservative auto-selection recommendation.

    Source-supported exact/proxy mappings are preferred only when the named dataset is
    actually present in the returned candidates. Uncertain mappings are never
    auto-approved. Search-only recommendations need a high score, a clear margin, and
    strong lexical evidence.
    """
    usable = [candidate for candidate in candidates if "error" not in candidate]
    if not usable:
        return None, "No candidate is available for automatic selection."

    relation = _normalise_text(mapping_relation)
    source_hint = _normalise_dataset_name(source_activity_hint)

    if relation == "uncertain":
        return None, "The source mapping is marked uncertain, so it requires manual approval."

    if source_hint and relation in {"exact", "proxy"}:
        for index, candidate in enumerate(usable):
            if _normalise_dataset_name(candidate.get("name")) == source_hint:
                label = "exact source mapping" if relation == "exact" else "source-supported proxy"
                return index, f"Preselected because the candidate matches the {label}."
        return None, "The source names a background dataset, but that exact dataset was not returned; review manually."

    top = usable[0]
    top_score = float(top.get("match_score") or 0)
    second_score = float(usable[1].get("match_score") or 0) if len(usable) > 1 else 0.0
    margin = top_score - second_score
    reasons = _normalise_text(top.get("match_reasons"))

    if target == "biosphere":
        strong_name = (
            "biosphere name exactly matches query" in reasons
            or "biosphere name contains query" in reasons
        )
        strong_context = (
            "unit matches" in reasons
            or "compartment matches" in reasons
        )
        if top_score >= 90 and margin >= 10 and strong_name and strong_context:
            return 0, "Preselected because the biosphere match is strong and clearly separated from alternatives."
        return None, "Biosphere match is not unambiguous enough for automatic approval."

    strong_identity = (
        "activity name exactly matches query" in reasons
        or "reference product exactly matches query" in reasons
    )
    if top_score >= 90 and margin >= 8 and strong_identity:
        return 0, "Preselected because the technosphere match is strong and clearly separated from alternatives."

    return None, "Search result is plausible but not unambiguous enough for automatic approval."
