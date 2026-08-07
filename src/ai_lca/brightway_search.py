from __future__ import annotations

from typing import Iterable
import re

import bw2data as bd


LOCATION_ALIASES = {
    "uk": "GB",
    "united kingdom": "GB",
    "great britain": "GB",
    "gb": "GB",
    "norway": "NO",
    "no": "NO",
    "europe": "RER",
    "rer": "RER",
    "global": "GLO",
    "glo": "GLO",
    "rest of world": "RoW",
    "row": "RoW",
}


def set_project(project_name: str) -> None:
    if not project_name.strip():
        raise ValueError("Brightway project name is required")
    bd.projects.set_current(project_name.strip())


def list_databases(project_name: str | None = None) -> list[str]:
    if project_name:
        set_project(project_name)
    return sorted(list(bd.databases))


def _activity_to_dict(activity) -> dict:
    return {
        "id": getattr(activity, "id", None),
        "database": activity.get("database", ""),
        "code": activity.get("code", ""),
        "name": activity.get("name", ""),
        "reference_product": activity.get("reference product", activity.get("product", "")),
        "location": activity.get("location", ""),
        "unit": activity.get("unit", ""),
        "comment": (activity.get("comment", "") or "")[:500],
    }


def _normalise_location(location: str) -> str:
    cleaned = (location or "").strip()
    return LOCATION_ALIASES.get(cleaned.lower(), cleaned)


def _activity_type(name: str) -> str:
    lower = (name or "").lower()
    if lower.startswith("market for ") or lower.startswith("market group for "):
        return "market"
    if lower.startswith("treatment of "):
        return "treatment"
    if "transport" in lower:
        return "transport"
    if "construction" in lower:
        return "construction"
    if "operation" in lower:
        return "operation"
    return "transforming"


def _query_variants(query: str, activity_type_hint: str | None = None) -> list[str]:
    query = (query or "").strip()
    if not query:
        return []

    variants = [query]
    hint = (activity_type_hint or "").strip().lower()
    if hint == "market" and not query.lower().startswith("market for "):
        variants.append(f"market for {query}")
    elif hint == "treatment" and not query.lower().startswith("treatment of "):
        variants.append(f"treatment of {query}")
    elif hint == "operation" and "operation" not in query.lower():
        variants.append(f"{query} operation")
    elif hint == "construction" and "construction" not in query.lower():
        variants.append(f"{query} construction")

    return list(dict.fromkeys(variants))


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(token) > 2}


def _score_candidate(
    candidate: dict,
    *,
    query: str,
    preferred_locations: list[str],
    unit: str | None,
    activity_type_hint: str | None,
    technology_hint: str | None,
    retrieval_rank: int,
) -> tuple[float, list[str]]:
    score = max(0.0, 20.0 - retrieval_rank * 0.25)
    reasons: list[str] = []

    q = query.lower().strip()
    name = (candidate.get("name") or "").lower()
    product = (candidate.get("reference_product") or "").lower()

    if q and q == product:
        score += 28
        reasons.append("reference product exactly matches search concept")
    elif q and q in product:
        score += 20
        reasons.append("reference product contains search concept")

    if q and q in name:
        score += 12
        reasons.append("activity name contains search concept")

    candidate_location = _normalise_location(candidate.get("location") or "")
    if candidate_location in preferred_locations:
        location_rank = preferred_locations.index(candidate_location)
        score += max(4, 18 - 3 * location_rank)
        reasons.append(f"preferred geography {candidate_location}")

    if unit and (candidate.get("unit") or "").strip().lower() == unit.strip().lower():
        score += 10
        reasons.append("unit matches extracted exchange")

    if activity_type_hint and activity_type_hint != "unknown":
        if _activity_type(candidate.get("name") or "") == activity_type_hint:
            score += 14
            reasons.append(f"activity type matches {activity_type_hint} hint")

    if technology_hint:
        tech_tokens = _tokens(technology_hint)
        candidate_tokens = _tokens(f"{candidate.get('name', '')} {candidate.get('reference_product', '')}")
        overlap = tech_tokens & candidate_tokens
        if overlap:
            score += min(10, 2.5 * len(overlap))
            reasons.append("supplier/technology wording overlaps")

    return score, reasons


def search_candidates(
    *,
    project_name: str,
    database_name: str,
    query: str,
    locations: Iterable[str] | None = None,
    limit: int = 12,
    unit: str | None = None,
    activity_type_hint: str | None = None,
    technology_hint: str | None = None,
) -> list[dict]:
    """Retrieve and rank real Brightway activities without fabricating dataset names.

    Location and activity-type hints influence ranking but do not exclude the global search pool.
    """
    set_project(project_name)
    if database_name not in bd.databases:
        raise KeyError(f"Database '{database_name}' not found in Brightway project '{project_name}'")

    db = bd.Database(database_name)
    query = (query or "").strip()
    if not query:
        return []

    preferred_locations = [
        _normalise_location(x) for x in (locations or []) if x and x.strip()
    ]
    preferred_locations = list(dict.fromkeys(preferred_locations))

    variants = _query_variants(query, activity_type_hint)
    pool_limit = max(25, limit * 4)
    collected: list[tuple[object, int]] = []
    seen: set[tuple] = set()
    retrieval_rank = 0

    # Always search globally first; preferences must not become hard filters.
    for variant in variants:
        for act in db.search(variant, limit=pool_limit):
            key = (act.get("database"), act.get("code"))
            if key not in seen:
                seen.add(key)
                collected.append((act, retrieval_rank))
                retrieval_rank += 1

    # Also ask Brightway for preferred-location hits so relevant local datasets are not buried.
    for location in preferred_locations:
        for variant in variants:
            try:
                results = db.search(variant, limit=max(limit, 8), filter={"location": location})
            except Exception:
                results = []
            for act in results:
                key = (act.get("database"), act.get("code"))
                if key not in seen:
                    seen.add(key)
                    collected.append((act, retrieval_rank))
                    retrieval_rank += 1

    ranked: list[dict] = []
    for act, rank in collected:
        candidate = _activity_to_dict(act)
        score, reasons = _score_candidate(
            candidate,
            query=query,
            preferred_locations=preferred_locations,
            unit=unit,
            activity_type_hint=activity_type_hint,
            technology_hint=technology_hint,
            retrieval_rank=rank,
        )
        candidate["match_score"] = round(score, 2)
        candidate["match_reasons"] = "; ".join(reasons)
        ranked.append(candidate)

    ranked.sort(key=lambda item: item["match_score"], reverse=True)
    return ranked[:limit]
