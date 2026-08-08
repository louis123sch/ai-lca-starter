from __future__ import annotations

import re

import pycountry
import bw2data as bd


REGIONAL_LOCATION_ALIASES: dict[str, list[str]] = {
    "europe": ["RER"],
    "european union": ["RER"],
    "eu": ["RER"],
    "global": ["GLO"],
    "world": ["GLO"],
    "worldwide": ["GLO"],
    "rest of world": ["RoW"],
}

ACTIVITY_TYPES = {
    "market",
    "transforming",
    "treatment",
    "transport",
    "construction",
    "operation",
    "unknown",
}


def set_project(project_name: str) -> None:
    if not project_name.strip():
        raise ValueError("Brightway project name is required")
    bd.projects.set_current(project_name.strip())


def list_databases(project_name: str | None = None) -> list[str]:
    if project_name:
        set_project(project_name)
    return sorted(list(bd.databases))


def _activity_type(name: str) -> str:
    lower = (name or "").strip().lower()
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


def _activity_to_dict(activity) -> dict:
    name = activity.get("name", "")
    return {
        "id": getattr(activity, "id", None),
        "database": activity.get("database", ""),
        "code": activity.get("code", ""),
        "name": name,
        "reference_product": activity.get("reference product", activity.get("product", "")),
        "location": activity.get("location", ""),
        "unit": activity.get("unit", ""),
        "activity_type": _activity_type(name),
        "comment": (activity.get("comment", "") or "")[:800],
    }


def _append_unique(values: list[str], value: str | None) -> None:
    cleaned = (value or "").strip()
    if cleaned and cleaned.casefold() not in {x.casefold() for x in values}:
        values.append(cleaned)


def location_preferences_from_context(location_hint: str | None) -> list[str]:
    """Translate evidence-derived geography into likely ecoinvent location labels.

    The hint may be a clean country name (``Germany``) or a longer model-generated
    phrase (``Germany primary; country-specific European results also reported``).
    Geography only affects ranking; it never removes otherwise relevant candidates.
    """
    hint = (location_hint or "").strip()
    if not hint:
        return []

    preferences: list[str] = []
    lowered = hint.casefold()
    _append_unique(preferences, hint)

    for phrase, aliases in REGIONAL_LOCATION_ALIASES.items():
        if phrase in lowered:
            for alias in aliases:
                _append_unique(preferences, alias)

    common = {
        "uk": "GB",
        "u.k.": "GB",
        "great britain": "GB",
        "united kingdom": "GB",
        "united states": "US",
        "usa": "US",
        "u.s.a.": "US",
        "south korea": "KR",
        "republic of korea": "KR",
    }
    for phrase, code in common.items():
        if phrase in lowered:
            _append_unique(preferences, code)

    try:
        country = pycountry.countries.lookup(hint)
        _append_unique(preferences, country.alpha_2)
        _append_unique(preferences, country.alpha_3)
        _append_unique(preferences, country.name)
    except LookupError:
        # Longer prose often contains a country name that lookup cannot parse directly.
        for country in pycountry.countries:
            names = [country.name]
            official_name = getattr(country, "official_name", None)
            common_name = getattr(country, "common_name", None)
            if official_name:
                names.append(official_name)
            if common_name:
                names.append(common_name)
            if any(re.search(rf"\b{re.escape(name.casefold())}\b", lowered) for name in names):
                _append_unique(preferences, country.alpha_2)
                _append_unique(preferences, country.alpha_3)
                _append_unique(preferences, country.name)

    return preferences


def _location_rank(location: str, preferences: list[str]) -> int:
    if not preferences:
        return 0
    loc = (location or "").strip().casefold()
    if not loc:
        return 4

    pref_keys = [p.casefold() for p in preferences]
    if loc in pref_keys:
        return 0
    if any(pref in loc or loc in pref for pref in pref_keys if len(pref) > 2):
        return 1
    if loc == "rer":
        return 2
    if loc in {"glo", "row"}:
        return 3
    return 4


def _query_variants(query: str, activity_type_hint: str | None = None) -> list[str]:
    query = (query or "").strip()
    if not query:
        return []

    variants = [query]
    hint = (activity_type_hint or "unknown").strip().lower()
    if hint == "market" and not query.lower().startswith(("market for ", "market group for ")):
        variants.append(f"market for {query}")
    elif hint == "treatment" and not query.lower().startswith("treatment of "):
        variants.append(f"treatment of {query}")
    elif hint == "transport" and "transport" not in query.lower():
        variants.append(f"transport, {query}")
    elif hint == "construction" and "construction" not in query.lower():
        variants.append(f"{query} construction")
    elif hint == "operation" and "operation" not in query.lower():
        variants.append(f"{query} operation")

    return list(dict.fromkeys(variants))


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) > 2 and token not in {"for", "the", "and", "with"}
    }


def _score_candidate(
    candidate: dict,
    *,
    query: str,
    preferences: list[str],
    unit: str | None,
    activity_type_hint: str | None,
    retrieval_rank: int,
) -> tuple[float, list[str]]:
    score = max(0.0, 12.0 - retrieval_rank * 0.08)
    reasons: list[str] = []

    q = query.strip().lower()
    name = (candidate.get("name") or "").strip().lower()
    product = (candidate.get("reference_product") or "").strip().lower()

    if q and q == product:
        score += 30
        reasons.append("reference product exactly matches query")
    elif q and q in product:
        score += 22
        reasons.append("reference product contains query")
    elif product and product in q:
        score += 14
        reasons.append("query contains reference product")

    if q and q in name:
        score += 14
        reasons.append("activity name contains query")

    query_tokens = _tokens(q)
    candidate_tokens = _tokens(f"{name} {product}")
    overlap = query_tokens & candidate_tokens
    if overlap:
        score += min(12, 3 * len(overlap))
        reasons.append(f"keyword overlap: {', '.join(sorted(overlap))}")

    if preferences:
        rank = _location_rank(candidate.get("location", ""), preferences)
        if rank == 0:
            score += 20
            reasons.append("exact evidence-derived geography match")
        elif rank == 1:
            score += 14
            reasons.append("compatible evidence-derived geography")
        elif rank == 2:
            score += 7
            reasons.append("European fallback geography")
        elif rank == 3:
            score += 3
            reasons.append("global/rest-of-world fallback geography")

    if unit and (candidate.get("unit") or "").strip().casefold() == unit.strip().casefold():
        score += 8
        reasons.append("unit matches foreground flow")

    hint = (activity_type_hint or "unknown").strip().lower()
    if hint in ACTIVITY_TYPES and hint != "unknown" and candidate.get("activity_type") == hint:
        score += 16
        reasons.append(f"activity type matches {hint} preference")

    return score, reasons


def search_candidates(
    *,
    project_name: str,
    database_name: str,
    query: str,
    location_hint: str | None = None,
    limit: int = 12,
    unit: str | None = None,
    activity_type_hint: str | None = None,
) -> list[dict]:
    """Retrieve and explain ranked real Brightway/ecoinvent candidates.

    Search variants and ranking preferences improve recall and ordering, but they do
    not fabricate activities and they do not hard-filter geography or activity type.
    """
    set_project(project_name)
    if database_name not in bd.databases:
        raise KeyError(f"Database '{database_name}' not found in Brightway project '{project_name}'")

    db = bd.Database(database_name)
    query = (query or "").strip()
    if not query:
        return []

    preferences = location_preferences_from_context(location_hint)
    variants = _query_variants(query, activity_type_hint)
    pool_limit = max(30, limit * 5)

    collected: list[tuple[object, int]] = []
    seen: set[tuple] = set()
    retrieval_rank = 0

    for variant in variants:
        for activity in db.search(variant, limit=pool_limit):
            key = (activity.get("database"), activity.get("code"))
            if key not in seen:
                seen.add(key)
                collected.append((activity, retrieval_rank))
                retrieval_rank += 1

    # A location-specific supplementary pass helps surface a relevant national
    # dataset that Brightway's general full-text search may rank too low.
    for location in preferences:
        if len(location) > 3 and location not in {"GLO", "RoW", "RER"}:
            continue
        for variant in variants:
            try:
                results = db.search(variant, limit=max(10, limit * 2), filter={"location": location})
            except Exception:
                results = []
            for activity in results:
                key = (activity.get("database"), activity.get("code"))
                if key not in seen:
                    seen.add(key)
                    collected.append((activity, retrieval_rank))
                    retrieval_rank += 1

    ranked: list[dict] = []
    for activity, rank in collected:
        candidate = _activity_to_dict(activity)
        score, reasons = _score_candidate(
            candidate,
            query=query,
            preferences=preferences,
            unit=unit,
            activity_type_hint=activity_type_hint,
            retrieval_rank=rank,
        )
        candidate["match_score"] = round(score, 1)
        candidate["match_reasons"] = "; ".join(reasons) or "Brightway text-search result"
        ranked.append(candidate)

    ranked.sort(key=lambda item: item["match_score"], reverse=True)
    return ranked[:limit]
