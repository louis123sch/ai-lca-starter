from __future__ import annotations

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


def location_preferences_from_context(location_hint: str | None) -> list[str]:
    """Translate source-derived geography into likely ecoinvent location labels.

    This is deliberately derived from the paper/process map rather than user-entered
    preferred locations. It only affects ranking; it never filters out candidates.
    """
    hint = (location_hint or "").strip()
    if not hint:
        return []

    preferences = [hint]
    lowered = hint.casefold()
    preferences.extend(REGIONAL_LOCATION_ALIASES.get(lowered, []))

    # Common terminology that pycountry does not always resolve as users write it.
    common = {
        "uk": "GB",
        "u.k.": "GB",
        "great britain": "GB",
        "united states": "US",
        "usa": "US",
        "u.s.a.": "US",
        "south korea": "KR",
        "republic of korea": "KR",
    }
    if lowered in common:
        preferences.append(common[lowered])

    try:
        country = pycountry.countries.lookup(hint)
        preferences.extend([country.alpha_2, country.alpha_3, country.name])
        official_name = getattr(country, "official_name", None)
        if official_name:
            preferences.append(official_name)
    except LookupError:
        pass

    deduplicated: list[str] = []
    seen: set[str] = set()
    for value in preferences:
        key = value.strip().casefold()
        if key and key not in seen:
            seen.add(key)
            deduplicated.append(value.strip())
    return deduplicated


def _location_rank(location: str, preferences: list[str]) -> int:
    if not preferences:
        return 0
    loc = (location or "").strip().casefold()
    if not loc:
        return 3

    pref_keys = [p.casefold() for p in preferences]
    if loc in pref_keys:
        return 0
    if any(pref in loc or loc in pref for pref in pref_keys if len(pref) > 2):
        return 1
    if loc in {"rer", "glo", "row"}:
        return 2
    return 3


def search_candidates(
    *,
    project_name: str,
    database_name: str,
    query: str,
    location_hint: str | None = None,
    limit: int = 12,
) -> list[dict]:
    """Return real Brightway/ecoinvent activities; never fabricate a candidate.

    When the source paper supplies a geography, candidates matching that geography
    are ranked first. Geography is a preference derived from the source, not a hard
    filter, so a missing/ambiguous location cannot hide otherwise relevant datasets.
    """
    set_project(project_name)
    if database_name not in bd.databases:
        raise KeyError(f"Database '{database_name}' not found in Brightway project '{project_name}'")

    db = bd.Database(database_name)
    query = (query or "").strip()
    if not query:
        return []

    preferences = location_preferences_from_context(location_hint)
    retrieval_limit = max(limit * 5, 30) if preferences else limit
    collected = list(db.search(query, limit=retrieval_limit))

    if preferences:
        collected = sorted(
            enumerate(collected),
            key=lambda pair: (_location_rank(pair[1].get("location", ""), preferences), pair[0]),
        )
        collected = [activity for _, activity in collected]

    return [_activity_to_dict(act) for act in collected[:limit]]
