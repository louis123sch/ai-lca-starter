from __future__ import annotations

from collections.abc import Iterable

import bw2data as bd


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


def search_candidates(
    *,
    project_name: str,
    database_name: str,
    query: str,
    preferred_locations: Iterable[str] | None = None,
    limit: int = 12,
) -> list[dict]:
    """Return real Brightway/ecoinvent activities with an optional soft geography boost.

    Geography never filters candidates out. When the paper provides an operational
    geography, matching ecoinvent locations are moved upward while Brightway's search
    order is otherwise retained.
    """
    set_project(project_name)
    if database_name not in bd.databases:
        raise KeyError(f"Database '{database_name}' not found in Brightway project '{project_name}'")

    db = bd.Database(database_name)
    query = (query or "").strip()
    if not query:
        return []

    expanded_limit = max(limit * 3, 20)
    results = list(db.search(query, limit=expanded_limit))
    preferred = {x.strip() for x in (preferred_locations or []) if x and x.strip()}

    if preferred:
        indexed = list(enumerate(results))
        indexed.sort(key=lambda pair: (0 if pair[1].get("location", "") in preferred else 1, pair[0]))
        results = [activity for _, activity in indexed]

    return [_activity_to_dict(act) for act in results[:limit]]
