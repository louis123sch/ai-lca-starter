from __future__ import annotations

from typing import Iterable
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
    locations: Iterable[str] | None = None,
    limit: int = 12,
) -> list[dict]:
    """Return real Brightway/ecoinvent activities; never fabricate a candidate."""
    set_project(project_name)
    if database_name not in bd.databases:
        raise KeyError(f"Database '{database_name}' not found in Brightway project '{project_name}'")

    db = bd.Database(database_name)
    query = (query or "").strip()
    if not query:
        return []

    collected = []
    seen: set[tuple] = set()
    location_list = [x.strip() for x in (locations or []) if x and x.strip()]

    if location_list:
        per_location_limit = max(limit, 5)
        for location in location_list:
            results = db.search(query, limit=per_location_limit, filter={"location": location})
            for act in results:
                key = (act.get("database"), act.get("code"))
                if key not in seen:
                    seen.add(key)
                    collected.append(act)
    else:
        collected = list(db.search(query, limit=limit))

    return [_activity_to_dict(act) for act in collected[:limit]]
