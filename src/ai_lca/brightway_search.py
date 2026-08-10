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


def list_biosphere_databases(project_name: str | None = None) -> list[str]:
    return [name for name in list_databases(project_name) if "biosphere" in name.casefold()]


def _node_to_dict(node) -> dict:
    categories = node.get("categories", ()) or ()
    if isinstance(categories, str):
        categories = (categories,)
    return {
        "id": getattr(node, "id", None),
        "database": node.get("database", ""),
        "code": node.get("code", ""),
        "name": node.get("name", ""),
        "reference_product": node.get("reference product", node.get("product", "")),
        "location": node.get("location", ""),
        "unit": node.get("unit", ""),
        "categories": " / ".join(str(x) for x in categories),
        "type": node.get("type", ""),
        "comment": (node.get("comment", "") or "")[:500],
    }


def search_candidates(
    *,
    project_name: str,
    database_name: str,
    query: str,
    preferred_locations: Iterable[str] | None = None,
    limit: int = 12,
) -> list[dict]:
    """Return real Brightway nodes with an optional soft geography boost.

    The function works for technosphere databases and biosphere databases. Geography
    never filters candidates out. When the paper provides an operational geography,
    matching locations are moved upward while Brightway's search order is otherwise retained.
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
        results = [node for _, node in indexed]

    return [_node_to_dict(node) for node in results[:limit]]
