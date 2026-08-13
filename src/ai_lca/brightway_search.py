from __future__ import annotations

import re
from collections.abc import Iterable

import bw2data as bd

from .geography import parse_flow_location_hint


# ecoinvent/SimaPro system-model labels (allocation approach + process-type letter, e.g.
# "Cutoff, U" or "APOS, S"). These describe how the *source* study modelled background data,
# not an identity attribute of a technosphere node, so a node with the same name/product/
# location is the right equivalent match regardless of which system model the local
# Brightway database uses. Stripped from search queries so a paper citing "Cutoff" data
# still matches an "APOS" (or any other system model) database, and vice versa.
_SYSTEM_MODEL_QUALIFIER = re.compile(
    r"\b(?:allocation,?\s*)?"
    r"(?:cut[- ]?off|apos|consequential|allocation at the point of substitution)\b"
    r",?\s*[US]?\s*(?:-\s*)?",
    re.IGNORECASE,
)


def normalize_search_query(query: str) -> str:
    """Strip ecoinvent/SimaPro system-model qualifiers (Cutoff, APOS, Consequential, ...).

    Leaves activity name, reference product and location text untouched.
    """
    cleaned = _SYSTEM_MODEL_QUALIFIER.sub("", query or "")
    return re.sub(r"\s+", " ", cleaned).strip(" ,|")


def split_off_location(query: str) -> tuple[str, str | None]:
    """Detect and remove a trailing ecoinvent-style location code from a search query.

    Brightway's full-text search combines every query word as a required match, so a
    location code left in the query text acts as a hard filter, not a ranking hint --
    silently returning zero results whenever the source's printed location (e.g. a
    generic "RoW" in a supplementary export) differs from the real matching node's
    location (e.g. "RNA"). The location is instead returned separately so callers can
    apply it only as a soft preferred-location ranking boost.
    """
    hints = parse_flow_location_hint(query)
    if not hints:
        return query, None
    location = hints[0]
    stripped = re.sub(rf"[,|]\s*{re.escape(location)}\s*$", "", query)
    return stripped.strip(" ,|-"), location


# Large enough to reach real regional/country variants that a naive text-relevance search
# ranks far down (an ecoinvent activity commonly has 100-200+ near-duplicate location
# variants), while staying fast against a local Brightway index.
_LOCATION_RERANK_POOL = 300


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
    never filters candidates out. When the paper provides an operational geography or a
    flow explicitly cites a location (including a generic "RoW"/"GLO"), that exact
    location -- whatever it is -- is moved upward while Brightway's search order is
    otherwise retained, so the result matches what the source study actually used.
    """
    set_project(project_name)
    if database_name not in bd.databases:
        raise KeyError(f"Database '{database_name}' not found in Brightway project '{project_name}'")

    db = bd.Database(database_name)
    query = normalize_search_query(query)
    query, detected_location = split_off_location(query)
    if not query:
        return []

    preferred = {x.strip() for x in (preferred_locations or []) if x and x.strip()}
    if detected_location:
        preferred.add(detected_location)

    # A real ecoinvent activity commonly has one location variant per country/region, and
    # raw text relevance does not correlate with location -- the wanted variant can rank
    # far outside a small window. Fetch a much larger pool whenever there is a location to
    # rank by, so it is actually present to be promoted.
    expanded_limit = _LOCATION_RERANK_POOL if preferred else max(limit * 3, 20)
    results = list(db.search(query, limit=expanded_limit))

    if preferred:
        indexed = list(enumerate(results))
        indexed.sort(key=lambda pair: (0 if pair[1].get("location", "") in preferred else 1, pair[0]))
        results = [node for _, node in indexed]

    return [_node_to_dict(node) for node in results[:limit]]
