from __future__ import annotations

import re


# Deterministic aliases used only to soft-rank real Brightway candidates. The LLM never
# fabricates ecoinvent geography codes; it returns the paper's human-readable context.
_LOCATION_ALIASES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("united kingdom", "great britain", "britain", "uk"), ("GB",)),
    (("germany",), ("DE",)),
    (("france",), ("FR",)),
    (("netherlands", "the netherlands"), ("NL",)),
    (("norway",), ("NO",)),
    (("sweden",), ("SE",)),
    (("denmark",), ("DK",)),
    (("spain",), ("ES",)),
    (("italy",), ("IT",)),
    (("united states", "usa", "u.s.", "us"), ("US",)),
    (("china",), ("CN",)),
    (("europe", "european"), ("RER",)),
    (("global", "worldwide", "world"), ("GLO",)),
)


def ecoinvent_location_hints(geography: str | None) -> list[str]:
    """Convert human-readable study geography into conservative ecoinvent location hints."""
    if not geography:
        return []
    text = geography.lower()
    hints: list[str] = []
    for aliases, codes in _LOCATION_ALIASES:
        matched = False
        for alias in aliases:
            if len(alias) <= 3 and alias.isalpha():
                if re.search(rf"\b{re.escape(alias)}\b", text):
                    matched = True
                    break
            elif alias in text:
                matched = True
                break
        if matched:
            for code in codes:
                if code not in hints:
                    hints.append(code)
    return hints
