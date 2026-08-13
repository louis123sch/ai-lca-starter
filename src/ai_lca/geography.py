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


_LOCATION_CODE_PATTERN = re.compile(r"^(?:RoW|[A-Z]{2,3}(?:-[A-Z0-9]{2,6})?)$")


def parse_flow_location_hint(name: str | None) -> list[str]:
    """Detect an ecoinvent-style location code already printed at the end of a flow name.

    Source papers/supplements sometimes print flow names in ecoinvent's own
    "<activity>, <location code>" convention (e.g. "Electricity, medium voltage, US-SERC"),
    or in a pipe-delimited technical-export convention
    ("<activity> | <reference product> | <system model> - <location code>"). When the
    trailing comma- or pipe-delimited segment looks like a real location code, surface it
    as a per-flow ranking hint instead of relying only on the paper's single overall
    operational geography.
    """
    if not name:
        return []
    segment = re.split(r"[,|]", name)[-1].strip()
    segment = re.sub(r"^-\s*", "", segment)
    if segment and _LOCATION_CODE_PATTERN.match(segment):
        return [segment]
    return []
