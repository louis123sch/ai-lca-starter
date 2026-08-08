from __future__ import annotations

import re
from dataclasses import dataclass

from .models import InventoryFlow, SourceEvidence


DOCUMENT_RE = re.compile(r"^\[DOCUMENT\s+(.+?)\]\s*$", re.IGNORECASE)
TABLE_RE = re.compile(r"^\[TABLE\s+(.+?)\]\s*$", re.IGNORECASE)
ACTIVITY_MARKERS = (
    "market for ",
    "market group for ",
    "treatment of ",
)


@dataclass
class MappingRecord:
    source_label: str
    activity: str
    location: str | None
    unit: str | None
    evidence: SourceEvidence


def _normalise(value: str | None) -> str:
    text = " ".join((value or "").split()).casefold()
    return text.replace("aluminum", "aluminium")


def _unit_compatible(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return True
    aliases = {
        "unit": "unit",
        "units": "unit",
        "piece": "unit",
        "pieces": "unit",
        "kg": "kilogram",
        "kilogram": "kilogram",
        "m3": "cubic meter",
        "m³": "cubic meter",
        "cubic metre": "cubic meter",
        "cubic meter": "cubic meter",
        "kwh": "kilowatt hour",
        "kilowatt hour": "kilowatt hour",
    }
    l = aliases.get(_normalise(left), _normalise(left))
    r = aliases.get(_normalise(right), _normalise(right))
    return l == r


def extract_mapping_records(source_text: str) -> list[MappingRecord]:
    """Extract simple source-provided product -> background-activity table rows.

    This intentionally handles only explicit pipe-delimited table rows produced by the
    document readers. It does not attempt to invent mappings from prose.
    """
    records: list[MappingRecord] = []
    current_document: str | None = None
    current_table: str | None = None

    for raw_line in (source_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        document_match = DOCUMENT_RE.match(line)
        if document_match:
            current_document = document_match.group(1).strip()
            current_table = None
            continue

        table_match = TABLE_RE.match(line)
        if table_match:
            current_table = f"Table {table_match.group(1).strip()}"
            continue

        if "|" not in line:
            continue

        cells = [" ".join(cell.split()) for cell in line.split("|")]
        activity_index = next(
            (
                index
                for index, cell in enumerate(cells)
                if any(marker in cell.casefold() for marker in ACTIVITY_MARKERS)
            ),
            None,
        )
        if activity_index is None or activity_index == 0:
            continue

        source_label = cells[0].strip()
        activity = cells[activity_index].strip()
        if not source_label or not activity:
            continue

        location = cells[activity_index + 1].strip() if len(cells) > activity_index + 1 else None
        unit = cells[activity_index + 2].strip() if len(cells) > activity_index + 2 else None
        location = location or None
        unit = unit or None

        records.append(
            MappingRecord(
                source_label=source_label,
                activity=activity,
                location=location,
                unit=unit,
                evidence=SourceEvidence(
                    source_document=current_document,
                    table=current_table,
                    evidence_text=line,
                ),
            )
        )

    return records


def apply_source_mapping_hint(flow: InventoryFlow, source_text: str) -> None:
    """Fill missing explicit/proxy technosphere hints from source mapping tables.

    Exact source-label matches are deterministic. A deliberately narrow turbine rule
    recovers a source-applied gas-turbine proxy for a foreground steam turbine when the
    mapping table contains exactly one unit-compatible turbine activity. Other fuzzy
    product substitutions are left unresolved.
    """
    if flow.exchange_type != "technosphere" or flow.ecoinvent_activity_hint:
        return

    records = extract_mapping_records(source_text)
    if not records:
        return

    names = {_normalise(flow.name), _normalise(flow.source_label)} - {""}
    exact = [record for record in records if _normalise(record.source_label) in names]
    exact = [record for record in exact if _unit_compatible(flow.unit, record.unit)]

    if len(exact) == 1:
        record = exact[0]
        flow.ecoinvent_activity_hint = record.activity
        flow.ecoinvent_location_hint = record.location
        flow.background_mapping_relation = "exact"
        flow.background_mapping_rationale = (
            "Recovered deterministically from an explicit source product-to-background-activity table row."
        )
        flow.background_mapping_evidence = [record.evidence]
        return

    # Narrow proxy recovery for the currently evidenced equipment pattern. This is
    # intentionally not a general fuzzy mapper.
    if "turbine" not in _normalise(flow.name):
        return

    turbine_records = [
        record
        for record in records
        if "turbine" in _normalise(record.source_label)
        and "turbine" in _normalise(record.activity)
        and _unit_compatible(flow.unit, record.unit)
    ]
    if len(turbine_records) != 1:
        return

    record = turbine_records[0]
    if _normalise(record.source_label) == _normalise(flow.name):
        return

    flow.ecoinvent_activity_hint = record.activity
    flow.ecoinvent_location_hint = record.location
    flow.background_mapping_relation = "proxy"
    flow.background_mapping_rationale = (
        f"Foreground evidence identifies '{flow.name}', while the source-applied background-data table contains "
        f"the sole unit-compatible turbine dataset '{record.activity}'. The foreground identity is retained and the background link is labelled as a proxy."
    )
    flow.background_mapping_evidence = [record.evidence]
