from __future__ import annotations

from .models import InventoryExtraction, ProcessMap


ELECTRICITY_VOLTAGE_QUALIFIERS = ("low voltage", "medium voltage", "high voltage")


def normalise_process_ids(process_map: ProcessMap) -> ProcessMap:
    """Assign deterministic IDs and discard process candidates with no direct evidence."""
    result = process_map.model_copy(deep=True)
    warnings = list(result.assumptions_or_warnings)
    counter = 1

    for group in result.technology_groups:
        supported = []
        for process in group.processes:
            if not process.evidence or not any(e.evidence_text.strip() for e in process.evidence):
                warnings.append(
                    f"Discarded unsupported process candidate '{process.name}' because no direct source evidence was supplied."
                )
                continue
            process.process_id = f"p{counter:03d}"
            counter += 1
            supported.append(process)
        group.processes = supported

    result.technology_groups = [g for g in result.technology_groups if g.processes]
    result.assumptions_or_warnings = warnings
    return result


def process_index(process_map: ProcessMap) -> dict[str, tuple[str, object]]:
    index: dict[str, tuple[str, object]] = {}
    for group in process_map.technology_groups:
        for process in group.processes:
            index[process.process_id] = (group.name, process)
    return index


def validate_inventory_against_process_map(
    extraction: InventoryExtraction,
    process_map: ProcessMap,
    source_text: str,
    allowed_process_ids: set[str] | None = None,
) -> InventoryExtraction:
    """Deterministically prevent invented/unapproved processes and unsupported specificity."""
    result = extraction.model_copy(deep=True)
    index = process_index(process_map)
    if allowed_process_ids is not None:
        index = {key: value for key, value in index.items() if key in allowed_process_ids}

    source_lower = (source_text or "").lower()
    warnings = list(result.assumptions_or_warnings)
    validated = []
    seen = set()

    for flow in result.flows:
        process_info = index.get(flow.process_id)
        if process_info is None:
            warnings.append(
                f"Discarded flow '{flow.name}' because it was assigned to unapproved process ID '{flow.process_id}'."
            )
            continue

        technology_group, process = process_info
        flow.technology_group = technology_group
        flow.process_name = process.name

        flow_name_lower = flow.name.lower()
        if "electricity" in flow_name_lower:
            for qualifier in ELECTRICITY_VOLTAGE_QUALIFIERS:
                if qualifier in flow_name_lower and qualifier not in source_lower:
                    original_name = flow.name
                    flow.name = "electricity"
                    note = (
                        f"Removed unsupported voltage-level qualifier from '{original_name}'; "
                        "the source does not state that voltage level."
                    )
                    flow.notes = f"{flow.notes}; {note}" if flow.notes else note
                    warnings.append(note)
                    break

        key = (
            flow.process_id,
            flow.name.strip().casefold(),
            flow.amount,
            (flow.unit or "").strip().casefold(),
            flow.direction,
            (flow.basis or "").strip().casefold(),
        )
        if key in seen:
            warnings.append(
                f"Discarded exact duplicate flow '{flow.name}' in process '{process.name}'."
            )
            continue
        seen.add(key)
        validated.append(flow)

    result.flows = validated
    result.assumptions_or_warnings = warnings
    return result
