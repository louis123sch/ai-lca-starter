from __future__ import annotations

from .models import InventoryExtraction, ProcessMap


ELECTRICITY_VOLTAGE_QUALIFIERS = ("low voltage", "medium voltage", "high voltage")
CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def _normalise_name(value: str | None) -> str:
    return " ".join((value or "").split()).casefold()


def _contexts_compatible(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return True
    return _normalise_name(left) == _normalise_name(right)


def _evidence_key(evidence) -> tuple:
    return (
        _normalise_name(evidence.source_document),
        evidence.page,
        _normalise_name(evidence.section),
        _normalise_name(evidence.table),
        _normalise_name(evidence.evidence_text),
    )


def _merge_evidence(target: list, incoming: list) -> None:
    seen = {_evidence_key(item) for item in target}
    for evidence in incoming:
        key = _evidence_key(evidence)
        if key not in seen:
            target.append(evidence)
            seen.add(key)


def normalise_process_ids(process_map: ProcessMap) -> ProcessMap:
    """Merge duplicate cross-document process descriptions, require evidence, and assign IDs."""
    result = process_map.model_copy(deep=True)
    warnings = list(result.assumptions_or_warnings)

    for group in result.technology_groups:
        supported = []
        for process in group.processes:
            if not process.evidence or not any(e.evidence_text.strip() for e in process.evidence):
                warnings.append(
                    f"Discarded unsupported process candidate '{process.name}' because no direct source evidence was supplied."
                )
                continue

            duplicate = next(
                (
                    existing
                    for existing in supported
                    if _normalise_name(existing.name) == _normalise_name(process.name)
                    and _contexts_compatible(existing.geographic_context, process.geographic_context)
                    and _contexts_compatible(existing.temporal_context, process.temporal_context)
                ),
                None,
            )

            if duplicate is None:
                supported.append(process)
                continue

            _merge_evidence(duplicate.evidence, process.evidence)
            duplicate.geographic_context = duplicate.geographic_context or process.geographic_context
            duplicate.temporal_context = duplicate.temporal_context or process.temporal_context
            if CONFIDENCE_ORDER[process.confidence] > CONFIDENCE_ORDER[duplicate.confidence]:
                duplicate.confidence = process.confidence
            if process.reason_for_separate_process not in duplicate.reason_for_separate_process:
                duplicate.reason_for_separate_process = (
                    f"{duplicate.reason_for_separate_process} {process.reason_for_separate_process}"
                ).strip()

            for operation in process.operations:
                existing_operation = next(
                    (
                        item
                        for item in duplicate.operations
                        if _normalise_name(item.name) == _normalise_name(operation.name)
                    ),
                    None,
                )
                if existing_operation is None:
                    duplicate.operations.append(operation)
                else:
                    _merge_evidence(existing_operation.evidence, operation.evidence)
                    if not existing_operation.notes and operation.notes:
                        existing_operation.notes = operation.notes

            warnings.append(
                f"Merged repeated descriptions of foreground process '{process.name}' across the evidence corpus."
            )

        group.processes = supported

    result.technology_groups = [g for g in result.technology_groups if g.processes]

    counter = 1
    for group in result.technology_groups:
        for process in group.processes:
            process.process_id = f"p{counter:03d}"
            counter += 1

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
    """Prevent invented/unapproved flows and merge repeated cross-document evidence."""
    result = extraction.model_copy(deep=True)
    index = process_index(process_map)
    if allowed_process_ids is not None:
        index = {key: value for key, value in index.items() if key in allowed_process_ids}

    warnings = list(result.assumptions_or_warnings)
    validated = []
    seen: dict[tuple, int] = {}

    for flow in result.flows:
        process_info = index.get(flow.process_id)
        if process_info is None:
            warnings.append(
                f"Discarded flow '{flow.name}' because it was assigned to unapproved process ID '{flow.process_id}'."
            )
            continue

        if not flow.evidence or not any(e.evidence_text.strip() for e in flow.evidence):
            warnings.append(
                f"Discarded flow '{flow.name}' in process '{flow.process_id}' because it had no direct source evidence."
            )
            continue

        technology_group, process = process_info
        flow.technology_group = technology_group
        flow.process_name = process.name

        flow_name_lower = flow.name.lower()
        evidence_lower = " ".join(e.evidence_text for e in flow.evidence).lower()
        if "electricity" in flow_name_lower:
            for qualifier in ELECTRICITY_VOLTAGE_QUALIFIERS:
                if qualifier in flow_name_lower and qualifier not in evidence_lower:
                    original_name = flow.name
                    flow.name = "electricity"
                    note = (
                        f"Removed unsupported voltage-level qualifier from '{original_name}'; "
                        "the evidence cited for this flow does not state that voltage level."
                    )
                    flow.notes = f"{flow.notes}; {note}" if flow.notes else note
                    warnings.append(note)
                    break

        key = (
            flow.process_id,
            _normalise_name(flow.name),
            flow.amount,
            _normalise_name(flow.unit),
            flow.direction,
            _normalise_name(flow.basis),
        )
        if key in seen:
            existing = validated[seen[key]]
            _merge_evidence(existing.evidence, flow.evidence)
            if not existing.operation_context and flow.operation_context:
                existing.operation_context = flow.operation_context
            if not existing.component_or_stage and flow.component_or_stage:
                existing.component_or_stage = flow.component_or_stage
            if flow.notes and flow.notes not in (existing.notes or ""):
                existing.notes = f"{existing.notes}; {flow.notes}" if existing.notes else flow.notes
            warnings.append(
                f"Merged repeated evidence for flow '{flow.name}' in process '{process.name}' rather than creating a duplicate exchange."
            )
            continue

        seen[key] = len(validated)
        validated.append(flow)

    result.flows = validated
    result.assumptions_or_warnings = warnings
    return result
