from __future__ import annotations

import re

from .mapping_evidence import apply_source_mapping_hint
from .models import InventoryExtraction, ProcessMap


ELECTRICITY_VOLTAGE_QUALIFIERS = ("low voltage", "medium voltage", "high voltage")
CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
CONTEXTUAL_SUFFIX_TERMS = (
    "plant construction",
    "plant manufacturing",
    "construction",
    "capital input",
    "capital inputs",
    "capital equipment",
    "for electricity recovery",
    "for energy recovery",
)
STEEL_SUBTYPE_TERMS = (
    "low-alloyed",
    "low alloyed",
    "high-alloyed",
    "high alloyed",
    "unalloyed",
    "chromium steel",
    "steel 18/8",
)
TRAILING_PARENTHETICAL = re.compile(r"\s*\(([^()]*)\)\s*$")


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


def _strip_contextual_suffix(value: str | None) -> tuple[str, str | None]:
    cleaned = " ".join((value or "").split()).strip()
    removed: list[str] = []
    while cleaned:
        match = TRAILING_PARENTHETICAL.search(cleaned)
        if not match:
            break
        content = match.group(1).strip()
        lowered = content.casefold()
        if not any(term in lowered for term in CONTEXTUAL_SUFFIX_TERMS):
            break
        removed.append(content)
        cleaned = cleaned[: match.start()].strip()
    return cleaned, "; ".join(reversed(removed)) or None


def _clean_search_term(value: str | None, fallback: str) -> str:
    cleaned, _ = _strip_contextual_suffix(value or fallback)
    if cleaned.casefold() == "aluminum":
        return "aluminium"
    return cleaned


def normalise_process_ids(process_map: ProcessMap) -> ProcessMap:
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


def _normalise_exchange_class(flow, warnings: list[str]) -> None:
    """Deterministically enforce obvious exchange-class relationships."""
    if flow.direction == "emission" or flow.flow_kind == "emission":
        if flow.exchange_type != "biosphere":
            warnings.append(
                f"Reclassified direct emission '{flow.name}' as a biosphere exchange."
            )
            flow.exchange_type = "biosphere"

    if flow.direction == "output" and flow.flow_kind == "product":
        if flow.exchange_type == "technosphere":
            flow.exchange_type = "production"

    if flow.exchange_type == "biosphere":
        flow.biosphere_search_term = _clean_search_term(flow.biosphere_search_term, flow.name)
        flow.ecoinvent_search_term = None
        flow.ecoinvent_activity_hint = None
        flow.ecoinvent_location_hint = None
        flow.background_mapping_relation = None
        flow.background_mapping_rationale = None
        flow.background_mapping_evidence = []
        return

    if flow.exchange_type == "production":
        flow.ecoinvent_search_term = None
        flow.ecoinvent_activity_hint = None
        flow.ecoinvent_location_hint = None
        flow.background_mapping_relation = None
        flow.background_mapping_rationale = None
        flow.background_mapping_evidence = []
        flow.biosphere_search_term = None
        flow.biosphere_compartment_hint = None
        return

    if flow.exchange_type == "technosphere":
        flow.biosphere_search_term = None
        flow.biosphere_compartment_hint = None


def validate_inventory_against_process_map(
    extraction: InventoryExtraction,
    process_map: ProcessMap,
    source_text: str,
    allowed_process_ids: set[str] | None = None,
) -> InventoryExtraction:
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

        original_name = flow.name
        clean_name, removed_context = _strip_contextual_suffix(flow.name)
        if clean_name and clean_name != flow.name:
            flow.name = clean_name
            if removed_context and not flow.component_or_stage:
                flow.component_or_stage = removed_context
            note = (
                f"Removed lifecycle/context qualifier from exchange name '{original_name}'; "
                f"canonical exchange is '{flow.name}'."
            )
            flow.notes = f"{flow.notes}; {note}" if flow.notes else note
            warnings.append(note)

        _normalise_exchange_class(flow, warnings)

        if flow.exchange_type == "technosphere":
            flow.ecoinvent_search_term = _clean_search_term(flow.ecoinvent_search_term, flow.name)

            flow_name_lower = flow.name.lower()
            evidence_lower = " ".join(e.evidence_text for e in flow.evidence).lower()
            if "electricity" in flow_name_lower:
                for qualifier in ELECTRICITY_VOLTAGE_QUALIFIERS:
                    if qualifier in flow_name_lower and qualifier not in evidence_lower:
                        original_name = flow.name
                        flow.name = "electricity"
                        flow.ecoinvent_search_term = "electricity"
                        note = (
                            f"Removed unsupported voltage-level qualifier from '{original_name}'; "
                            "the evidence cited for the foreground quantity does not state that voltage level."
                        )
                        flow.notes = f"{flow.notes}; {note}" if flow.notes else note
                        warnings.append(note)
                        break

            # Recover explicit cross-document mappings even when the model misses them.
            apply_source_mapping_hint(flow, source_text)

            if flow.ecoinvent_activity_hint and not flow.background_mapping_evidence:
                warnings.append(
                    f"Removed unsupported ecoinvent activity hint '{flow.ecoinvent_activity_hint}' for flow '{flow.name}' because no mapping evidence was supplied."
                )
                flow.ecoinvent_activity_hint = None
                flow.ecoinvent_location_hint = None
                flow.background_mapping_relation = None
                flow.background_mapping_rationale = None

            if flow.ecoinvent_activity_hint and flow.background_mapping_relation is None:
                flow.background_mapping_relation = "uncertain"
                warnings.append(
                    f"Background activity hint for '{flow.name}' has source evidence but no stated exact/proxy relation; marked uncertain for review."
                )

            if not flow.ecoinvent_activity_hint:
                flow.ecoinvent_location_hint = None
                flow.background_mapping_relation = None
                flow.background_mapping_rationale = None

            if flow.ecoinvent_activity_hint and _normalise_name(flow.name) == "steel":
                hint_lower = flow.ecoinvent_activity_hint.casefold()
                quantity_supports_subtype = any(term in evidence_lower for term in STEEL_SUBTYPE_TERMS)
                hint_is_specific_subtype = any(term in hint_lower for term in STEEL_SUBTYPE_TERMS)
                if hint_is_specific_subtype and not quantity_supports_subtype and flow.background_mapping_relation != "proxy":
                    rejected_hint = flow.ecoinvent_activity_hint
                    flow.ecoinvent_activity_hint = None
                    flow.ecoinvent_location_hint = None
                    flow.background_mapping_relation = None
                    flow.background_mapping_rationale = None
                    flow.background_mapping_evidence = []
                    warnings.append(
                        f"Removed over-specific steel mapping hint '{rejected_hint}' because the quantitative foreground evidence identifies only generic steel and no proxy relationship was supported."
                    )

            if flow.ecoinvent_activity_hint and "steam turbine" in _normalise_name(flow.name):
                if "gas turbine" in flow.ecoinvent_activity_hint.casefold():
                    if flow.background_mapping_relation != "proxy":
                        flow.background_mapping_relation = "proxy"
                        warnings.append(
                            "Treated the source-supported gas-turbine background activity as a proxy for the foreground steam turbine; the foreground exchange remains 'steam turbine'."
                        )
                    if not flow.background_mapping_rationale:
                        flow.background_mapping_rationale = (
                            "Foreground evidence identifies a steam turbine while the source's applied background-data list identifies a gas-turbine activity; this is represented as a proxy rather than an exact identity."
                        )

        key = (
            flow.process_id,
            flow.exchange_type,
            _normalise_name(flow.name),
            flow.amount,
            _normalise_name(flow.unit),
            flow.direction,
            _normalise_name(flow.basis),
            _normalise_name(flow.biosphere_compartment_hint) if flow.exchange_type == "biosphere" else "",
        )
        if key in seen:
            existing = validated[seen[key]]
            _merge_evidence(existing.evidence, flow.evidence)
            _merge_evidence(existing.background_mapping_evidence, flow.background_mapping_evidence)
            if not existing.operation_context and flow.operation_context:
                existing.operation_context = flow.operation_context
            if not existing.component_or_stage and flow.component_or_stage:
                existing.component_or_stage = flow.component_or_stage
            if not existing.source_label and flow.source_label:
                existing.source_label = flow.source_label

            for field_name, label in (
                ("exchange_geography_hint", "exchange geography"),
                ("supplier_technology_hint", "supplier/technology"),
                ("biosphere_compartment_hint", "biosphere compartment"),
            ):
                existing_value = getattr(existing, field_name)
                incoming_value = getattr(flow, field_name)
                if not existing_value and incoming_value:
                    setattr(existing, field_name, incoming_value)
                elif existing_value and incoming_value and _normalise_name(existing_value) != _normalise_name(incoming_value):
                    warnings.append(
                        f"Conflicting {label} hints for repeated flow '{flow.name}': '{existing_value}' versus '{incoming_value}'. Kept the first for review."
                    )

            if not existing.interpretation_reason and flow.interpretation_reason:
                existing.interpretation_reason = flow.interpretation_reason
            elif (
                existing.interpretation_reason
                and flow.interpretation_reason
                and flow.interpretation_reason not in existing.interpretation_reason
            ):
                existing.interpretation_reason = (
                    f"{existing.interpretation_reason} {flow.interpretation_reason}"
                ).strip()

            if not existing.ecoinvent_activity_hint and flow.ecoinvent_activity_hint:
                existing.ecoinvent_activity_hint = flow.ecoinvent_activity_hint
                existing.ecoinvent_location_hint = flow.ecoinvent_location_hint
                existing.background_mapping_relation = flow.background_mapping_relation
                existing.background_mapping_rationale = flow.background_mapping_rationale
            if not existing.biosphere_search_term and flow.biosphere_search_term:
                existing.biosphere_search_term = flow.biosphere_search_term
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
