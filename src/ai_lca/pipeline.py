from __future__ import annotations

from collections.abc import Iterable

from .llm import extract_inventory_from_text
from .models import ForegroundProcessProposal, InventoryExtraction


def merge_inventory_extractions(
    extracted_sources: list[tuple[str, InventoryExtraction]],
) -> InventoryExtraction:
    """Merge independently interpreted sources while preserving evidence and process context."""
    if not extracted_sources:
        raise ValueError("No extracted sources supplied")

    flows = []
    warnings: list[str] = []
    process_names: list[str] = []
    technology_names: list[str] = []
    functional_units: list[str] = []
    system_descriptions: list[str] = []
    foreground_processes: list[ForegroundProcessProposal] = []
    seen_processes: set[tuple[str, str]] = set()

    for source_name, extraction in extracted_sources:
        for flow in extraction.flows:
            flow.evidence.source_document = source_name
            flows.append(flow)

        for process in extraction.foreground_processes:
            process.source_document = source_name
            key = (process.name.strip().lower(), source_name)
            if key not in seen_processes:
                seen_processes.add(key)
                foreground_processes.append(process)

        warnings.extend(
            f"{source_name}: {warning}" for warning in extraction.assumptions_or_warnings
        )
        if extraction.process_name:
            process_names.append(extraction.process_name)
        if extraction.technology_name:
            technology_names.append(extraction.technology_name)
        if extraction.functional_unit:
            functional_units.append(extraction.functional_unit)
        if extraction.system_description:
            system_descriptions.append(f"{source_name}: {extraction.system_description}")

    unique_process_names = list(dict.fromkeys(process_names))
    unique_technology_names = list(dict.fromkeys(technology_names))
    unique_functional_units = list(dict.fromkeys(functional_units))

    if len(unique_process_names) > 1:
        warnings.append(
            "Multiple process names were identified across sources: "
            + "; ".join(unique_process_names)
        )
    if len(unique_technology_names) > 1:
        warnings.append(
            "Multiple technology descriptions were identified across sources: "
            + "; ".join(unique_technology_names)
        )
    if len(unique_functional_units) > 1:
        warnings.append(
            "Multiple functional units were identified across sources: "
            + "; ".join(unique_functional_units)
        )

    source_names = [source_name for source_name, _ in extracted_sources]
    return InventoryExtraction(
        process_name=unique_process_names[0] if len(unique_process_names) == 1 else None,
        technology_name=unique_technology_names[0] if len(unique_technology_names) == 1 else None,
        functional_unit=unique_functional_units[0] if len(unique_functional_units) == 1 else None,
        system_description="\n".join(system_descriptions) if system_descriptions else None,
        foreground_processes=foreground_processes,
        source_summary=(
            f"Independent two-pass foreground interpretation from {len(source_names)} source(s): "
            + ", ".join(source_names)
        ),
        assumptions_or_warnings=warnings,
        flows=flows,
    )


def extract_inventory_from_sources(
    sources: Iterable[tuple[str, str]],
    *,
    model: str | None = None,
    api_key: str | None = None,
    extra_instructions: str = "",
) -> InventoryExtraction:
    """Understand and extract each document independently, then merge for human review."""
    source_list = [(name, text) for name, text in sources if text and text.strip()]
    if not source_list:
        raise ValueError("No source material supplied")

    extracted_sources: list[tuple[str, InventoryExtraction]] = []
    for source_name, source_text in source_list:
        extraction = extract_inventory_from_text(
            source_text,
            model=model,
            api_key=api_key,
            extra_instructions=extra_instructions,
            source_document=source_name,
        )
        extracted_sources.append((source_name, extraction))

    return merge_inventory_extractions(extracted_sources)
