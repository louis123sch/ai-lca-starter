from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import pandas as pd
import bw2data as bd

from .models import InventoryExtraction


@dataclass
class WritePlan:
    ready: bool
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    exchanges: list[dict] = field(default_factory=list)


def _text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _amount(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _unit_key(value: str) -> str:
    value = value.casefold().strip()
    aliases = {
        "kg": "kilogram",
        "kilograms": "kilogram",
        "kwh": "kilowatt hour",
        "kw h": "kilowatt hour",
        "mj": "megajoule",
        "m3": "cubic meter",
        "m^3": "cubic meter",
    }
    return aliases.get(value, value)


def build_write_plan(
    extraction: InventoryExtraction,
    inventory_df: pd.DataFrame,
    mapping_df: pd.DataFrame | None = None,
) -> WritePlan:
    """Validate that reviewed rows can be written without inventing modelling choices.

    The writer is intentionally strict. It supports mapped technosphere inputs,
    mapped biosphere emissions, and explicit foreground-to-foreground inputs.
    Additional co-products/outputs remain a human modelling decision and block the
    write until they are excluded or represented explicitly in a future extension.
    """
    blockers: list[str] = []
    warnings: list[str] = []
    exchanges: list[dict] = []

    process_by_id = {p.process_id: p for p in extraction.processes}
    for process in extraction.processes:
        if not _text(process.reference_product):
            blockers.append(
                f"Process {process.process_id!r} has no reviewed reference product."
            )
        if not _text(process.reference_unit):
            blockers.append(
                f"Process {process.process_id!r} has no reviewed reference unit."
            )

    mappings: dict[int, dict] = {}
    if mapping_df is not None and not mapping_df.empty:
        for _, row in mapping_df.iterrows():
            raw_flow_id = row.get("flow_id")
            if raw_flow_id is None or pd.isna(raw_flow_id):
                continue
            mappings[int(raw_flow_id)] = row.to_dict()

    included = inventory_df[inventory_df["include"] == True] if "include" in inventory_df else inventory_df  # noqa: E712
    for row_index, row in included.iterrows():
        raw_flow_id = row.get("flow_id", row_index)
        flow_id = int(raw_flow_id) if raw_flow_id is not None and not pd.isna(raw_flow_id) else int(row_index)
        process_id = _text(row.get("process_id"))
        flow_name = _text(row.get("name")) or f"flow {flow_id}"
        direction = _text(row.get("direction")).casefold() or "unknown"
        linked_process_id = _text(row.get("linked_process_id"))
        amount = _amount(row.get("amount"))
        unit = _text(row.get("unit"))

        if process_id not in process_by_id:
            blockers.append(f"Flow {flow_id} ({flow_name}) references unknown process {process_id!r}.")
            continue
        if amount is None:
            blockers.append(f"Flow {flow_id} ({flow_name}) has no reviewed numeric amount.")
            continue

        if linked_process_id:
            if direction != "input":
                blockers.append(
                    f"Flow {flow_id} ({flow_name}) links a foreground process but is not an input."
                )
                continue
            target = process_by_id.get(linked_process_id)
            if target is None:
                blockers.append(
                    f"Flow {flow_id} ({flow_name}) links unknown foreground process {linked_process_id!r}."
                )
                continue
            target_unit = _text(target.reference_unit)
            if unit and target_unit and _unit_key(unit) != _unit_key(target_unit):
                blockers.append(
                    f"Flow {flow_id} ({flow_name}) unit {unit!r} does not match linked process unit {target_unit!r}; "
                    "no automatic conversion is allowed."
                )
                continue
            exchanges.append(
                {
                    "flow_id": flow_id,
                    "process_id": process_id,
                    "flow_name": flow_name,
                    "amount": amount,
                    "unit": unit,
                    "exchange_type": "technosphere_foreground",
                    "target_process_id": linked_process_id,
                }
            )
            continue

        if direction == "output":
            blockers.append(
                f"Flow {flow_id} ({flow_name}) is an additional output/co-product. "
                "The strict v1 writer will not invent a production/allocation model; review or exclude it before writing."
            )
            continue
        if direction not in {"input", "emission"}:
            blockers.append(
                f"Flow {flow_id} ({flow_name}) has direction {direction!r}; review it before writing."
            )
            continue

        mapping = mappings.get(flow_id)
        if not mapping:
            blockers.append(f"Flow {flow_id} ({flow_name}) has no selected Brightway mapping.")
            continue
        target_database = _text(mapping.get("database"))
        target_code = _text(mapping.get("code"))
        if not target_database or not target_code:
            blockers.append(f"Flow {flow_id} ({flow_name}) has an incomplete Brightway mapping.")
            continue
        mapped_unit = _text(mapping.get("unit"))
        if unit and mapped_unit and _unit_key(unit) != _unit_key(mapped_unit):
            blockers.append(
                f"Flow {flow_id} ({flow_name}) unit {unit!r} does not match mapped node unit {mapped_unit!r}; "
                "no automatic conversion is allowed."
            )
            continue

        exchanges.append(
            {
                "flow_id": flow_id,
                "process_id": process_id,
                "flow_name": flow_name,
                "amount": amount,
                "unit": unit,
                "exchange_type": "biosphere" if direction == "emission" else "technosphere_background",
                "target_database": target_database,
                "target_code": target_code,
                "target_name": _text(mapping.get("name")),
            }
        )

    if extraction.functional_unit:
        warnings.append(
            "Foreground activities are written on their reviewed reference-product units. "
            f"The extracted study functional unit ({extraction.functional_unit}) is retained as metadata, not converted automatically."
        )

    return WritePlan(
        ready=not blockers,
        blockers=list(dict.fromkeys(blockers)),
        warnings=list(dict.fromkeys(warnings)),
        exchanges=exchanges,
    )


def _safe_code(process_id: str) -> str:
    code = re.sub(r"[^A-Za-z0-9_.-]+", "-", process_id.strip()).strip("-")
    return code or "foreground-process"


def write_foreground_database(
    *,
    project_name: str,
    database_name: str,
    extraction: InventoryExtraction,
    inventory_df: pd.DataFrame,
    mapping_df: pd.DataFrame | None = None,
) -> dict:
    """Write a reviewed, fully validated foreground database into the local Brightway project."""
    project_name = project_name.strip()
    database_name = database_name.strip()
    if not project_name:
        raise ValueError("Brightway project name is required")
    if not database_name:
        raise ValueError("Foreground database name is required")

    plan = build_write_plan(extraction, inventory_df, mapping_df)
    if not plan.ready:
        raise ValueError("Foreground write is blocked:\n- " + "\n- ".join(plan.blockers))

    bd.projects.set_current(project_name)
    if database_name in bd.databases:
        raise ValueError(
            f"Brightway database {database_name!r} already exists. Choose a new name; the writer never overwrites databases."
        )

    db = bd.Database(database_name)
    db.register()

    created = {}
    try:
        for process in extraction.processes:
            kwargs = {
                "code": _safe_code(process.process_id),
                "name": process.name,
                "unit": process.reference_unit,
                "reference product": process.reference_product,
                "type": "process",
                "ai_lca_process_id": process.process_id,
                "ai_lca_source_summary": extraction.source_summary,
            }
            if extraction.study_context.operational_geography:
                kwargs["location"] = extraction.study_context.operational_geography
            if extraction.provenance:
                kwargs["ai_lca_extractor_version"] = extraction.provenance.extractor_version
                kwargs["ai_lca_model"] = extraction.provenance.model
                kwargs["ai_lca_git_sha"] = extraction.provenance.git_sha or ""
            activity = db.new_activity(**kwargs)
            activity.save()
            activity.new_exchange(input=activity.key, amount=1.0, type="production").save()
            created[process.process_id] = activity

        for exchange in plan.exchanges:
            source = created[exchange["process_id"]]
            if exchange["exchange_type"] == "technosphere_foreground":
                target = created[exchange["target_process_id"]]
                source.new_exchange(
                    input=target.key,
                    amount=exchange["amount"],
                    type="technosphere",
                ).save()
            else:
                target_key = (exchange["target_database"], exchange["target_code"])
                source.new_exchange(
                    input=target_key,
                    amount=exchange["amount"],
                    type="biosphere" if exchange["exchange_type"] == "biosphere" else "technosphere",
                ).save()
    except Exception:
        # Never leave a silently partial database after a failed write.
        try:
            del bd.databases[database_name]
        except Exception:
            pass
        raise

    return {
        "database": database_name,
        "processes_created": len(created),
        "exchanges_created": len(plan.exchanges),
        "warnings": plan.warnings,
    }
