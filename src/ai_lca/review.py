from __future__ import annotations

import pandas as pd

from .models import ForegroundProcess, InventoryExtraction


def _clean_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _resolve_process_review(
    extraction: InventoryExtraction,
    review_df: pd.DataFrame,
) -> tuple[
    dict[str, pd.Series],
    dict[str, str | None],
    dict[str, str],
    set[str],
    list[str],
]:
    original = {process.process_id: process for process in extraction.processes}
    rows = {
        _clean_text(row.get("process_id")): row
        for _, row in review_df.iterrows()
        if _clean_text(row.get("process_id")) in original
    }

    retained_ids: set[str] = set()
    merge_requests: dict[str, str] = {}
    warnings = list(extraction.assumptions_or_warnings)

    for process_id in original:
        row = rows.get(process_id)
        if row is None:
            retained_ids.add(process_id)
            continue
        include = bool(row.get("include", True))
        merge_into = _clean_text(row.get("merge_into"))
        if merge_into == process_id:
            merge_into = ""
        if merge_into:
            merge_requests[process_id] = merge_into
        elif include:
            retained_ids.add(process_id)

    valid_merges: dict[str, str] = {}
    for source_id, target_id in merge_requests.items():
        if target_id not in original:
            warnings.append(
                f"Human review requested merge {source_id!r} -> unknown process {target_id!r}; merge ignored."
            )
            retained_ids.add(source_id)
            continue
        target_row = rows.get(target_id)
        target_included = True if target_row is None else bool(target_row.get("include", True))
        target_merges_again = target_id in merge_requests
        if not target_included or target_merges_again:
            warnings.append(
                f"Human review requested merge {source_id!r} -> {target_id!r}, but the target is not a retained process; merge ignored."
            )
            retained_ids.add(source_id)
            continue
        valid_merges[source_id] = target_id
        retained_ids.add(target_id)

    process_id_map: dict[str, str | None] = {}
    for process_id in original:
        if process_id in valid_merges:
            process_id_map[process_id] = valid_merges[process_id]
        elif process_id in retained_ids:
            process_id_map[process_id] = process_id
        else:
            process_id_map[process_id] = None

    return rows, process_id_map, valid_merges, retained_ids, warnings


def process_review_id_map(
    extraction: InventoryExtraction,
    review_df: pd.DataFrame,
) -> dict[str, str | None]:
    """Return the deterministic original-process -> reviewed-process mapping."""
    _, process_id_map, _, _, _ = _resolve_process_review(extraction, review_df)
    return process_id_map


def remap_inventory_dataframe(
    inventory_df: pd.DataFrame,
    *,
    process_id_map: dict[str, str | None],
    process_names: dict[str, str],
) -> pd.DataFrame:
    """Remap process references while preserving existing stable flow IDs and evidence rows."""
    result = inventory_df.copy()
    kept_rows = []
    for _, row in result.iterrows():
        source_process = _clean_text(row.get("process_id"))
        mapped_process = process_id_map.get(source_process, source_process)
        if mapped_process is None:
            continue
        row = row.copy()
        row["process_id"] = mapped_process
        row["process_name"] = process_names.get(mapped_process, "")

        linked = _clean_text(row.get("linked_process_id"))
        if linked:
            mapped_link = process_id_map.get(linked, linked)
            row["linked_process_id"] = mapped_link or ""
        kept_rows.append(row)

    if not kept_rows:
        return result.iloc[0:0].copy()
    return pd.DataFrame(kept_rows).reset_index(drop=True)


def apply_process_review(
    extraction: InventoryExtraction,
    review_df: pd.DataFrame,
) -> InventoryExtraction:
    """Apply merge/remove/rename/re-parent edits without losing the original evidence trail.

    Process IDs are stable. A merge reattaches source-supported flows to the retained
    target process. Removing a process without a merge removes its attached flows.
    """
    original = {process.process_id: process for process in extraction.processes}
    rows, process_id_map, valid_merges, retained_ids, warnings = _resolve_process_review(
        extraction,
        review_df,
    )

    reviewed_processes: list[ForegroundProcess] = []
    for process_id, process in original.items():
        if process_id_map[process_id] != process_id:
            continue
        row = rows.get(process_id)
        if row is None:
            reviewed_processes.append(process)
            continue

        parent = _clean_text(row.get("parent")) or None
        if parent:
            parent = process_id_map.get(parent)
        if parent == process_id:
            parent = None
            warnings.append(f"Human review removed self-parent relationship for {process_id!r}.")
        if parent and parent not in retained_ids:
            parent = None

        reviewed_processes.append(
            process.model_copy(
                update={
                    "name": _clean_text(row.get("process")) or process.name,
                    "parent_process_id": parent,
                    "reference_product": _clean_text(row.get("reference_product")) or None,
                    "reference_unit": _clean_text(row.get("reference_unit")) or None,
                }
            )
        )

    reviewed_flows = []
    for flow in extraction.flows:
        mapped_process = process_id_map.get(flow.process_id)
        if mapped_process is None:
            continue
        linked = flow.linked_process_id
        if linked:
            linked = process_id_map.get(linked)
        reviewed_flows.append(
            flow.model_copy(
                update={
                    "process_id": mapped_process,
                    "linked_process_id": linked,
                }
            )
        )

    if valid_merges:
        merge_text = ", ".join(f"{src}->{dst}" for src, dst in sorted(valid_merges.items()))
        warnings.append(f"Human-reviewed foreground process merges applied: {merge_text}.")

    removed = sorted(pid for pid, mapped in process_id_map.items() if mapped is None)
    if removed:
        warnings.append(
            "Human review removed foreground process(es) and their attached flows: " + ", ".join(removed)
        )

    return extraction.model_copy(
        update={
            "processes": reviewed_processes,
            "flows": reviewed_flows,
            "assumptions_or_warnings": list(dict.fromkeys(warnings)),
        }
    )
