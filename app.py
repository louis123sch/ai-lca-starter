from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ai_lca.brightway_search import (
    list_biosphere_databases,
    list_databases,
    normalize_search_query,
    search_candidates,
)
from ai_lca.brightway_writer import build_write_plan, write_foreground_database
from ai_lca.documents import combine_document_evidence
from ai_lca.export import (
    candidate_structure_to_dataframe,
    dataframe_to_json,
    extraction_to_dataframe,
    normalize_inventory_review,
    process_structure_to_dataframe,
    review_bundle_to_json,
)
from ai_lca.geography import ecoinvent_location_hints, parse_flow_location_hint
from ai_lca.llm import extract_inventory_from_documents, extract_inventory_from_text
from ai_lca.review import (
    apply_process_review,
    process_review_id_map,
    remap_inventory_dataframe,
)
from ai_lca.runtime import extractor_version, git_sha

def _default_search_text(row) -> str:
    """Prefer an explicit source-printed background-process identifier over the plain flow name."""
    hint = str(row.get("background_process_hint", "") or "").strip()
    if hint and hint.lower() != "nan":
        return hint
    return str(row.get("name", "") or "").strip()


def _row_unit(row) -> str:
    unit = str(row.get("unit", "") or "").strip()
    return "" if unit.lower() == "nan" else unit


def _search_display_text(row) -> str:
    """Default search box text: the search query, annotated with the flow's unit for clarity."""
    base = _default_search_text(row)
    unit = _row_unit(row)
    if unit and not base.rstrip().endswith(f"({unit})"):
        return f"{base} ({unit})"
    return base


def _strip_unit_suffix(query: str, unit: str) -> str:
    """Remove a trailing '(<unit>)' annotation before the text is actually searched.

    The unit is shown in the search box purely for human readability -- Brightway's
    full-text search requires every query word to match, so a literal unit word left in
    the query risks the same silent hard-filtering that a location code caused.
    """
    unit = (unit or "").strip()
    if not unit:
        return query
    suffix = f"({unit})"
    stripped = query.rstrip()
    if stripped.endswith(suffix):
        return stripped[: -len(suffix)].strip()
    return query


load_dotenv()

st.set_page_config(page_title="AI-LCA Foreground Builder", layout="wide")
st.title("AI-LCA Foreground Builder")
st.caption(
    "Source evidence → visual/native ingestion → activity-role classification → locked foreground flows → "
    "human review → Brightway matching/write"
)

with st.sidebar:
    st.header("Configuration")
    st.caption(f"Extractor {extractor_version()} | commit {(git_sha() or 'unknown')[:10]}")
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    project_name = os.getenv("BRIGHTWAY_PROJECT", "")
    st.caption(f"OpenAI model: {model} · Brightway project: {project_name or 'not set in .env'}")
    candidate_limit = st.slider("Candidates per flow", 3, 20, 8)

    database_name = ""
    biosphere_databases: list[str] = []
    if project_name:
        try:
            dbs = list_databases(project_name)
            biosphere_databases = list_biosphere_databases(project_name)
            background_dbs = [db for db in dbs if db not in biosphere_databases]
            if background_dbs:
                default_index = next(
                    (i for i, x in enumerate(background_dbs) if "ecoinvent" in x.lower()),
                    0,
                )
                database_name = st.selectbox(
                    "Technosphere/background database",
                    background_dbs,
                    index=default_index,
                )
            else:
                st.warning("No technosphere/background databases found in this Brightway project.")
            if biosphere_databases:
                st.caption(f"Biosphere search: {biosphere_databases[0]}")
            else:
                st.caption("No biosphere database detected; emission mappings will be blocked.")
        except Exception as exc:
            st.warning(f"Brightway project not available yet: {exc}")

paste_tab, document_tab = st.tabs(["Paste text", "Upload document"])

with paste_tab:
    pasted_text = st.text_area(
        "Paste paper text, technical documentation, datasheet text, or engineering notes",
        height=320,
        placeholder="Paste the relevant source material here…",
    )

uploaded_payloads: list[tuple[str, bytes]] = []
document_preview = ""
detected_visuals = 0
ingestion_warnings: list[str] = []

with document_tab:
    uploaded_documents = st.file_uploader(
        "Upload the paper and any supplementary PDF/Word/Excel documents",
        type=["pdf", "docx", "xlsx", "xlsm"],
        accept_multiple_files=True,
    )
    if uploaded_documents:
        try:
            uploaded_payloads = [(doc.name, doc.getvalue()) for doc in uploaded_documents]
            document_preview, visual_assets, ingestion_warnings = combine_document_evidence(uploaded_payloads)
            detected_visuals = len(visual_assets)
            names = ", ".join(doc.name for doc in uploaded_documents)
            st.success(
                f"Combined {len(uploaded_documents)} source document(s) "
                f"({len(document_preview):,} text characters, {detected_visuals} selected visual asset(s)): {names}"
            )
            if detected_visuals:
                st.caption(
                    "Relevant embedded figures/scanned pages will be transcribed with vision before process interpretation."
                )
            for warning in ingestion_warnings:
                st.warning(warning)
            with st.expander("Preview combined native source text"):
                st.text(document_preview[:16000])
        except Exception as exc:
            st.error(str(exc))

extra_instructions = st.text_area(
    "Optional study instructions",
    placeholder=(
        "Use only when the paper itself needs disambiguation. Do not provide desired process names or missing inventory values."
    ),
    height=90,
)

source_text = pasted_text.strip() or document_preview.strip()

if st.button("1. Interpret paper and extract foreground", type="primary", disabled=not bool(source_text)):
    try:
        spinner_text = (
            "Classifying process roles, locking the foreground graph, then extracting evidence-backed flows…"
            if pasted_text.strip()
            else "Reading text and figures, classifying process roles, locking the foreground graph, then extracting flows…"
        )
        with st.spinner(spinner_text):
            if uploaded_payloads and not pasted_text.strip():
                extraction = extract_inventory_from_documents(
                    uploaded_payloads,
                    model=model,
                    extra_instructions=extra_instructions,
                )
            else:
                extraction = extract_inventory_from_text(
                    source_text,
                    model=model,
                    extra_instructions=extra_instructions,
                )
        original_inventory_df = extraction_to_dataframe(extraction)
        st.session_state["original_extraction"] = extraction
        st.session_state["extraction"] = extraction
        st.session_state["process_review_df"] = process_structure_to_dataframe(extraction)
        st.session_state["original_inventory_df"] = original_inventory_df.copy()
        st.session_state["inventory_df"] = original_inventory_df.copy()
        st.session_state.pop("candidates", None)
        st.session_state.pop("mapping_df", None)
    except Exception as exc:
        st.exception(exc)

if "extraction" in st.session_state:
    extraction = st.session_state["extraction"]
    st.subheader("Paper interpretation")

    context = extraction.study_context
    meta1, meta2, meta3 = st.columns(3)
    meta1.write(f"**Primary process/system:** {extraction.process_name or 'Not identified'}")
    meta2.write(f"**Functional unit:** {extraction.functional_unit or 'Not identified'}")
    meta3.write(f"**Operational geography:** {context.operational_geography or 'Not identified'}")
    if extraction.provenance:
        st.caption(
            f"Extraction provenance: v{extraction.provenance.extractor_version} · "
            f"model {extraction.provenance.model} · "
            f"commit {(extraction.provenance.git_sha or 'unknown')[:10]} · "
            f"{extraction.provenance.generated_at_utc}"
        )
    if context.operational_geography:
        st.caption(
            f"Geography basis: {context.geography_basis}. "
            f"{context.geography_rationale or ''}".strip()
        )
    if context.additional_geographies:
        st.write(f"**Additional geographic scenarios:** {', '.join(context.additional_geographies)}")
    if context.system_boundary:
        st.write(f"**System boundary:** {context.system_boundary}")
    if context.temporal_context:
        st.write(f"**Temporal context:** {context.temporal_context}")
    st.write(extraction.source_summary)

    if extraction.candidate_activities:
        with st.expander("Why process-like activities were retained or rejected", expanded=False):
            st.dataframe(
                candidate_structure_to_dataframe(extraction),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Only assessed product systems and explicitly interconnected foreground activities are locked as processes. "
                "Internal stages, shared supporting activities, background supplies and descriptive-only entities remain audit evidence."
            )

    st.subheader("Review foreground process structure")
    process_ids = [p.process_id for p in st.session_state["original_extraction"].processes]
    reviewed_process_df = st.data_editor(
        st.session_state["process_review_df"],
        width="stretch",
        hide_index=True,
        column_config={
            "include": st.column_config.CheckboxColumn("Keep"),
            "process_id": st.column_config.TextColumn("Process ID", disabled=True),
            "process": st.column_config.TextColumn("Process name"),
            "merge_into": st.column_config.SelectboxColumn(
                "Merge into",
                options=[""] + process_ids,
                help="Optional: reattach this process's flows to another retained process.",
            ),
            "parent": st.column_config.SelectboxColumn("Parent", options=[""] + process_ids),
            "role": st.column_config.TextColumn("AI role", disabled=True),
            "reference_product": st.column_config.TextColumn("Reference product"),
            "reference_unit": st.column_config.TextColumn("Reference unit"),
            "classification_rationale": st.column_config.TextColumn("AI rationale", disabled=True, width="large"),
            "stage": st.column_config.TextColumn("Stage", disabled=True),
            "evidence": st.column_config.TextColumn("Evidence", disabled=True, width="large"),
        },
        key="process_editor",
    )
    st.session_state["process_review_df"] = reviewed_process_df

    if st.button("Apply process review"):
        try:
            original_extraction = st.session_state["original_extraction"]
            reviewed_extraction = apply_process_review(
                original_extraction,
                reviewed_process_df,
            )
            if not reviewed_extraction.processes:
                raise ValueError("At least one foreground process must remain after review.")
            id_map = process_review_id_map(original_extraction, reviewed_process_df)
            process_names = {
                process.process_id: process.name for process in reviewed_extraction.processes
            }
            original_inventory_df = st.session_state.get(
                "original_inventory_df",
                extraction_to_dataframe(original_extraction),
            )
            reviewed_inventory_df = remap_inventory_dataframe(
                original_inventory_df,
                process_id_map=id_map,
                process_names=process_names,
            )
            st.session_state["extraction"] = reviewed_extraction
            st.session_state["inventory_df"] = reviewed_inventory_df
            st.session_state.pop("candidates", None)
            st.session_state.pop("mapping_df", None)
            st.success(
                "Process review applied. Source flow IDs and evidence were preserved while process assignments were updated."
            )
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    extraction = st.session_state["extraction"]
    st.caption(
        "You can rename, remove, re-parent or merge AI-proposed processes before any Brightway matching. "
        "The original role classifications and evidence remain in the extraction audit trail."
    )

    if extraction.assumptions_or_warnings:
        with st.expander("Assumptions / warnings", expanded=True):
            for warning in extraction.assumptions_or_warnings:
                st.write(f"• {warning}")

    st.subheader("Review foreground inventory")
    current_process_ids = [p.process_id for p in extraction.processes]
    edited_df = st.data_editor(
        st.session_state["inventory_df"],
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "include": st.column_config.CheckboxColumn("Include"),
            "flow_id": st.column_config.NumberColumn("Flow ID", disabled=True),
            "review_status": st.column_config.TextColumn("Review status", disabled=True),
            "process_id": st.column_config.SelectboxColumn("Process ID", options=current_process_ids),
            "process_name": st.column_config.TextColumn("Process", disabled=True),
            "direction": st.column_config.SelectboxColumn(
                "Direction", options=["input", "output", "emission", "unknown"]
            ),
            "linked_process_id": st.column_config.SelectboxColumn(
                "Linked foreground process",
                options=[""] + current_process_ids,
                help="Use only for a reviewed explicit foreground-to-foreground input link.",
            ),
            "background_process_hint": st.column_config.TextColumn(
                "Background process hint",
                help=(
                    "Verbatim technical background-process identifier, only when the source explicitly printed one "
                    "(e.g. a supplementary LCA-software process export). Used as the default Brightway search text "
                    "instead of the plain flow name when present. Edit or clear it if it looks wrong."
                ),
            ),
            "document": st.column_config.TextColumn("Document", disabled=True),
            "page": st.column_config.NumberColumn("Page", disabled=True),
            "paragraph": st.column_config.NumberColumn("Paragraph", disabled=True),
            "table": st.column_config.TextColumn("Table", disabled=True),
            "evidence_text": st.column_config.TextColumn("Source evidence", disabled=True, width="large"),
        },
        key="inventory_editor",
    )
    edited_df = normalize_inventory_review(
        edited_df,
        extraction=extraction,
        original_extraction=st.session_state["original_extraction"],
    )
    st.session_state["inventory_df"] = edited_df
    st.caption(
        "Source evidence is read-only. Rows are labelled AI proposed, human edited, or user added; the untouched AI extraction is preserved separately."
    )

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "Download reviewed inventory CSV",
            edited_df.to_csv(index=False).encode("utf-8"),
            file_name="reviewed_foreground_inventory.csv",
            mime="text/csv",
        )
    with dl2:
        st.download_button(
            "Download reviewed inventory JSON",
            dataframe_to_json(edited_df),
            file_name="reviewed_foreground_inventory.json",
            mime="application/json",
        )
    with dl3:
        st.download_button(
            "Download original AI extraction JSON",
            st.session_state["original_extraction"].model_dump_json(indent=2),
            file_name="ai_lca_original_extraction.json",
            mime="application/json",
        )

    can_search = bool(project_name and database_name)
    if st.button("2. Search Brightway candidates", disabled=not can_search):
        location_hints = ecoinvent_location_hints(context.operational_geography)
        candidate_map: dict[int, list[dict]] = {}
        query_map: dict[int, str] = {}
        db_map: dict[int, str] = {}
        flow_hint_map: dict[int, list[str]] = {}
        candidate_cache: dict[tuple[str, str, tuple[str, ...]], list[dict]] = {}
        progress = st.progress(0.0)
        included = edited_df[edited_df["include"] == True].reset_index(drop=True)  # noqa: E712

        for n, (_, row) in enumerate(included.iterrows(), start=1):
            flow_id_raw = row.get("flow_id", n - 1)
            flow_id = int(flow_id_raw) if pd.notna(flow_id_raw) else n - 1
            base_query = _default_search_text(row)
            unit = _row_unit(row)
            query = _search_display_text(row)
            direction = str(row.get("direction", "unknown")).strip().casefold()
            linked_process = str(row.get("linked_process_id", "") or "").strip()
            query_map[flow_id] = query

            if linked_process and linked_process.lower() != "nan":
                candidate_map[flow_id] = [
                    {
                        "foreground_link": linked_process,
                        "name": f"Foreground process: {linked_process}",
                    }
                ]
                progress.progress(n / max(len(included), 1))
                continue
            search_db = database_name
            flow_hints = [
                hint for hint in [*location_hints, *parse_flow_location_hint(base_query)]
            ]
            flow_hints = list(dict.fromkeys(flow_hints))  # dedupe, keep order
            preferred_locations = flow_hints
            if direction == "emission":
                if not biosphere_databases:
                    candidate_map[flow_id] = [{"error": "No biosphere database is available in this Brightway project."}]
                    progress.progress(n / max(len(included), 1))
                    continue
                search_db = biosphere_databases[0]
                preferred_locations = []
            db_map[flow_id] = search_db
            flow_hint_map[flow_id] = preferred_locations

            search_text = _strip_unit_suffix(query, unit)
            cache_key = (search_db, search_text.casefold(), tuple(preferred_locations))
            try:
                if cache_key not in candidate_cache:
                    candidate_cache[cache_key] = search_candidates(
                        project_name=project_name,
                        database_name=search_db,
                        query=search_text,
                        preferred_locations=preferred_locations,
                        limit=candidate_limit,
                    )
                candidate_map[flow_id] = candidate_cache[cache_key]
            except Exception as exc:
                candidate_map[flow_id] = [{"error": str(exc)}]
            progress.progress(n / max(len(included), 1))

        st.session_state["candidates"] = candidate_map
        st.session_state["candidate_queries"] = query_map
        st.session_state["original_queries"] = dict(query_map)
        st.session_state["candidate_dbs"] = db_map
        st.session_state["candidate_flow_hints"] = flow_hint_map
        st.session_state["candidate_location_hints"] = location_hints

if "candidates" in st.session_state and "inventory_df" in st.session_state:
    extraction = st.session_state["extraction"]
    st.subheader("Review Brightway mappings")
    hints = st.session_state.get("candidate_location_hints", [])
    if hints:
        st.caption(
            "Technosphere candidates matching paper-derived geography are softly promoted, never filtered. "
            f"Location hints: {', '.join(hints)}. Emissions are searched in the biosphere database."
        )
    else:
        st.caption(
            "No supported operational geography was found, so technosphere search order is unchanged. "
            "Emissions are searched separately in the biosphere database."
        )

    mapping_rows = []
    inv_df = st.session_state["inventory_df"]
    process_order = [p.process_id for p in extraction.processes]

    for process_id in process_order:
        process_rows = inv_df[(inv_df["process_id"] == process_id) & (inv_df["include"] == True)]  # noqa: E712
        if process_rows.empty:
            continue
        process_name = str(process_rows.iloc[0].get("process_name", process_id))
        st.markdown(f"## {process_name}")

        for _, row in process_rows.iterrows():
            flow_id = int(row["flow_id"])
            flow_name = str(row.get("name", f"Flow {flow_id}"))
            direction = str(row.get("direction", "unknown")).strip().casefold()
            linked_process = str(row.get("linked_process_id", "") or "").strip()
            if linked_process.lower() == "nan":
                linked_process = ""
            st.markdown(f"### {flow_name}")

            if linked_process:
                st.info(f"Explicit foreground link → {linked_process}. No ecoinvent mapping is needed for this exchange.")
                continue
            if direction == "output":
                exclude_output = st.checkbox(
                    "Exclude this output/co-product from the write — I don't want to model it",
                    key=f"exclude_output_{flow_id}",
                )
                if exclude_output:
                    inv_df.loc[inv_df["flow_id"] == flow_id, "include"] = False
                    st.session_state["inventory_df"] = inv_df
                    st.info("Excluded from the write. Amount/evidence stay in the reviewed inventory export.")
                    continue
                st.caption(
                    "Additional output/co-product (e.g. a recycling credit/avoided-burden flow). Map it to a real "
                    "Brightway node below and it will be written as a technosphere exchange using its reviewed "
                    f"signed amount ({row.get('amount')} {row.get('unit', '')}) — no allocation/production model is invented."
                )

            default_db = (
                biosphere_databases[0] if direction == "emission" and biosphere_databases else database_name
            )
            search_db_for_flow = st.session_state.get("candidate_dbs", {}).get(flow_id, default_db)
            default_query = st.session_state.get("candidate_queries", {}).get(
                flow_id, _search_display_text(row)
            )
            flow_hints = st.session_state.get("candidate_flow_hints", {}).get(flow_id, [])
            unit = _row_unit(row)
            original_query = st.session_state.get("original_queries", {}).get(
                flow_id, _search_display_text(row)
            )

            query_col, button_col = st.columns([5, 1])
            with query_col:
                edited_query = st.text_input(
                    "Search query",
                    value=default_query,
                    key=f"query_{flow_id}",
                    label_visibility="collapsed",
                    help=f"Searched against '{search_db_for_flow}'. Edit and re-search if the automatic match is wrong.",
                )
            with button_col:
                rerun_search = st.button("Search", key=f"research_{flow_id}")
            st.caption(f"Original: {original_query}")
            search_text_preview = _strip_unit_suffix(edited_query, unit)
            normalized_query = normalize_search_query(search_text_preview)
            caption = f"Geography hint: {', '.join(flow_hints) if flow_hints else 'none'} · DB: {search_db_for_flow}"
            if normalized_query != edited_query.strip():
                caption += f' · Searching as: "{normalized_query}" (unit/system-model labels are shown but not searched)'
            st.caption(caption)

            if rerun_search:
                try:
                    search_text = _strip_unit_suffix(edited_query, unit)
                    new_hints = (
                        []
                        if direction == "emission"
                        else list(
                            dict.fromkeys(
                                [
                                    *ecoinvent_location_hints(context.operational_geography),
                                    *parse_flow_location_hint(search_text),
                                ]
                            )
                        )
                    )
                    new_candidates = search_candidates(
                        project_name=project_name,
                        database_name=search_db_for_flow,
                        query=search_text,
                        preferred_locations=new_hints,
                        limit=candidate_limit,
                    )
                    st.session_state["candidates"][flow_id] = new_candidates
                    st.session_state.setdefault("candidate_queries", {})[flow_id] = edited_query
                    st.session_state.setdefault("candidate_flow_hints", {})[flow_id] = new_hints
                    st.session_state.setdefault("candidate_dbs", {})[flow_id] = search_db_for_flow
                except Exception as exc:
                    st.session_state["candidates"][flow_id] = [{"error": str(exc)}]
                st.rerun()

            candidates = st.session_state["candidates"].get(flow_id, [])
            if not candidates:
                st.warning("No candidates returned. Edit the search text above and press Search.")
                continue
            if "error" in candidates[0]:
                st.error(candidates[0]["error"])
                continue

            display = pd.DataFrame(candidates)
            display.insert(0, "rank", range(1, len(display) + 1))
            display_columns = [
                column
                for column in [
                    "rank",
                    "name",
                    "reference_product",
                    "location",
                    "categories",
                    "unit",
                    "database",
                    "id",
                    "code",
                ]
                if column in display.columns
            ]
            st.dataframe(display[display_columns], width="stretch", hide_index=True)

            labels = [
                f"{c.get('name', '')} | {c.get('reference_product', '')} | "
                f"{c.get('location', c.get('categories', ''))} | {c.get('unit', '')}"
                for c in candidates
            ]
            no_selection = "— no selection —"
            choice = st.selectbox(
                "Selected mapping",
                labels + [no_selection],
                index=0,
                key=f"mapping_{flow_id}",
            )
            if choice != no_selection:
                chosen_index = labels.index(choice)
                chosen = candidates[chosen_index]
                mapping_rows.append(
                    {
                        "flow_id": flow_id,
                        "process_id": process_id,
                        "process_name": process_name,
                        "flow_name": flow_name,
                        "mapping_kind": "biosphere" if direction == "emission" else "technosphere",
                        **chosen,
                    }
                )

    mapping_df = pd.DataFrame(mapping_rows)
    st.session_state["mapping_df"] = mapping_df

    if not mapping_df.empty:
        st.subheader("Selected mappings")
        st.dataframe(mapping_df, width="stretch", hide_index=True)
        st.download_button(
            "Download selected mappings CSV",
            mapping_df.to_csv(index=False).encode("utf-8"),
            file_name="selected_brightway_mappings.csv",
            mime="text/csv",
        )

    st.download_button(
        "Download reproducible review bundle",
        review_bundle_to_json(
            extraction,
            inv_df,
            mapping_df,
            original_extraction=st.session_state["original_extraction"],
        ),
        file_name="ai_lca_review_bundle.json",
        mime="application/json",
    )

    st.subheader("3. Create reviewed Brightway foreground database")
    plan = build_write_plan(extraction, inv_df, mapping_df)
    if plan.ready:
        st.success(
            f"Write validation passed: {len(extraction.processes)} process(es), "
            f"{len(plan.exchanges)} quantitative exchange(s)."
        )
    else:
        st.warning(
            "The database writer is deliberately strict. Resolve these items before writing so it does not invent units, "
            "conversions, co-product treatment or missing mappings."
        )
        with st.expander("Write blockers", expanded=True):
            for blocker in plan.blockers:
                st.write(f"• {blocker}")

    foreground_db_name = st.text_input(
        "New foreground database name",
        value="ai_lca_reviewed_foreground",
        help="The writer never overwrites an existing Brightway database.",
    )
    confirm_write = st.checkbox(
        "I have reviewed the process structure, amounts/units and selected Brightway mappings."
    )
    if st.button(
        "Create Brightway foreground database",
        disabled=not (plan.ready and project_name and foreground_db_name.strip() and confirm_write),
    ):
        try:
            report = write_foreground_database(
                project_name=project_name,
                database_name=foreground_db_name,
                extraction=extraction,
                inventory_df=inv_df,
                mapping_df=mapping_df,
            )
            st.success(
                f"Created {report['database']}: {report['processes_created']} activities and "
                f"{report['exchanges_created']} inventory exchanges."
            )
            for warning in report["warnings"]:
                st.info(warning)
        except Exception as exc:
            st.exception(exc)
