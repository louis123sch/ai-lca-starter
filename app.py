from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ai_lca.brightway_search import (
    list_biosphere_databases,
    list_databases,
    search_candidates,
)
from ai_lca.brightway_writer import build_write_plan, write_foreground_database
from ai_lca.documents import combine_document_evidence
from ai_lca.export import (
    candidate_structure_to_dataframe,
    dataframe_to_json,
    extraction_to_dataframe,
    process_structure_to_dataframe,
    review_bundle_to_json,
)
from ai_lca.geography import ecoinvent_location_hints
from ai_lca.llm import extract_inventory_from_documents, extract_inventory_from_text
from ai_lca.review import apply_process_review
from ai_lca.runtime import extractor_version, git_sha

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
    model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    project_name = st.text_input("Brightway project", value=os.getenv("BRIGHTWAY_PROJECT", ""))
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
        "Upload the paper and any supplementary PDF/Word documents",
        type=["pdf", "docx"],
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
        st.session_state["original_extraction"] = extraction
        st.session_state["extraction"] = extraction
        st.session_state["process_review_df"] = process_structure_to_dataframe(extraction)
        st.session_state["inventory_df"] = extraction_to_dataframe(extraction)
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
            reviewed_extraction = apply_process_review(
                st.session_state["original_extraction"],
                reviewed_process_df,
            )
            if not reviewed_extraction.processes:
                raise ValueError("At least one foreground process must remain after review.")
            st.session_state["extraction"] = reviewed_extraction
            st.session_state["inventory_df"] = extraction_to_dataframe(reviewed_extraction)
            st.session_state.pop("candidates", None)
            st.session_state.pop("mapping_df", None)
            st.success("Process review applied. Flow assignments were updated deterministically.")
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
    edited_df = st.data_editor(
        st.session_state["inventory_df"],
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "include": st.column_config.CheckboxColumn("Include"),
            "flow_id": st.column_config.NumberColumn("Flow ID", disabled=True),
            "process_id": st.column_config.TextColumn("Process ID", disabled=True),
            "process_name": st.column_config.TextColumn("Process", disabled=True),
            "linked_process_id": st.column_config.TextColumn("Linked foreground process", disabled=True),
            "evidence_text": st.column_config.TextColumn("Evidence", width="large"),
        },
        key="inventory_editor",
    )
    st.session_state["inventory_df"] = edited_df

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
        candidate_cache: dict[tuple[str, str], list[dict]] = {}
        progress = st.progress(0.0)
        included = edited_df[edited_df["include"] == True].reset_index(drop=True)  # noqa: E712

        for n, (_, row) in enumerate(included.iterrows(), start=1):
            flow_id_raw = row.get("flow_id", n - 1)
            flow_id = int(flow_id_raw) if pd.notna(flow_id_raw) else n - 1
            query = str(row.get("name", "")).strip()
            direction = str(row.get("direction", "unknown")).strip().casefold()
            linked_process = str(row.get("linked_process_id", "") or "").strip()

            if linked_process and linked_process.lower() != "nan":
                candidate_map[flow_id] = [
                    {
                        "foreground_link": linked_process,
                        "name": f"Foreground process: {linked_process}",
                    }
                ]
                progress.progress(n / max(len(included), 1))
                continue
            if direction == "output":
                candidate_map[flow_id] = []
                progress.progress(n / max(len(included), 1))
                continue

            search_db = database_name
            preferred_locations = location_hints
            if direction == "emission":
                if not biosphere_databases:
                    candidate_map[flow_id] = [{"error": "No biosphere database is available in this Brightway project."}]
                    progress.progress(n / max(len(included), 1))
                    continue
                search_db = biosphere_databases[0]
                preferred_locations = []

            cache_key = (search_db, query.casefold())
            try:
                if cache_key not in candidate_cache:
                    candidate_cache[cache_key] = search_candidates(
                        project_name=project_name,
                        database_name=search_db,
                        query=query,
                        preferred_locations=preferred_locations,
                        limit=candidate_limit,
                    )
                candidate_map[flow_id] = candidate_cache[cache_key]
            except Exception as exc:
                candidate_map[flow_id] = [{"error": str(exc)}]
            progress.progress(n / max(len(included), 1))

        st.session_state["candidates"] = candidate_map
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
            candidates = st.session_state["candidates"].get(flow_id, [])
            st.markdown(f"### {flow_name}")

            if linked_process:
                st.info(f"Explicit foreground link → {linked_process}. No ecoinvent mapping is needed for this exchange.")
                continue
            if direction == "output":
                st.warning(
                    "Additional output/co-product: no background mapping is proposed. The strict writer will require "
                    "this to be excluded or explicitly modelled rather than inventing allocation/production structure."
                )
                continue
            if not candidates:
                st.warning("No candidates returned.")
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
