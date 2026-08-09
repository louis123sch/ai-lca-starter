from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ai_lca.brightway_search import list_databases, search_candidates
from ai_lca.documents import extract_document_text
from ai_lca.export import dataframe_to_json, extraction_to_dataframe, process_structure_to_dataframe
from ai_lca.geography import ecoinvent_location_hints
from ai_lca.llm import extract_inventory_from_text

load_dotenv()

st.set_page_config(page_title="AI-LCA Foreground Builder", layout="wide")
st.title("AI-LCA Foreground Builder")
st.caption("Source evidence → process structure → foreground flows → human review → Brightway matching")

with st.sidebar:
    st.header("Configuration")
    model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    project_name = st.text_input("Brightway project", value=os.getenv("BRIGHTWAY_PROJECT", ""))
    candidate_limit = st.slider("Candidates per flow", 3, 20, 8)

    database_name = ""
    if project_name:
        try:
            dbs = list_databases(project_name)
            if dbs:
                default_index = next((i for i, x in enumerate(dbs) if "ecoinvent" in x.lower()), 0)
                database_name = st.selectbox("Background database", dbs, index=default_index)
            else:
                st.warning("No databases found in this Brightway project.")
        except Exception as exc:
            st.warning(f"Brightway project not available yet: {exc}")

paste_tab, document_tab = st.tabs(["Paste text", "Upload document"])

with paste_tab:
    pasted_text = st.text_area(
        "Paste paper text, technical documentation, datasheet text, or engineering notes",
        height=320,
        placeholder="Paste the relevant source material here…",
    )

with document_tab:
    uploaded_document = st.file_uploader(
        "Upload a text-readable PDF or Word document",
        type=["pdf", "docx"],
    )
    document_preview = ""
    if uploaded_document is not None:
        try:
            document_preview = extract_document_text(
                uploaded_document.getvalue(),
                uploaded_document.name,
            )
            st.success(
                f"Extracted machine-readable text from {uploaded_document.name} "
                f"({len(document_preview):,} characters)."
            )
            with st.expander("Preview extracted text"):
                st.text(document_preview[:12000])
        except Exception as exc:
            st.error(str(exc))

extra_instructions = st.text_area(
    "Optional study instructions",
    placeholder="Example: Focus on cradle-to-gate foreground inputs for 1 kg H2; keep infrastructure and operation separate only where the paper does.",
    height=90,
)

source_text = pasted_text.strip() or document_preview.strip()

if st.button("1. Interpret paper and extract foreground", type="primary", disabled=not bool(source_text)):
    try:
        with st.spinner("Identifying process structure, then extracting evidence-backed flows…"):
            extraction = extract_inventory_from_text(
                source_text,
                model=model,
                extra_instructions=extra_instructions,
            )
        st.session_state["extraction"] = extraction
        st.session_state["inventory_df"] = extraction_to_dataframe(extraction)
        st.session_state.pop("candidates", None)
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
    if context.operational_geography:
        st.caption(
            f"Geography basis: {context.geography_basis}. "
            f"{context.geography_rationale or ''}".strip()
        )
    if context.system_boundary:
        st.write(f"**System boundary:** {context.system_boundary}")
    if context.temporal_context:
        st.write(f"**Temporal context:** {context.temporal_context}")
    st.write(extraction.source_summary)

    st.subheader("Detected foreground process structure")
    process_df = process_structure_to_dataframe(extraction)
    st.dataframe(process_df, width="stretch", hide_index=True)
    st.caption(
        "Subprocesses are only retained when the source supports them as separately modelled steps. "
        "Background supplies such as electricity or natural gas are not promoted into processes by default."
    )

    if extraction.assumptions_or_warnings:
        with st.expander("Assumptions / warnings", expanded=True):
            for warning in extraction.assumptions_or_warnings:
                st.write(f"• {warning}")

    st.subheader("Proposed foreground inventory")
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
            "evidence_text": st.column_config.TextColumn("Evidence", width="large"),
        },
        key="inventory_editor",
    )
    st.session_state["inventory_df"] = edited_df

    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download reviewed inventory CSV",
            edited_df.to_csv(index=False).encode("utf-8"),
            file_name="reviewed_foreground_inventory.csv",
            mime="text/csv",
        )
    with right:
        st.download_button(
            "Download reviewed inventory JSON",
            dataframe_to_json(edited_df),
            file_name="reviewed_foreground_inventory.json",
            mime="application/json",
        )

    can_search = bool(project_name and database_name)
    if st.button("2. Search ecoinvent candidates", disabled=not can_search):
        location_hints = ecoinvent_location_hints(context.operational_geography)
        candidate_map: dict[int, list[dict]] = {}
        candidate_cache: dict[str, list[dict]] = {}
        progress = st.progress(0.0)
        included = edited_df[edited_df["include"] == True].reset_index(drop=True)  # noqa: E712

        for n, (_, row) in enumerate(included.iterrows(), start=1):
            flow_id_raw = row.get("flow_id", n - 1)
            flow_id = int(flow_id_raw) if pd.notna(flow_id_raw) else n - 1
            query = str(row.get("name", "")).strip()
            cache_key = query.casefold()
            try:
                if cache_key not in candidate_cache:
                    candidate_cache[cache_key] = search_candidates(
                        project_name=project_name,
                        database_name=database_name,
                        query=query,
                        preferred_locations=location_hints,
                        limit=candidate_limit,
                    )
                candidate_map[flow_id] = candidate_cache[cache_key]
            except Exception as exc:
                candidate_map[flow_id] = [{"error": str(exc)}]
            progress.progress(n / max(len(included), 1))

        st.session_state["candidates"] = candidate_map
        st.session_state["candidate_location_hints"] = location_hints

if "candidates" in st.session_state and "inventory_df" in st.session_state:
    st.subheader("Candidate background processes")
    hints = st.session_state.get("candidate_location_hints", [])
    if hints:
        st.caption(
            "Candidates matching the paper-derived operational geography are softly promoted, not filtered. "
            f"Location hints: {', '.join(hints)}."
        )
    else:
        st.caption("No supported operational geography was found, so Brightway's search order is used unchanged.")
    st.caption(
        "Candidate results come from your selected Brightway database. Identical flow queries are searched once and reused. "
        "The top-ranked candidate is preselected, but you can change it or choose no selection."
    )

    mapping_rows = []
    inv_df = st.session_state["inventory_df"]
    process_order = [p.process_id for p in st.session_state["extraction"].processes]

    for process_id in process_order:
        process_rows = inv_df[(inv_df["process_id"] == process_id) & (inv_df["include"] == True)]  # noqa: E712
        if process_rows.empty:
            continue
        process_name = str(process_rows.iloc[0].get("process_name", process_id))
        st.markdown(f"## {process_name}")

        for _, row in process_rows.iterrows():
            flow_id = int(row["flow_id"])
            flow_name = str(row.get("name", f"Flow {flow_id}"))
            candidates = st.session_state["candidates"].get(flow_id, [])
            st.markdown(f"### {flow_name}")

            if not candidates:
                st.warning("No candidates returned.")
                continue
            if "error" in candidates[0]:
                st.error(candidates[0]["error"])
                continue

            display = pd.DataFrame(candidates)
            display.insert(0, "rank", range(1, len(display) + 1))
            st.dataframe(
                display[["rank", "name", "reference_product", "location", "unit", "database", "id", "code"]],
                width="stretch",
                hide_index=True,
            )

            labels = [
                f"{c['name']} | {c['reference_product']} | {c['location']} | {c['unit']}"
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
                        **chosen,
                    }
                )

    if mapping_rows:
        mapping_df = pd.DataFrame(mapping_rows)
        st.subheader("Selected mappings")
        st.dataframe(mapping_df, width="stretch", hide_index=True)
        st.download_button(
            "Download selected mappings CSV",
            mapping_df.to_csv(index=False).encode("utf-8"),
            file_name="selected_ecoinvent_mappings.csv",
            mime="text/csv",
        )
