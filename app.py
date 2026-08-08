from __future__ import annotations

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ai_lca.documents import extract_pdf_text
from ai_lca.llm import extract_inventory_from_text, extract_process_map_from_text
from ai_lca.export import dataframe_to_json, extraction_to_dataframe, process_map_to_dataframe
from ai_lca.brightway_search import list_databases, search_candidates

load_dotenv()

st.set_page_config(page_title="AI-LCA Foreground Builder", layout="wide")
st.title("AI-LCA Foreground Builder")
st.caption("AI maps the paper → human confirms foreground processes → AI extracts exchanges → human approves → Brightway calculates")

with st.sidebar:
    st.header("Configuration")
    model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    project_name = st.text_input("Brightway project", value=os.getenv("BRIGHTWAY_PROJECT", ""))
    locations_text = st.text_input("Preferred ecoinvent locations", value="GB,RER,GLO,RoW")
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

paste_tab, pdf_tab = st.tabs(["Paste text", "Upload PDF"])

with paste_tab:
    pasted_text = st.text_area(
        "Paste paper text, technical documentation, datasheet text, or engineering notes",
        height=320,
        placeholder="Paste the relevant source material here…",
    )

with pdf_tab:
    uploaded_pdf = st.file_uploader("Upload a text-readable PDF", type=["pdf"])
    pdf_preview = ""
    if uploaded_pdf is not None:
        try:
            pdf_preview = extract_pdf_text(uploaded_pdf.getvalue())
            st.success(f"Extracted machine-readable text from PDF ({len(pdf_preview):,} characters).")
            with st.expander("Preview extracted text"):
                st.text(pdf_preview[:12000])
        except Exception as exc:
            st.error(str(exc))

extra_instructions = st.text_area(
    "Optional study instructions",
    placeholder="Example: Focus on cradle-to-gate foreground inputs for 1 kg H2; keep infrastructure and operation separate.",
    height=90,
)

source_text = pasted_text.strip() or pdf_preview.strip()

if st.button("1. Analyse paper structure", type="primary", disabled=not bool(source_text)):
    try:
        with st.spinner("Identifying paper-supported foreground processes and descriptive operations…"):
            process_map = extract_process_map_from_text(
                source_text,
                model=model,
                extra_instructions=extra_instructions,
            )
        st.session_state["process_map"] = process_map
        st.session_state["process_map_df"] = process_map_to_dataframe(process_map)
        st.session_state.pop("extraction", None)
        st.session_state.pop("inventory_df", None)
        st.session_state.pop("candidates", None)
    except Exception as exc:
        st.exception(exc)

if "process_map" in st.session_state:
    process_map = st.session_state["process_map"]
    st.subheader("Review AI foreground interpretation")
    st.caption(
        "Only paper-supported foreground processes are listed as processes. Engineering steps shown under Operations are context only and will not become separate Brightway activities or ecoinvent searches."
    )

    meta1, meta2, meta3, meta4 = st.columns(4)
    meta1.write(f"**Functional unit:** {process_map.functional_unit or 'Not identified'}")
    meta2.write(f"**System boundary:** {process_map.system_boundary or 'Not identified'}")
    meta3.write(f"**Study geography:** {process_map.geographic_context or 'Not identified'}")
    meta4.write(f"**Time context:** {process_map.temporal_context or 'Not identified'}")
    st.write(process_map.source_summary)

    if process_map.assumptions_or_warnings:
        with st.expander("Process-map warnings", expanded=True):
            for warning in process_map.assumptions_or_warnings:
                st.write(f"• {warning}")

    for group in process_map.technology_groups:
        with st.expander(f"{group.name} — {len(group.processes)} foreground process(es)", expanded=True):
            if group.description:
                st.write(group.description)
            for process in group.processes:
                st.markdown(f"**{process.process_id}: {process.name}**")
                context_bits = []
                if process.geographic_context:
                    context_bits.append(f"geography: {process.geographic_context}")
                if process.temporal_context:
                    context_bits.append(f"time: {process.temporal_context}")
                if context_bits:
                    st.caption(" | ".join(context_bits))
                st.write(process.reason_for_separate_process)
                if process.evidence:
                    evidence = process.evidence[0]
                    where = [
                        f"p. {evidence.page}" if evidence.page is not None else None,
                        evidence.table,
                        evidence.section,
                    ]
                    where_text = " · ".join(x for x in where if x)
                    if where_text:
                        st.caption(where_text)
                    st.code(evidence.evidence_text, language=None)
                if process.operations:
                    st.markdown("**Operations described inside this process — not separate foreground processes:**")
                    for operation in process.operations:
                        st.write(f"• {operation.name}")

    process_df = st.data_editor(
        st.session_state["process_map_df"],
        width="stretch",
        hide_index=True,
        num_rows="fixed",
        disabled=[
            "process_id",
            "technology_group",
            "process_name",
            "geographic_context",
            "temporal_context",
            "evidence_type",
            "confidence",
            "reason_for_separate_process",
            "operations_not_separate_processes",
            "page",
            "table",
            "evidence_text",
        ],
        column_config={
            "include": st.column_config.CheckboxColumn("Use as foreground process"),
            "process_id": st.column_config.TextColumn("Process ID", width="small"),
            "process_name": st.column_config.TextColumn("Foreground process", width="large"),
            "technology_group": st.column_config.TextColumn("Technology / pathway"),
            "operations_not_separate_processes": st.column_config.TextColumn("Operations (context only)", width="large"),
            "reason_for_separate_process": st.column_config.TextColumn("Why this is a separate process", width="large"),
            "evidence_text": st.column_config.TextColumn("Process evidence", width="large"),
        },
        key="process_map_editor",
    )
    st.session_state["process_map_df"] = process_df

    selected_process_ids = process_df.loc[process_df["include"] == True, "process_id"].astype(str).tolist()  # noqa: E712
    if st.button("2. Extract inventory for selected processes", disabled=not bool(selected_process_ids)):
        try:
            with st.spinner("Extracting exchanges only under the confirmed process map…"):
                extraction = extract_inventory_from_text(
                    source_text,
                    process_map=process_map,
                    approved_process_ids=selected_process_ids,
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
    st.subheader("Proposed foreground inventory")
    st.caption(
        "Flows are tied to the confirmed foreground process IDs. Only quantified inputs default to background matching; outputs, emissions and unquantified mentions remain visible but are not searched automatically."
    )

    if extraction.assumptions_or_warnings:
        with st.expander("Inventory warnings", expanded=True):
            for warning in extraction.assumptions_or_warnings:
                st.write(f"• {warning}")

    edited_df = st.data_editor(
        st.session_state["inventory_df"],
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        disabled=["background_match_eligible", "flow_id", "process_id"],
        column_config={
            "include": st.column_config.CheckboxColumn("Search ecoinvent"),
            "background_match_eligible": st.column_config.CheckboxColumn("Eligible", disabled=True),
            "flow_id": st.column_config.NumberColumn("Flow ID", disabled=True),
            "process_id": st.column_config.TextColumn("Process ID", disabled=True),
            "process_name": st.column_config.TextColumn("Foreground process", width="large"),
            "technology_group": st.column_config.TextColumn("Technology / pathway"),
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
    if st.button("3. Search ecoinvent candidates", disabled=not can_search):
        locations = [x.strip() for x in locations_text.split(",") if x.strip()]
        candidate_map: dict[int, list[dict]] = {}
        progress = st.progress(0.0)
        included = edited_df[
            (edited_df["include"] == True)  # noqa: E712
            & (edited_df["background_match_eligible"] == True)  # noqa: E712
        ].reset_index(drop=True)

        blocked = edited_df[
            (edited_df["include"] == True)  # noqa: E712
            & (edited_df["background_match_eligible"] != True)  # noqa: E712
        ]
        if not blocked.empty:
            st.warning(
                f"Skipped {len(blocked)} selected row(s) because they are not quantified foreground inputs."
            )

        for n, (_, row) in enumerate(included.iterrows(), start=1):
            flow_id = int(row.get("flow_id", n - 1))
            query = str(row.get("name", "")).strip()
            try:
                candidate_map[flow_id] = search_candidates(
                    project_name=project_name,
                    database_name=database_name,
                    query=query,
                    locations=locations,
                    limit=candidate_limit,
                )
            except Exception as exc:
                candidate_map[flow_id] = [{"error": str(exc)}]
            progress.progress(n / max(len(included), 1))
        st.session_state["candidates"] = candidate_map

if "candidates" in st.session_state and "inventory_df" in st.session_state:
    st.subheader("Candidate background processes")
    st.caption("These are real results returned by your selected Brightway database. The prototype does not auto-approve them.")
    mapping_rows = []
    inv_df = st.session_state["inventory_df"]

    for flow_id, candidates in st.session_state["candidates"].items():
        matching = inv_df[inv_df["flow_id"] == flow_id]
        flow_name = matching.iloc[0]["name"] if not matching.empty else f"Flow {flow_id}"
        process_name = matching.iloc[0]["process_name"] if not matching.empty else "Unknown process"
        st.markdown(f"### {flow_name}")
        st.caption(f"Foreground process: {process_name}")

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
        choice = st.selectbox(
            "Approve mapping",
            ["— no selection —"] + labels,
            key=f"mapping_{flow_id}",
        )
        if choice != "— no selection —":
            chosen_index = labels.index(choice)
            chosen = candidates[chosen_index]
            mapping_rows.append({"flow_id": flow_id, "flow_name": flow_name, "process_name": process_name, **chosen})

    if mapping_rows:
        mapping_df = pd.DataFrame(mapping_rows)
        st.subheader("Approved mappings")
        st.dataframe(mapping_df, width="stretch", hide_index=True)
        st.download_button(
            "Download approved mappings CSV",
            mapping_df.to_csv(index=False).encode("utf-8"),
            file_name="approved_ecoinvent_mappings.csv",
            mime="text/csv",
        )
