from __future__ import annotations

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ai_lca.documents import extract_pdf_text
from ai_lca.llm import extract_inventory_from_text
from ai_lca.export import extraction_to_dataframe, dataframe_to_json
from ai_lca.brightway_search import list_databases, search_candidates

load_dotenv()

ITEM_TYPES = [
    "technosphere_flow",
    "biosphere_flow",
    "parameter",
    "reference_product",
]

st.set_page_config(page_title="AI-LCA Foreground Builder", layout="wide")
st.title("AI-LCA Foreground Builder")
st.caption("AI proposes → deterministic validation → human approves → Brightway calculates")

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

if st.button("1. Extract proposed inventory", type="primary", disabled=not bool(source_text)):
    try:
        with st.spinner("Extracting and classifying LCA-relevant information…"):
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
    st.subheader("Proposed LCA information")

    meta1, meta2 = st.columns(2)
    meta1.write(f"**Process:** {extraction.process_name or 'Not identified'}")
    meta2.write(f"**Functional unit:** {extraction.functional_unit or 'Not identified'}")
    st.write(extraction.source_summary)

    if extraction.assumptions_or_warnings:
        with st.expander("Assumptions / warnings", expanded=True):
            for warning in extraction.assumptions_or_warnings:
                st.write(f"• {warning}")

    edited_df = st.data_editor(
        st.session_state["inventory_df"],
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "include": st.column_config.CheckboxColumn("Include"),
            "flow_id": st.column_config.NumberColumn("Flow ID", disabled=True),
            "item_type": st.column_config.SelectboxColumn(
                "Item type",
                options=ITEM_TYPES,
                required=True,
                help="Only technosphere_flow items are sent to the ecoinvent candidate search.",
            ),
            "evidence_text": st.column_config.TextColumn("Evidence", width="large"),
        },
        key="inventory_editor",
    )
    st.session_state["inventory_df"] = edited_df

    included_df = edited_df[edited_df["include"] == True]  # noqa: E712
    if not included_df.empty:
        counts = included_df["item_type"].value_counts().to_dict()
        st.caption(
            "Routing: "
            + ", ".join(f"{item_type}={counts.get(item_type, 0)}" for item_type in ITEM_TYPES)
        )

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
        locations = [x.strip() for x in locations_text.split(",") if x.strip()]
        candidate_map: dict[int, list[dict]] = {}
        progress = st.progress(0.0)

        searchable = edited_df[
            (edited_df["include"] == True)  # noqa: E712
            & (edited_df["item_type"] == "technosphere_flow")
        ].reset_index(drop=True)

        skipped = edited_df[
            (edited_df["include"] == True)  # noqa: E712
            & (edited_df["item_type"] != "technosphere_flow")
        ]
        if not skipped.empty:
            st.info(
                f"Skipped {len(skipped)} included item(s) that are parameters, biosphere flows, or reference products. "
                "They remain in the reviewed model information but are not searched in ecoinvent."
            )

        if searchable.empty:
            st.warning("No included technosphere flows are available for ecoinvent search.")
        else:
            for n, (_, row) in enumerate(searchable.iterrows(), start=1):
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
                progress.progress(n / max(len(searchable), 1))

        st.session_state["candidates"] = candidate_map

if "candidates" in st.session_state and "inventory_df" in st.session_state:
    st.subheader("Candidate background processes")
    st.caption("These are real results returned by your selected Brightway database. The prototype does not auto-approve them.")
    mapping_rows = []
    inv_df = st.session_state["inventory_df"]

    for flow_id, candidates in st.session_state["candidates"].items():
        matching = inv_df[inv_df["flow_id"] == flow_id]
        flow_name = matching.iloc[0]["name"] if not matching.empty else f"Flow {flow_id}"
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
        choice = st.selectbox(
            "Approve mapping",
            ["— no selection —"] + labels,
            key=f"mapping_{flow_id}",
        )
        if choice != "— no selection —":
            chosen_index = labels.index(choice)
            chosen = candidates[chosen_index]
            mapping_rows.append({"flow_id": flow_id, "flow_name": flow_name, **chosen})

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
