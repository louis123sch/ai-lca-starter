from __future__ import annotations

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ai_lca.documents import extract_document_text
from ai_lca.pipeline import extract_inventory_from_sources
from ai_lca.export import extraction_to_dataframe, dataframe_to_json, searchable_exchanges
from ai_lca.brightway_search import list_databases, search_candidates

load_dotenv()

ITEM_TYPES = [
    "technosphere_flow",
    "biosphere_flow",
    "parameter",
    "reference_product",
]
ACTIVITY_TYPES = [
    "market",
    "transforming",
    "treatment",
    "transport",
    "service",
    "construction",
    "operation",
    "unknown",
]

st.set_page_config(page_title="AI-LCA Foreground Builder", layout="wide")
st.title("AI-LCA Foreground Builder")
st.caption("AI reads and interprets → deterministic validation → human approves → Brightway retrieves")

with st.sidebar:
    st.header("Configuration")
    model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    project_name = st.text_input("Brightway project", value=os.getenv("BRIGHTWAY_PROJECT", ""))
    locations_text = st.text_input("Preferred ecoinvent locations", value="GB,RER,GLO,RoW")
    candidate_limit = st.slider("Candidates per exchange", 3, 20, 8)

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

st.subheader("1. Add source material")
upload_tab, paste_tab = st.tabs(["Upload documents", "Paste text"])

extracted_documents: list[tuple[str, str]] = []

with upload_tab:
    uploaded_files = st.file_uploader(
        "Drop PDF or Word files here",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Each source is interpreted independently before the proposed foregrounds are merged.",
    )

    if uploaded_files:
        file_status_rows = []
        for uploaded_file in uploaded_files:
            try:
                extracted_text = extract_document_text(uploaded_file.getvalue(), uploaded_file.name)
                extracted_documents.append((uploaded_file.name, extracted_text))
                file_status_rows.append(
                    {
                        "document": uploaded_file.name,
                        "type": uploaded_file.name.rsplit(".", 1)[-1].lower(),
                        "characters_extracted": len(extracted_text),
                        "status": "ready",
                    }
                )
            except Exception as exc:
                file_status_rows.append(
                    {
                        "document": uploaded_file.name,
                        "type": uploaded_file.name.rsplit(".", 1)[-1].lower(),
                        "characters_extracted": 0,
                        "status": f"error: {exc}",
                    }
                )

        st.dataframe(pd.DataFrame(file_status_rows), hide_index=True, width="stretch")
        for source_name, source_text_preview in extracted_documents:
            with st.expander(f"Preview extracted text — {source_name}"):
                st.text(source_text_preview[:12000])

with paste_tab:
    pasted_text = st.text_area(
        "Paste additional paper text, datasheet text, or engineering notes",
        height=260,
        placeholder="Optional: paste additional source material here…",
    )

extra_instructions = st.text_area(
    "Optional study instructions",
    placeholder=(
        "Example: Model cradle-to-gate hydrogen production in GB. Preserve distinct capture/compression stages "
        "only where the source supports them."
    ),
    height=90,
)

source_parts: list[tuple[str, str]] = list(extracted_documents)
if pasted_text.strip():
    source_parts.append(("pasted_text", pasted_text.strip()))

if source_parts:
    st.info(
        f"Ready to analyse {len(source_parts)} source(s): "
        + ", ".join(name for name, _ in source_parts)
        + ". Each source uses two AI passes: process understanding first, selective foreground extraction second."
    )

if st.button("2. Read documents and propose foreground", type="primary", disabled=not bool(source_parts)):
    try:
        with st.spinner(
            f"Understanding {len(source_parts)} source(s), then extracting ecoinvent-linkable foreground exchanges…"
        ):
            extraction = extract_inventory_from_sources(
                source_parts,
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
    st.subheader("3. Review AI foreground interpretation")

    meta1, meta2, meta3 = st.columns(3)
    meta1.write(f"**Technology:** {extraction.technology_name or 'Needs review'}")
    meta2.write(f"**Top-level process:** {extraction.process_name or 'Needs review'}")
    meta3.write(f"**Functional unit:** {extraction.functional_unit or 'Needs review'}")

    if extraction.system_description:
        st.write(extraction.system_description)

    process_names = []
    if extraction.foreground_processes:
        st.markdown("#### Proposed foreground process context")
        process_df = pd.DataFrame(
            [
                {
                    "process": process.name,
                    "role": process.role,
                    "source": process.source_document,
                    "evidence": process.evidence_text,
                }
                for process in extraction.foreground_processes
            ]
        )
        st.dataframe(process_df, hide_index=True, width="stretch")
        process_names = list(dict.fromkeys(process_df["process"].dropna().astype(str).tolist()))

    if extraction.assumptions_or_warnings:
        with st.expander("Interpretation notes / warnings", expanded=False):
            for warning in extraction.assumptions_or_warnings:
                st.write(f"• {warning}")

    current_df = st.session_state["inventory_df"]
    existing_parent_processes = current_df["parent_process"].dropna().astype(str).tolist()
    parent_process_options = [""] + list(dict.fromkeys(process_names + existing_parent_processes))

    st.markdown("#### Proposed exchanges and parameters")
    st.caption(
        "For technosphere items, 'Search concept' is the canonical product/service sent to Brightway. "
        "Foreground technology/scenario wording belongs in Parent process or Supplier technology, not in the search concept."
    )

    edited_df = st.data_editor(
        current_df,
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "include": st.column_config.CheckboxColumn("Include"),
            "flow_id": st.column_config.NumberColumn("ID", disabled=True),
            "name": st.column_config.TextColumn("Canonical concept", width="medium"),
            "source_label": st.column_config.TextColumn("Source wording", width="medium"),
            "item_type": st.column_config.SelectboxColumn(
                "Type", options=ITEM_TYPES, required=True
            ),
            "parent_process": st.column_config.SelectboxColumn(
                "Parent process", options=parent_process_options
            ),
            "search_worthy": st.column_config.CheckboxColumn(
                "Search ecoinvent?",
                help="Deterministic routing still requires Type = technosphere_flow and a non-empty search concept.",
            ),
            "ecoinvent_search_term": st.column_config.TextColumn("Search concept", width="medium"),
            "ecoinvent_activity_type_hint": st.column_config.SelectboxColumn(
                "Activity hint", options=ACTIVITY_TYPES
            ),
            "geography_hint": st.column_config.TextColumn("Geography hint"),
            "supplier_technology_hint": st.column_config.TextColumn("Supplier technology"),
            "interpretation_reason": st.column_config.TextColumn("Why", width="large"),
            "source_document": st.column_config.TextColumn("Source", disabled=True),
            "evidence_text": st.column_config.TextColumn("Evidence", width="large"),
        },
        key="inventory_editor",
    )
    st.session_state["inventory_df"] = edited_df

    searchable = searchable_exchanges(edited_df)
    st.caption(
        f"{len(searchable)} exchange(s) currently eligible for ecoinvent search; "
        f"{len(edited_df) - len(searchable)} row(s) remain foreground context/parameters/emissions/outputs or are excluded."
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
    if st.button("4. Retrieve and rank ecoinvent candidates", disabled=not can_search):
        default_locations = [x.strip() for x in locations_text.split(",") if x.strip()]
        candidate_map: dict[int, list[dict]] = {}
        progress = st.progress(0.0)
        searchable = searchable_exchanges(edited_df).reset_index(drop=True)

        if searchable.empty:
            st.warning("No approved technosphere search concepts are available for ecoinvent retrieval.")
        else:
            for n, (_, row) in enumerate(searchable.iterrows(), start=1):
                flow_id = int(row.get("flow_id", n - 1))
                query = str(row.get("ecoinvent_search_term", "")).strip()
                row_locations = []
                geography_hint = str(row.get("geography_hint") or "").strip()
                if geography_hint:
                    row_locations.append(geography_hint)
                row_locations.extend(default_locations)
                row_locations = list(dict.fromkeys(row_locations))

                try:
                    candidate_map[flow_id] = search_candidates(
                        project_name=project_name,
                        database_name=database_name,
                        query=query,
                        locations=row_locations,
                        limit=candidate_limit,
                        unit=str(row.get("unit") or "").strip() or None,
                        activity_type_hint=str(row.get("ecoinvent_activity_type_hint") or "").strip() or None,
                        technology_hint=str(row.get("supplier_technology_hint") or "").strip() or None,
                    )
                except Exception as exc:
                    candidate_map[flow_id] = [{"error": str(exc)}]
                progress.progress(n / max(len(searchable), 1))

        st.session_state["candidates"] = candidate_map

if "candidates" in st.session_state and "inventory_df" in st.session_state:
    st.subheader("5. Review real ecoinvent candidates")
    st.caption(
        "All candidates below come from your selected Brightway database. Match score is deterministic triage, not AI approval."
    )
    mapping_rows = []
    inv_df = st.session_state["inventory_df"]

    for flow_id, candidates in st.session_state["candidates"].items():
        matching = inv_df[inv_df["flow_id"] == flow_id]
        if matching.empty:
            continue
        row = matching.iloc[0]
        flow_name = str(row.get("name") or f"Flow {flow_id}")
        parent = str(row.get("parent_process") or "").strip()
        heading = f"{parent} → {flow_name}" if parent else flow_name
        st.markdown(f"### {heading}")
        st.caption(
            f"AI search concept: {row.get('ecoinvent_search_term') or flow_name}"
            + (f" | geography: {row.get('geography_hint')}" if row.get("geography_hint") else "")
            + (f" | activity hint: {row.get('ecoinvent_activity_type_hint')}" if row.get("ecoinvent_activity_type_hint") else "")
        )

        manual_query = st.text_input(
            "Manual Brightway search",
            value=str(row.get("ecoinvent_search_term") or flow_name),
            key=f"manual_query_{flow_id}",
        )
        if st.button("Run manual search", key=f"manual_search_{flow_id}"):
            try:
                preferred_locations = [
                    x for x in [str(row.get("geography_hint") or "").strip()]
                    if x
                ] + [x.strip() for x in locations_text.split(",") if x.strip()]
                st.session_state["candidates"][flow_id] = search_candidates(
                    project_name=project_name,
                    database_name=database_name,
                    query=manual_query,
                    locations=preferred_locations,
                    limit=candidate_limit,
                    unit=str(row.get("unit") or "").strip() or None,
                )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        if not candidates:
            st.warning("No candidates returned.")
            continue
        if "error" in candidates[0]:
            st.error(candidates[0]["error"])
            continue

        display = pd.DataFrame(candidates)
        display.insert(0, "rank", range(1, len(display) + 1))
        display_columns = [
            "rank",
            "match_score",
            "name",
            "reference_product",
            "location",
            "unit",
            "match_reasons",
            "database",
            "code",
        ]
        st.dataframe(display[display_columns], width="stretch", hide_index=True)

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
            mapping_rows.append(
                {
                    "flow_id": flow_id,
                    "parent_process": parent,
                    "flow_name": flow_name,
                    "search_concept": row.get("ecoinvent_search_term"),
                    **chosen,
                }
            )

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
