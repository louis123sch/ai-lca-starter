from __future__ import annotations

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ai_lca.documents import combine_document_texts, extract_docx_text, extract_pdf_text
from ai_lca.llm import extract_inventory_from_text, extract_process_map_from_text
from ai_lca.export import dataframe_to_json, extraction_to_dataframe, process_map_to_dataframe
from ai_lca.brightway_search import list_databases, search_candidates

load_dotenv()

st.set_page_config(page_title="AI-LCA Foreground Builder", layout="wide")
st.title("AI-LCA Foreground Builder")
st.caption("AI maps the evidence corpus → human confirms foreground processes → AI extracts exchanges → human approves → Brightway calculates")


def process_geography(process_map, process_id: str) -> str | None:
    """Return process geography, falling back to the corpus-wide study geography."""
    for group in process_map.technology_groups:
        for process in group.processes:
            if process.process_id == process_id:
                return process.geographic_context or process_map.geographic_context
    return process_map.geographic_context


def evidence_location_text(evidence) -> str:
    bits = []
    if evidence.source_document:
        bits.append(evidence.source_document)
    if evidence.page is not None:
        bits.append(f"p. {evidence.page}")
    if evidence.table:
        bits.append(evidence.table)
    if evidence.section:
        bits.append(evidence.section)
    return " · ".join(bits) or "Source location not identified"


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

paste_tab, upload_tab = st.tabs(["Paste text", "Upload documents"])

with paste_tab:
    pasted_text = st.text_area(
        "Paste additional evidence, technical documentation, datasheet text, or engineering notes",
        height=320,
        placeholder="Optional: pasted text is treated as another source in the same evidence corpus…",
    )

with upload_tab:
    uploaded_documents = st.file_uploader(
        "Upload PDFs and Word documents",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload the paper, supplementary information, appendices, reports, or other supporting sources together. All readable files are treated as one evidence corpus.",
    )

    extracted_documents: list[tuple[str, str]] = []
    if uploaded_documents:
        for uploaded_document in uploaded_documents:
            try:
                filename = uploaded_document.name
                lower_name = filename.lower()
                document_bytes = uploaded_document.getvalue()
                if lower_name.endswith(".pdf"):
                    extracted_text = extract_pdf_text(document_bytes)
                    document_kind = "PDF"
                elif lower_name.endswith(".docx"):
                    extracted_text = extract_docx_text(document_bytes)
                    document_kind = "Word"
                else:
                    raise ValueError("Unsupported document type. Upload PDF or .docx.")

                extracted_documents.append((filename, extracted_text))
                st.success(
                    f"{filename}: extracted {document_kind} content ({len(extracted_text):,} characters)."
                )
                with st.expander(f"Preview: {filename}"):
                    st.text(extracted_text[:12000])
            except Exception as exc:
                st.error(f"{uploaded_document.name}: {exc}")

extra_instructions = st.text_area(
    "Optional study instructions",
    placeholder="Example: Focus on cradle-to-gate foreground inputs for 1 kg H2; keep infrastructure and operation separate.",
    height=90,
)

corpus_documents = list(extracted_documents)
if pasted_text.strip():
    corpus_documents.append(("pasted-text", pasted_text.strip()))
source_text = combine_document_texts(corpus_documents)

if corpus_documents:
    st.caption(
        f"Evidence corpus: {len(corpus_documents)} source(s). Processes and flows will be inferred from the combined evidence, not document-by-document."
    )

if st.button("1. Analyse evidence corpus", type="primary", disabled=not bool(source_text)):
    try:
        with st.spinner("Building one process map from all uploaded evidence…"):
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
        st.session_state.pop("candidate_geographies", None)
        st.session_state.pop("candidate_queries", None)
    except Exception as exc:
        st.exception(exc)

if "process_map" in st.session_state:
    process_map = st.session_state["process_map"]
    st.subheader("Review AI foreground interpretation")
    st.caption(
        "The process map is synthesized across the complete evidence corpus. Evidence from several files can jointly support one process. Repeated descriptions are merged; engineering operations remain context only and do not become extra Brightway activities."
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
                elif process_map.geographic_context:
                    context_bits.append(f"geography: {process_map.geographic_context} (corpus-wide)")
                if process.temporal_context:
                    context_bits.append(f"time: {process.temporal_context}")
                elif process_map.temporal_context:
                    context_bits.append(f"time: {process_map.temporal_context} (corpus-wide)")
                if context_bits:
                    st.caption(" | ".join(context_bits))
                st.write(process.reason_for_separate_process)

                if process.evidence:
                    st.markdown("**Supporting evidence:**")
                    for evidence in process.evidence:
                        st.caption(evidence_location_text(evidence))
                        st.code(evidence.evidence_text, language=None)

                if process.operations:
                    st.markdown("**Operations described inside this process — not separate foreground processes:**")
                    for operation in process.operations:
                        sources = sorted(
                            {
                                evidence.source_document
                                for evidence in operation.evidence
                                if evidence.source_document
                            }
                        )
                        source_suffix = f" — {', '.join(sources)}" if sources else ""
                        st.write(f"• {operation.name}{source_suffix}")

    process_df = st.session_state["process_map_df"]
    if process_df.empty:
        st.warning(
            "No evidence-backed foreground processes were identified. Nothing will be sent to ecoinvent matching; inspect the evidence corpus or adjust the study instructions before continuing."
        )
        selected_process_ids = []
    else:
        process_df = st.data_editor(
            process_df,
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
                "source_documents",
                "pages",
                "tables",
                "evidence_text",
            ],
            column_config={
                "include": st.column_config.CheckboxColumn("Use as foreground process"),
                "process_id": st.column_config.TextColumn("Process ID", width="small"),
                "process_name": st.column_config.TextColumn("Foreground process", width="large"),
                "technology_group": st.column_config.TextColumn("Technology / pathway"),
                "geographic_context": st.column_config.TextColumn("Evidence-derived geography"),
                "source_documents": st.column_config.TextColumn("Supporting documents", width="large"),
                "operations_not_separate_processes": st.column_config.TextColumn("Operations (context only)", width="large"),
                "reason_for_separate_process": st.column_config.TextColumn("Why this is a separate process", width="large"),
                "evidence_text": st.column_config.TextColumn("Process evidence", width="large"),
            },
            key="process_map_editor",
        )
        st.session_state["process_map_df"] = process_df
        selected_process_ids = process_df.loc[
            process_df["include"] == True, "process_id"
        ].astype(str).tolist()  # noqa: E712

    if st.button("2. Extract inventory for selected processes", disabled=not bool(selected_process_ids)):
        try:
            with st.spinner("Building process inventories from all relevant evidence across the corpus…"):
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
            st.session_state.pop("candidate_geographies", None)
            st.session_state.pop("candidate_queries", None)
        except Exception as exc:
            st.exception(exc)

if "extraction" in st.session_state:
    extraction = st.session_state["extraction"]
    st.subheader("Proposed foreground inventory")
    st.caption(
        "Each flow is tied to a confirmed process and may combine supporting evidence from several documents. Only quantified inputs default to background matching; outputs, emissions and unquantified mentions remain visible but are not searched automatically."
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
            "background_match_eligible": st.column_config.CheckboxColumn("Eligible"),
            "flow_id": st.column_config.NumberColumn("Flow ID", disabled=True),
            "process_id": st.column_config.TextColumn("Process ID", disabled=True),
            "process_name": st.column_config.TextColumn("Foreground process", width="large"),
            "technology_group": st.column_config.TextColumn("Technology / pathway"),
            "source_documents": st.column_config.TextColumn("Supporting documents", width="large"),
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
        candidate_map: dict[int, list[dict]] = {}
        candidate_geographies: dict[int, str | None] = {}
        candidate_queries: dict[int, str] = {}
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

        process_map = st.session_state.get("process_map")
        for n, (_, row) in enumerate(included.iterrows(), start=1):
            flow_id = int(row.get("flow_id", n - 1))
            query = str(row.get("name", "")).strip()
            process_id = str(row.get("process_id", "")).strip()
            geography = process_geography(process_map, process_id) if process_map else None
            candidate_geographies[flow_id] = geography
            candidate_queries[flow_id] = query
            try:
                candidate_map[flow_id] = search_candidates(
                    project_name=project_name,
                    database_name=database_name,
                    query=query,
                    location_hint=geography,
                    limit=candidate_limit,
                )
            except Exception as exc:
                candidate_map[flow_id] = [{"error": str(exc)}]
            progress.progress(n / max(len(included), 1))
        st.session_state["candidates"] = candidate_map
        st.session_state["candidate_geographies"] = candidate_geographies
        st.session_state["candidate_queries"] = candidate_queries

if "candidates" in st.session_state and "inventory_df" in st.session_state:
    st.subheader("Candidate background processes")
    st.caption(
        "The extracted flow name is only the starting search query. Edit the query for any flow and search again without changing the evidence-backed foreground inventory. Evidence-derived geography remains a soft ranking signal."
    )
    mapping_rows = []
    inv_df = st.session_state["inventory_df"]
    candidate_geographies = st.session_state.get("candidate_geographies", {})
    candidate_queries = st.session_state.setdefault("candidate_queries", {})

    for flow_id, candidates in list(st.session_state["candidates"].items()):
        matching = inv_df[inv_df["flow_id"] == flow_id]
        flow_name = matching.iloc[0]["name"] if not matching.empty else f"Flow {flow_id}"
        process_name = matching.iloc[0]["process_name"] if not matching.empty else "Unknown process"
        geography = candidate_geographies.get(flow_id)
        st.markdown(f"### {flow_name}")
        context = f"Foreground process: {process_name}"
        if geography:
            context += f" | Evidence-derived geography: {geography}"
        else:
            context += " | Evidence-derived geography: not identified"
        st.caption(context)

        query_col, search_col = st.columns([5, 1])
        with query_col:
            edited_query = st.text_input(
                "Ecoinvent search query",
                value=candidate_queries.get(flow_id, flow_name),
                key=f"search_query_{flow_id}",
                help="This changes only the Brightway/ecoinvent search. It does not alter the extracted LCI flow name or evidence.",
            )
        with search_col:
            st.write("")
            st.write("")
            search_again = st.button(
                "Search again",
                key=f"search_again_{flow_id}",
                use_container_width=True,
                disabled=not bool(project_name and database_name and edited_query.strip()),
            )

        if search_again:
            candidate_queries[flow_id] = edited_query.strip()
            try:
                st.session_state["candidates"][flow_id] = search_candidates(
                    project_name=project_name,
                    database_name=database_name,
                    query=edited_query.strip(),
                    location_hint=geography,
                    limit=candidate_limit,
                )
            except Exception as exc:
                st.session_state["candidates"][flow_id] = [{"error": str(exc)}]
            st.session_state["candidate_queries"] = candidate_queries
            st.session_state.pop(f"mapping_{flow_id}", None)
            st.rerun()

        current_query = candidate_queries.get(flow_id, flow_name)

        if not candidates:
            st.warning("No candidates returned. Edit the search query above and search again.")
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
            mapping_rows.append(
                {
                    "flow_id": flow_id,
                    "flow_name": flow_name,
                    "search_query": current_query,
                    "process_name": process_name,
                    "evidence_geography": geography,
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