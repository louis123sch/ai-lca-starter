from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ai_lca.brightway_search import list_databases, search_candidates
from ai_lca.documents import combine_document_texts, extract_docx_text, extract_pdf_text
from ai_lca.export import dataframe_to_json, extraction_to_dataframe, process_map_to_dataframe
from ai_lca.llm import extract_inventory_from_text, extract_process_map_from_text

load_dotenv()

st.set_page_config(page_title="AI-LCA Foreground Builder", layout="wide")
st.title("AI-LCA Foreground Builder")
st.caption(
    "AI maps the evidence corpus → human confirms foreground processes → "
    "AI extracts exchanges → human approves background mappings → Brightway calculates"
)

ACTIVITY_TYPE_OPTIONS = [
    "market",
    "transforming",
    "treatment",
    "transport",
    "construction",
    "operation",
    "unknown",
]


def _text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


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


def suggested_activity_type(row) -> str:
    """Return a soft initial ecoinvent activity-type preference for an exchange."""
    explicit_hint = _text(row.get("ecoinvent_activity_hint"))
    lower_hint = explicit_hint.lower()
    if lower_hint.startswith("market for ") or lower_hint.startswith("market group for "):
        return "market"
    if lower_hint.startswith("treatment of "):
        return "treatment"

    flow_kind = _text(row.get("flow_kind")).lower()
    direction = _text(row.get("direction")).lower()
    if flow_kind == "transport":
        return "transport"
    if flow_kind == "waste":
        return "treatment"
    if direction == "input" and flow_kind in {"material", "energy", "water"}:
        return "market"
    return "unknown"


def initial_search_query(row) -> str:
    """Prefer explicit source-provided background mapping, then clean search concept."""
    return (
        _text(row.get("ecoinvent_activity_hint"))
        or _text(row.get("ecoinvent_search_term"))
        or _text(row.get("name"))
    )


def search_geography(row, process_map) -> tuple[str | None, str]:
    """Prefer an explicit source-provided exchange location over process geography."""
    explicit = _text(row.get("ecoinvent_location_hint"))
    if explicit:
        return explicit, "source-provided background mapping"
    process_id = _text(row.get("process_id"))
    geography = process_geography(process_map, process_id) if process_map else None
    return geography, "foreground operating context"


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
                default_index = next((i for i, name in enumerate(dbs) if "ecoinvent" in name.lower()), 0)
                database_name = st.selectbox("Background database", dbs, index=default_index)
            else:
                st.warning("No databases found in this Brightway project.")
        except Exception as exc:
            st.warning(f"Brightway project not available yet: {exc}")


st.subheader("1. Add evidence")
paste_tab, upload_tab = st.tabs(["Paste text", "Upload documents"])

with paste_tab:
    pasted_text = st.text_area(
        "Paste additional evidence, technical documentation, datasheet text, or engineering notes",
        height=280,
        placeholder="Optional: pasted text is treated as another source in the same evidence corpus…",
    )

with upload_tab:
    uploaded_documents = st.file_uploader(
        "Upload PDFs and Word documents",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help=(
            "Upload papers, supplementary information, appendices and reports together. "
            "All readable files are treated as one evidence corpus."
        ),
    )

    extracted_documents: list[tuple[str, str]] = []
    if uploaded_documents:
        status_rows = []
        for uploaded_document in uploaded_documents:
            try:
                filename = uploaded_document.name
                data = uploaded_document.getvalue()
                if filename.lower().endswith(".pdf"):
                    extracted_text = extract_pdf_text(data)
                    kind = "PDF"
                elif filename.lower().endswith(".docx"):
                    extracted_text = extract_docx_text(data)
                    kind = "Word"
                else:
                    raise ValueError("Unsupported document type. Upload PDF or .docx.")

                extracted_documents.append((filename, extracted_text))
                status_rows.append(
                    {"document": filename, "type": kind, "characters": len(extracted_text), "status": "ready"}
                )
                with st.expander(f"Preview — {filename}"):
                    st.text(extracted_text[:12000])
            except Exception as exc:
                status_rows.append(
                    {"document": uploaded_document.name, "type": "", "characters": 0, "status": f"error: {exc}"}
                )

        if status_rows:
            st.dataframe(pd.DataFrame(status_rows), hide_index=True, width="stretch")

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
    st.info(
        f"Evidence corpus: {len(corpus_documents)} source(s). Processes and flows are built from the combined evidence."
    )

if st.button("2. Analyse evidence corpus", type="primary", disabled=not bool(source_text)):
    try:
        with st.spinner("Building one process map from all uploaded evidence…"):
            process_map = extract_process_map_from_text(
                source_text,
                model=model,
                extra_instructions=extra_instructions,
            )
        st.session_state["process_map"] = process_map
        st.session_state["process_map_df"] = process_map_to_dataframe(process_map)
        for key in (
            "extraction",
            "inventory_df",
            "candidates",
            "candidate_geographies",
            "candidate_geography_sources",
            "candidate_queries",
            "candidate_activity_types",
        ):
            st.session_state.pop(key, None)
    except Exception as exc:
        st.exception(exc)


if "process_map" in st.session_state:
    process_map = st.session_state["process_map"]
    st.subheader("3. Review AI foreground interpretation")
    st.caption(
        "Processes are synthesized across the complete evidence corpus. Engineering operations remain context only; "
        "they do not become extra foreground activities."
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
                    st.markdown("**Operations inside this process — not separate foreground processes:**")
                    for operation in process.operations:
                        sources = sorted(
                            {
                                evidence.source_document
                                for evidence in operation.evidence
                                if evidence.source_document
                            }
                        )
                        suffix = f" — {', '.join(sources)}" if sources else ""
                        st.write(f"• {operation.name}{suffix}")

    process_df = st.session_state["process_map_df"]
    if process_df.empty:
        st.warning("No evidence-backed foreground processes were identified.")
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
                "process_name": st.column_config.TextColumn("Foreground process", width="large"),
                "technology_group": st.column_config.TextColumn("Technology / pathway"),
                "geographic_context": st.column_config.TextColumn("Evidence-derived geography"),
                "source_documents": st.column_config.TextColumn("Supporting documents", width="large"),
                "operations_not_separate_processes": st.column_config.TextColumn("Operations (context only)", width="large"),
                "reason_for_separate_process": st.column_config.TextColumn("Why separate", width="large"),
                "evidence_text": st.column_config.TextColumn("Process evidence", width="large"),
            },
            key="process_map_editor",
        )
        st.session_state["process_map_df"] = process_df
        selected_process_ids = process_df.loc[
            process_df["include"] == True, "process_id"  # noqa: E712
        ].astype(str).tolist()

    if st.button("4. Extract inventory for selected processes", disabled=not bool(selected_process_ids)):
        try:
            with st.spinner("Building foreground inventories from all relevant evidence…"):
                extraction = extract_inventory_from_text(
                    source_text,
                    process_map=process_map,
                    approved_process_ids=selected_process_ids,
                    model=model,
                    extra_instructions=extra_instructions,
                )
            st.session_state["extraction"] = extraction
            st.session_state["inventory_df"] = extraction_to_dataframe(extraction)
            for key in (
                "candidates",
                "candidate_geographies",
                "candidate_geography_sources",
                "candidate_queries",
                "candidate_activity_types",
            ):
                st.session_state.pop(key, None)
        except Exception as exc:
            st.exception(exc)


if "extraction" in st.session_state:
    extraction = st.session_state["extraction"]
    st.subheader("5. Review foreground inventory")
    st.caption(
        "Exchange names are bare LCI concepts. Lifecycle context such as plant construction is stored separately and is not part of the ecoinvent search term. "
        "If an uploaded source explicitly specifies a database activity, that is shown as a separate source-provided mapping hint."
    )

    if extraction.assumptions_or_warnings:
        with st.expander("Inventory warnings", expanded=False):
            for warning in extraction.assumptions_or_warnings:
                st.write(f"• {warning}")

    inventory_df = st.session_state["inventory_df"]
    edited_df = st.data_editor(
        inventory_df,
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        disabled=[
            "background_match_eligible",
            "flow_id",
            "process_id",
            "source_documents",
            "pages",
            "tables",
            "evidence_text",
            "ecoinvent_activity_hint",
            "ecoinvent_location_hint",
            "mapping_source_documents",
            "mapping_pages",
            "mapping_tables",
            "mapping_evidence_text",
        ],
        column_config={
            "include": st.column_config.CheckboxColumn("Search ecoinvent"),
            "background_match_eligible": st.column_config.CheckboxColumn("Eligible"),
            "flow_id": st.column_config.NumberColumn("ID", disabled=True),
            "name": st.column_config.TextColumn("Exchange", width="medium"),
            "source_label": st.column_config.TextColumn("Source wording"),
            "component_or_stage": st.column_config.TextColumn("Stage / component", width="medium"),
            "ecoinvent_search_term": st.column_config.TextColumn(
                "Clean search concept",
                width="medium",
                help="Editable retrieval concept; stage labels such as plant construction should not appear here.",
            ),
            "ecoinvent_activity_hint": st.column_config.TextColumn("Source-provided activity hint", width="large"),
            "ecoinvent_location_hint": st.column_config.TextColumn("Source-provided activity location"),
            "source_documents": st.column_config.TextColumn("Quantity evidence documents", width="large"),
            "mapping_source_documents": st.column_config.TextColumn("Mapping evidence documents", width="large"),
            "evidence_text": st.column_config.TextColumn("Quantity evidence", width="large"),
            "mapping_evidence_text": st.column_config.TextColumn("Mapping evidence", width="large"),
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

    searchable = edited_df[
        (edited_df["include"] == True)  # noqa: E712
        & (edited_df["background_match_eligible"] == True)  # noqa: E712
    ]
    st.caption(f"{len(searchable)} quantified input exchange(s) selected for ecoinvent matching.")

    can_search = bool(project_name and database_name and not searchable.empty)
    if st.button("6. Retrieve and rank ecoinvent candidates", disabled=not can_search):
        candidate_map: dict[int, list[dict]] = {}
        candidate_geographies: dict[int, str | None] = {}
        candidate_geography_sources: dict[int, str] = {}
        candidate_queries: dict[int, str] = {}
        candidate_activity_types: dict[int, str] = {}
        progress = st.progress(0.0)

        process_map = st.session_state.get("process_map")
        searchable = searchable.reset_index(drop=True)
        for n, (_, row) in enumerate(searchable.iterrows(), start=1):
            flow_id = int(row.get("flow_id", n - 1))
            query = initial_search_query(row)
            geography, geography_source = search_geography(row, process_map)
            activity_type = suggested_activity_type(row)
            unit = _text(row.get("unit")) or None

            candidate_geographies[flow_id] = geography
            candidate_geography_sources[flow_id] = geography_source
            candidate_queries[flow_id] = query
            candidate_activity_types[flow_id] = activity_type

            try:
                candidate_map[flow_id] = search_candidates(
                    project_name=project_name,
                    database_name=database_name,
                    query=query,
                    location_hint=geography,
                    limit=candidate_limit,
                    unit=unit,
                    activity_type_hint=activity_type,
                )
            except Exception as exc:
                candidate_map[flow_id] = [{"error": str(exc)}]
            progress.progress(n / max(len(searchable), 1))

        st.session_state["candidates"] = candidate_map
        st.session_state["candidate_geographies"] = candidate_geographies
        st.session_state["candidate_geography_sources"] = candidate_geography_sources
        st.session_state["candidate_queries"] = candidate_queries
        st.session_state["candidate_activity_types"] = candidate_activity_types


if "candidates" in st.session_state and "inventory_df" in st.session_state:
    st.subheader("7. Review real ecoinvent candidates")
    st.caption(
        "Manual search controls change retrieval only. They never rewrite the evidence-backed foreground exchange. "
        "The highest-ranked candidate is pre-selected but remains subject to your approval."
    )

    inv_df = st.session_state["inventory_df"]
    mapping_rows = []
    candidate_geographies = st.session_state.get("candidate_geographies", {})
    candidate_geography_sources = st.session_state.get("candidate_geography_sources", {})
    candidate_queries = st.session_state.setdefault("candidate_queries", {})
    candidate_activity_types = st.session_state.setdefault("candidate_activity_types", {})

    for flow_id, candidates in list(st.session_state["candidates"].items()):
        matching = inv_df[inv_df["flow_id"] == flow_id]
        if matching.empty:
            continue

        row = matching.iloc[0]
        flow_name = _text(row.get("name")) or f"Flow {flow_id}"
        process_name = _text(row.get("process_name")) or "Unknown process"
        stage = _text(row.get("component_or_stage"))
        unit = _text(row.get("unit"))
        explicit_activity_hint = _text(row.get("ecoinvent_activity_hint"))
        explicit_location_hint = _text(row.get("ecoinvent_location_hint"))
        geography = candidate_geographies.get(flow_id)
        geography_source = candidate_geography_sources.get(flow_id, "foreground operating context")

        st.markdown(f"### {flow_name}")
        context_bits = [f"Foreground process: {process_name}"]
        if stage:
            context_bits.append(f"stage: {stage}")
        if unit:
            context_bits.append(f"unit: {unit}")
        if geography:
            context_bits.append(f"ranking geography: {geography} ({geography_source})")
        st.caption(" | ".join(context_bits))

        if explicit_activity_hint:
            mapping_text = f"**Source-provided background hint:** `{explicit_activity_hint}`"
            if explicit_location_hint:
                mapping_text += f" · location `{explicit_location_hint}`"
            st.info(mapping_text)

        current_type = candidate_activity_types.get(flow_id, suggested_activity_type(row))
        if current_type not in ACTIVITY_TYPE_OPTIONS:
            current_type = "unknown"

        query_col, type_col, search_col = st.columns([5, 2, 1.4])
        with query_col:
            edited_query = st.text_input(
                "Ecoinvent search query",
                value=candidate_queries.get(flow_id, initial_search_query(row)),
                key=f"search_query_{flow_id}",
                help="Search-only control. The foreground exchange name and evidence are unchanged.",
            )
        with type_col:
            edited_activity_type = st.selectbox(
                "Activity type",
                ACTIVITY_TYPE_OPTIONS,
                index=ACTIVITY_TYPE_OPTIONS.index(current_type),
                key=f"activity_type_{flow_id}",
                help="Soft ranking/retrieval preference. Market is normally appropriate for purchased background products.",
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
            candidate_activity_types[flow_id] = edited_activity_type
            try:
                st.session_state["candidates"][flow_id] = search_candidates(
                    project_name=project_name,
                    database_name=database_name,
                    query=edited_query.strip(),
                    location_hint=geography,
                    limit=candidate_limit,
                    unit=unit or None,
                    activity_type_hint=edited_activity_type,
                )
            except Exception as exc:
                st.session_state["candidates"][flow_id] = [{"error": str(exc)}]
            st.session_state["candidate_queries"] = candidate_queries
            st.session_state["candidate_activity_types"] = candidate_activity_types
            st.session_state.pop(f"mapping_{flow_id}", None)
            st.rerun()

        current_query = candidate_queries.get(flow_id, initial_search_query(row))
        current_type = candidate_activity_types.get(flow_id, current_type)

        if not candidates:
            st.warning("No candidates returned. Edit the search controls above and search again.")
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
            "activity_type",
            "match_reasons",
        ]
        st.dataframe(
            display[[column for column in display_columns if column in display.columns]],
            width="stretch",
            hide_index=True,
        )

        labels = [
            f"{candidate['name']} | {candidate['reference_product']} | {candidate['location']} | {candidate['unit']}"
            for candidate in candidates
        ]
        no_selection = "— no selection —"
        choice = st.selectbox(
            "Approve mapping",
            labels + [no_selection],
            index=0,
            key=f"mapping_{flow_id}",
            help="Highest-ranked candidate is pre-selected. Change it or choose no selection if needed.",
        )

        if choice != no_selection:
            chosen = candidates[labels.index(choice)]
            mapping_rows.append(
                {
                    "flow_id": flow_id,
                    "flow_name": flow_name,
                    "component_or_stage": stage,
                    "process_name": process_name,
                    "source_activity_hint": explicit_activity_hint,
                    "source_activity_location": explicit_location_hint,
                    "search_query": current_query,
                    "activity_type_preference": current_type,
                    "ranking_geography": geography,
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
