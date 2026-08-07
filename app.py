from __future__ import annotations

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from document_reader import extract_document_text
from foreground_pipeline import extract_inventory_from_sources
from export_helpers import extraction_to_dataframe, dataframe_to_json, searchable_exchanges
from ecoinvent_search import list_databases, search_candidates

load_dotenv()

ITEM_TYPES = ["technosphere_flow", "biosphere_flow", "parameter", "reference_product"]
ACTIVITY_TYPES = ["market", "transforming", "treatment", "transport", "service", "construction", "operation", "unknown"]


def _suggested_study_location(extraction) -> str:
    """Return one supported study-level ecoinvent geography, otherwise leave it for review."""
    locations = [
        context.ecoinvent_location_hint.strip()
        for context in extraction.operating_contexts
        if context.geography_basis != "not_specified"
        and context.ecoinvent_location_hint
        and context.ecoinvent_location_hint.strip()
    ]
    unique = list(dict.fromkeys(locations))
    return unique[0] if len(unique) == 1 else ""


st.set_page_config(page_title="AI-LCA Foreground Builder", layout="wide")
st.title("AI-LCA Foreground Builder")
st.caption("AI reads and interprets → deterministic validation → human approves → Brightway retrieves")

with st.sidebar:
    st.header("Configuration")
    model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    project_name = st.text_input("Brightway project", value=os.getenv("BRIGHTWAY_PROJECT", ""))
    locations_text = st.text_input(
        "Additional fallback ecoinvent locations",
        value="GB,RER,GLO,RoW",
        help=(
            "Used after any exchange-specific geography and the reviewed study operating geography. "
            "These are ranking preferences, not hard filters."
        ),
    )
    candidate_limit = st.slider("Candidates per exchange", 3, 20, 8)

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
        status_rows = []
        for uploaded_file in uploaded_files:
            try:
                text = extract_document_text(uploaded_file.getvalue(), uploaded_file.name)
                extracted_documents.append((uploaded_file.name, text))
                status_rows.append({"document": uploaded_file.name, "characters": len(text), "status": "ready"})
            except Exception as exc:
                status_rows.append({"document": uploaded_file.name, "characters": 0, "status": f"error: {exc}"})
        st.dataframe(pd.DataFrame(status_rows), hide_index=True, width="stretch")
        for source_name, preview in extracted_documents:
            with st.expander(f"Preview — {source_name}"):
                st.text(preview[:12000])

with paste_tab:
    pasted_text = st.text_area("Paste additional technical text or notes", height=260)

extra_instructions = st.text_area(
    "Optional study instructions",
    placeholder="Example: Model cradle-to-gate hydrogen production in GB; keep capture and compression distinct only where supported.",
    height=90,
)

sources = list(extracted_documents)
if pasted_text.strip():
    sources.append(("pasted_text", pasted_text.strip()))

if sources:
    st.info(
        f"Ready to analyse {len(sources)} source(s). Each source uses two AI passes: "
        "process understanding first, selective foreground extraction second."
    )

if st.button("2. Read documents and propose foreground", type="primary", disabled=not bool(sources)):
    try:
        with st.spinner("Understanding the technology, operating context, then extracting ecoinvent-linkable exchanges…"):
            extraction = extract_inventory_from_sources(sources, model=model, extra_instructions=extra_instructions)
        st.session_state["extraction"] = extraction
        st.session_state["inventory_df"] = extraction_to_dataframe(extraction)
        st.session_state["study_location_preference"] = _suggested_study_location(extraction)
        st.session_state.pop("candidates", None)
    except Exception as exc:
        st.exception(exc)

if "extraction" in st.session_state:
    extraction = st.session_state["extraction"]
    st.subheader("3. Review AI foreground interpretation")

    c1, c2, c3 = st.columns(3)
    c1.write(f"**Technology:** {extraction.technology_name or 'Needs review'}")
    c2.write(f"**Top-level process:** {extraction.process_name or 'Needs review'}")
    c3.write(f"**Functional unit:** {extraction.functional_unit or 'Needs review'}")
    if extraction.system_description:
        st.write(extraction.system_description)

    st.markdown("#### Intended operating context")
    st.caption(
        "This is the proposed location/setting of the foreground system itself. It is kept separate from "
        "input provenance such as Norwegian natural gas or imported electricity."
    )

    supported_contexts = []
    if extraction.operating_contexts:
        context_rows = []
        for context in extraction.operating_contexts:
            context_rows.append(
                {
                    "intended geography": context.intended_geography,
                    "ecoinvent location hint": context.ecoinvent_location_hint,
                    "basis": context.geography_basis,
                    "operating setting": context.operating_setting,
                    "temporal context": context.temporal_context,
                    "source": context.source_document,
                    "evidence": context.evidence_text,
                    "note": context.note,
                }
            )
            if context.geography_basis != "not_specified" and context.ecoinvent_location_hint:
                supported_contexts.append(context)

        context_df = pd.DataFrame(context_rows)
        st.dataframe(context_df, hide_index=True, width="stretch")

        if supported_contexts:
            unique_hints = list(dict.fromkeys(c.ecoinvent_location_hint for c in supported_contexts if c.ecoinvent_location_hint))
            if len(unique_hints) == 1:
                chosen_context = supported_contexts[0]
                st.success(
                    f"AI operating-geography proposal: {chosen_context.intended_geography or unique_hints[0]} "
                    f"→ ecoinvent preference {unique_hints[0]} ({chosen_context.geography_basis.replace('_', ' ')})."
                )
            else:
                st.warning(
                    "The sources imply different operating geographies. Review them and choose the study geography manually below."
                )
        else:
            st.warning(
                "The source does not establish a sufficiently supported operating geography. "
                "Choose the study geography manually if your LCA requires one."
            )
    else:
        st.warning("No operating-context proposal is available; choose the study geography manually if needed.")

    study_location_preference = st.text_input(
        "Study geography preference for ecoinvent matching",
        key="study_location_preference",
        help=(
            "Review/edit the proposed ecoinvent geography for where the foreground system operates. "
            "This becomes the first geography preference for exchanges that do not have their own more specific geography. "
            "It never hard-filters Brightway results."
        ),
        placeholder="e.g. GB, DE, RER, GLO — leave blank if genuinely unspecified",
    ).strip()

    process_names: list[str] = []
    if extraction.foreground_processes:
        st.markdown("#### Proposed foreground process context")
        process_df = pd.DataFrame([
            {"process": p.name, "role": p.role, "source": p.source_document, "evidence": p.evidence_text}
            for p in extraction.foreground_processes
        ])
        st.dataframe(process_df, hide_index=True, width="stretch")
        process_names = list(dict.fromkeys(process_df["process"].dropna().astype(str).tolist()))

    if extraction.assumptions_or_warnings:
        with st.expander("Interpretation notes / warnings"):
            for warning in extraction.assumptions_or_warnings:
                st.write(f"• {warning}")

    current_df = st.session_state["inventory_df"]
    existing_parents = current_df["parent_process"].dropna().astype(str).tolist()
    parent_options = [""] + list(dict.fromkeys(process_names + existing_parents))

    st.markdown("#### Proposed exchanges and parameters")
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
            "item_type": st.column_config.SelectboxColumn("Type", options=ITEM_TYPES, required=True),
            "parent_process": st.column_config.SelectboxColumn("Parent process", options=parent_options),
            "search_worthy": st.column_config.CheckboxColumn("Search ecoinvent?"),
            "ecoinvent_search_term": st.column_config.TextColumn("Search concept", width="medium"),
            "ecoinvent_activity_type_hint": st.column_config.SelectboxColumn("Activity hint", options=ACTIVITY_TYPES),
            "geography_hint": st.column_config.TextColumn("Exchange geography"),
            "supplier_technology_hint": st.column_config.TextColumn("Supplier technology"),
            "interpretation_reason": st.column_config.TextColumn("Why", width="large"),
            "source_document": st.column_config.TextColumn("Source", disabled=True),
            "evidence_text": st.column_config.TextColumn("Evidence", width="large"),
        },
        key="inventory_editor",
    )
    st.session_state["inventory_df"] = edited_df
    searchable = searchable_exchanges(edited_df)
    st.caption(f"{len(searchable)} exchange(s) currently eligible for ecoinvent search.")

    left, right = st.columns(2)
    with left:
        st.download_button("Download reviewed inventory CSV", edited_df.to_csv(index=False).encode("utf-8"), "reviewed_foreground_inventory.csv", "text/csv")
    with right:
        st.download_button("Download reviewed inventory JSON", dataframe_to_json(edited_df), "reviewed_foreground_inventory.json", "application/json")

    if st.button("4. Retrieve and rank ecoinvent candidates", disabled=not bool(project_name and database_name)):
        default_locations = [x.strip() for x in locations_text.split(",") if x.strip()]
        candidate_map: dict[int, list[dict]] = {}
        searchable = searchable_exchanges(edited_df).reset_index(drop=True)
        progress = st.progress(0.0)
        for n, (_, row) in enumerate(searchable.iterrows(), start=1):
            flow_id = int(row.get("flow_id", n - 1))
            exchange_geography = str(row.get("geography_hint") or "").strip()
            locations = list(
                dict.fromkeys(
                    ([exchange_geography] if exchange_geography else [])
                    + ([study_location_preference] if study_location_preference else [])
                    + default_locations
                )
            )
            try:
                candidate_map[flow_id] = search_candidates(
                    project_name=project_name,
                    database_name=database_name,
                    query=str(row.get("ecoinvent_search_term") or "").strip(),
                    locations=locations,
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
    inv_df = st.session_state["inventory_df"]
    mapping_rows = []
    study_location_preference = str(st.session_state.get("study_location_preference") or "").strip()

    for flow_id, candidates in st.session_state["candidates"].items():
        matching = inv_df[inv_df["flow_id"] == flow_id]
        if matching.empty:
            continue
        row = matching.iloc[0]
        flow_name = str(row.get("name") or f"Flow {flow_id}")
        parent = str(row.get("parent_process") or "").strip()
        st.markdown(f"### {parent + ' → ' if parent else ''}{flow_name}")

        manual_query = st.text_input("Manual Brightway search", value=str(row.get("ecoinvent_search_term") or flow_name), key=f"manual_query_{flow_id}")
        if st.button("Run manual search", key=f"manual_search_{flow_id}"):
            exchange_geography = str(row.get("geography_hint") or "").strip()
            fallback_locations = [x.strip() for x in locations_text.split(",") if x.strip()]
            locations = list(
                dict.fromkeys(
                    ([exchange_geography] if exchange_geography else [])
                    + ([study_location_preference] if study_location_preference else [])
                    + fallback_locations
                )
            )
            st.session_state["candidates"][flow_id] = search_candidates(
                project_name=project_name,
                database_name=database_name,
                query=manual_query,
                locations=locations,
                limit=candidate_limit,
                unit=str(row.get("unit") or "").strip() or None,
            )
            st.rerun()

        if not candidates:
            st.warning("No candidates returned.")
            continue
        if "error" in candidates[0]:
            st.error(candidates[0]["error"])
            continue

        display = pd.DataFrame(candidates)
        display.insert(0, "rank", range(1, len(display) + 1))
        st.dataframe(display[["rank", "match_score", "name", "reference_product", "location", "unit", "match_reasons", "database", "code"]], width="stretch", hide_index=True)

        labels = [f"{c['name']} | {c['reference_product']} | {c['location']} | {c['unit']}" for c in candidates]
        choice = st.selectbox("Approve mapping", ["— no selection —"] + labels, key=f"mapping_{flow_id}")
        if choice != "— no selection —":
            chosen = candidates[labels.index(choice)]
            mapping_rows.append({"flow_id": flow_id, "parent_process": parent, "flow_name": flow_name, "search_concept": row.get("ecoinvent_search_term"), "study_geography": study_location_preference, **chosen})

    if mapping_rows:
        mapping_df = pd.DataFrame(mapping_rows)
        st.subheader("Approved mappings")
        st.dataframe(mapping_df, width="stretch", hide_index=True)
        st.download_button("Download approved mappings CSV", mapping_df.to_csv(index=False).encode("utf-8"), "approved_ecoinvent_mappings.csv", "text/csv")
