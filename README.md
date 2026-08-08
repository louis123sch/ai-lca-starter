# AI-LCA Starter

A human-in-the-loop prototype for reconstructing foreground LCA models from papers, supplementary information, Word documents, and technical evidence.

> **AI proposes → deterministic validation → human approves → Brightway calculates.**

## Current workflow

1. Upload multiple text-readable PDFs and/or modern Word (`.docx`) documents, optionally with pasted text.
2. Treat all supplied material as **one provenance-tagged evidence corpus**.
3. AI reconstructs the paper-supported foreground process map.
4. Human reviews which proposed foreground processes are genuine modelled processes.
5. AI extracts a complete foreground inventory across all approved processes.
6. Exchanges are classified as:
   - **technosphere** — products/services supplied by background activities;
   - **biosphere** — direct elementary emissions/resources crossing the environment boundary;
   - **production** — reference products and co-products.
7. Human reviews names, quantities, exchange class, stage, evidence, and search concepts.
8. Technosphere exchanges are matched against the selected ecoinvent/background database.
9. Biosphere exchanges are matched against the installed Brightway biosphere database (normally `biosphere3`).
10. Candidate mappings remain human-approved before later Brightway foreground writing/calculation.

## Important modelling behaviour

- Uploaded files contribute jointly to the same foreground processes and exchanges. The paper and supplement are not treated as separate LCAs.
- PDF page markers and Word paragraphs/tables are retained; Word tables remain in their original document order.
- Engineering operations are not automatically turned into separate foreground activities.
- Lifecycle context such as `plant construction` is stored separately from canonical exchange/search names.
- Explicit construction materials/equipment such as concrete, steel, aluminium, cast iron and turbines remain separate exchanges.
- Direct process emissions and direct resource uptake are retained as **biosphere exchanges** instead of being discarded because they are not ecoinvent technosphere inputs.
- LCIA results such as `kg CO2-eq` are **not** elementary-flow emissions. A deterministic guard drops indicator-like rows that the AI might otherwise misread as biosphere flows.
- Purchased water is technosphere; direct environmental water withdrawal is biosphere only when the source supports that interpretation.
- Source-provided technosphere mappings can be represented as **exact**, **proxy**, or **uncertain** without renaming the foreground exchange.
- Cross-document mapping tables are used as evidence. A source-applied gas-turbine dataset can therefore be retained as a labelled proxy for a foreground steam-turbine item while the foreground identity remains `steam turbine`.
- Generic materials do not inherit unsupported specific subtypes. Generic `steel`, for example, stays unresolved when several incompatible steel datasets are plausible.
- Exchange-specific supply provenance is kept separate from foreground operating geography.
- Candidate ranking uses product/activity similarity, unit, evidence-derived geography, activity type, and source-derived supplier/technology hints.
- Search controls are editable without modifying the evidence-backed foreground inventory.
- Automatic approval is conservative: source-backed exact/proxy mappings can be preselected; search-only candidates are preselected only when the match is strong and clearly separated from alternatives. Ambiguous results default to **no selection**.

## Install

Use the Brightway environment you already work in:

```bash
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

Run the app:

```bash
streamlit run app.py
```

## Configuration

Copy `.env.example` to `.env` and provide your API key/project name, or export the environment variables from your shell.

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-mini
BRIGHTWAY_PROJECT=your_existing_brightway_project
```

Do not commit `.env`.

## Review and matching

The inventory table lets the reviewer correct the exchange class before matching. Matching eligibility is recalculated from the reviewed row:

- quantified **technosphere input** → technosphere/eecoinvent matching;
- quantified **biosphere exchange** → biosphere-flow matching;
- production output, unquantified mention, or unresolved item → no automatic background matching.

Technosphere candidate search remains per-flow and editable. Biosphere exchanges have a separate elementary-flow search query and optional compartment hint. Approved mappings can be downloaded as a combined Brightway mapping file plus separate technosphere and biosphere CSVs.

## Deliberately not included yet

- OCR/multimodal interpretation for scanned or image-only PDFs;
- legacy `.doc` ingestion;
- automatic unit conversion;
- persistent Brightway foreground database writing;
- LCIA/Monte Carlo/scenario execution in the app.

Those steps should build on the reviewed process map and approved technosphere/biosphere mappings rather than bypassing them.
