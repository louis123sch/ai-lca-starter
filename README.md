# AI-LCA Starter

A local research prototype for:

**upload multiple PDF / Word sources → combine them into one evidence corpus → evidence-backed process map → human confirms foreground processes → AI extracts exchanges across the corpus → human review → Brightway/ecoinvent candidate search → human-approved mapping**

The architectural rule is:

> **AI proposes → deterministic validation → human approves → Brightway calculates.**

This prototype deliberately **does not** let the LLM calculate LCIA or fabricate ecoinvent datasets.

## What works

- Upload multiple text-readable PDFs and modern Word (`.docx`) documents in one analysis.
- Optionally add pasted text as another source in the same evidence corpus.
- Preserve a `[DOCUMENT filename]` boundary around each source so provenance remains traceable while all sources are reasoned over together.
- Preserve PDF page markers and Word paragraphs/tables in document order.
- Reconstruct one foreground **process structure across the complete evidence corpus**, rather than extracting a separate model from each file.
- Allow one foreground process to be supported by evidence from several documents.
- Allow one foreground flow to be supported by evidence from several documents.
- Merge repeated descriptions of the same process/flow across documents instead of creating duplicate foreground activities or duplicate ecoinvent searches.
- Treat conflicting evidence as a warning rather than silently averaging, choosing, or duplicating it.
- Group related processes by technology/pathway.
- Keep engineering operations such as plasma generation, purification or separation inside a foreground process unless the combined evidence explicitly supports them as separate modelled processes.
- Require direct evidence for each proposed foreground process.
- Capture evidence-derived study/process geography and time context.
- Let the user approve which foreground processes proceed to exchange extraction.
- Deterministically reject flows assigned to invented or unapproved processes.
- Remove unsupported electricity voltage specificity when the evidence corpus only states `electricity`.
- Search a real database already installed in your Brightway 2.5 project.
- Rank candidate activities using geography extracted from the evidence corpus when available; there is no manual preferred-geography setting.
- Review and manually approve candidate ecoinvent activities.
- Export the reviewed inventory and approved mappings.

## Deliberately not included yet

- OCR for scanned/image-only PDFs.
- Legacy Word `.doc` ingestion; use `.docx`.
- LLM ranking of ecoinvent candidates.
- Unit conversion between foreground and candidate datasets.
- Automatic construction of the persistent Brightway foreground database.
- LCIA, Monte Carlo, scenario APIs or dynamic electricity.

Those should be added after the extraction and matching steps are tested against known LCA inventories.

## Recommended setup on your Mac / VS Code

Use your **existing working Brightway 2.5 environment** rather than creating another Brightway installation unnecessarily.

Open a terminal in VS Code and activate the environment you already use for Brightway. Then, from this project directory:

```bash
python -m pip install -e .
```

If Streamlit or the other dependencies are missing, the command above installs them into that environment. The project only directly imports `bw2data`; your existing Brightway environment can retain its current solver setup.

For Apple Silicon, Brightway's current installation documentation recommends the `brightway25` stack with `scikit-umfpack` rather than `pypardiso`.

## Configure the API key

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env`:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-mini
BRIGHTWAY_PROJECT=the_exact_name_of_your_existing_project
```

Do not commit `.env` to GitHub.

Alternatively, export the key from your shell:

```bash
export OPENAI_API_KEY="your_key_here"
```

## Run the app

```bash
streamlit run app.py
```

## Workflow

1. Upload **all documents that should count as evidence**: paper, supplementary information, appendices, reports, Word tables, etc. You can mix PDF and `.docx` files.
2. Optionally paste additional evidence; it is added to the same corpus.
3. Add study instructions if useful, e.g. `Focus on cradle-to-gate inputs for 1 kg H2; keep infrastructure separate from operation.`
4. Click **Analyse evidence corpus**.
5. Review the single synthesized process map. A process can show supporting evidence from multiple documents.
6. Uncheck any process that the evidence corpus does not genuinely support as a separate foreground process.
7. Click **Extract inventory for selected processes**.
8. Review flows built from all relevant sources. A single flow can show several supporting documents.
9. Edit/remove rows as required.
10. Enter/select the Brightway project and ecoinvent database in the sidebar.
11. Click **Search ecoinvent candidates**.
12. Where geography is supported by the evidence corpus, matching locations are ranked first without being used as a hard filter.
13. Select a mapping only when you agree with it.
14. Export the reviewed inventory and mappings.

## Multi-document evidence model

Uploaded files are not processed as independent LCAs. They are wrapped with source markers and concatenated into one evidence corpus before process discovery. The LLM is explicitly instructed to use evidence across documents jointly.

For example, if `paper.pdf` establishes that the study models one thermal-plasma methane-pyrolysis process and `supplement.docx` supplies its electricity and natural-gas inventory, the expected result is **one foreground methane-pyrolysis process** whose flows are supported by both files. The supplement should not generate another foreground process merely because its quantities live in another document.

Likewise, if the same electricity exchange appears in both files, it should remain one exchange with multiple evidence records. If the documents genuinely disagree about its amount or basis, the program should flag that conflict for review rather than average the values or silently create duplicate flows.

## Document extraction and provenance

For PDFs, local extraction retains markers such as `[PAGE 8]`, allowing page provenance for proposed processes and flows.

For Word `.docx` files, pagination is not stable enough to treat page numbers as reliable source metadata. Paragraphs are retained in document order and tables are represented explicitly as `[TABLE N]` blocks. This is important because many LCI inventories are stored in Word tables.

Each uploaded source is additionally wrapped in `[DOCUMENT filename] ... [END DOCUMENT filename]` markers. Process and flow evidence can therefore record which file, page/table/section and evidence text support a modelling decision while still reasoning across the complete corpus.

Scanned PDFs, complicated figures and image-only tables need a multimodal PDF path later. Do not silently OCR or infer values in the benchmark workflow.

## Project structure

```text
ai-lca-starter/
├── app.py
├── src/ai_lca/
│   ├── models.py             # process-map and multi-source evidence schemas
│   ├── documents.py          # local PDF / Word extraction + corpus construction
│   ├── llm.py                # corpus-wide process discovery + constrained inventory extraction
│   ├── validation.py         # deterministic process/flow safeguards
│   ├── brightway_search.py   # real Brightway candidate retrieval + evidence-derived geography ranking
│   └── export.py
├── notebooks/
├── tests/
├── data/
├── .env.example
└── pyproject.toml
```

## Next development step

The next version should strengthen the **candidate-ranking layer** with lexical/semantic similarity, unit, reference product, geography and activity type while still requiring user approval. After that, approved mappings can be written into a persistent Brightway foreground database.
