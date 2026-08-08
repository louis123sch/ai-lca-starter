# AI-LCA Starter

A local research prototype for:

**paste text / upload PDF or Word → evidence-backed process map → human confirms foreground processes → AI extracts exchanges → human review → Brightway/ecoinvent candidate search → human-approved mapping**

The architectural rule is:

> **AI proposes → deterministic validation → human approves → Brightway calculates.**

This prototype deliberately **does not** let the LLM calculate LCIA or fabricate ecoinvent datasets.

## What works

- Paste technical text into a local Streamlit interface.
- Upload a text-readable PDF; text is extracted locally with page markers.
- Upload a modern Word (`.docx`) document; paragraphs and tables are extracted locally in document order.
- First reconstruct the foreground **process structure** represented by the source rather than flattening the whole paper into a bag of flows.
- Group related processes by technology/pathway.
- Keep engineering operations such as plasma generation, purification or separation inside a foreground process unless the source LCA explicitly models them as separate processes.
- Require direct source evidence for each proposed foreground process.
- Capture source-derived study/process geography and time context.
- Let the user approve which foreground processes proceed to exchange extraction.
- Extract foreground flows with amount, unit, basis, process ID, stage/operation context and evidence text.
- Deterministically reject flows assigned to invented or unapproved processes.
- Remove unsupported electricity voltage specificity when the source only states `electricity`.
- Review and edit the proposed inventory in the browser.
- Search a real database already installed in your Brightway 2.5 project.
- Rank candidate activities using geography extracted from the paper when available; there is no manual preferred-geography setting.
- Review candidate ecoinvent activities including database, activity ID/code, reference product, location and unit.
- Manually approve a background mapping.
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

Streamlit will print a local URL, normally `http://localhost:8501`, and usually opens it automatically.

## Workflow

1. Paste source text **or** upload a text-readable PDF / Word `.docx` document.
2. Add study instructions if useful, e.g. `Focus on cradle-to-gate inputs for 1 kg H2; keep infrastructure separate from operation.`
3. Click **Analyse paper structure**.
4. Review the technology groups, foreground processes, operations and the evidence supporting each process.
5. Uncheck any process that the paper does not genuinely model as a separate foreground process.
6. Click **Extract inventory for selected processes**.
7. Inspect the value, unit, basis and especially the **evidence** field for every exchange.
8. Edit/remove rows as required.
9. Enter/select the Brightway project and ecoinvent database in the sidebar.
10. Click **Search ecoinvent candidates**.
11. Where geography is stated in the source, candidates matching that source-derived geography are ranked first without being used as a hard filter.
12. Select a mapping only when you agree with it.
13. Export the reviewed inventory and mappings.

## Document extraction and provenance

For PDFs, local extraction retains markers such as `[PAGE 8]`, which allows the model to return page provenance for proposed processes and flows.

For Word `.docx` files, pagination is not stable enough to treat page numbers as reliable source metadata. Instead, paragraphs are retained in document order and tables are represented explicitly as `[TABLE N]` blocks. This is important because many published or draft LCI inventories are stored in Word tables.

Scanned PDFs, complicated figures and image-only tables need a multimodal PDF path later. Do not silently OCR or infer values in the benchmark workflow.

## Project structure

```text
ai-lca-starter/
├── app.py
├── src/ai_lca/
│   ├── models.py             # process-map and inventory schemas
│   ├── documents.py          # local PDF / Word extraction
│   ├── llm.py                # process discovery + constrained inventory extraction
│   ├── validation.py         # deterministic process/flow safeguards
│   ├── brightway_search.py   # real Brightway candidate retrieval + source-geography ranking
│   └── export.py
├── notebooks/
├── tests/
├── data/
├── .env.example
└── pyproject.toml
```

## Next development step

The next version should strengthen the **candidate-ranking layer** with lexical/semantic similarity, unit, reference product, geography and activity type while still requiring user approval. After that, approved mappings can be written into a persistent Brightway foreground database.
