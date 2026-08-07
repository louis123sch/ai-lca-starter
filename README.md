# AI-LCA Starter

A first local prototype for:

**paste text / upload PDF → AI-proposed foreground inventory → human review → Brightway/ecoinvent candidate search → human-approved mapping**

The architectural rule is:

> **AI proposes → deterministic validation → human approves → Brightway calculates.**

This prototype deliberately **does not** let the LLM calculate LCIA or fabricate ecoinvent datasets.

## What works in v0.1

- Paste technical text into a local Streamlit interface.
- Upload a text-readable PDF; text is extracted locally with page markers.
- Send that source text to the OpenAI API using Pydantic-backed Structured Outputs.
- Extract foreground flows with amount, unit, basis, stage/component, page and evidence text.
- Review and edit the proposed inventory in the browser.
- Search a real database already installed in your Brightway 2.5 project.
- Review candidate ecoinvent activities including database, activity ID/code, reference product, location and unit.
- Manually approve a background mapping.
- Export the reviewed inventory and approved mappings.

## Deliberately not in v0.1

- OCR for scanned/image-only PDFs.
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

1. Paste source text **or** upload a text-readable PDF.
2. Add study instructions if useful, e.g. `Focus on cradle-to-gate inputs for 1 kg H2; keep infrastructure separate from operation.`
3. Click **Extract proposed inventory**.
4. Inspect the value, unit, basis and especially the **evidence** field for every row.
5. Edit/remove/add rows as required.
6. Enter/select the Brightway project and ecoinvent database in the sidebar.
7. Click **Search ecoinvent candidates**.
8. Review the real candidates returned from your local database.
9. Select a mapping only when you agree with it.
10. Export the reviewed inventory and mappings.

## Why PDF text is extracted locally first

For this first research prototype, local extraction makes provenance easy to inspect. Each page is converted to text with a marker such as `[PAGE 8]`, and the model must return evidence for each proposed flow. That makes it much easier to test extraction accuracy against the original source.

Scanned PDFs, complicated figures and some tables need a multimodal PDF path later. Do not silently OCR or infer values in the first benchmark version.

## Project structure

```text
ai-lca-starter/
├── app.py
├── src/ai_lca/
│   ├── models.py             # strict foreground schema
│   ├── documents.py          # local PDF text extraction
│   ├── llm.py                # structured LLM extraction
│   ├── brightway_search.py   # real Brightway candidate retrieval
│   └── export.py
├── notebooks/
│   └── 01_document_to_inventory.ipynb
├── tests/
├── data/
├── .env.example
└── pyproject.toml
```

## Next development step

The next version should add an explicit **candidate-ranking layer** which scores lexical/semantic similarity, unit, reference product, geography and activity type, while still requiring user approval. After that, approved mappings can be written into a persistent Brightway foreground database.
