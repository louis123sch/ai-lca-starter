# AI-LCA Foreground Builder — starter prototype

This repository is a first working prototype for a human-in-the-loop AI-assisted LCA workflow built around Brightway.

The design principle is:

> **AI proposes → deterministic validation → human approves → Brightway calculates.**

The current prototype supports:

1. Paste technical text or drag and drop multiple source documents.
2. Read machine-readable PDF (`.pdf`) and Word (`.docx`) files locally.
3. Preserve source filename provenance, PDF page markers, and Word table markers.
4. Use an OpenAI model with Structured Outputs to extract and classify LCA-relevant information.
5. Review/edit the proposed information in a Streamlit table.
6. Separate technosphere flows from biosphere flows, model parameters, and reference products.
7. Search a local Brightway/ecoinvent database only for approved technosphere flows.
8. Manually approve a real ecoinvent candidate and export the reviewed mappings.

## Important boundary

The prototype **does not yet write the approved inventory into a persistent Brightway foreground database or run LCIA automatically**. The next intended milestone is an explicit foreground-database builder after extraction and background-process matching have been validated.

Scanned/image-only PDFs are also not OCR'd in this version.

## Project structure

```text
ai-lca-starter/
├── app.py
├── src/ai_lca/
│   ├── models.py
│   ├── documents.py
│   ├── llm.py
│   ├── brightway_search.py
│   └── export.py
├── notebooks/
├── tests/
├── data/
├── .env.example
├── pyproject.toml
└── README.md
```

## Installation

Use the same Python environment in which your Brightway installation already works.

```bash
python -m pip install -e .
```

The editable install is useful during development. Re-run the command whenever `pyproject.toml` gains a new dependency.

Create a local `.env` file from `.env.example`:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-mini
BRIGHTWAY_PROJECT=your_existing_project
```

Never commit `.env` or API keys to GitHub.

## Run the interface

```bash
streamlit run app.py
```

Streamlit opens a local browser interface, normally at `http://localhost:8501`.

### Document input

The **Upload documents** tab accepts several files at once. Streamlit's file uploader is itself the drag-and-drop zone, so files can be dragged directly from Finder onto that area or selected through the file picker.

Supported formats in this version:

- `.pdf` — machine-readable PDFs, with page markers retained
- `.docx` — Word paragraphs and tables

Pasted text and uploaded documents can be analysed together. Each extracted item can retain:

- source document
- PDF page where available
- Word/PDF table marker where available
- evidence snippet

## Classification and routing

Every extracted item is proposed as one of:

- `technosphere_flow` — materials, energy, fuels, transport, services, infrastructure inputs; eligible for ecoinvent candidate search
- `biosphere_flow` — direct elementary emissions/resources; retained for later biosphere matching
- `parameter` — plant lifetime, operating hours, capacity, efficiency, yield, load factor, etc.; retained but never sent to ecoinvent search
- `reference_product` — foreground output/product information

The item type is visible and editable before candidate search.

## Current development direction

Planned milestones include:

1. Better hybrid ecoinvent retrieval and ranking.
2. AI ranking/explanation of real candidates with human approval.
3. Persistent Brightway foreground-database construction.
4. Parameter handling and scenario generation.
5. Dynamic/API-driven inputs.
6. Uncertainty and provenance-aware validation.
