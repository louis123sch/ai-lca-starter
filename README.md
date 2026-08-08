# AI-LCA Foreground Builder — starter prototype

Human-in-the-loop AI-assisted LCA prototype built around Brightway.

> **AI interprets and proposes → deterministic software validates → human approves → Brightway retrieves/calculates.**

## Working files are at the repository root

The project is deliberately flattened so the code you actually edit is immediately visible in VS Code.

```text
ai-lca-starter/
├── app.py                              # Streamlit interface and workflow
├── document_reader.py                  # PDF/DOCX ingestion and provenance
├── ai_foreground_interpreter.py        # OpenAI prompts + two-pass document reasoning
├── foreground_pipeline.py              # Multi-document interpretation/merging
├── ecoinvent_search.py                 # Brightway retrieval and candidate ranking
├── data_models.py                      # Pydantic schemas
├── export_helpers.py                   # Review tables, exports, deterministic routing
├── notebook_document_to_foreground.ipynb # Interactive workbench
├── PROJECT_MAP.md                      # What each file does
├── tests/                              # Automated tests only
├── data/                               # Project/local data
├── src/ai_lca/                         # Compatibility import shims only
├── .env.example
├── pyproject.toml
└── README.md
```

**Edit the top-level files, not `src/ai_lca/`.** The nested package now contains only compatibility redirects so older imports/tests do not break while the project is being reorganised.

See `PROJECT_MAP.md` for a direct “what file do I edit?” guide.

## Current workflow

1. Paste technical text or drag and drop multiple PDF/DOCX source documents.
2. Preserve source/page/table provenance.
3. Run two AI passes per source:
   - understand the technology and foreground process context;
   - selectively propose foreground exchanges, parameters, emissions and outputs.
4. Review the draft foreground interpretation.
5. Send only approved technosphere concepts to the local Brightway/ecoinvent matcher.
6. Retrieve and rank real ecoinvent activities.
7. Human approves the final mapping.

Example interpretation:

```text
Source wording: "Natural gas - SMR + CCS 90%"

canonical concept: natural gas
parent process: SMR
ecoinvent search concept: natural gas
activity hint: market
geography hint: GB (only if source-supported)
90% capture: separate parameter, not a background search term
```

A row reaches technosphere search only when it is included, classified as `technosphere_flow`, marked `search_worthy`, and has a non-empty search concept. Brightway remains the authority on which ecoinvent datasets actually exist.

## Install and run

Use the Python environment where Brightway already works:

```bash
python -m pip install -e .
pytest
streamlit run app.py
```

Create a local `.env` from `.env.example`:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-mini
BRIGHTWAY_PROJECT=your_existing_project
```

Never commit `.env` or API keys.

## Notebook workflow

Open `notebook_document_to_foreground.ipynb` directly from the root to inspect the stages without Streamlit. It imports the same top-level modules as the app.

## Current boundary

The prototype does **not yet** write the approved inventory into a persistent Brightway foreground database or run LCIA automatically. Foreground network/subprocess construction comes after document interpretation and ecoinvent matching are reliable.

Scanned/image-only PDFs also still need a later OCR/multimodal path.
