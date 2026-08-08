# AI-LCA project map

The working code is deliberately kept at the repository root so it is easy to open and edit directly in VS Code.

## Main files

| File | What to edit it for |
|---|---|
| `app.py` | Streamlit interface, buttons, review tables, user workflow |
| `document_reader.py` | PDF/DOCX reading and source/page/table provenance |
| `ai_foreground_interpreter.py` | OpenAI prompts and the two-pass AI reasoning/extraction behaviour |
| `foreground_pipeline.py` | Multi-document processing and merging of proposed foregrounds |
| `ecoinvent_search.py` | Brightway search, ecoinvent query variants, geography preferences and candidate scoring |
| `data_models.py` | Pydantic schemas: process proposals, exchanges, parameters and evidence |
| `export_helpers.py` | Conversion to review tables/CSV/JSON and deterministic search routing |
| `notebook_document_to_foreground.ipynb` | Interactive workbench for testing the pipeline without the Streamlit UI |

## Normal workflow

```text
PDF / DOCX / pasted text
        |
        v
document_reader.py
        |
        v
ai_foreground_interpreter.py
  pass 1: understand technology/process context
  pass 2: propose selective foreground exchanges
        |
        v
foreground_pipeline.py
  merge multiple source interpretations
        |
        v
export_helpers.py
  reviewable dataframe + deterministic routing
        |
        v
ecoinvent_search.py
  retrieve/rank real Brightway activities
        |
        v
app.py
  human review and approval
```

## If you want to change...

- **What the AI extracts or how deeply it interprets a document:** edit `ai_foreground_interpreter.py`.
- **How PDF/Word files are read:** edit `document_reader.py`.
- **How several documents are combined:** edit `foreground_pipeline.py`.
- **What counts as ecoinvent-searchable:** edit `export_helpers.py` and the schemas in `data_models.py`.
- **How ecoinvent candidates are found/ranked:** edit `ecoinvent_search.py`.
- **What the browser window looks like:** edit `app.py`.

## Secondary folders

- `tests/` contains automated tests only.
- `data/` is reserved for local/project data.
- `src/ai_lca/` contains compatibility import shims only; do not edit business logic there.

## Run

```bash
python -m pip install -e .
pytest
streamlit run app.py
```
