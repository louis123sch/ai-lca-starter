# AI-LCA Foreground Builder — starter prototype

This repository is a human-in-the-loop AI-assisted LCA prototype built around Brightway.

The design principle is:

> **AI interprets and proposes → deterministic software validates → human approves → Brightway retrieves/calculates.**

## Current workflow

1. Paste technical text or drag and drop multiple source documents.
2. Read machine-readable PDF (`.pdf`) and Word (`.docx`) files locally.
3. Preserve source filename provenance, PDF page markers, and Word table markers.
4. Run a **two-pass AI interpretation for each source**:
   - Pass 1: understand the technology/system and foreground process context.
   - Pass 2: selectively extract foreground exchanges that plausibly link to ecoinvent, plus useful parameters/emissions/outputs.
5. Review/edit the proposed foreground interpretation in Streamlit.
6. Retrieve real candidate activities from the selected local Brightway/ecoinvent database.
7. Rank those candidates deterministically using product wording, activity type, unit and geography preferences.
8. Manually approve a real ecoinvent candidate and export the reviewed mappings.

## What the AI is now asked to do

The AI is not supposed to copy source labels directly into ecoinvent search.

For example:

```text
Source wording: "Natural gas - SMR + CCS 90%"

AI foreground interpretation:
- canonical concept: natural gas
- parent foreground process: SMR
- ecoinvent search concept: natural gas
- likely activity type: market
- geography hint: GB (only if supported)
- 90% capture: separate model parameter, not an ecoinvent-searchable flow
```

The extraction keeps separate fields for:

- canonical LCA concept
- original/source wording
- parent foreground process/stage
- amount and unit
- ecoinvent search concept
- activity-type hint (`market`, `transforming`, `treatment`, etc.)
- geography hint
- explicitly stated supplier technology/provenance
- short interpretation rationale
- source/page/table/evidence provenance

The AI may use ecoinvent naming semantics to propose how an exchange should be searched, but it **must not invent an exact ecoinvent dataset**. Brightway is the authority on which activities actually exist in the installed database.

## Deterministic search routing

A row reaches the ecoinvent technosphere matcher only when all of the following are true:

- the user has included it;
- `item_type == technosphere_flow`;
- `search_worthy == true`;
- `ecoinvent_search_term` is non-empty.

This means parameters such as plant lifetime, capture rate or operating hours cannot enter the background search merely because they contain numbers or technology wording.

Location is a ranking preference rather than a hard exclusion. The candidate retriever also understands common ecoinvent naming patterns such as `market for ...` and `treatment of ...` when generating search variants.

## Important boundary

The prototype **does not yet write the approved inventory into a persistent Brightway foreground database or run LCIA automatically**. Foreground network/subprocess construction will come after document interpretation and background-process matching are reliable.

Scanned/image-only PDFs are also not OCR'd in this version.

## Project structure

```text
ai-lca-starter/
├── app.py
├── src/ai_lca/
│   ├── models.py
│   ├── documents.py
│   ├── llm.py
│   ├── pipeline.py
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

Use the same Python environment in which Brightway already works.

```bash
python -m pip install -e .
```

Create a local `.env` file from `.env.example`:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-mini
BRIGHTWAY_PROJECT=your_existing_project
```

Never commit `.env` or API keys to GitHub.

## Run

```bash
streamlit run app.py
```

The document interpretation now uses **two OpenAI calls per source document**, so it is deliberately more expensive than the earlier one-pass extractor.

## Current development direction

Next milestones:

1. Validate the two-pass interpretation against representative technical papers/datasheets.
2. Add AI assessment/explanation of the **real** candidates returned by Brightway.
3. Improve hybrid lexical/semantic ecoinvent retrieval.
4. Add explicit foreground network/subprocess review and persistent Brightway foreground construction.
5. Add parameter/scenario handling, dynamic inputs, uncertainty and provenance-aware validation.
