# AI-LCA Starter

A local research prototype for:

**paste text / upload PDF or DOCX → identify the paper's foreground process structure and study context → extract evidence-backed flows → human review → Brightway/ecoinvent candidate search → human-selected mapping**

The architectural rule remains:

> **AI proposes → deterministic validation → human reviews → Brightway supplies real background datasets.**

The LLM does not calculate LCIA and does not fabricate ecoinvent datasets.

## What changed in v0.2

The main change is **schema-first paper interpretation**. The tool no longer asks one model call to produce a flat inventory immediately.

1. **Process/context pass** — identify only the foreground processes and subprocesses the source actually models, plus functional unit, system boundary and operational geography where supported.
2. **Flow pass** — extract materials, energy, transport, outputs and emissions, but each flow must attach to one of those locked process IDs. The second pass is not allowed to create extra processes.
3. **Human review** — inspect the process hierarchy, context, values and evidence before background matching.
4. **Brightway matching** — search only real activities in the installed database. Paper-derived geography can softly promote matching locations but never filters the candidate list or comes from a manual preference field.

This is intended to reduce two specific failure modes from the first prototype:

- **invented detail**, such as turning generic `electricity` into an unsupported medium-voltage electricity dataset;
- **process fragmentation**, where one foreground process in the paper is unnecessarily split into multiple pseudo-subprocesses, causing redundant candidate searches.

Identical flow queries are now searched once and reused, and the highest-ranked real candidate is preselected in the dropdown while remaining editable.

## Supported inputs

- pasted text;
- text-readable PDF, extracted locally with `[PAGE N]` provenance markers;
- Word `.docx`, extracted locally with paragraph and table provenance markers.

Scanned/image-only PDFs still need a later multimodal path.

## Study geography

There is no longer a manual `GB,RER,GLO,RoW` geography preference box. The extraction shows the **operational geography represented by the paper** when the source supports one and labels it as explicit or inferred. A small deterministic mapping converts common country/region names into conservative ecoinvent location hints for **soft ranking only**. If the source does not establish geography, no location is forced.

## Relationship to Zhang et al. (2026), *Sustainability assessment using multimodal artificial intelligence agents*

This project does **not copy their code**. It adopts a useful architectural idea from the paper/repository: establish a structured LCI representation first, then populate and review it with evidence rather than asking an LLM for an unconstrained final inventory in one shot.

Their system uses an LCA agent to choose/design a schema and critique an inventory, while a stakeholder agent fills it iteratively using retrieval tools. This project narrows that concept to literature-to-Brightway foreground modelling:

- the **first pass** plays the role of structure/schema definition, but the schema is the process hierarchy evidenced in the supplied paper;
- the **second pass** fills that locked structure with cited foreground flows;
- Pydantic validation prevents flows from referencing process IDs that were not identified in the first pass;
- the human remains the final reviewer before ecoinvent mappings are used.

Unlike the published agent system, this prototype does not search the public web to fill missing data and deliberately does not let iterative agents fill evidence gaps with external estimates. That restriction is important for the current research goal: reconstruct what the source paper actually modelled before adding any gap-filling functionality.

Reference: Zhang, Z. et al. (2026), *Nature Electronics*, DOI 10.1038/s41928-026-01653-w.

## Recommended setup on your Mac / VS Code

Use your existing working Brightway 2.5 environment. From the project directory:

```bash
python -m pip install -e .
```

The update adds `python-docx` for Word ingestion.

## Configure the API key

```bash
cp .env.example .env
```

Then edit `.env`:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-mini
BRIGHTWAY_PROJECT=the_exact_name_of_your_existing_project
```

Do not commit `.env`.

## Run the app

```bash
streamlit run app.py
```

## Workflow

1. Paste source text or upload a text-readable PDF/DOCX.
2. Add study-specific instructions only if necessary.
3. Click **Interpret paper and extract foreground**.
4. Check the detected process hierarchy first. If the paper has one process, the tool should normally show one process.
5. Check operational geography, functional unit, system boundary and warnings.
6. Review/edit the foreground inventory and its evidence.
7. Select the Brightway project and ecoinvent database.
8. Click **Search ecoinvent candidates**.
9. Review the preselected top candidate for each included flow and change it where needed.
10. Export the reviewed inventory and selected mappings.

## Project structure

```text
ai-lca-starter/
├── app.py
├── src/ai_lca/
│   ├── models.py             # process hierarchy, context and flow schemas
│   ├── documents.py          # local PDF + DOCX text extraction
│   ├── geography.py          # conservative paper-context → ecoinvent location hints
│   ├── llm.py                # two-pass structured extraction
│   ├── brightway_search.py   # real Brightway candidate retrieval + soft geo ranking
│   ├── benchmark.py          # repeated paper-grounded regression evaluation
│   └── export.py
├── benchmarks/
│   └── hermesmann_2022/      # Benchmark 001 ground truth
├── notebooks/
├── tests/
├── data/
├── .env.example
└── pyproject.toml
```

## Still deliberately out of scope

- OCR/multimodal extraction for scanned PDFs and complex figures;
- automatic creation of the persistent Brightway foreground database;
- LCIA, Monte Carlo or dynamic grid APIs;
- external web gap filling;
- automatic approval of mappings.

Those should come only after the paper-to-foreground extraction has been benchmarked against known inventories.

## Regression benchmark: Hermesmann & Müller (2022)

The repository now includes a paper-grounded benchmark under `benchmarks/hermesmann_2022/`.
It treats the paper's explicitly modeled LCI in Tables 2–4 as the core foreground ground truth: nine hydrogen-production configurations and 119 quantified foreground flows. The supplementary information supplies expected ecoinvent background mappings, while remaining separate from foreground flow naming.

The benchmark is designed to catch the failure modes that matter for paper-to-Brightway reconstruction:

- missing modeled processes or flows;
- invented/over-decomposed foreground subprocesses;
- materials mentioned only in review prose being mistaken for modeled LCI flows;
- background ecoinvent names such as `market for ...` leaking into foreground flow names;
- wrong amount, unit, direction, functional unit, system boundary, or reference geography.

The application can now ingest the paper and supplementary documents together. Each source is wrapped in a `[DOCUMENT: filename]` marker so evidence provenance remains unambiguous even when page numbers restart in different files.

### Run a repeated live benchmark

With `OPENAI_API_KEY` configured, run:

```bash
ai-lca-benchmark live \
  --expected benchmarks/hermesmann_2022/expected.json \
  --source "path/to/main-paper.pdf" "path/to/supplement.docx" \
  --runs 5
```

Each run saves the structured extraction and a scored report under `benchmark_runs/hermesmann_2022/`, plus an aggregate summary. Repeating the same source several times is intentional: it measures extraction stability as well as one-off accuracy.

To score an extraction JSON that already exists:

```bash
ai-lca-benchmark evaluate \
  --expected benchmarks/hermesmann_2022/expected.json \
  --extraction benchmark_runs/hermesmann_2022/extraction_run_01.json
```

The benchmark source documents themselves are not committed to the repository; only the derived factual ground truth and evaluation rules are stored.
