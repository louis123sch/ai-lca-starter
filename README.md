# AI-LCA Paper Extractor

A local, human-in-the-loop research tool for turning LCA papers into evidence-linked foreground inventories and reviewed Brightway models.

**PDF/DOCX/text → native + visual evidence → process-role classification → locked foreground graph → flow extraction → human review → Brightway candidate mapping → optional strict foreground database write**

The architectural rule is:

> **AI proposes from source evidence → deterministic rules lock/validate structure → human reviews → Brightway supplies real background nodes.**

The extractor does not calculate LCIA, fill missing LCI from the public web, invent ecoinvent datasets, silently convert units, or choose co-product allocation on the user's behalf.

## v0.4 candidate: moving out of benchmark tuning

The current development phase replaces repeated paper-specific prompt tuning with an explicit, testable process-role decision layer.

Before a process can enter the locked foreground graph, each process-like source entity is classified as one of:

- `assessed_product_system`
- `interconnected_foreground_process`
- `internal_stage`
- `shared_supporting_activity`
- `background_supply`
- `descriptive_only`

Only the first two roles are deterministically promoted into foreground process IDs. This is intended to address the general failure class exposed by development/holdout work: internal inventory stages and shared supporting activities being mistaken for independently assessed foreground processes.

The app now also supports human process review before mapping: keep/remove, rename, re-parent, or merge proposed foreground processes. Flow reassignment after a merge is deterministic and the original AI classifications remain in the audit trail.

## Evidence ingestion

Supported inputs:

- pasted text;
- PDF, including native text plus selected embedded figures/pages for multimodal transcription;
- Word `.docx`, including paragraphs, tables and selected embedded visual evidence;
- Excel `.xlsx`/`.xlsm`, read sheet-by-sheet as tab-separated tabular evidence;
- multiple main/supplementary documents in one extraction.

Visual evidence is transcribed before LCA interpretation. The visual stage is evidence transcription only; it does not decide process structure or map to ecoinvent.

## Extraction architecture

1. **Evidence ingestion** — combine source documents with provenance markers and visual transcriptions.
2. **Candidate activity classification** — identify process-like source entities and classify their role in the actual LCA model.
3. **Deterministic foreground locking** — only assessed product systems and explicitly interconnected foreground processes enter the locked graph.
4. **Locked flow extraction** — extract source-supported materials, energy, transport, products and emissions without creating new process IDs.
5. **Human process review** — merge/remove/rename/re-parent processes and review reference products/units.
6. **Human inventory review** — edit/include/exclude flows while retaining source evidence.
7. **Brightway matching** — search only real nodes in the selected local Brightway databases. Paper geography is a soft ranking hint for technosphere search; emissions are searched in a biosphere database.
8. **Strict database write** — create a new local Brightway foreground database only after explicit confirmation and only when the reviewed model can be represented without hidden modelling assumptions.

## Strict Brightway writer

The v0.4 writer supports:

- one reviewed production activity per retained foreground process;
- mapped technosphere inputs;
- mapped technosphere outputs/co-products (e.g. recycling credits, avoided-burden flows) — written using the flow's own reviewed signed amount, exactly as reviewed, with no allocation/production percentage invented;
- mapped biosphere emissions;
- explicit foreground-to-foreground input links;
- extraction/version metadata on created foreground activities.

It deliberately blocks writing when:

- an included quantitative exchange has no amount;
- a process lacks a reviewed reference product/unit;
- an input/emission/output has no selected mapping and has not been explicitly excluded;
- source and mapped units differ and would require an unapproved conversion;
- a flow direction is unresolved.

The writer never overwrites an existing Brightway database.

## Reproducibility

Every extraction records:

- extractor package version;
- OpenAI model name;
- Git commit SHA when available;
- UTC generation time;
- whether the source path was text or documents.

The app can export a reproducible review bundle containing the structured extraction, reviewed inventory rows and selected Brightway mappings.

## Recommended setup on Mac / VS Code

Use the Python environment in which your Brightway project already works.

```bash
git clone <your-repository-clone-url>
cd ai-lca-starter
python -m pip install -e ".[test]"
cp .env.example .env
```

Edit `.env`:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-mini
BRIGHTWAY_PROJECT=the_exact_name_of_your_existing_project
```

Do not commit `.env`.

Run deterministic tests:

```bash
python -m pytest -q
```

Run the app:

```bash
streamlit run app.py
```

## App workflow

1. Paste text or upload the paper and supplementary PDF/DOCX files.
2. Click **Interpret paper and extract foreground**.
3. Inspect the retained/rejected candidate-activity classifications.
4. Review the process structure. Merge/remove/rename/re-parent only where the paper supports the edit.
5. Apply process review.
6. Review amounts, units, directions, foreground links and evidence in the inventory.
7. Select the local Brightway project and technosphere database.
8. Search candidates and review each selected mapping.
9. Download the audit/review bundle.
10. If the strict write validator passes, explicitly confirm and create a new Brightway foreground database.

## Current finish line

The paper extractor is considered v1-ready when a user can upload a previously unseen paper and supplement, receive an evidence-linked foreground graph and inventory, correct it in the UI, map retained exchanges to real Brightway nodes, and create/export a reproducible local foreground model without manually retyping the paper.

The remaining research validation task is blind testing on a larger untouched paper set. New features such as LCIA, Monte Carlo, dynamic grid APIs or web-based gap filling are separate follow-on work and should not delay the extractor v1 release.
