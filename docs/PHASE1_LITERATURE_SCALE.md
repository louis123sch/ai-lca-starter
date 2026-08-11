# Phase 1: large-scale machine-readable LCA corpus

This branch extends the current AI-LCA extraction core; it does not replace it.

## Objective

Build and process a large, diverse LCA corpus using machine-readable sources first, with no dependency on manually collected PDFs. PDF-only validation is deliberately deferred until the core extraction logic has been exposed to broad LCA structure.

## Pipeline

1. Discover journal records by DOI/metadata (Crossref; RSS later for incremental updates).
2. Cheap deterministic metadata screening before any LLM spend.
3. Acquire the best programmatically available machine-readable source:
   - publisher JATS/XML
   - HTML/native tables
   - supplementary XLSX/CSV
   - other structured supplementary files
4. Skip records that require manual retrieval during the large-scale phase.
5. Feed retained source evidence into the existing AI-LCA process/flow extraction core.
6. Store extraction outputs, evidence provenance, warnings, runtime and model version per DOI.
7. Analyze recurring failures and improve only generic extraction architecture.
8. After the core stabilizes, select approximately 10 diverse papers for PDF-only validation.

## First implemented slice

`python -m ai_lca.literature` streams journal metadata from Crossref with cursor paging and writes `catalogue.jsonl` plus `catalogue.csv`.

The default journal is The International Journal of Life Cycle Assessment (online ISSN 1614-7502).

Example:

```bash
python -m ai_lca.literature \
  --from-year 2020 \
  --until-year 2026 \
  --mailto YOUR_EMAIL \
  --output-dir literature_runs/ijlca_2020_2026
```

For a fast smoke run:

```bash
python -m ai_lca.literature --max-records 25 --output-dir literature_runs/smoke
```

## Source acquisition policy

No browser-login scraping and no systematic authenticated PDF downloading. The acquisition layer should use publisher/open-access/TDM endpoints under their applicable licence terms. Springer Nature currently exposes an Open Access API with JATS full text where available and a Full Text TDM XML endpoint for users covered by the relevant agreement.

A missing programmatic source is a queue status, not a reason for manual intervention during Phase 1.

## Design rule

The DOI is the durable paper identity. All Phase-1 records retain DOI metadata so a clean structured extraction can later be paired with the PDF of the same paper during Phase 2.
