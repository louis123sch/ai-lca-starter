# Benchmark 002 — Gerloff (2021)

Source study: Niklas Gerloff, *Comparative Life-Cycle-Assessment analysis of three major water electrolysis technologies while applying various energy scenarios for a greener hydrogen production*, Journal of Energy Storage 43 (2021) 102759.

## Why this benchmark matters

This is the first unseen-paper validation after the Hermesmann benchmark. The extractor is not changed before the first Gerloff run.

Gerloff is structurally different from Hermesmann:

- three 1 MW foreground technology systems: AEC, PEMEC, SOEC;
- cradle-to-grave rather than cradle-to-gate;
- 20-year equipment lifetime and three stack replacements;
- explicit BoP and stack component inventories;
- operating water, electricity, KOH and heat data per kg H2;
- transport, disposal and additional manufacturing-energy assumptions;
- Germany-focused electricity scenarios (2019, 2030, 2050, RE);
- ecoinvent v3.5 cut-off background data.

The benchmark deliberately treats physical BoP/stack items as component/stage information inside AEC, PEMEC and SOEC rather than as dozens of independent foreground production processes. This follows the project's anti-fragmentation objective: detailed capital inventory should not automatically cause one Brightway search per physical component.

## Critical ingestion test

The supplementary DOCX contains the detailed numerical component/eCoinvent inventories in Figures 1-5 as embedded images. The current application DOCX ingestion path reads Word paragraphs and native Word tables, but not text inside embedded images.

For the blind first run:

- `source_main_excerpt.txt` contains machine-readable main-paper evidence;
- `source_supplement_machine_readable.txt` mirrors what the current text-only DOCX path can expose;
- the source fixture does **not** contain transcribed values from the embedded image figures;
- the gold standard **does** include 30 representative quantities from those figures.

This is intentional. If the first run misses those quantities, that is a real end-to-end ingestion failure rather than an LLM hallucination/extraction failure. The benchmark must reveal it before any multimodal/OCR improvement is added.

## Gold-standard scope

`flows.csv` contains 116 targets:

- 12 operating flows from the reported per-kg-H2 operating table;
- 6 additional manufacturing-energy flows per functional unit;
- 68 grouped capital-component anchors from the main-paper AEC/PEMEC/SOEC component tables;
- 30 representative detailed material quantities from supplementary Figures 1-5.

The benchmark also checks:

- functional unit: 1 kg H2;
- system boundary: cradle to grave;
- Germany as the operational/scenario context;
- process structure: AEC, PEMEC, SOEC;
- no conversion of physical components into redundant foreground processes;
- no leakage of raw ecoinvent dataset names into foreground-flow names.

## First-run rule

Do not change the extractor before measuring this benchmark. The first result is a blind generalisation test. Only after the result is recorded should failures be classified as:

1. document ingestion failure;
2. process-structure interpretation failure;
3. flow extraction/quantity failure;
4. foreground/background separation failure;
5. benchmark-design ambiguity.
