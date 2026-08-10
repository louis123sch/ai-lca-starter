# Paper Extractor v1 Validation Plan

## Purpose

This document defines the remaining work needed to move from the current v0.4 candidate to a defensible v1 release without returning to open-ended tuning against the existing benchmark suite.

The current architecture is:

`source evidence -> native/visual ingestion -> candidate activity role classification -> deterministic foreground locking -> locked flow extraction -> human process/inventory review -> Brightway matching -> strict foreground write`

## Stage A — Candidate consolidation

Before any new blind evaluation:

- deterministic CI must pass;
- the Streamlit application and package must compile;
- the strict Brightway writer must pass an integration test in a temporary Brightway project;
- extraction outputs must record model, package version, commit SHA, timestamp and source mode;
- original source-grounded activity-role classifications must remain auditable after human review;
- existing benchmark gold standards and historical metrics remain unchanged.

No automatic paper-specific prompt repair is used during this stage.

## Stage B — One regression check of the architectural change

The new role-classification architecture may be checked once against the existing 001-006 material to answer a narrow question: did the architectural change preserve established capabilities while addressing the known structural failure class?

Interpret results according to benchmark history:

- Hermesmann and Gerloff are regression evidence;
- Yang is development evidence;
- Gonzales-Calienes is no longer independent unseen evidence after prior use;
- Afzal and Terlouw retain their original frozen-holdout results as historical evidence; any new run is post-holdout regression/development evidence only.

Do not repeatedly tune v0.4 against these six papers. If the role-classification architecture causes a serious regression, diagnose it before moving forward. Small benchmark-score fluctuations are not a reason to restart autonomous prompt optimisation.

## Stage C — New blind validation set

Freeze a candidate commit SHA before selecting or inspecting holdout outputs.

Target 8-12 previously untouched LCA papers, selected to cover structural diversity rather than a single technology family. The set should include examples of:

- one assessed product system described through multiple internal unit operations;
- multiple independent technology/pathway alternatives;
- explicit interconnected foreground subprocesses;
- shared supporting infrastructure/services;
- capital plus operation inventories;
- supplementary LCI tables;
- image/table-heavy evidence;
- co-products or avoided-product/system-expansion treatment;
- transport/storage stages;
- descriptive engineering detail that is broader than the actual LCA model.

Where possible, reserve at least four papers as frozen holdouts that cannot influence repairs until all frozen papers have run on the exact same SHA.

## Stage D — Evaluation dimensions

Report paper-level and aggregate metrics separately. At minimum record:

- process recall;
- process precision;
- flow recall;
- flow precision;
- amount accuracy where quantities are expected;
- unit accuracy;
- direction accuracy;
- functional-unit accuracy;
- system-boundary/context accuracy;
- unsupported/invented quantity count;
- over-decomposition count;
- under-decomposition count;
- repeated-run stability;
- human correction burden.

Human correction burden should distinguish:

1. approval/no substantive correction;
2. minor rename/unit/include-exclude correction;
3. process merge/re-parent or several flow corrections;
4. substantial reconstruction required.

A human-in-the-loop extractor succeeds when review is normally verification/correction rather than reconstruction from scratch.

## Stage E — Provisional v1 release criteria

The following are targets, not benchmark thresholds to be retroactively weakened:

- process recall: >= 90% across the blind set;
- process precision: >= 90%;
- flow recall: >= 85%;
- flow precision: >= 90%;
- no systematic invented quantities or unsupported ecoinvent detail;
- no common paper structure produces catastrophic foreground fragmentation;
- a majority of blind papers require only approval or minor correction;
- reviewed mappings can be exported reproducibly;
- the strict Brightway writer can create a foreground database whenever the reviewed paper model contains sufficient explicit information and no unresolved allocation/unit-conversion decision.

If a paper is blocked from automatic Brightway writing because the source leaves a modelling choice unresolved, that is preferable to silently inventing the choice and is not itself a failure of evidence extraction.

## Stage F — Release

When blind validation is complete:

1. record the frozen validation SHA and results;
2. repair only general defects justified across the blind set;
3. run regression tests after any repair;
4. perform a final untouched holdout run if any holdouts remain;
5. tag the accepted commit as v1.0.0;
6. archive benchmark/validation summaries with the release;
7. treat LCIA, uncertainty, dynamic data APIs and external gap filling as separate follow-on projects.

## Definition of done

Paper Extractor v1 is complete when a user can upload a previously unseen paper and supplement, obtain an evidence-linked proposed foreground graph and inventory, inspect why activities were retained or rejected, correct the structure and flows, map retained exchanges to real local Brightway nodes, export the complete review/provenance record, and create a reviewed local foreground database when the source provides enough information — without manually retyping the paper.