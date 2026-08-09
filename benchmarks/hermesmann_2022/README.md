# Benchmark 001 — Hermesmann & Müller (2022)

This benchmark is derived from the user-supplied main paper and supplementary information for *Green, Turquoise, Blue, or Grey? Environmentally friendly Hydrogen Production in Transforming Energy Systems*.

The full journal PDF and DOCX are not committed. Two private-repository text fixtures retain the LCA-relevant source content needed for automated regression testing, including deliberate technology-review distractors, the goal/scope, Tables 2–4, background mapping information, and published reference-case GWI values.

## Ground-truth scope

The core foreground target is the LCI explicitly reported in Tables 2–4 of the main paper:

- SMR;
- SMR-CCS (56% capture);
- SMR-CCS (90% capture);
- MP-H2;
- MP-NG;
- MP-C;
- MP-E;
- PEMEL today;
- PEMEL future.

These nine configurations contain 119 quantified foreground rows in the benchmark.

The process-flow diagrams contain internal unit operations (for example pre-treatment, reforming, shift, PSA, liquid-tin reactor, carbon filtration and separators). The benchmark does **not** treat these boxes as separate foreground LCA processes unless the inventory itself is separately modeled on that basis. This is deliberate: it tests over-decomposition.

## Context targets

- Functional unit: 1 kg H2 at 30 bar at the production site.
- System boundary: cradle-to-gate.
- Primary/reference case: Germany.
- Additional country-specific scenarios are assessed separately in the paper.

## Background-mapping targets

Supplementary Table S1 supplies expected ecoinvent v3.6 cut-off background process names and regions. These are stored as mapping expectations, not foreground-flow names. For example, the foreground flow `Electricity` may later map to `Market for electricity, high voltage` in Germany; the foreground extractor should not rename the flow to the ecoinvent dataset during extraction.

## Published-result targets

The benchmark stores the reference-case GWI values without by-product credits reported in Table 7. These targets are kept separate from extraction scoring because a reconstructed Brightway model can differ from the paper due to ecoinvent and LCIA version differences even when the foreground inventory is correct.

## Automated GitHub Actions loop

`.github/workflows/hermesmann-benchmark.yml` runs deterministic tests and then repeated live LLM extractions. Pushes to `main` that change the extractor, tests, benchmark, workflow, or package configuration trigger the workflow automatically. A manual `workflow_dispatch` run defaults to five independent LLM runs; push-triggered runs default to three to limit API cost while still checking stochastic stability.

The live job requires one encrypted repository secret:

`OPENAI_API_KEY`

Add it in GitHub under **Settings → Secrets and variables → Actions → New repository secret**. Do not commit the key to `.env`, source files, workflow YAML, issues, logs, or benchmark artifacts.

The workflow uploads run-by-run extraction JSON, score reports, and `summary.json` as a 30-day artifact. The quality gate currently requires:

- mean overall score >= 0.85;
- minimum individual-run score >= 0.80;
- mean process recall and precision >= 0.90;
- mean flow recall and precision >= 0.85.

The thresholds and gold-standard values should not be weakened simply to make CI pass. Failures should drive generalizable extractor improvements instead.
