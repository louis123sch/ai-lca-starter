# Frozen holdout baseline — 2026-08-10

Frozen extractor commit: `ee10b61eaeba8b8a05d422f95cc19ce2d17eb179`

These are the original, immutable generalisation results from the first frozen holdout sequence. They must remain historical evidence even if evaluator logic or extraction logic is improved later.

## Benchmark 005 — Afzal et al. 2023

Workflow run: `31354970065`
Artifact: `benchmark-005-holdout-ee10b61eaeba8b8a05d422f95cc19ce2d17eb179-31354970065` (artifact id `9050373290`)

- runs: 3
- model: `gpt-5-mini`
- mean overall: `0.20000000000000004`
- minimum overall: `0.2`
- mean process recall: `0.0`
- mean process precision: `0.0`
- mean flow recall: `0.0`
- mean flow precision: `0.0`

Interpretation note recorded after both holdouts completed: this raw score contains a benchmark-evaluator false-negative. All three extractions produced exactly two MPW foreground product systems, but the name matcher failed to recognize semantically equivalent forms such as `MPW to methanol (foreground product system)` against gold `MPW-methanol`, which prevented all attached flows from being evaluated as matches. The raw score above is not revised or replaced.

## Benchmark 006 — Terlouw et al. 2021

Workflow run: `31355209899`
Artifact: `benchmark-006-final-holdout-ee10b61eaeba8b8a05d422f95cc19ce2d17eb179-31355209899` (artifact id `9050499558`)

- runs: 3
- model: `gpt-5-mini`
- mean overall: `0.783122538295968`
- minimum overall: `0.6805760810071156`
- mean process recall: `1.0`
- mean process precision: `0.8095238095238096`
- mean flow recall: `0.6896551724137931`
- mean flow precision: `0.4843459470325142`

Per-run overall scores: `0.8149829903913391`, `0.6805760810071156`, `0.8538085434894495`.

Interpretation note: process recall was perfect in all three runs. Two runs over-decomposed shared DAC-plant and CO2 transport/storage activities into additional foreground processes; the third retained exactly the five expected processes but still emitted many extra flows, leaving flow precision low. This is extraction-quality evidence, not an infrastructure failure.

## Pre-freeze diagnostic — Benchmark 004

Gonzales-Calienes et al. 2025 passed on the same commit before the holdout sequence.

Workflow run: `31354518022`
Artifact id: `9050300133`

- mean overall: `0.9862732732732735`
- minimum overall: `0.9858468468468471`
- mean process recall: `1.0`
- mean process precision: `1.0`
- mean flow recall: `1.0`
- mean flow precision: `0.9955555555555556`

No gold standards or thresholds were changed in recording this file.
