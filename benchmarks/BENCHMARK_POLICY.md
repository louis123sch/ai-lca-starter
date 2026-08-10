# AI-LCA benchmark generalisation policy

## Objective

The benchmark programme is intended to improve general paper-reading and LCA foreground reconstruction, not to optimise scores on a small set of repeatedly inspected papers.

## Benchmark roles

### Development / regression suite

- Hermesmann 2022: established regression benchmark.
- Gerloff 2021: originally unseen multimodal benchmark; after repeated repair cycles it is now part of the development/regression suite. Its original blind metrics remain the historical baseline and must never be overwritten.
- Yang et al. 2024 (Benchmark 003): development benchmark. It may expose general extraction or evaluator defects. Any repair must remain paper-agnostic and preserve earlier regressions.

Once a benchmark has influenced a prompt, evaluator, ingestion, or extraction change, its later scores are regression evidence, not independent evidence of generalisation.

### Unseen diagnostic

- Gonzales-Calienes et al. 2025 (Benchmark 004): next unseen diagnostic benchmark. Run it only after the development suite is stable. If it exposes a genuine general defect, it may become development material; after any repair, rerun the full relevant regression suite before accepting the change.

### Frozen holdouts

- Afzal et al. 2023 (Benchmark 005): holdout validation.
- Terlouw et al. 2021 (Benchmark 006): final holdout validation.

After Benchmark 004 passes, freeze the extractor commit SHA. Benchmarks 005 and 006 must both run against that exact SHA. No prompt, evaluator, ingestion, extraction, fixture, expected-value, or threshold changes are permitted between the two holdout runs.

A failure on Benchmark 005 must not be repaired before Benchmark 006 runs. Benchmark 006 must still run on the identical frozen SHA if Benchmark 005 completed successfully at the extraction/infrastructure level but failed its quality gate.

Infrastructure-only failures may be retried because they do not reveal benchmark content to the extractor. Quality failures may not trigger holdout-specific repair until the frozen holdout sequence is complete and its original metrics have been preserved.

## Repair constraints

All development repairs must satisfy these rules:

1. Do not mention a benchmark paper, technology name, expected process name, expected flow, or benchmark-specific vocabulary in a general extraction prompt unless that concept is independently justified as a general LCA rule.
2. Do not lower benchmark thresholds or edit expected values merely to improve scores.
3. Do not invent missing LCI.
4. Treat evaluator false negatives as evaluator-design issues; do not weaken the gold standard to accommodate them.
5. Preserve the architecture: source evidence -> visual/native evidence ingestion -> process structure -> locked flow extraction -> human review -> Brightway matching.
6. Prefer rules that express general modelling distinctions, such as the evidence required to promote a life-cycle stage or unit operation to a separate interconnected foreground activity.
7. After a benchmark influences a repair, move it conceptually into the regression suite and stop citing subsequent performance on that paper as independent generalisation evidence.

## Evidence of generalisation

The strongest evidence is performance on papers that did not influence the extractor version being tested. Report development/regression results separately from frozen-holdout results. Periodically add new heterogeneous LCA papers and reserve some permanently as untouched holdouts for future frozen releases.
