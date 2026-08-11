# Corpus-driven inventory recall development

This is the post-Phase-1 development architecture for Paper Extractor v1.

## Frozen baseline

The active immutable development baseline is `benchmarks/corpus_baseline_v1_1_2026-08-11` and is derived directly from the frozen Phase 1 SQLite artifact rather than a hand-built paper manifest.

The exact selection contains 44 papers that reached inventory processing:

- 37 `UNRESOLVED_INVENTORY`;
- 7 `COMPLETE`.

The earlier `corpus_baseline_v1_2026-08-11` directory is retained as historical provenance. Its summary correctly recorded 37 + 7, but its later `papers.json` was not an exact export and is superseded by v1.1.

Every replay run executes the frozen SQLite selection query and verifies both the total (44) and status counts (37 + 7) before development work proceeds. Count drift is a hard failure.

These papers are development/regression material, not unseen validation after this point.

## Efficiency rule

Do not rerun whole papers when the upstream result is already trusted. The replay path reparses source material deterministically, reuses the locked structure and existing resolved decisions, and spends model calls only on candidates that are new, missing, or ambiguous.

Model routing remains asymmetric:

- deterministic Python for enumeration, diagnostics, comparison, validation and merge logic;
- `gpt-5-nano` for targeted candidate assignment repair;
- `gpt-5-mini` for targeted semantic flow repair;
- no larger model unless measured evidence justifies escalation.

## Development gates

A proposed general code repair must pass, in order:

1. deterministic tests;
2. micro replay (2 difficult unresolved cases + 1 resolved control);
3. 8-paper canary;
4. the affected failure cohort;
5. the full 44-paper regression.

A failed gate stops the iteration. The expensive downstream gates do not run.

## Autonomous repair safeguards

The autonomous repair agent may change only an allowlisted set of extractor source/test files selected for the current recurring failure class.

It may not:

- edit benchmark, baseline, holdout or gold data;
- edit GitHub workflow/policy files;
- add DOI/title/author-specific logic;
- encode expected benchmark answers;
- weaken validation thresholds or evidence requirements;
- invent LCI quantities, units, geography, datasets or processes;
- automatically merge to `main`.

A patch that passes all corpus gates is committed only to `inventory-recall-development`. A draft PR to `main` is opened or updated for human review.

Rejected repair proposals are retained as workflow evidence. The controller normalizes harmless unified-diff formatting variants, but disallowed paths or scientific-integrity violations remain hard failures.

## Current zero-token diagnostic signal

Analysis of the exact frozen state found overlapping recurring signals including:

- table-heavy unresolved evidence: 24 papers;
- candidate ambiguity: 28 papers;
- locked processes with no assigned candidates: 14 papers;
- cross-process assignment: 25 papers;
- sparse flow extraction: 16 papers.

These categories overlap. They are diagnostic signals, not ground-truth causal labels. The controller recomputes their ranking from the exact frozen state on every iteration rather than hard-coding a target paper.

## Human gold control

`python -m ai_lca.gold_review` generates a candidate-level review queue from the canary corpus without API calls. The queue remains `PENDING_HUMAN_REVIEW` until a person verifies the source-supported disposition/process/flow information.

Self-QC improvement is therefore useful for automated screening, but it is not treated as proof of true precision/recall. Human-checked labels are required before the final v1 claim.

## Workflows

- `inventory-development-ci.yml`: deterministic branch CI.
- `inventory-replay.yml`: manual diagnostics/micro/canary/cohort/full replay.
- `inventory-autonomous-development.yml`: one bounded autonomous repair every six hours, with progressive gates and no automatic merge.

The broad hourly literature acquisition loop is paused. Hermesmann and Gerloff are manual regression benchmarks rather than automatic development loops. New literature acquisition resumes only after the extractor has materially improved and the v1 candidate has been validated.
