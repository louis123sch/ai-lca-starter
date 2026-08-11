# AI-LCA Paper Extractor: Corpus-Driven Development Plan

This repository follows a corpus-driven development strategy for Paper Extractor v1.

Core rules:

- `main` is the best validated product branch.
- `inventory-recall-development` is the experimental branch.
- The existing 44-paper corpus is development/regression material, not unseen validation.
- Preserve a frozen baseline before changing extraction behaviour.
- Prefer deterministic parsing and replay over repeated full-paper LLM calls.
- Reuse cached upstream stages whenever only a downstream component changes.
- Repair only unresolved or ambiguous candidates where possible.
- Evaluate changes through micro -> canary -> failure-cohort -> full-44 gates.
- Stop at the first failed gate.
- Measure both quality and efficiency (calls, tokens, estimated cost, runtime).
- Never weaken gold data, thresholds, or provenance to make an experiment pass.
- Never add paper-specific title/author rules.
- Development automation may experiment on the development branch, but promotion to `main` must occur through a pull request after the full regression gate.
- After corpus performance stabilises, freeze a candidate and validate on 8-12 untouched papers, with at least four frozen holdouts where practical.

## Immediate implementation sequence

1. Freeze the current 44-paper Phase 1 baseline.
2. Disable the obsolete hourly literature acquisition schedule.
3. Integrate Phase 1 into `main`.
4. Create `inventory-recall-development` from integrated `main`.
5. Add deterministic corpus diagnostics and failure classification.
6. Add stage-aware replay and unresolved-only repair.
7. Strengthen deterministic table/list enumeration.
8. Add canary and gold-development manifests.
9. Add micro, canary, cohort, and full-regression workflows with strict budgets.
10. Add automatic baseline-vs-candidate comparison and an autonomous iteration controller.
11. Iterate on the highest-value recurring failure class.
12. Open a PR to `main` only when a general change passes the full regression gate.
