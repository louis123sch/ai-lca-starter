# Corpus development v1

This directory defines development controls derived from the immutable 44-paper baseline.

- `micro.json`: two difficult unresolved cases plus one resolved regression control.
- `canary.json`: eight-paper fast gate spanning leading recurring failure signals and two resolved controls.

These are development/regression sets, not blind validation.

Human-checked labels are deliberately not fabricated. Use `python -m ai_lca.gold_review --state-dir literature_state` to create a `PENDING_HUMAN_REVIEW` queue from the frozen source evidence.
