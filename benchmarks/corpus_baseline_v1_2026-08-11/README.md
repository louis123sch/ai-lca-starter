# Corpus Baseline v1 — 2026-08-11

This directory records the immutable baseline for corpus-driven inventory recall development.

Source workflow run: `31473783816` (`Scheduled Phase 1 literature scale batches`, run #17).

Integrated product baseline immediately before development branch creation:

- Phase 1 integration merge: `c7d83ff02d42c7237af6cddfc56bae9e651ac112`
- Development-plan documentation commit: `b62413b2d779ce4435b2b4a5494380b3e842726e`
- Hourly Phase 1 scheduler paused: `93c41220d87c15f7d79d19d48cd38a7f93a5e33c`

Persisted Phase 1 state summary from the final productive corpus state:

- `COMPLETE`: 7
- `UNRESOLVED_INVENTORY`: 37
- `SCREEN_REJECTED`: 30
- `ACQUISITION_FAILED`: 27
- API calls recorded: 526
- Total tokens recorded: 4,385,673
- Cached tokens recorded: 82,176
- Estimated OpenAI API cost recorded: USD 2.559404
- `FLOW_ENUMERATION_FAILURE`: 37
- `SOURCE_ACQUISITION_FAILURE`: 27

The 44 papers consisting of the 37 unresolved inventory cases plus 7 currently resolved cases are the development/regression corpus. They are not an unseen validation set after this baseline.

Do not overwrite baseline-derived manifests or metrics. Candidate experiments must write to separate experiment state/results.
