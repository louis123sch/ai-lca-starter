# Corpus Baseline v1.1 — 2026-08-11

This is the corrected immutable development baseline derived directly from the frozen Phase 1 SQLite artifact (`artifact_id=9094408729`, workflow run `31473783816`).

The preceding `corpus_baseline_v1_2026-08-11` metadata summary correctly recorded 37 `UNRESOLVED_INVENTORY` + 7 `COMPLETE`, but its later-added `papers.json` manifest was not an exact export of those SQLite rows. It is retained as historical provenance and is superseded by this directory for all replay/development work.

Exact selection rule:

```sql
SELECT doi,title,status,source_hash,paper_dir,last_error
FROM papers
WHERE status IN ('COMPLETE','UNRESOLVED_INVENTORY')
```

Result:

- `UNRESOLVED_INVENTORY`: 37
- `COMPLETE`: 7
- development/regression papers: 44

These 44 papers are development/regression material after this freeze and must not be described as unseen validation evidence.

Do not modify this baseline directory after creation. Future corrections require a new versioned baseline.
