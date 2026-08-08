# Project map

Current implementation lives under `src/ai_lca/` and `app.py`.

Core workflow:

```text
multiple PDF/DOCX sources
        ↓
one evidence corpus
        ↓
process map
        ↓
human process review
        ↓
complete foreground inventory
        ├── technosphere exchanges
        ├── biosphere elementary flows
        └── production exchanges
        ↓
human inventory review
        ↓
Brightway matching
        ├── ecoinvent/background activities
        └── biosphere database flows
        ↓
human-approved mappings
```

The branch intentionally keeps process structure, exchange identity, lifecycle stage, source provenance, source-provided background mappings, and search controls as separate concepts.
