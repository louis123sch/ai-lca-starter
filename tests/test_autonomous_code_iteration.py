from __future__ import annotations

import pytest

from ai_lca.autonomous_code_iteration import (
    PatchProposal,
    _changed_paths,
    _normalise_unified_diff,
    _validate_patch,
)


def test_changed_paths_reads_unified_diff_targets() -> None:
    diff = """--- a/src/ai_lca/jats.py
+++ b/src/ai_lca/jats.py
@@ -1 +1 @@
-old
+new
"""
    assert _changed_paths(diff) == {"src/ai_lca/jats.py"}


def test_changed_paths_accepts_bare_paths_and_markdown_fence() -> None:
    diff = """```diff
--- src/ai_lca/jats.py
+++ src/ai_lca/jats.py
@@ -1 +1 @@
-old
+new
```
"""
    normalised = _normalise_unified_diff(diff)
    assert normalised.startswith("--- a/src/ai_lca/jats.py\n+++ b/src/ai_lca/jats.py\n")
    assert "```" not in normalised
    assert _changed_paths(normalised) == {"src/ai_lca/jats.py"}


def test_changed_paths_accepts_diff_git_header_fallback() -> None:
    diff = """diff --git a/src/ai_lca/jats.py b/src/ai_lca/jats.py
@@ -1 +1 @@
-old
+new
"""
    assert _changed_paths(diff) == {"src/ai_lca/jats.py"}


def test_guardrail_blocks_benchmark_specific_patch() -> None:
    proposal = PatchProposal(
        summary="bad",
        rationale="bad",
        files_touched=["src/ai_lca/jats.py"],
        unified_diff="""--- a/src/ai_lca/jats.py
+++ b/src/ai_lca/jats.py
@@ -1 +1 @@
-old
+SPECIAL_DOI = '10.1007/example'
""",
    )
    with pytest.raises(ValueError, match="anti-overfitting"):
        _validate_patch(proposal, {"src/ai_lca/jats.py"})


def test_guardrail_blocks_disallowed_path() -> None:
    proposal = PatchProposal(
        summary="bad",
        rationale="bad",
        files_touched=["benchmarks/gold.json"],
        unified_diff="""--- a/benchmarks/gold.json
+++ b/benchmarks/gold.json
@@ -1 +1 @@
-old
+new
""",
    )
    with pytest.raises(ValueError, match="disallowed"):
        _validate_patch(proposal, {"src/ai_lca/jats.py"})
