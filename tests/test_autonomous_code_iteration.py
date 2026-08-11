from __future__ import annotations

from pathlib import Path

import pytest

from ai_lca.autonomous_code_iteration import (
    PatchProposal,
    _apply_exact_context_patch,
    _changed_paths,
    _has_unnumbered_hunks,
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


def test_exact_context_fallback_applies_unique_unnumbered_hunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "src" / "ai_lca" / "jats.py"
    target.parent.mkdir(parents=True)
    target.write_text("def f():\n    old = 1\n    keep = 2\n", encoding="utf-8")
    diff = """--- a/src/ai_lca/jats.py
+++ b/src/ai_lca/jats.py
@@
 def f():
-    old = 1
+    old = 3
     keep = 2
"""
    assert _has_unnumbered_hunks(diff)
    changed = _apply_exact_context_patch(diff)
    assert changed == ["src/ai_lca/jats.py"]
    assert target.read_text(encoding="utf-8") == "def f():\n    old = 3\n    keep = 2\n"


def test_exact_context_fallback_fails_closed_on_ambiguous_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "src" / "ai_lca" / "jats.py"
    target.parent.mkdir(parents=True)
    target.write_text("old = 1\nold = 1\n", encoding="utf-8")
    diff = """--- a/src/ai_lca/jats.py
+++ b/src/ai_lca/jats.py
@@
-old = 1
+old = 2
"""
    with pytest.raises(ValueError, match="matched 2 times"):
        _apply_exact_context_patch(diff)
    assert target.read_text(encoding="utf-8") == "old = 1\nold = 1\n"


def test_exact_context_fallback_rejects_unmarked_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "src" / "ai_lca" / "jats.py"
    target.parent.mkdir(parents=True)
    target.write_text("old = 1\n", encoding="utf-8")
    diff = """--- a/src/ai_lca/jats.py
+++ b/src/ai_lca/jats.py
@@
old = 1
+old = 2
"""
    with pytest.raises(ValueError, match="without a diff marker"):
        _apply_exact_context_patch(diff)


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
