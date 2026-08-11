from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_lca import autonomous_code_iteration as iteration
from ai_lca.autonomous_code_iteration import (
    PatchProposal,
    _anonymized_failure_metrics,
    _changed_paths,
    _normalise_unified_diff,
    _repair_bare_hunk_headers,
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


def test_normalise_strips_apply_patch_footer() -> None:
    diff = """--- a/src/ai_lca/jats.py
+++ b/src/ai_lca/jats.py
@@
-old
+new
*** End Patch
"""
    normalised = _normalise_unified_diff(diff)
    assert "*** End Patch" not in normalised


def test_repair_bare_hunk_header_from_unique_source_context(tmp_path) -> None:
    source = tmp_path / "src" / "ai_lca" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    diff = """--- a/src/ai_lca/example.py
+++ b/src/ai_lca/example.py
@@
 alpha
-beta
+BETA
 gamma
"""
    repaired = _repair_bare_hunk_headers(diff, root=tmp_path)
    assert "@@ -1,3 +1,3 @@" in repaired


def test_repair_numbered_hunk_with_wrong_location_from_unique_context(tmp_path) -> None:
    source = tmp_path / "src" / "ai_lca" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("zero\none\ntwo\nthree\n", encoding="utf-8")
    diff = """--- a/src/ai_lca/example.py
+++ b/src/ai_lca/example.py
@@ -99,3 +99,3 @@
 one
-two
+TWO
 three
"""
    repaired = _repair_bare_hunk_headers(diff, root=tmp_path)
    assert "@@ -2,3 +2,3 @@" in repaired
    assert "@@ -99,3 +99,3 @@" not in repaired


def test_repair_bare_hunk_rejects_ambiguous_context(tmp_path) -> None:
    source = tmp_path / "src" / "ai_lca" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("same\nx\nsame\ny\n", encoding="utf-8")
    diff = """--- a/src/ai_lca/example.py
+++ b/src/ai_lca/example.py
@@
 same
+extra
"""
    with pytest.raises(ValueError, match="found 2"):
        _repair_bare_hunk_headers(diff, root=tmp_path)


def test_guardrail_blocks_benchmark_specific_patch() -> None:
    proposal = PatchProposal(
        summary="bad",
        rationale="bad",
        files_touched=["src/ai_lca/jats.py", "tests/test_jats.py"],
        unified_diff="""--- a/src/ai_lca/jats.py
+++ b/src/ai_lca/jats.py
@@ -1 +1 @@
-old
+SPECIAL_DOI = '10.1007/example'
--- a/tests/test_jats.py
+++ b/tests/test_jats.py
@@ -1 +1 @@
-old
+new
""",
    )
    with pytest.raises(ValueError, match="anti-overfitting"):
        _validate_patch(
            proposal, {"src/ai_lca/jats.py", "tests/test_jats.py"}
        )


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


def test_guardrail_requires_test_for_source_behavior_change() -> None:
    proposal = PatchProposal(
        summary="missing test",
        rationale="bad",
        files_touched=["src/ai_lca/jats.py"],
        unified_diff="""--- a/src/ai_lca/jats.py
+++ b/src/ai_lca/jats.py
@@ -1 +1 @@
-old
+new
""",
    )
    with pytest.raises(ValueError, match="must include a test-file change"):
        _validate_patch(
            proposal, {"src/ai_lca/jats.py", "tests/test_jats.py"}
        )


def test_anonymized_failure_metrics_excludes_paper_identity() -> None:
    diagnostics = {
        "papers": [
            {
                "doi": "10.1007/should-not-leak",
                "title": "Secret paper title",
                "failure_classes": ["INCOMPLETE_CANDIDATE_REVIEW"],
                "process_count": 2,
                "candidate_count": 12,
                "modeled_candidate_count": 5,
                "candidate_coverage": 0.42,
                "ambiguous_or_missing_candidate_count": 7,
                "flow_count": 5,
                "amount_coverage": 0.8,
                "unit_coverage": 1.0,
                "evidence_type_counts": {"table_row": 12},
                "multi_process_assignment_count": 0,
                "duplicate_process_reference_count": 0,
            }
        ]
    }
    rows = _anonymized_failure_metrics(diagnostics, "INCOMPLETE_CANDIDATE_REVIEW")
    assert rows == [
        {
            "process_count": 2,
            "candidate_count": 12,
            "modeled_candidate_count": 5,
            "candidate_coverage": 0.42,
            "ambiguous_or_missing_candidate_count": 7,
            "flow_count": 5,
            "amount_coverage": 0.8,
            "unit_coverage": 1.0,
            "evidence_type_counts": {"table_row": 12},
            "multi_process_assignment_count": 0,
            "duplicate_process_reference_count": 0,
        }
    ]
    assert "doi" not in rows[0]
    assert "title" not in rows[0]


def test_noop_proposals_retry_then_exit_cleanly(tmp_path, monkeypatch) -> None:
    diagnostics_path = tmp_path / "diagnostics.json"
    diagnostics_path.write_text(
        """{
  "failure_class_ranking": [
    {
      "failure_class": "INCOMPLETE_CANDIDATE_REVIEW",
      "affected_papers": 34,
      "priority_score": 68.0
    }
  ],
  "papers": []
}
""",
        encoding="utf-8",
    )

    proposal = PatchProposal(
        summary="no safe patch",
        rationale="insufficient basis",
        files_touched=[],
        unified_diff="",
    )

    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def parse(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=proposal))]
            )

    completions = FakeCompletions()
    fake_client = SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=completions))
    )
    monkeypatch.setattr(iteration, "_client", lambda: fake_client)
    monkeypatch.chdir(tmp_path)

    args = SimpleNamespace(
        diagnostics=diagnostics_path,
        failure_class="INCOMPLETE_CANDIDATE_REVIEW",
        model="test-model",
        reasoning_effort="low",
        max_context_chars=1000,
        max_proposal_attempts=3,
        output_dir=tmp_path / "artifacts",
    )
    result = iteration.propose_and_apply(args)

    assert completions.calls == 3
    assert result["applied"] is False
    assert result["proposal_attempts"] == 3
    assert len(result["rejection_reasons"]) == 3
    assert all(
        "proposal contains no changed file paths" in reason
        for reason in result["rejection_reasons"]
    )
    assert (tmp_path / "artifacts" / "iteration.json").exists()
