from __future__ import annotations

import json
from pathlib import Path

from ai_lca.autonomous_continuous_iteration import (
    _history_context,
    _load_rejection_history,
)


def test_rejection_history_is_same_class_and_identity_free(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    records = [
        {
            "failure_class": "TABLE_HEAVY_UNRESOLVED",
            "summary": "bad table expansion",
            "rationale": "added too many candidates",
            "rejected_at_gate": "micro",
            "regression_reasons": ["ambiguity increased"],
            "doi": "10.1007/should-not-leak",
            "title": "Secret title",
        },
        {
            "failure_class": "CANDIDATE_AMBIGUITY",
            "summary": "different class",
            "rationale": "not relevant",
            "rejected_at_gate": "micro",
            "regression_reasons": ["regressed"],
        },
    ]
    history.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    loaded = _load_rejection_history(
        history, "TABLE_HEAVY_UNRESOLVED", max_entries=12
    )

    assert loaded == [
        {
            "summary": "bad table expansion",
            "rationale": "added too many candidates",
            "rejected_at_gate": "micro",
            "regression_reasons": ["ambiguity increased"],
        }
    ]
    rendered = _history_context(loaded)
    assert "bad table expansion" in rendered
    assert "Do not repeat" in rendered
    assert "10.1007/should-not-leak" not in rendered
    assert "Secret title" not in rendered


def test_rejection_history_keeps_only_latest_entries(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    history.write_text(
        "\n".join(
            json.dumps(
                {
                    "failure_class": "TABLE_HEAVY_UNRESOLVED",
                    "summary": f"attempt {index}",
                    "rationale": "reason",
                    "rejected_at_gate": "micro",
                    "regression_reasons": [],
                }
            )
            for index in range(5)
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = _load_rejection_history(
        history, "TABLE_HEAVY_UNRESOLVED", max_entries=2
    )
    assert [item["summary"] for item in loaded] == ["attempt 3", "attempt 4"]
