from pathlib import Path

import pytest

from ai_lca.repair_agent import REQUIRED_PROMPT_INVARIANTS, _validate_replacement


def test_repair_validator_accepts_current_llm_prompt():
    current = Path("src/ai_lca/llm.py").read_text()
    _validate_replacement(current, current)


def test_repair_validator_rejects_removing_prompt_invariant():
    current = Path("src/ai_lca/llm.py").read_text()
    invariant = REQUIRED_PROMPT_INVARIANTS[-1]
    assert invariant in current.casefold()
    modified = current.replace(
        "Do not reclassify such tabulated components as background subprocesses.",
        "Keep tabulated components attached to the assessed foreground system.",
    )
    with pytest.raises(ValueError, match="required prompt invariants"):
        _validate_replacement(current, modified)
