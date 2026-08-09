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

    # Remove every occurrence of the protected invariant. A valid repair may
    # repeat protected wording for emphasis, so the regression test must check
    # the validator's semantic invariant rather than assume a single occurrence.
    modified = current.replace(
        "do not reclassify such tabulated components as background subprocesses",
        "keep tabulated components attached to the assessed foreground system",
    )
    assert modified != current
    assert invariant not in modified.casefold()

    with pytest.raises(ValueError, match="required prompt invariants"):
        _validate_replacement(current, modified)
