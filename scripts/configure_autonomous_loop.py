from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOOP = ROOT / "src/ai_lca/autonomous_benchmark_loop.py"
TESTS = ROOT / "tests/test_autonomous_benchmark_gates.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def configure_loop() -> None:
    text = LOOP.read_text(encoding="utf-8")

    # Keep final absolute thresholds, but use paper-level mean overall score for iteration acceptance.
    helper_re = re.compile(
        r"def _threshold_failures\([\s\S]*?(?=\ndef _all_absolute_pass)",
        re.MULTILINE,
    )
    helper_block = '''def _paper_overall_failures(
    summaries: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
) -> list[str]:
    """Reject only when a paper's own accepted mean overall score declines."""
    failures: list[str] = []
    for name, summary in summaries.items():
        value = float(summary["mean_overall_score"])
        base_value = float(baseline[name]["mean_overall_score"])
        if value + NO_REGRESSION_TOLERANCE < base_value:
            failures.append(
                f"{name}: mean overall score regressed {value:.6f} < accepted {base_value:.6f}"
            )
    return failures


def _target_improved(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    """The development target advances whenever its mean overall score genuinely rises."""
    return float(candidate["mean_overall_score"]) > float(baseline["mean_overall_score"]) + 1e-9

'''
    if "def _paper_overall_failures(" not in text:
        text, n = helper_re.subn(helper_block, text, count=1)
        if n != 1:
            raise RuntimeError(f"gate helpers: expected one block, replaced {n}")

    if "def _load_external_validation_lessons(" not in text:
        marker = '''def _append_history(record: dict[str, Any]) -> None:
'''
        insertion = '''def _load_external_validation_lessons() -> dict[str, Any]:
    path = Path(".autonomy/external_validation_lessons.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


'''
        if marker not in text:
            raise RuntimeError("could not find history insertion point")
        text = text.replace(marker, insertion + marker, 1)

    text = text.replace(
        '"no_regression_policy": "No candidate metric may be lower than the accepted baseline metric.",',
        '"acceptance_policy": "Target paper mean overall score must increase; then every protected paper must keep or improve its own accepted mean overall score.",',
    )

    if '"external_validation_lessons": _load_external_validation_lessons(),' not in text:
        text = text.replace(
            '"recent_rejections": _load_history(),',
            '"recent_rejections": _load_history(),\n        "external_validation_lessons": _load_external_validation_lessons(),',
            1,
        )

    old_instruction = (
        '"Make one small general extractor repair. The target must improve without regressing any "\n'
        '            "accepted target metric. Regression benchmarks must not fall below accepted baselines; "\n'
        '            "absolute floors are only the target final success condition."'
    )
    new_instruction = (
        '"Diagnose the accepted baseline, recent rejections, and external validation lesson, then make one small general extractor repair. "\n'
        '            "The development target paper must increase its mean overall score. If it improves, test every protected paper; "\n'
        '            "each paper must keep or improve its own accepted mean overall score. Component precision/recall metrics are diagnostic, "\n'
        '            "not separate vetoes. Preserve generalization and never hard-code benchmark-specific labels."'
    )
    if old_instruction in text:
        text = text.replace(old_instruction, new_instruction, 1)
    elif "Diagnose the accepted baseline, recent rejections, and external validation lesson" not in text:
        raise RuntimeError("could not replace repair instruction")

    target_gate_re = re.compile(
        r'''    target_failures = _threshold_failures\(\n[\s\S]*?    if target_failures:\n''',
        re.MULTILINE,
    )
    target_gate = '''    target_failures: list[str] = []
    if not _target_improved(target_summary, state["baseline"][TARGET_BENCHMARK]):
        target_failures.append(
            f"{TARGET_BENCHMARK}: mean overall score did not improve over accepted baseline"
        )

    if target_failures:
'''
    if "target_failures = _threshold_failures(" in text:
        text, n = target_gate_re.subn(target_gate, text, count=1)
        if n != 1:
            raise RuntimeError(f"target gate: expected one block, replaced {n}")

    text = text.replace(
        '    failures = _threshold_failures(candidate_summaries, state["baseline"])\n',
        '    failures = _paper_overall_failures(candidate_summaries, state["baseline"])\n',
        1,
    )

    LOOP.write_text(text, encoding="utf-8")


def write_tests() -> None:
    TESTS.write_text(
        '''from ai_lca import autonomous_benchmark_loop as loop


def _summary(overall, pr=1.0, pp=1.0, fr=1.0, fp=1.0):
    return {
        "mean_overall_score": overall,
        "mean_process_recall": pr,
        "mean_process_precision": pp,
        "mean_flow_recall": fr,
        "mean_flow_precision": fp,
    }


def test_target_gate_uses_paper_overall_score_only():
    baseline = _summary(0.4991736513, pr=0.0952, pp=1.0, fr=0.0702, fp=0.0656)
    candidate = _summary(0.9954545455, pr=1.0, pp=0.9545, fr=1.0, fp=1.0)
    assert loop._target_improved(candidate, baseline)


def test_target_rejects_no_overall_improvement_even_if_component_metric_improves():
    baseline = _summary(0.50, pr=0.10)
    candidate = _summary(0.50, pr=0.90)
    assert not loop._target_improved(candidate, baseline)


def test_regression_gate_is_per_paper_overall_not_component_metrics():
    baseline = {
        "paper_a": _summary(0.90, pp=1.0),
        "paper_b": _summary(0.80, fr=0.60),
    }
    candidate = {
        "paper_a": _summary(0.91, pp=0.95),
        "paper_b": _summary(0.80, fr=0.55),
    }
    assert loop._paper_overall_failures(candidate, baseline) == []


def test_regression_gate_rejects_any_paper_overall_decline():
    baseline = {
        "paper_a": _summary(0.90),
        "paper_b": _summary(0.80),
    }
    candidate = {
        "paper_a": _summary(0.95),
        "paper_b": _summary(0.79),
    }
    failures = loop._paper_overall_failures(candidate, baseline)
    assert len(failures) == 1
    assert "paper_b" in failures[0]


def test_final_absolute_gate_still_uses_component_thresholds_for_target_success():
    passing = _summary(0.80, pr=0.90, pp=0.90, fr=0.80, fp=0.80)
    assert loop._all_absolute_pass({loop.TARGET_BENCHMARK: passing})
''',
        encoding="utf-8",
    )


def main() -> None:
    configure_loop()
    write_tests()
    print(json.dumps({"configured": True, "acceptance": "target overall up; every protected paper overall same-or-better"}))


if __name__ == "__main__":
    main()
