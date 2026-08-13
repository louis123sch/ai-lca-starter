from ai_lca import autonomous_benchmark_loop as loop


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
