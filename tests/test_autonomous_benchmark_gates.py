from ai_lca import autonomous_benchmark_loop as loop

def _summary(overall, pr, pp, fr, fp):
    return {'mean_overall_score': overall, 'mean_process_recall': pr, 'mean_process_precision': pp, 'mean_flow_recall': fr, 'mean_flow_precision': fp}

def test_incremental_target_improvement_is_not_forced_to_final_floors():
    base = _summary(0.4991736513, 0.0952380952, 1.0, 0.0701754386, 0.0655737705)
    candidate = _summary(0.5334611118, 0.0952380952, 1.0, 0.0877192982, 0.0819672131)
    assert loop._threshold_failures({'mycelium_2024': candidate}, {'mycelium_2024': base}) == []
    assert loop._target_improved(candidate, base)

def test_target_metric_regression_is_rejected_even_when_other_metrics_improve():
    base = _summary(0.4991736513, 0.0952380952, 1.0, 0.0701754386, 0.0655737705)
    candidate = _summary(0.995, 1.0, 0.955, 1.0, 1.0)
    failures = loop._threshold_failures({'mycelium_2024': candidate}, {'mycelium_2024': base})
    assert any('mean_process_precision' in failure and 'regression' in failure for failure in failures)

def test_weak_regression_benchmark_is_protected_at_accepted_baseline_not_global_floor():
    base = _summary(0.8037537810, 1.0, 1.0, 0.6551724138, 0.6333333333)
    assert loop._threshold_failures({'terlouw_2021': dict(base)}, {'terlouw_2021': base}) == []
    candidate = dict(base)
    candidate['mean_flow_recall'] -= 0.001
    assert loop._threshold_failures({'terlouw_2021': candidate}, {'terlouw_2021': base})

def test_final_absolute_gate_applies_to_target_only():
    passing = _summary(0.80, 0.90, 0.90, 0.80, 0.80)
    assert loop._all_absolute_pass({'mycelium_2024': passing})
