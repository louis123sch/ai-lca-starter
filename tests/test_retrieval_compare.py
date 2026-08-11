from ai_lca.retrieval_compare import compare_reports


def report(*, status="UNRESOLVED_INVENTORY", ambiguity=2, flows=3, tokens=100, cost=0.1, audit=None):
    payload = {
        "results": [
            {
                "doi": "10.0000/example",
                "status": status,
                "ambiguous_or_missing_candidate_count": ambiguity,
                "flow_count": flows,
            }
        ],
        "usage": {
            "calls_this_run": 2,
            "tokens_this_run": tokens,
            "estimated_cost_this_run_usd": cost,
        },
    }
    if audit is not None:
        payload["router_audit"] = audit
    return payload


def safe_audit(*, baseline_disagreements=0):
    return {
        "inventory_safety_pass": True,
        "unsafe_exclusion_count": 0,
        "baseline_modeled_disagreement_count": baseline_disagreements,
    }


def test_retrieval_gate_accepts_same_quality_with_lower_tokens():
    control = report(tokens=100, cost=0.10)
    routed = report(tokens=80, cost=0.08, audit=safe_audit())
    comparison = compare_reports(control, routed)
    assert comparison["pass_gate"] is True
    assert comparison["efficiency_noninferior"] is True


def test_baseline_disagreements_do_not_fail_structurally_safe_router():
    control = report(tokens=100, cost=0.10)
    routed = report(tokens=80, cost=0.08, audit=safe_audit(baseline_disagreements=12))
    comparison = compare_reports(control, routed)
    assert comparison["pass_gate"] is True
    assert comparison["router_safety_pass"] is True


def test_retrieval_gate_rejects_flow_regression_even_if_cheaper():
    control = report(flows=3, tokens=100, cost=0.10)
    routed = report(flows=2, tokens=60, cost=0.05, audit=safe_audit())
    comparison = compare_reports(control, routed)
    assert comparison["pass_gate"] is False
    assert comparison["regressions"]


def test_retrieval_gate_rejects_router_safety_failure():
    control = report()
    routed = report(
        tokens=80,
        cost=0.08,
        audit={
            "inventory_safety_pass": False,
            "unsafe_exclusion_count": 1,
            "baseline_modeled_disagreement_count": 0,
        },
    )
    comparison = compare_reports(control, routed)
    assert comparison["pass_gate"] is False
    assert comparison["router_safety_pass"] is False
