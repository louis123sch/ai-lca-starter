from ai_lca.retrieval_compare import compare_reports


DOI = "10.0000/example"


def report(
    *,
    status="UNRESOLVED_INVENTORY",
    ambiguity=2,
    flows=3,
    coverage=1.0,
    process_count=1,
    tokens=100,
    cost=0.1,
    audit=None,
    safe_excluded=None,
    corrected_baseline=None,
):
    payload = {
        "dois": [DOI],
        "results": [
            {
                "doi": DOI,
                "status": status,
                "ambiguous_or_missing_candidate_count": ambiguity,
                "candidate_coverage": coverage,
                "process_count": process_count,
                "flow_count": flows,
                "retrieval_safe_excluded_candidate_ids": list(safe_excluded or []),
                "retrieval_corrected_baseline_modeled_candidate_ids": list(corrected_baseline or []),
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


def test_safe_lcia_flow_removal_is_an_improvement_not_a_regression():
    control = report(tokens=100, cost=0.10)
    routed = report(
        tokens=90,
        cost=0.09,
        audit=safe_audit(),
        safe_excluded=["cand_lcia"],
        corrected_baseline=["cand_lcia"],
    )
    comparison = compare_reports(
        control,
        routed,
        control_flow_candidates={DOI: {"cand_lci", "cand_lcia"}},
        routed_flow_candidates={DOI: {"cand_lci"}},
    )
    assert comparison["pass_gate"] is True
    assert comparison["quality_improved"] is True
    assert comparison["regressions"] == []
    assert comparison["improvements"][0]["corrected_lcia_flow_candidate_ids"] == ["cand_lcia"]


def test_unprotected_flow_candidate_loss_is_a_regression_even_if_cheaper():
    control = report(tokens=100, cost=0.10)
    routed = report(tokens=60, cost=0.05, audit=safe_audit())
    comparison = compare_reports(
        control,
        routed,
        control_flow_candidates={DOI: {"cand_lci"}},
        routed_flow_candidates={DOI: set()},
    )
    assert comparison["pass_gate"] is False
    assert comparison["regressions"]
    assert comparison["regressions"][0]["lost_unprotected_flow_candidate_ids"] == ["cand_lci"]


def test_retrieval_gate_rejects_candidate_coverage_regression():
    control = report(coverage=1.0)
    routed = report(coverage=0.9, tokens=80, cost=0.08, audit=safe_audit())
    comparison = compare_reports(control, routed)
    assert comparison["pass_gate"] is False
    assert "candidate coverage decreased" in comparison["regressions"][0]["reasons"]


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
