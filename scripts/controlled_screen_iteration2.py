from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_lca.autonomous_literature import ApiRunner, Budget, EligibilityDecision, RunConfig, StateStore, _slug
from ai_lca.jats import parse_jats_bytes

SCREEN_PROMPT_V2 = """You are a low-cost gate for an autonomous life-cycle assessment literature pipeline.
Decide whether the supplied article is a case/application LCA that is useful for reconstructing a physical/environmental foreground life-cycle inventory for a Brightway-style model.
Pass only when the article actually performs an LCA, assesses concrete product systems/processes/configurations, and contains enough evidence that a physical foreground inventory of material, energy, service, waste, or elementary-emission exchanges is plausibly reconstructable.
Reject reviews, editorials, corrections, award notices, and purely methodological discussions without a reconstructable case inventory.
A social life cycle assessment (S-LCA), social organizational LCA, social-risk/hotspot assessment, or product social impact study does NOT qualify merely because it calls stakeholder indicators, social scores, working-condition data, or monetary/social variables an 'inventory'. Reject a social-only study when it does not provide a reconstructable physical/environmental foreground LCI.
Do NOT reject a study merely because it also contains social assessment: a mixed environmental + social study should pass when its environmental LCA contains a reconstructable physical foreground inventory.
Do not invent missing information."""


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _screen_pass(decision: EligibilityDecision) -> bool:
    return bool(decision.eligible and decision.reconstructable_foreground and decision.likely_inventory)


def run(state_dir: Path, selection_path: Path, scope: str, output: Path, max_calls: int, max_tokens: int, max_cost: float) -> dict:
    selection = _read(selection_path)
    targets = set(selection["targets"])
    controls = set(selection["controls"])
    if scope == "single":
        dois = [selection["single_doi"]]
    else:
        dois = selection["targets"] + selection["controls"]

    config = RunConfig(
        state_dir=state_dir,
        screen_model="gpt-5-nano",
        core_model="gpt-5-mini",
        screen_reasoning="low",
        max_concurrent_requests=1,
        max_process_workers=1,
        max_paper_workers=1,
        max_total_calls=max_calls,
        max_calls_per_paper=1,
        max_total_tokens=max_tokens,
        max_estimated_cost_usd=max_cost,
        max_repair_calls_per_process=0,
        infrastructure_retries=1,
    )
    store = StateStore(state_dir)
    budget = Budget(config, store)
    api = ApiRunner(config, store, budget)

    rows = []
    for doi in dois:
        paper_dir = state_dir / "corpus" / _slug(doi)
        article = paper_dir / "source" / "article.xml"
        baseline_path = paper_dir / "extraction" / "screen.json"
        if not article.exists() or not baseline_path.exists():
            raise FileNotFoundError(f"missing frozen source/screen for {doi}")
        doc = parse_jats_bytes(article.read_bytes(), expected_doi=doi)
        baseline = EligibilityDecision.model_validate(_read(baseline_path))
        revised = api.parse(
            doi=doi,
            stage="screen_accuracy_v2",
            model="gpt-5-nano",
            reasoning_effort="low",
            system_prompt=SCREEN_PROMPT_V2,
            user_prompt=doc.screening_text(),
            response_format=EligibilityDecision,
        )
        baseline_pass = _screen_pass(baseline)
        revised_pass = _screen_pass(revised)
        expected = "reject_social_only" if doi in targets else "retain_environmental_lca"
        passed = (not revised_pass) if doi in targets else revised_pass
        rows.append({
            "doi": doi,
            "title": doc.title,
            "expected": expected,
            "baseline_pass": baseline_pass,
            "revised_pass": revised_pass,
            "pass": passed,
            "baseline": baseline.model_dump(),
            "revised": revised.model_dump(),
        })

    result = {
        "experiment": "accuracy_iteration_v2_social_only_screening",
        "scope": scope,
        "pass_gate": bool(rows) and all(row["pass"] for row in rows),
        "targets_rejected": sum(1 for row in rows if row["expected"] == "reject_social_only" and not row["revised_pass"]),
        "target_count": sum(1 for row in rows if row["expected"] == "reject_social_only"),
        "controls_retained": sum(1 for row in rows if row["expected"] == "retain_environmental_lca" and row["revised_pass"]),
        "control_count": sum(1 for row in rows if row["expected"] == "retain_environmental_lca"),
        "papers": rows,
        "usage": budget.summary(),
        "limits": {
            "max_total_calls": max_calls,
            "max_calls_per_paper": 1,
            "max_total_tokens": max_tokens,
            "max_estimated_cost_usd": max_cost,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=Path("benchmarks/accuracy_iteration_v2/selection.json"))
    parser.add_argument("--scope", choices=["single", "canary"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-total-calls", type=int, default=8)
    parser.add_argument("--max-total-tokens", type=int, default=100000)
    parser.add_argument("--max-estimated-cost-usd", type=float, default=0.01)
    args = parser.parse_args()
    result = run(args.state_dir, args.selection, args.scope, args.output, args.max_total_calls, args.max_total_tokens, args.max_estimated_cost_usd)
    raise SystemExit(0 if result["pass_gate"] else 3)


if __name__ == "__main__":
    main()
