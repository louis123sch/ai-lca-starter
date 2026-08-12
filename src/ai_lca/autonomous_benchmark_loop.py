from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .autonomous_benchmark_iteration import propose_and_apply


BRANCH = "agent/autonomous-benchmark-loop"
STATE_PATH = Path(".autonomy/benchmark_loop_state.json")
HISTORY_PATH = Path(".autonomy/benchmark_rejection_history.jsonl")
ARTIFACTS = Path("artifacts/autonomous_benchmark_loop")

ABSOLUTE_THRESHOLDS = {
    "mean_overall_score": 0.80,
    "mean_process_recall": 0.90,
    "mean_process_precision": 0.90,
    "mean_flow_recall": 0.80,
    "mean_flow_precision": 0.80,
}
NO_REGRESSION_TOLERANCE = 0.0
TARGET_MIN_GAIN = 0.005

BENCHMARKS: dict[str, dict[str, Any]] = {
    "mycelium_2024": {
        "expected": "benchmarks/mycelium_2024/expected.json",
        "sources": ["benchmarks/mycelium_2024/source_excerpt.txt"],
    },
    "hermesmann_2022": {
        "expected": "benchmarks/hermesmann_2022/expected.json",
        "sources": [
            "benchmarks/hermesmann_2022/source_main_excerpt.txt",
            "benchmarks/hermesmann_2022/source_supplement_excerpt.txt",
        ],
    },
    "yang_2024": {
        "expected": "benchmarks/yang_2024/expected.json",
        "sources": ["benchmarks/yang_2024/source_excerpt.txt"],
    },
    "gonzales_calienes_2025": {
        "expected": "benchmarks/gonzales_calienes_2025/expected.json",
        "sources": ["benchmarks/gonzales_calienes_2025/source_excerpt.txt"],
    },
    "afzal_2023": {
        "expected": "benchmarks/afzal_2023/expected.json",
        "sources": ["benchmarks/afzal_2023/source_excerpt.txt"],
    },
    "terlouw_2021": {
        "expected": "benchmarks/terlouw_2021/expected.json",
        "sources": ["benchmarks/terlouw_2021/source_excerpt.txt"],
    },
}
TARGET_BENCHMARK = "mycelium_2024"
REGRESSION_BENCHMARKS = [
    "hermesmann_2022",
    "yang_2024",
    "gonzales_calienes_2025",
    "afzal_2023",
    "terlouw_2021",
]


def _run(
    cmd: list[str],
    *,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if proc.stdout:
        print(proc.stdout, flush=True)
    if proc.stderr:
        print(proc.stderr, flush=True)
    if check and proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], check=check)


def _bootstrap_evaluator_fix() -> bool:
    # Apply two general evaluator corrections already diagnosed from equivalent labels.
    path = Path("src/ai_lca/benchmark.py")
    original = path.read_text(encoding="utf-8")
    text = original
    old_substring = '''        elif min(len(a), len(b)) >= 5 and (a in b or b in a):
            scores.append(0.93)
        elif min(len(a_core), len(b_core)) >= 5 and (a_core in b_core or b_core in a_core):
            scores.append(0.91)
'''
    new_substring = '''        elif min(len(a), len(b)) >= 5 and (a in b or b in a):
            # Reward substring equivalence while preferring the more specific label.
            # This avoids assigning a qualified activity to a shorter generic activity
            # when both would otherwise receive an identical substring score.
            a_tokens, b_tokens = a.split(), b.split()
            closeness = min(len(a_tokens), len(b_tokens)) / max(len(a_tokens), len(b_tokens))
            scores.append(0.90 + 0.03 * closeness)
        elif min(len(a_core), len(b_core)) >= 5 and (a_core in b_core or b_core in a_core):
            a_tokens, b_tokens = a_core.split(), b_core.split()
            closeness = min(len(a_tokens), len(b_tokens)) / max(len(a_tokens), len(b_tokens))
            scores.append(0.88 + 0.03 * closeness)
'''
    if old_substring in text:
        text = text.replace(old_substring, new_substring, 1)
    elif new_substring not in text:
        raise RuntimeError("benchmark substring-matching block was not found")

    old_parent = '''        if process.parent_process_id:
            forbidden_processes.append(f"{process.name} (unexpected child process)")
'''
    new_parent = '''        if process.parent_process_id and not is_expected_match:
            forbidden_processes.append(f"{process.name} (unexpected child process)")
'''
    if old_parent in text:
        text = text.replace(old_parent, new_parent, 1)
    elif new_parent not in text:
        raise RuntimeError("benchmark child-process diagnostic block was not found")

    changed = text != original
    if changed:
        path.write_text(text, encoding="utf-8")

    test_path = Path("tests/test_benchmark.py")
    tests = test_path.read_text(encoding="utf-8")
    marker = "test_process_matching_prefers_more_specific_substring_candidate"
    if marker not in tests:
        tests += r'''


def test_process_matching_prefers_more_specific_substring_candidate():
    from types import SimpleNamespace

    from ai_lca.benchmark import _score_name, _unique_match

    expected = [
        {"name": "Transport to waste treatment", "aliases": ["Transport to waste treatment"]},
        {"name": "Waste treatment", "aliases": ["Waste treatment"]},
    ]
    actual = [
        SimpleNamespace(name="Transport to waste treatment (C2)"),
        SimpleNamespace(name="Waste treatment (C3)"),
    ]
    matches, missing, extra = _unique_match(
        expected,
        actual,
        lambda e, a: _score_name(a.name, e["aliases"]),
        0.60,
    )
    assert matches == {0: 0, 1: 1}
    assert missing == []
    assert extra == []


def test_expected_child_process_is_not_reported_as_unexpected_child():
    extraction = _perfect_extraction()
    extraction.processes[0].parent_process_id = "assessed_product_system"
    report = evaluate_extraction(extraction, _expected())
    assert not any("unexpected child process" in name for name in report.forbidden_processes)
'''
        test_path.write_text(tests, encoding="utf-8")
        changed = True
    return changed


def _benchmark_command(name: str, output_dir: Path) -> list[str]:
    spec = BENCHMARKS[name]
    return [
        "bash",
        "scripts/run_resilient_benchmark.sh",
        "python",
        "-m",
        "ai_lca.benchmark",
        "live",
        "--expected",
        spec["expected"],
        "--source",
        *spec["sources"],
        "--runs",
        "1",
        "--model",
        os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        "--output-dir",
        str(output_dir),
    ]


def _run_benchmark(name: str, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _run(_benchmark_command(name, output_dir))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    report = json.loads((output_dir / "report_run_01.json").read_text(encoding="utf-8"))
    return summary, report


def _report_excerpt(report: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "overall_score",
        "process_recall",
        "process_precision",
        "flow_recall",
        "flow_precision",
        "amount_accuracy",
        "unit_accuracy",
        "direction_accuracy",
        "missing_processes",
        "unexpected_processes",
        "missing_flows",
        "unexpected_flows",
        "forbidden_processes",
        "forbidden_foreground_names",
    ]
    excerpt: dict[str, Any] = {}
    for key in keys:
        value = report.get(key)
        if isinstance(value, list):
            value = value[:20]
        excerpt[key] = value
    return excerpt


def _load_history(limit: int = 8) -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    rows = []
    for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:]


def _append_history(record: dict[str, Any]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _threshold_failures(
    summaries: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
) -> list[str]:
    """Protect accepted metrics while permitting incremental progress to final floors."""
    failures: list[str] = []
    for name, summary in summaries.items():
        base = baseline[name]
        for metric, threshold in ABSOLUTE_THRESHOLDS.items():
            value = float(summary[metric])
            base_value = float(base[metric])
            if base_value >= threshold and value < threshold:
                failures.append(
                    f"{name}: lost achieved absolute floor for {metric}: "
                    f"{value:.4f} < {threshold:.4f}"
                )
            if value + NO_REGRESSION_TOLERANCE < base_value:
                failures.append(
                    f"{name}: regression {metric} {value:.4f} < accepted {base_value:.4f}"
                )
    return failures

def _target_improved(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    if float(candidate["mean_overall_score"]) >= float(baseline["mean_overall_score"]) + TARGET_MIN_GAIN:
        return True
    for metric in (
        "mean_process_recall",
        "mean_process_precision",
        "mean_flow_recall",
        "mean_flow_precision",
    ):
        if float(candidate[metric]) >= float(baseline[metric]) + 0.01:
            return True
    return False


def _all_absolute_pass(summaries: dict[str, dict[str, Any]]) -> bool:
    """Absolute thresholds are the final success condition for the development target."""
    summary = summaries[TARGET_BENCHMARK]
    return not any(
        float(summary[metric]) < threshold
        for metric, threshold in ABSOLUTE_THRESHOLDS.items()
    )

def _commit_and_push(message: str, *, include_code: bool) -> str | None:
    _git("config", "user.name", "ai-lca-autonomous-benchmark")
    _git("config", "user.email", "actions@users.noreply.github.com")
    if include_code:
        _git("add", "-A", "src/ai_lca", "tests", ".autonomy")
    else:
        _git("add", "-A", ".autonomy")
    if _git("diff", "--cached", "--quiet", check=False).returncode == 0:
        print("No changes to commit.")
        return None
    _git("commit", "-m", message)
    sha = _git("rev-parse", "HEAD").stdout.strip().splitlines()[-1]
    _git("push", "origin", f"HEAD:{BRANCH}")
    return sha


def _restore_candidate_changes() -> None:
    _git(
        "restore",
        "--",
        "src/ai_lca/llm.py",
        "src/ai_lca/structure.py",
        "src/ai_lca/models.py",
        "tests",
        check=False,
    )


def _initialise_state(max_iterations: int) -> None:
    print("No autonomous benchmark state found; establishing a fresh accepted baseline.")
    _bootstrap_evaluator_fix()
    _run(["python", "-m", "pytest", "-q"])
    baseline: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    root = ARTIFACTS / "baseline"
    for name in [TARGET_BENCHMARK, *REGRESSION_BENCHMARKS]:
        summary, report = _run_benchmark(name, root / name)
        baseline[name] = summary
        diagnostics[name] = _report_excerpt(report)

    state = {
        "version": 1,
        "status": "active",
        "iteration": 0,
        "max_iterations": max_iterations,
        "target_benchmark": TARGET_BENCHMARK,
        "blocking_regression_benchmarks": REGRESSION_BENCHMARKS,
        "absolute_thresholds": ABSOLUTE_THRESHOLDS,
        "no_regression_tolerance": NO_REGRESSION_TOLERANCE,
        "target_min_gain": TARGET_MIN_GAIN,
        "baseline": baseline,
        "diagnostics": diagnostics,
        "last_result": "baseline_initialized",
        "source_run_id": os.getenv("GITHUB_RUN_ID"),
    }
    _write_state(state)
    _append_history(
        {
            "kind": "baseline",
            "run_id": os.getenv("GITHUB_RUN_ID"),
            "iteration": 0,
            "absolute_gate_passed": _all_absolute_pass(baseline),
        }
    )
    _commit_and_push(
        "Initialize autonomous benchmark loop and corrected evaluator diagnostics",
        include_code=True,
    )


def _mark_terminal(state: dict[str, Any], status: str) -> None:
    state["status"] = status
    state["last_result"] = status
    _write_state(state)
    _append_history(
        {
            "kind": "terminal",
            "run_id": os.getenv("GITHUB_RUN_ID"),
            "iteration": state.get("iteration"),
            "status": status,
        }
    )
    _commit_and_push(f"Mark autonomous benchmark loop {status}", include_code=False)


def _iterate(state: dict[str, Any]) -> None:
    iteration = int(state["iteration"]) + 1
    run_root = ARTIFACTS / f"iteration_{iteration:02d}"
    run_root.mkdir(parents=True, exist_ok=True)

    diagnostics_payload = {
        "iteration": iteration,
        "development_target": TARGET_BENCHMARK,
        "absolute_thresholds": ABSOLUTE_THRESHOLDS,
        "no_regression_policy": "No candidate metric may be lower than the accepted baseline metric.",
        "accepted_baseline": state["baseline"],
        "accepted_diagnostics": state.get("diagnostics", {}),
        "recent_rejections": _load_history(),
        "instruction": (
            "Make one small general extractor repair. The target must improve without regressing any "
            "accepted target metric. Regression benchmarks must not fall below accepted baselines; "
            "absolute floors are only the target final success condition."
        ),
    }
    diagnostics_path = run_root / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics_payload, indent=2) + "\n", encoding="utf-8"
    )

    repair = propose_and_apply(
        diagnostics_path,
        run_root / "repair",
        model=os.getenv("OPENAI_REPAIR_MODEL", "gpt-5-mini"),
        reasoning_effort=os.getenv("OPENAI_REPAIR_REASONING_EFFORT", "medium"),
        max_context_chars=90000,
        max_attempts=3,
    )
    if not repair.get("applied", True) or not repair.get("changed_paths"):
        state["iteration"] = iteration
        state["last_result"] = "no_valid_patch"
        _write_state(state)
        _append_history(
            {
                "kind": "rejected",
                "run_id": os.getenv("GITHUB_RUN_ID"),
                "iteration": iteration,
                "reason": "no_valid_patch",
                "repair": repair,
            }
        )
        _restore_candidate_changes()
        _commit_and_push(
            f"Record autonomous benchmark rejection {iteration}: no valid patch",
            include_code=False,
        )
        return

    candidate_summaries: dict[str, dict[str, Any]] = {}
    candidate_reports: dict[str, dict[str, Any]] = {}

    target_summary, target_report = _run_benchmark(
        TARGET_BENCHMARK, run_root / "candidate" / TARGET_BENCHMARK
    )
    candidate_summaries[TARGET_BENCHMARK] = target_summary
    candidate_reports[TARGET_BENCHMARK] = _report_excerpt(target_report)

    target_failures = _threshold_failures(
        {TARGET_BENCHMARK: target_summary},
        {TARGET_BENCHMARK: state["baseline"][TARGET_BENCHMARK]},
    )
    if not _target_improved(target_summary, state["baseline"][TARGET_BENCHMARK]):
        target_failures.append(f"{TARGET_BENCHMARK}: target did not improve over accepted baseline")

    if target_failures:
        state["iteration"] = iteration
        state["last_result"] = "target_gate_rejected"
        _write_state(state)
        _append_history(
            {
                "kind": "rejected",
                "run_id": os.getenv("GITHUB_RUN_ID"),
                "iteration": iteration,
                "reason": "target_gate",
                "failures": target_failures,
                "repair_summary": repair.get("summary"),
                "repair_rationale": repair.get("rationale"),
                "target_summary": target_summary,
                "target_diagnostics": candidate_reports[TARGET_BENCHMARK],
            }
        )
        _restore_candidate_changes()
        _commit_and_push(
            f"Record autonomous benchmark rejection {iteration}: target gate",
            include_code=False,
        )
        return

    for name in REGRESSION_BENCHMARKS:
        summary, report = _run_benchmark(name, run_root / "candidate" / name)
        candidate_summaries[name] = summary
        candidate_reports[name] = _report_excerpt(report)

    failures = _threshold_failures(candidate_summaries, state["baseline"])
    if failures:
        state["iteration"] = iteration
        state["last_result"] = "regression_gate_rejected"
        _write_state(state)
        _append_history(
            {
                "kind": "rejected",
                "run_id": os.getenv("GITHUB_RUN_ID"),
                "iteration": iteration,
                "reason": "regression_gate",
                "failures": failures,
                "repair_summary": repair.get("summary"),
                "repair_rationale": repair.get("rationale"),
                "candidate_summaries": candidate_summaries,
                "candidate_diagnostics": candidate_reports,
            }
        )
        _restore_candidate_changes()
        _commit_and_push(
            f"Record autonomous benchmark rejection {iteration}: regression gate",
            include_code=False,
        )
        return

    state["iteration"] = iteration
    state["baseline"] = candidate_summaries
    state["diagnostics"] = candidate_reports
    state["last_result"] = "accepted"
    _write_state(state)
    _append_history(
        {
            "kind": "accepted",
            "run_id": os.getenv("GITHUB_RUN_ID"),
            "iteration": iteration,
            "repair_summary": repair.get("summary"),
            "repair_rationale": repair.get("rationale"),
            "candidate_summaries": candidate_summaries,
        }
    )
    message = f"Accept autonomous benchmark repair {iteration}: {repair.get('summary', 'general extractor improvement')}"
    _commit_and_push(message[:220], include_code=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression-controlled autonomous benchmark loop.")
    parser.add_argument("--max-iterations", type=int, default=8)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    _bootstrap_evaluator_fix()
    _run(["python", "-m", "pytest", "-q"])

    if not STATE_PATH.exists():
        _initialise_state(args.max_iterations)

    while True:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if state.get("status") != "active":
            print(f"Loop is not active: {state.get('status')}")
            return

        state["max_iterations"] = int(state.get("max_iterations", args.max_iterations))
        if _all_absolute_pass(state["baseline"]):
            _mark_terminal(state, "quality_gate_passed")
            return

        if int(state.get("iteration", 0)) >= int(state["max_iterations"]):
            _mark_terminal(state, "max_iterations_reached")
            return

        _iterate(state)


if __name__ == "__main__":
    main()
