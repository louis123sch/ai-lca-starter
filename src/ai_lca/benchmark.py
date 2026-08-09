from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .documents import combine_document_texts
from .models import InventoryExtraction

BENCHMARK_EXTRA_INSTRUCTIONS = """Reconstruct only the foreground LCI actually modeled in the study.
Prioritize goal/scope, system-boundary figures, inventory-analysis sections, and explicit LCI tables over technology-review prose.
Do not turn internal boxes in process-flow diagrams into separate foreground processes unless the LCA separately inventories them.
Do not turn background ecoinvent dataset names into foreground flow names.
"""


@dataclass
class BenchmarkReport:
    benchmark_id: str
    expected_processes: int
    extracted_processes: int
    matched_processes: int
    unexpected_processes: list[str]
    missing_processes: list[str]
    forbidden_processes: list[str]
    expected_flows: int
    extracted_flows: int
    matched_flows: int
    unexpected_flows: list[str]
    missing_flows: list[str]
    forbidden_foreground_names: list[str]
    process_recall: float
    process_precision: float
    flow_recall: float
    flow_precision: float
    amount_accuracy: float
    unit_accuracy: float
    direction_accuracy: float
    functional_unit_accuracy: float
    system_boundary_accuracy: float
    geography_accuracy: float
    overall_score: float


def _norm(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    text = text.replace("₂", "2").replace("³", "3").replace("™", "")
    text = text.replace("–", "-").replace("—", "-")
    return " ".join(re.sub(r"[^a-z0-9%]+", " ", text).split())


def _score_name(name: str, aliases: list[str]) -> float:
    a = _norm(name)
    scores = []
    for alias in aliases:
        b = _norm(alias)
        if not a or not b:
            scores.append(0.0)
        elif a == b:
            scores.append(1.0)
        elif min(len(a), len(b)) >= 5 and (a in b or b in a):
            scores.append(0.93)
        else:
            scores.append(SequenceMatcher(None, a, b).ratio())
    return max(scores, default=0.0)


def _unique_match(expected, actual, score_fn, threshold):
    pairs = sorted(
        (
            (score_fn(exp, got), ei, ai)
            for ei, exp in enumerate(expected)
            for ai, got in enumerate(actual)
            if score_fn(exp, got) >= threshold
        ),
        reverse=True,
    )
    matches, used = {}, set()
    for _, ei, ai in pairs:
        if ei not in matches and ai not in used:
            matches[ei] = ai
            used.add(ai)
    return matches, [i for i in range(len(expected)) if i not in matches], [i for i in range(len(actual)) if i not in used]


def _unit_equal(expected, actual) -> bool:
    aliases = {
        "m3": {"m3", "cubic metre", "cubic meter"},
        "kg": {"kg", "kilogram", "kilograms"},
        "kwh": {"kwh", "kilowatt hour", "kilowatt hours"},
        "unit": {"unit", "units"},
    }
    e, a = _norm(expected), _norm(actual)
    for canonical, values in aliases.items():
        e = canonical if e in values else e
        a = canonical if a in values else a
    return e == a


def _amount_equal(expected, actual) -> bool:
    if expected is None:
        return actual is None
    if actual is None:
        return False
    return math.isclose(float(actual), float(expected), rel_tol=0.01, abs_tol=max(abs(float(expected)) * 1e-6, 1e-12))


def evaluate_extraction(extraction: InventoryExtraction, expected: dict[str, Any]) -> BenchmarkReport:
    exp_proc = expected["processes"]
    proc_matches, missing_pi, extra_pi = _unique_match(
        exp_proc,
        extraction.processes,
        lambda e, a: _score_name(a.name, e.get("aliases", [e["name"]])),
        0.60,
    )
    process_ids = {exp_proc[ei]["key"]: extraction.processes[ai].process_id for ei, ai in proc_matches.items()}
    actual_by_process: dict[str, list[Any]] = {}
    for flow in extraction.flows:
        actual_by_process.setdefault(flow.process_id, []).append(flow)

    exp_groups: dict[str, list[dict[str, Any]]] = {}
    for flow in expected["flows"]:
        exp_groups.setdefault(flow["process_key"], []).append(flow)

    matched_actual, missing_flows = set(), []
    amount_hits = unit_hits = direction_hits = matched_flows = 0
    global_index = {id(flow): i for i, flow in enumerate(extraction.flows)}
    for key, exp_flows in exp_groups.items():
        pid = process_ids.get(key)
        if not pid:
            missing_flows.extend(f"{key}: {f['name']}" for f in exp_flows)
            continue
        got_flows = actual_by_process.get(pid, [])
        matches, missing, _ = _unique_match(
            exp_flows,
            got_flows,
            lambda e, a: min(
                1.0,
                _score_name(a.name, e.get("aliases", [e["name"]]))
                + (0.03 if _unit_equal(e.get("unit"), a.unit) else 0)
                + (0.02 if _norm(e.get("direction")) == _norm(a.direction) else 0),
            ),
            0.62,
        )
        for ei, ai in matches.items():
            e, a = exp_flows[ei], got_flows[ai]
            matched_flows += 1
            matched_actual.add(global_index[id(a)])
            amount_hits += _amount_equal(e.get("amount"), a.amount)
            unit_hits += _unit_equal(e.get("unit"), a.unit)
            direction_hits += _norm(e.get("direction")) == _norm(a.direction)
        missing_flows.extend(f"{key}: {exp_flows[i]['name']}" for i in missing)

    unexpected_flows = [f"{f.process_id}: {f.name}" for i, f in enumerate(extraction.flows) if i not in matched_actual]
    forbidden_terms = [_norm(x) for x in expected.get("forbidden_process_terms", [])]
    forbidden_processes = []
    for process in extraction.processes:
        if any(term in _norm(process.name) for term in forbidden_terms if term):
            forbidden_processes.append(process.name)
        if process.parent_process_id:
            forbidden_processes.append(f"{process.name} (unexpected child process)")
    bad_flow_terms = [_norm(x) for x in expected.get("forbidden_foreground_name_terms", [])]
    forbidden_names = sorted({f.name for f in extraction.flows if any(term in _norm(f.name) for term in bad_flow_terms if term)})

    np, ngp = len(exp_proc), len(extraction.processes)
    nf, ngf = len(expected["flows"]), len(extraction.flows)
    pr = len(proc_matches) / np if np else 1.0
    pp = len(proc_matches) / ngp if ngp else 0.0
    fr = matched_flows / nf if nf else 1.0
    fp = matched_flows / ngf if ngf else 0.0
    aa = amount_hits / matched_flows if matched_flows else 0.0
    ua = unit_hits / matched_flows if matched_flows else 0.0
    da = direction_hits / matched_flows if matched_flows else 0.0
    context = expected.get("context", {})
    fu = _norm(extraction.functional_unit)
    terms = [_norm(x) for x in context.get("functional_unit_terms", [])]
    fua = sum(t in fu for t in terms) / len(terms) if terms else 1.0
    sba = float(_norm(context.get("system_boundary")) in _norm(extraction.study_context.system_boundary))
    ga = float(_norm(context.get("reference_geography")) in _norm(extraction.study_context.operational_geography))
    score = (
        .10 * pr + .10 * pp + .20 * fr + .20 * fp + .10 * aa + .05 * ua + .05 * da
        + .05 * fua + .05 * sba + .05 * ga + .03 * (not forbidden_processes) + .02 * (not forbidden_names)
    )
    return BenchmarkReport(
        benchmark_id=expected.get("benchmark_id", "unknown"),
        expected_processes=np,
        extracted_processes=ngp,
        matched_processes=len(proc_matches),
        unexpected_processes=[extraction.processes[i].name for i in extra_pi],
        missing_processes=[exp_proc[i]["name"] for i in missing_pi],
        forbidden_processes=sorted(set(forbidden_processes)),
        expected_flows=nf,
        extracted_flows=ngf,
        matched_flows=matched_flows,
        unexpected_flows=unexpected_flows,
        missing_flows=missing_flows,
        forbidden_foreground_names=forbidden_names,
        process_recall=pr,
        process_precision=pp,
        flow_recall=fr,
        flow_precision=fp,
        amount_accuracy=aa,
        unit_accuracy=ua,
        direction_accuracy=da,
        functional_unit_accuracy=fua,
        system_boundary_accuracy=sba,
        geography_accuracy=ga,
        overall_score=float(score),
    )


def report_to_dict(report: BenchmarkReport) -> dict[str, Any]:
    return asdict(report)


def format_report(report: BenchmarkReport) -> str:
    pct = lambda x: f"{100*x:5.1f}%"
    lines = [
        f"Benchmark: {report.benchmark_id}",
        f"Overall score:       {pct(report.overall_score)}",
        f"Process recall:      {pct(report.process_recall)} ({report.matched_processes}/{report.expected_processes})",
        f"Process precision:   {pct(report.process_precision)} ({report.matched_processes}/{report.extracted_processes})",
        f"Flow recall:         {pct(report.flow_recall)} ({report.matched_flows}/{report.expected_flows})",
        f"Flow precision:      {pct(report.flow_precision)} ({report.matched_flows}/{report.extracted_flows})",
        f"Amount accuracy:     {pct(report.amount_accuracy)}",
        f"Forbidden processes: {len(report.forbidden_processes)}",
        f"Dataset-name leakage:{len(report.forbidden_foreground_names):2d}",
    ]
    if report.missing_processes:
        lines.append("Missing processes: " + ", ".join(report.missing_processes))
    if report.unexpected_processes:
        lines.append("Unexpected processes: " + ", ".join(report.unexpected_processes))
    if report.forbidden_processes:
        lines.append("Over-decomposed: " + ", ".join(report.forbidden_processes))
    if report.missing_flows:
        lines.append("Missing flows: " + ", ".join(report.missing_flows[:12]) + (f" ... (+{len(report.missing_flows)-12})" if len(report.missing_flows) > 12 else ""))
    if report.unexpected_flows:
        lines.append("Unexpected flows: " + ", ".join(report.unexpected_flows[:12]) + (f" ... (+{len(report.unexpected_flows)-12})" if len(report.unexpected_flows) > 12 else ""))
    return "\n".join(lines)


def load_expected(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("flows_file") and "flows" not in data:
        with (path.parent / data["flows_file"]).open(newline="") as handle:
            data["flows"] = [
                {
                    "process_key": row["process_key"],
                    "name": row["name"],
                    "aliases": [x for x in row.get("aliases", "").split("|") if x],
                    "amount": float(row["amount"]) if row.get("amount") else None,
                    "unit": row.get("unit") or None,
                    "direction": row.get("direction") or "unknown",
                }
                for row in csv.DictReader(handle)
            ]
    if data.get("mappings_file") and "mapping_expectations" not in data:
        with (path.parent / data["mappings_file"]).open(newline="") as handle:
            data["mapping_expectations"] = list(csv.DictReader(handle))
    return data


def compare_published_gwi(calculated: dict[str, float], expected: dict[str, Any]) -> dict[str, Any]:
    targets = expected.get("published_gwi_reference_case_without_byproducts", {})
    rows = []
    for key, target in targets.items():
        value = calculated.get(key)
        err = None if value is None else 100 * (float(value) - float(target)) / float(target)
        rows.append({"process_key": key, "expected": target, "calculated": value, "percent_error": err})
    available = [abs(r["percent_error"]) for r in rows if r["percent_error"] is not None]
    return {
        "benchmark_id": expected.get("benchmark_id"),
        "matched_results": len(available),
        "expected_results": len(targets),
        "mean_absolute_percent_error": statistics.fmean(available) if available else None,
        "results": rows,
    }


def run_live_benchmark(source_paths: list[Path], expected_path: Path, *, runs: int, model: str | None, output_dir: Path):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Configure it before running the live benchmark.")
    from .llm import extract_inventory_from_text

    source_text = combine_document_texts([(p.name, p.read_bytes()) for p in source_paths])
    expected = load_expected(expected_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for n in range(1, runs + 1):
        extraction = extract_inventory_from_text(source_text, model=model, extra_instructions=BENCHMARK_EXTRA_INSTRUCTIONS)
        (output_dir / f"extraction_run_{n:02d}.json").write_text(extraction.model_dump_json(indent=2))
        report = evaluate_extraction(extraction, expected)
        (output_dir / f"report_run_{n:02d}.json").write_text(json.dumps(report_to_dict(report), indent=2) + "\n")
        reports.append(report)
        print(f"\n=== Run {n}/{runs} ===\n{format_report(report)}")
    summary = {
        "benchmark_id": expected.get("benchmark_id"),
        "runs": runs,
        "model": model or os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        "mean_overall_score": statistics.fmean(r.overall_score for r in reports),
        "min_overall_score": min(r.overall_score for r in reports),
        "mean_process_recall": statistics.fmean(r.process_recall for r in reports),
        "mean_process_precision": statistics.fmean(r.process_precision for r in reports),
        "mean_flow_recall": statistics.fmean(r.flow_recall for r in reports),
        "mean_flow_precision": statistics.fmean(r.flow_precision for r in reports),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== Aggregate ===\n" + json.dumps(summary, indent=2))
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-grounded AI-LCA benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--expected", required=True, type=Path)
    ev.add_argument("--extraction", required=True, type=Path)
    live = sub.add_parser("live")
    live.add_argument("--expected", required=True, type=Path)
    live.add_argument("--source", required=True, nargs="+", type=Path)
    live.add_argument("--runs", type=int, default=5)
    live.add_argument("--model")
    live.add_argument("--output-dir", type=Path, default=Path("benchmark_runs/hermesmann_2022"))
    res = sub.add_parser("results")
    res.add_argument("--expected", required=True, type=Path)
    res.add_argument("--calculated", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "evaluate":
        report = evaluate_extraction(InventoryExtraction.model_validate_json(args.extraction.read_text()), load_expected(args.expected))
        print(format_report(report))
    elif args.command == "live":
        run_live_benchmark(args.source, args.expected, runs=args.runs, model=args.model, output_dir=args.output_dir)
    else:
        print(json.dumps(compare_published_gwi(json.loads(args.calculated.read_text()), load_expected(args.expected)), indent=2))


if __name__ == "__main__":
    main()
