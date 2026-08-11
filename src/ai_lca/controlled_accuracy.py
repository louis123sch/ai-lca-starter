from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .autonomous_literature import (
    ASSIGN_SYSTEM_PROMPT, ApiRunner, Budget, CandidateAssignmentBatch,
    ProcessCandidateExtraction, RunConfig, StateStore, _candidate_payload,
    _load_model, _slug, _validate_assignments, _validate_process_extraction, _write_json,
)
from .corpus_diagnostics import BASELINE_MANIFEST, load_baseline_papers
from .inventory_replay import TargetedReplayProcessor, _paper_dir, _read

SELECTION = Path("benchmarks/accuracy_iteration_v1/selection.json")

RESULT = [re.compile(x, re.I) for x in (
    r"\bgwp\b", r"global warming", r"impact categor", r"environmental impact",
    r"life cycle impact", r"\blcia\b", r"acidification", r"eutrophication",
    r"toxicity", r"ozone depletion", r"photochemical", r"characteri[sz]",
    r"kg\s*(?:co2|co₂)[- ]?(?:eq|equiv)",
)]
METHOD = [re.compile(x, re.I) for x in (
    r"\ballocation\b", r"system expansion", r"\bsubstitution\b",
    r"attribution method", r"normalization", r"weighting",
)]
INVENTORY = [re.compile(x, re.I) for x in (
    r"\binventory\b", r"\blci\b", r"\binput(?:s)?\b", r"\boutput(?:s)?\b",
    r"consumption", r"demand", r"feedstock", r"raw material", r"electricity",
    r"natural gas", r"water", r"steam", r"diesel", r"fuel", r"steel",
    r"alumin", r"nickel", r"copper", r"cement", r"concrete", r"transport",
    r"freight", r"\bwaste\b", r"\bemission(?:s)?\b",
)]
RESULT_CONTEXT = [re.compile(x, re.I) for x in (
    r"impact result", r"environmental impact", r"impact categor",
    r"contribution .* impact", r"absolute .* impact", r"life cycle impact assessment",
)]
METHOD_CONTEXT = [re.compile(x, re.I) for x in (
    r"attribution method", r"allocation method", r"advantages and limitations",
    r"multifunctionality",
)]

ADDENDUM = """
Controlled accuracy experiment. The supplied rows were previously called modeled
inventory but are high-confidence result/method table candidates. Re-review them only.
LCIA result rows (GWP/CO2-eq, acidification, eutrophication, toxicity, characterized
impact scores, contribution-to-impact tables) are not LCI exchanges. Tables defining
or comparing allocation, substitution, system expansion, attribution, weighting or
normalization methods are not LCI exchanges. Do not apply this merely because CO2 or
an impact term appears: a direct source-supported elementary emission in an actual
LCI/input-output table can be modeled inventory. Do not alter the locked graph or
invent anything. Every supplied candidate gets exactly one disposition; when the
disposition is not_inventory, process_ids must be [].
"""


def _n(patterns, text: str) -> int:
    return sum(bool(p.search(text or "")) for p in patterns)


def candidate_risks(candidate: dict[str, Any], assignment: dict[str, Any]) -> list[str]:
    """High-confidence diagnostic rules only; this function never changes extraction."""
    if assignment.get("disposition") != "modeled_inventory" or candidate.get("evidence_type") != "table_row":
        return []
    evidence = str(candidate.get("evidence_text") or "")
    context = " ".join(str(candidate.get(k) or "") for k in ("context", "table", "source_location"))
    text = evidence + "\n" + context
    result, method, inv = _n(RESULT, text), _n(METHOD, text), _n(INVENTORY, text)
    risks = []
    if result >= 2 and _n(RESULT_CONTEXT, context) >= 1 and inv <= 1:
        risks.append("MODELED_LCIA_RESULT_TABLE_RISK")
    if method >= 2 and (_n(METHOD_CONTEXT, context) >= 1 or inv == 0):
        risks.append("MODELED_METHOD_TABLE_RISK")
    return risks


def _root(state: Path, paper: dict[str, Any]) -> Path:
    return _paper_dir(state, paper) / "extraction"


def target_ids(candidates: list[dict[str, Any]], assignments: list[dict[str, Any]]) -> list[str]:
    cmap = {str(c.get("candidate_id")): c for c in candidates if c.get("candidate_id")}
    return sorted({
        str(a["candidate_id"]) for a in assignments if a.get("candidate_id")
        and candidate_risks(cmap.get(str(a["candidate_id"]), {}), a)
    })


def diagnose(state: Path, manifest: Path = BASELINE_MANIFEST) -> dict[str, Any]:
    baseline_id, papers = load_baseline_papers(state, manifest)
    counts, affected, examples, rows = Counter(), defaultdict(set), defaultdict(list), []
    for paper in papers:
        root = _root(state, paper)
        qc = _read(root / "qc.json", {}) or {}
        cands = _read(root / "inventory_candidates.json", []) or []
        assigns = (_read(root / "assignments.json", {}) or {}).get("assignments") or []
        cmap = {str(c.get("candidate_id")): c for c in cands if c.get("candidate_id")}
        signals = Counter()
        for a in assigns:
            for signal in candidate_risks(cmap.get(str(a.get("candidate_id")), {}), a):
                signals[signal] += 1
                if len(examples[signal]) < 10:
                    c = cmap[str(a["candidate_id"])]
                    examples[signal].append({
                        "doi": paper.get("doi"), "candidate_id": a["candidate_id"],
                        "table": c.get("table"), "source_location": c.get("source_location"),
                        "snippet": " ".join(str(c.get("evidence_text") or "").split())[:400],
                    })
        unresolved = int(qc.get("ambiguous_or_missing_candidate_count") or 0)
        flows = int(qc.get("flow_count") or 0)
        if unresolved: signals["UNRESOLVED_CANDIDATES"] += unresolved
        if not flows: signals["ZERO_FLOW_EXTRACTION"] += 1
        if float(qc.get("candidate_coverage") or 0) < .95: signals["LOW_CANDIDATE_COVERAGE"] += 1
        if flows and float(qc.get("amount_coverage") or 0) < .8: signals["LOW_AMOUNT_COVERAGE"] += 1
        if flows and float(qc.get("unit_coverage") or 0) < .8: signals["LOW_UNIT_COVERAGE"] += 1
        doi = str(paper.get("doi") or "")
        for signal, n in signals.items(): counts[signal] += n; affected[signal].add(doi)
        rows.append({"doi": doi, "title": paper.get("title"), "baseline_status": paper.get("status"), "signals": dict(signals)})
    confidence = {"MODELED_LCIA_RESULT_TABLE_RISK":1.0,"MODELED_METHOD_TABLE_RISK":1.0,
                  "UNRESOLVED_CANDIDATES":.75,"ZERO_FLOW_EXTRACTION":.85,
                  "LOW_CANDIDATE_COVERAGE":.8,"LOW_AMOUNT_COVERAGE":.6,"LOW_UNIT_COVERAGE":.6}
    ranked=[]
    for signal, dois in affected.items():
        severity = 2.0 if signal.startswith("MODELED_") else 1.0
        ranked.append({"failure_signal":signal,"affected_papers":len(dois),"signal_count":counts[signal],
                       "priority_score":round(len(dois)*confidence.get(signal,.5)*severity,3),
                       "dois":sorted(dois),"examples":examples.get(signal,[])})
    ranked.sort(key=lambda x:(-x["priority_score"],-x["affected_papers"],x["failure_signal"]))
    semantic = sorted(set().union(*(affected[s] for s in ("MODELED_LCIA_RESULT_TABLE_RISK","MODELED_METHOD_TABLE_RISK"))))
    return {"baseline_id":baseline_id,"paper_count":len(rows),
            "status_counts":dict(Counter(str(p["status"]) for p in papers)),
            "failure_signal_ranking":ranked,
            "recommended_iteration_1_target":{
                "failure_class":"MODELED_NONINVENTORY_TABLE_RISK",
                "mechanism":"LCIA-result and methodology/comparison table rows are being classified as foreground inventory.",
                "affected_papers":len(semantic),"dois":semantic,
                "reason":"Source-verifiable contamination can make even a COMPLETE extraction scientifically wrong."
            },"papers":rows}


class Processor(TargetedReplayProcessor):
    """Re-review only diagnostic target assignments; make no flow-extraction model calls."""
    def _assign(self, doi, source_hash, structure, candidates, paper_dir):  # noqa: ANN001
        payload = _read(paper_dir / "extraction" / "assignments.json", {}) or {}
        baseline = CandidateAssignmentBatch.model_validate(payload)
        allowed = {p.process_id for p in structure.processes}
        accepted, missing = _validate_assignments(candidates, baseline, allowed)
        cmap = {str(c.candidate_id): c for c in candidates}
        ids = sorted({str(a.candidate_id) for a in accepted if candidate_risks(cmap[str(a.candidate_id)].as_dict(), a.model_dump())})
        exp = paper_dir / "extraction" / "accuracy_iteration_v1"
        _write_json(exp / "targets.json", {"doi":doi,"target_candidate_ids":ids})
        if not ids:
            _write_json(exp / "assignments.json", CandidateAssignmentBatch(assignments=accepted)); return accepted, missing
        chosen = set(ids); subset = [c for c in candidates if c.candidate_id in chosen]
        prompt = "LOCKED PROCESSES:\n" + json.dumps([
            {"process_id":p.process_id,"name":p.name,"stage":p.stage} for p in structure.processes
        ], ensure_ascii=False) + "\n\nREVIEW ONLY THESE FLAGGED CANDIDATES:\n" + json.dumps(_candidate_payload(subset), ensure_ascii=False)
        repair = self.api.parse(doi=doi, stage="candidate_assignment_accuracy_v1", model=self.config.screen_model,
                                reasoning_effort=self.config.screen_reasoning,
                                system_prompt=ASSIGN_SYSTEM_PROMPT+"\n\n"+ADDENDUM,
                                user_prompt=prompt, response_format=CandidateAssignmentBatch)
        revised, revised_missing = _validate_assignments(candidates, CandidateAssignmentBatch(
            assignments=[a for a in accepted if a.candidate_id not in chosen] + repair.assignments
        ), allowed)
        _write_json(exp / "assignments.json", CandidateAssignmentBatch(assignments=revised))
        return revised, revised_missing

    def _extract_process(self, doi, source_hash, process, allowed, assigned, paper_dir):  # noqa: ANN001
        path = paper_dir / "extraction" / "processes" / f"{_slug(process.process_id)}.json"
        baseline = _load_model(path, ProcessCandidateExtraction) if path.exists() else ProcessCandidateExtraction(process_id=process.process_id)
        cleaned, _, failures = _validate_process_extraction(process.process_id, assigned, baseline, allowed)
        failures = [f for f in failures if not f.startswith("flow referenced unknown candidate ")]
        return cleaned, failures


def _snapshot(state: Path, manifest: Path, dois: list[str]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    _, papers = load_baseline_papers(state, manifest); by_doi={str(p["doi"]):p for p in papers}; out={}
    for doi in dois:
        root=_root(state,by_doi[doi]); qc=_read(root/"qc.json",{}) or {}; c=_read(root/"inventory_candidates.json",[]) or []
        a=(_read(root/"assignments.json",{}) or {}).get("assignments") or []
        out[doi]={"target_ids":target_ids(c,a),"assignments":{str(x["candidate_id"]):x for x in a},
                  "ambiguity":int(qc.get("ambiguous_or_missing_candidate_count") or 0),
                  "coverage":float(qc.get("candidate_coverage") or 0),"flows":int(qc.get("flow_count") or 0)}
    return out, by_doi


def run(state: Path, scope: str, manifest: Path, selection: Path, max_calls: int, max_tokens: int, max_cost: float) -> dict[str, Any]:
    selected=_read(selection,{}) or {}; dois=[str(selected["single_doi"])] if scope=="single" else [str(x) for x in selected.get("canary_dois") or []]
    before,papers=_snapshot(state,manifest,dois)
    config=RunConfig(state_dir=state,screen_model=os.getenv("OPENAI_SCREEN_MODEL","gpt-5-nano"),core_model=os.getenv("OPENAI_MODEL","gpt-5-mini"),
                     max_concurrent_requests=1,max_process_workers=1,max_paper_workers=1,max_total_calls=max_calls,max_calls_per_paper=1,
                     max_total_tokens=max_tokens,max_estimated_cost_usd=max_cost,max_repair_calls_per_process=0,infrastructure_retries=1)
    store=StateStore(state); budget=Budget(config,store); processor=Processor(config,store,ApiRunner(config,store,budget),os.environ.get("SPRINGER_API_KEY",""))
    results=[processor.process(doi) for doi in dois]; rmap={str(r.get("doi")):r for r in results}; papers_out=[]
    for doi in dois:
        after=(_read(_root(state,papers[doi])/"accuracy_iteration_v1"/"assignments.json",{}) or {}).get("assignments") or []
        amap={str(a["candidate_id"]):a for a in after}; targets=set(before[doi]["target_ids"])
        corrected=sorted(x for x in targets if amap.get(x,{}).get("disposition")=="not_inventory" and not (amap.get(x,{}).get("process_ids") or []))
        outside=sorted(x for x,old in before[doi]["assignments"].items() if x not in targets and amap.get(x)!=old)
        new=rmap.get(doi,{})
        reasons=[]
        if len(corrected)!=len(targets): reasons.append("not all high-confidence false-positive targets were corrected")
        if outside: reasons.append("assignment outside target class changed")
        if int(new.get("ambiguous_or_missing_candidate_count") or 0)>before[doi]["ambiguity"]: reasons.append("ambiguity increased")
        if float(new.get("candidate_coverage") or 0)+1e-12<before[doi]["coverage"]: reasons.append("candidate coverage decreased")
        if not targets: reasons.append("selected paper had no target candidates")
        papers_out.append({"doi":doi,"target_count":len(targets),"corrected_target_count":len(corrected),"corrected_target_ids":corrected,
                           "outside_target_changes":outside,"before_flow_count":before[doi]["flows"],"after_flow_count":int(new.get("flow_count") or 0),
                           "pass":not reasons,"reasons":reasons})
    comparison={"pass_gate":bool(papers_out) and all(x["pass"] for x in papers_out),"papers":papers_out,
                "target_total":sum(x["target_count"] for x in papers_out),"corrected_total":sum(x["corrected_target_count"] for x in papers_out)}
    return {"experiment":"accuracy_iteration_v1_modeled_noninventory_tables","scope":scope,"dois":dois,"comparison":comparison,
            "usage":budget.summary(),"limits":{"max_total_calls":max_calls,"max_calls_per_paper":1,"max_total_tokens":max_tokens,"max_estimated_cost_usd":max_cost}}


def main() -> None:
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
    d=sub.add_parser("diagnose"); d.add_argument("--state-dir",type=Path,required=True); d.add_argument("--manifest",type=Path,default=BASELINE_MANIFEST); d.add_argument("--output",type=Path,required=True)
    r=sub.add_parser("run"); r.add_argument("--state-dir",type=Path,required=True); r.add_argument("--manifest",type=Path,default=BASELINE_MANIFEST); r.add_argument("--selection",type=Path,default=SELECTION); r.add_argument("--scope",choices=["single","canary"],default="single"); r.add_argument("--max-total-calls",type=int,default=10); r.add_argument("--max-total-tokens",type=int,default=200000); r.add_argument("--max-estimated-cost-usd",type=float,default=.10); r.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.command=="diagnose": report=diagnose(args.state_dir,args.manifest)
    else: report=run(args.state_dir,args.scope,args.manifest,args.selection,args.max_total_calls,args.max_total_tokens,args.max_estimated_cost_usd)
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in report if k not in {"papers","failure_signal_ranking"}},indent=2,ensure_ascii=False))
    if args.command=="run" and not report["comparison"]["pass_gate"]: raise SystemExit(3)

if __name__=="__main__": main()
