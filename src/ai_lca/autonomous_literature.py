from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from .jats import InventoryCandidate, JATSDocument, parse_jats_bytes
from .literature import IJLCA_ONLINE_ISSN, iter_crossref_journal, rough_lca_relevance
from .llm import STRUCTURE_SYSTEM_PROMPT, _client
from .models import ForegroundInterpretation, ForegroundStructure, InventoryExtraction, InventoryFlow, SourceEvidence
from .runtime import build_extraction_provenance
from .springer_oa import fetch_jats_for_doi
from .structure import lock_foreground_interpretation

PIPELINE_VERSION = "phase1-autonomous-v1"
TERMINAL_STATUSES = {"COMPLETE", "SCREEN_REJECTED", "ACQUISITION_FAILED", "UNRESOLVED_STRUCTURE", "UNRESOLVED_INVENTORY"}

SCREEN_SYSTEM_PROMPT = """You are a low-cost gate for an autonomous life-cycle assessment literature pipeline.
Decide whether the supplied article is a case/application LCA that is useful for reconstructing a physical/environmental foreground life-cycle inventory for a Brightway-style model.
Pass only when the article actually performs an LCA, assesses concrete product systems/processes/configurations, and contains enough evidence that a physical foreground inventory of material, energy, service, waste, or elementary-emission exchanges is plausibly reconstructable.
Reject reviews, editorials, corrections, award notices, and purely methodological discussions without a reconstructable case inventory.
A social life cycle assessment (S-LCA), social organizational LCA, social-risk/hotspot assessment, or product social impact study does NOT qualify merely because it calls stakeholder indicators, social scores, working-condition data, or monetary/social variables an 'inventory'. Reject a social-only study when it does not provide a reconstructable physical/environmental foreground LCI.
Do NOT reject a study merely because it also contains social assessment: a mixed environmental + social study should pass when its environmental LCA contains a reconstructable physical foreground inventory.
Do not invent missing information."""

STRUCTURE_REPAIR_SYSTEM_PROMPT = """You are repairing an already attempted foreground-process interpretation for a life-cycle assessment.
Correct only the failed aspects using supplied source evidence. Reclassify existing source-supported candidates when possible instead of inventing entities. Do not promote components, equipment, inventory sections, life-cycle stages, or background supplies to foreground processes unless the source explicitly models them as independent foreground activities or assessed systems. Keep genuinely separate assessed alternatives. Preserve source-derived names and evidence. Never introduce unsupported ecoinvent detail. Return a complete ForegroundInterpretation."""

ASSIGN_SYSTEM_PROMPT = """You are assigning deterministically enumerated LCI evidence rows to an already locked foreground process graph. You may not add, split, merge, or rename processes.
For EVERY candidate_id supplied choose modeled_inventory, not_inventory, or ambiguous. A modeled_inventory row may belong to multiple supplied process IDs when it genuinely contains values for multiple assessed alternatives. Use only supplied process IDs.
Results/impact tables and descriptive parameters are not inventory. In particular, LCIA midpoint or endpoint result rows such as GWP/climate-change scores, acidification, eutrophication, toxicity, depletion/resource scores, or other characterized impact-category values must be not_inventory, even when reported per functional unit or per assessed alternative. Do not confuse a CO2-equivalent impact score with a direct elementary CO2 emission from an actual LCI input/output table.
When a candidate is not_inventory, process_ids must be empty. Candidate IDs are opaque source identifiers: copy only supplied candidate_ids exactly and never invent wildcard or placeholder IDs. Return exactly one assignment per supplied candidate_id and no others."""

FLOW_CANDIDATE_SYSTEM_PROMPT = """You are converting deterministically enumerated, source-supported inventory candidates into foreground LCI flows for ONE locked process. You may not change the process graph.
Review every candidate_id. For a candidate that supports modeled flows, return those flows preserving candidate_id. Otherwise place its ID in non_inventory_candidate_ids or ambiguous_candidate_ids. Never omit a candidate from all categories. Never invent a flow, amount, unit, basis, stage, dataset name, geography, or process. Preserve source flow names. Amounts and units must come from the supplied row/context; use null if absent. evidence_text must be copied from the supplied candidate evidence_text. Ordinary material/energy/service inputs are exchanges attached to this process, not new foreground processes."""


class EligibilityDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    eligible: bool
    study_type: Literal["case_lca", "methodology", "review", "editorial", "correction", "other"]
    reconstructable_foreground: bool
    likely_inventory: bool
    confidence: float = Field(ge=0, le=1)
    reason: str


class CandidateAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    disposition: Literal["modeled_inventory", "not_inventory", "ambiguous"]
    process_ids: list[str] = Field(default_factory=list)
    rationale: str


class CandidateAssignmentBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assignments: list[CandidateAssignment] = Field(default_factory=list)


class CandidateFlow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    name: str
    amount: float | None = None
    unit: str | None = None
    direction: Literal["input", "output", "emission", "unknown"] = "unknown"
    linked_process_id: str | None = None
    component_or_stage: str | None = None
    basis: str | None = None
    notes: str | None = None
    evidence_text: str


class ProcessCandidateExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    process_id: str
    flows: list[CandidateFlow] = Field(default_factory=list)
    non_inventory_candidate_ids: list[str] = Field(default_factory=list)
    ambiguous_candidate_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass
class ApiUsage:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    duration_seconds: float = 0.0


@dataclass
class RunConfig:
    state_dir: Path
    screen_model: str = "gpt-5-nano"
    core_model: str = "gpt-5-mini"
    screen_reasoning: str = "low"
    structure_reasoning: str = "medium"
    flow_reasoning: str = "low"
    max_concurrent_requests: int = 4
    max_process_workers: int = 4
    max_paper_workers: int = 3
    max_total_calls: int = 120
    max_calls_per_paper: int = 15
    max_total_tokens: int = 5_000_000
    max_estimated_cost_usd: float = 5.0
    max_repair_calls_per_process: int = 1
    infrastructure_retries: int = 3
    structure_max_chars: int = 120_000
    assignment_chunk_size: int = 60


class BudgetExceeded(RuntimeError):
    pass


_PRICE_PER_MILLION = {
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-mini-2025-08-07": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5-nano-2025-08-07": (0.05, 0.005, 0.40),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    return hashlib.sha1(value.strip().lower().encode("utf-8")).hexdigest()[:16]


def _normalise(text: str | None) -> str:
    return " ".join((text or "").casefold().split())


def _usage_from_completion(completion, model: str, duration: float) -> ApiUsage:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return ApiUsage(duration_seconds=duration)
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", prompt + completion_tokens) or 0)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    cached = int(getattr(prompt_details, "cached_tokens", 0) or 0)
    reasoning = int(getattr(completion_details, "reasoning_tokens", 0) or 0)
    input_price, cached_price, output_price = _PRICE_PER_MILLION.get(model, (0.0, 0.0, 0.0))
    cost = max(0, prompt - cached) * input_price / 1_000_000 + cached * cached_price / 1_000_000 + completion_tokens * output_price / 1_000_000
    return ApiUsage(prompt, cached, completion_tokens, reasoning, total, cost, duration)


class StateStore:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = state_dir / "phase1.sqlite3"
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self.conn:
            self.conn.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS papers (doi TEXT PRIMARY KEY,title TEXT,status TEXT NOT NULL,source_hash TEXT,paper_dir TEXT,last_error TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs (job_key TEXT PRIMARY KEY,doi TEXT NOT NULL,stage TEXT NOT NULL,process_id TEXT,status TEXT NOT NULL,attempts INTEGER NOT NULL DEFAULT 0,model TEXT,result_path TEXT,error TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS processes (doi TEXT NOT NULL,process_id TEXT NOT NULL,name TEXT NOT NULL,status TEXT NOT NULL,PRIMARY KEY(doi,process_id));
            CREATE TABLE IF NOT EXISTS inventory_candidates (doi TEXT NOT NULL,candidate_id TEXT NOT NULL,source_location TEXT,evidence_type TEXT,status TEXT,PRIMARY KEY(doi,candidate_id));
            CREATE TABLE IF NOT EXISTS failures (id INTEGER PRIMARY KEY AUTOINCREMENT,doi TEXT NOT NULL,stage TEXT NOT NULL,process_id TEXT,taxonomy TEXT NOT NULL,detail TEXT,created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS api_usage (id INTEGER PRIMARY KEY AUTOINCREMENT,doi TEXT NOT NULL,stage TEXT NOT NULL,process_id TEXT,model TEXT NOT NULL,prompt_tokens INTEGER NOT NULL,cached_tokens INTEGER NOT NULL,completion_tokens INTEGER NOT NULL,reasoning_tokens INTEGER NOT NULL,total_tokens INTEGER NOT NULL,estimated_cost_usd REAL NOT NULL,duration_seconds REAL NOT NULL,created_at TEXT NOT NULL);
            """)

    def paper_status(self, doi: str) -> str | None:
        with self._lock:
            row = self.conn.execute("SELECT status FROM papers WHERE doi=?", (doi,)).fetchone()
            return str(row["status"]) if row else None

    def upsert_paper(self, doi: str, *, title=None, status: str, source_hash=None, paper_dir=None, last_error=None) -> None:
        now = _utc_now()
        with self._lock, self.conn:
            self.conn.execute("""INSERT INTO papers(doi,title,status,source_hash,paper_dir,last_error,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(doi) DO UPDATE SET title=COALESCE(excluded.title,papers.title),status=excluded.status,source_hash=COALESCE(excluded.source_hash,papers.source_hash),paper_dir=COALESCE(excluded.paper_dir,papers.paper_dir),last_error=excluded.last_error,updated_at=excluded.updated_at""", (doi,title,status,source_hash,paper_dir,last_error,now,now))

    def completed_job_result(self, job_key: str) -> Path | None:
        with self._lock:
            row = self.conn.execute("SELECT result_path,status FROM jobs WHERE job_key=?", (job_key,)).fetchone()
            if not row or row["status"] != "complete" or not row["result_path"]:
                return None
            path = Path(str(row["result_path"]))
            return path if path.exists() else None

    def start_job(self, job_key: str, doi: str, stage: str, *, process_id=None, model=None) -> None:
        now = _utc_now()
        with self._lock, self.conn:
            self.conn.execute("""INSERT INTO jobs(job_key,doi,stage,process_id,status,attempts,model,created_at,updated_at) VALUES(?,?,?,?,'running',1,?,?,?)
            ON CONFLICT(job_key) DO UPDATE SET status='running',attempts=jobs.attempts+1,error=NULL,updated_at=excluded.updated_at""", (job_key,doi,stage,process_id,model,now,now))

    def complete_job(self, job_key: str, result_path: Path) -> None:
        with self._lock, self.conn:
            self.conn.execute("UPDATE jobs SET status='complete',result_path=?,updated_at=? WHERE job_key=?", (str(result_path),_utc_now(),job_key))

    def fail_job(self, job_key: str, error: str, retryable: bool=False) -> None:
        with self._lock, self.conn:
            self.conn.execute("UPDATE jobs SET status=?,error=?,updated_at=? WHERE job_key=?", ("retryable" if retryable else "failed",error,_utc_now(),job_key))

    def record_processes(self, doi: str, structure: ForegroundStructure) -> None:
        with self._lock, self.conn:
            for p in structure.processes:
                self.conn.execute("INSERT OR REPLACE INTO processes(doi,process_id,name,status) VALUES(?,?,?,?)", (doi,p.process_id,p.name,"locked"))

    def record_candidates(self, doi: str, candidates: list[InventoryCandidate]) -> None:
        with self._lock, self.conn:
            for c in candidates:
                self.conn.execute("INSERT OR REPLACE INTO inventory_candidates(doi,candidate_id,source_location,evidence_type,status) VALUES(?,?,?,?,?)", (doi,c.candidate_id,c.source_location,c.evidence_type,"enumerated"))

    def record_failure(self, doi: str, stage: str, taxonomy: str, detail: str, process_id=None) -> None:
        with self._lock, self.conn:
            self.conn.execute("INSERT INTO failures(doi,stage,process_id,taxonomy,detail,created_at) VALUES(?,?,?,?,?,?)", (doi,stage,process_id,taxonomy,detail,_utc_now()))

    def record_usage(self, doi: str, stage: str, model: str, usage: ApiUsage, process_id=None) -> None:
        with self._lock, self.conn:
            self.conn.execute("""INSERT INTO api_usage(doi,stage,process_id,model,prompt_tokens,cached_tokens,completion_tokens,reasoning_tokens,total_tokens,estimated_cost_usd,duration_seconds,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", (doi,stage,process_id,model,usage.prompt_tokens,usage.cached_tokens,usage.completion_tokens,usage.reasoning_tokens,usage.total_tokens,usage.estimated_cost_usd,usage.duration_seconds,_utc_now()))

    def paper_api_calls(self, doi: str) -> int:
        with self._lock:
            return int(self.conn.execute("SELECT COUNT(*) n FROM api_usage WHERE doi=?", (doi,)).fetchone()["n"])

    def aggregate(self) -> dict:
        with self._lock:
            statuses = {r["status"]:r["n"] for r in self.conn.execute("SELECT status,COUNT(*) n FROM papers GROUP BY status")}
            u = self.conn.execute("SELECT COUNT(*) calls,COALESCE(SUM(total_tokens),0) tokens,COALESCE(SUM(cached_tokens),0) cached,COALESCE(SUM(estimated_cost_usd),0) cost FROM api_usage").fetchone()
            failures = {r["taxonomy"]:r["n"] for r in self.conn.execute("SELECT taxonomy,COUNT(*) n FROM failures GROUP BY taxonomy ORDER BY n DESC")}
        return {"paper_statuses":statuses,"api_calls":int(u["calls"]),"total_tokens":int(u["tokens"]),"cached_tokens":int(u["cached"]),"estimated_cost_usd":round(float(u["cost"]),6),"failure_taxonomy":failures}


class Budget:
    def __init__(self, config: RunConfig, store: StateStore):
        self.config=config; self.store=store; self._lock=threading.Lock(); self.calls_this_run=0; self.tokens_this_run=0; self.cost_this_run=0.0; self.calls_by_paper={}
    def reserve_call(self, doi: str) -> None:
        with self._lock:
            if self.calls_this_run >= self.config.max_total_calls: raise BudgetExceeded("MAX_TOTAL_API_CALLS reached")
            if self.calls_by_paper.get(doi,0) >= self.config.max_calls_per_paper: raise BudgetExceeded(f"MAX_CALLS_PER_PAPER reached for {doi}")
            if self.tokens_this_run >= self.config.max_total_tokens: raise BudgetExceeded("MAX_TOTAL_TOKENS_PER_RUN reached")
            if self.cost_this_run >= self.config.max_estimated_cost_usd: raise BudgetExceeded("MAX_ESTIMATED_COST_USD reached")
            self.calls_this_run += 1; self.calls_by_paper[doi]=self.calls_by_paper.get(doi,0)+1
    def add_usage(self, usage: ApiUsage) -> None:
        with self._lock: self.tokens_this_run += usage.total_tokens; self.cost_this_run += usage.estimated_cost_usd
    def summary(self) -> dict:
        with self._lock: return {"calls_this_run":self.calls_this_run,"tokens_this_run":self.tokens_this_run,"estimated_cost_this_run_usd":round(self.cost_this_run,6)}


T = TypeVar("T", bound=BaseModel)
class ApiRunner:
    def __init__(self, config: RunConfig, store: StateStore, budget: Budget):
        self.config=config; self.store=store; self.budget=budget; self.client=_client(); self.semaphore=threading.BoundedSemaphore(max(1,config.max_concurrent_requests))
    def parse(self, *, doi: str, stage: str, model: str, system_prompt: str, user_prompt: str, response_format: type[T], reasoning_effort: str, process_id=None) -> T:
        last=None
        for attempt in range(1,self.config.infrastructure_retries+1):
            self.budget.reserve_call(doi); started=time.monotonic()
            try:
                with self.semaphore:
                    completion=self.client.beta.chat.completions.parse(model=model,reasoning_effort=reasoning_effort,messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],response_format=response_format)
                usage=_usage_from_completion(completion,model,time.monotonic()-started); self.budget.add_usage(usage); self.store.record_usage(doi,stage,model,usage,process_id)
                message=completion.choices[0].message
                if getattr(message,"refusal",None): raise RuntimeError(f"Model refusal: {message.refusal}")
                if message.parsed is None: raise RuntimeError("Model returned no parsed structured output")
                return message.parsed
            except BudgetExceeded: raise
            except Exception as exc:
                last=exc; name=type(exc).__name__.casefold(); retryable=any(x in name for x in ("ratelimit","timeout","connection","internalserver","apistatus"))
                if not retryable or attempt>=self.config.infrastructure_retries: raise
                time.sleep(min(20.0,2.0**(attempt-1)))
        raise last


def _job_key(doi, source_hash, stage, *, process_id, model, prompt_version):
    return hashlib.sha256("|".join([PIPELINE_VERSION,doi,source_hash,stage,process_id or "",model,prompt_version]).encode()).hexdigest()

def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True,exist_ok=True)
    text=payload.model_dump_json(indent=2) if isinstance(payload,BaseModel) else json.dumps(payload,ensure_ascii=False,indent=2)
    path.write_text(text+"\n",encoding="utf-8"); return path

def _load_model(path: Path, cls: type[T]) -> T: return cls.model_validate_json(path.read_text(encoding="utf-8"))
def _candidate_payload(candidates): return [c.as_dict() for c in candidates]

def _deterministic_screen_rejection(doc: JATSDocument):
    title=doc.title.casefold()
    if any(x in title for x in ("correction to","erratum","editorial","young scientist lca award")): return "obvious non-case article title"
    corpus=f"{doc.title} {doc.abstract}".casefold()
    if "life cycle" not in corpus and "lca" not in corpus and len(doc.inventory_candidates)<3: return "no clear LCA signal and fewer than three inventory candidates"
    return None

def structure_gate(structure: ForegroundStructure, source_text: str) -> list[str]:
    failures=[]
    if not structure.processes: return ["no locked foreground process"]
    ids=[p.process_id for p in structure.processes]; names=[_normalise(p.name) for p in structure.processes]
    if len(ids)!=len(set(ids)): failures.append("duplicate process IDs")
    if len(names)!=len(set(names)): failures.append("duplicate process names")
    for p in structure.processes:
        if not p.evidence: failures.append(f"{p.process_id}: no explicit source evidence")
    if "functional unit" in source_text.casefold() and not (structure.functional_unit or "").strip(): failures.append("source discusses a functional unit but extraction did not identify one")
    if len(structure.processes)>25: failures.append("implausibly large foreground graph (>25 processes); likely over-segmentation")
    return failures

def _validate_assignments(candidates, batch, allowed):
    known={c.candidate_id for c in candidates}; accepted={}; warnings=[]
    for a in batch.assignments:
        if a.candidate_id not in known: warnings.append(f"unknown candidate {a.candidate_id}"); continue
        valid=[p for p in a.process_ids if p in allowed]
        if a.disposition=="modeled_inventory" and not valid: a=a.model_copy(update={"disposition":"ambiguous","process_ids":[]})
        elif valid!=a.process_ids: a=a.model_copy(update={"process_ids":valid})
        accepted.setdefault(a.candidate_id,a)
    missing=[c.candidate_id for c in candidates if c.candidate_id not in accepted]
    return list(accepted.values()),missing

def _validate_process_extraction(process_id, assigned, extraction, allowed):
    cmap={c.candidate_id:c for c in assigned}; valid=[]; failures=[]; reviewed=set()
    for f in extraction.flows:
        c=cmap.get(f.candidate_id)
        if c is None: failures.append(f"flow referenced unknown candidate {f.candidate_id}"); continue
        if _normalise(f.evidence_text) not in _normalise(c.evidence_text): failures.append(f"{f.candidate_id}: unsupported evidence_text"); continue
        if f.linked_process_id and f.linked_process_id not in allowed: f=f.model_copy(update={"linked_process_id":None})
        valid.append(f); reviewed.add(f.candidate_id)
    noninv=[x for x in extraction.non_inventory_candidate_ids if x in cmap]; amb=[x for x in extraction.ambiguous_candidate_ids if x in cmap]
    reviewed.update(noninv); reviewed.update(amb); missing=[c.candidate_id for c in assigned if c.candidate_id not in reviewed]
    if missing: failures.append(f"{len(missing)} assigned candidate(s) were not reviewed")
    if amb: failures.append(f"{len(amb)} candidate(s) remain ambiguous")
    return extraction.model_copy(update={"process_id":process_id,"flows":valid,"non_inventory_candidate_ids":list(dict.fromkeys(noninv)),"ambiguous_candidate_ids":list(dict.fromkeys(amb))}),missing,failures

def _to_inventory_flow(process_id, flow, candidate):
    return InventoryFlow(process_id=process_id,name=flow.name,amount=flow.amount,unit=flow.unit,direction=flow.direction,linked_process_id=flow.linked_process_id,component_or_stage=flow.component_or_stage,basis=flow.basis,notes=flow.notes,evidence=SourceEvidence(document="JATS XML",table=candidate.table,section=candidate.context if candidate.evidence_type=="section_statement" else None,evidence_text=candidate.evidence_text))


class PaperProcessor:
    def __init__(self, config, store, api, springer_api_key): self.config=config; self.store=store; self.api=api; self.springer_api_key=springer_api_key
    def _paper_dir(self, doi): return self.config.state_dir/"corpus"/_slug(doi)
    def _acquire(self, doi):
        path=self._paper_dir(doi)/"source"/"article.xml"
        if path.exists(): data=path.read_bytes(); return data,path,hashlib.sha256(data).hexdigest()
        data=fetch_jats_for_doi(doi,self.springer_api_key)
        if data is None: raise FileNotFoundError("Springer OA API returned no JATS article")
        path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data); return data,path,hashlib.sha256(data).hexdigest()
    def _screen(self, doi, source_hash, doc, paper_dir):
        path=paper_dir/"extraction"/"screen.json"; key=_job_key(doi,source_hash,"screen",process_id=None,model=self.config.screen_model,prompt_version="screen-v2")
        cached=self.store.completed_job_result(key)
        if cached: return _load_model(cached,EligibilityDecision)
        self.store.start_job(key,doi,"screen",model=self.config.screen_model)
        try:
            result=self.api.parse(doi=doi,stage="screen",model=self.config.screen_model,reasoning_effort=self.config.screen_reasoning,system_prompt=SCREEN_SYSTEM_PROMPT,user_prompt=doc.screening_text(),response_format=EligibilityDecision)
            _write_json(path,result); self.store.complete_job(key,path); return result
        except Exception as exc: self.store.fail_job(key,str(exc)); raise
    def _structure(self, doi, source_hash, doc, paper_dir):
        path=paper_dir/"extraction"/"structure.json"; key=_job_key(doi,source_hash,"structure",process_id=None,model=self.config.core_model,prompt_version="structure-v1")
        cached=self.store.completed_job_result(key)
        if cached: return _load_model(cached,ForegroundStructure)
        self.store.start_job(key,doi,"structure",model=self.config.core_model); text=doc.structure_text(self.config.structure_max_chars)
        try:
            interp=self.api.parse(doi=doi,stage="structure",model=self.config.core_model,reasoning_effort=self.config.structure_reasoning,system_prompt=STRUCTURE_SYSTEM_PROMPT,user_prompt="Identify and classify process-like entities, study context, and source-supported reference products/units. Do not decide Brightway mappings.\n\nSOURCE MATERIAL:\n"+text,response_format=ForegroundInterpretation)
            structure=lock_foreground_interpretation(interp); failures=structure_gate(structure,text)
            if failures:
                repair=self.api.parse(doi=doi,stage="structure_repair",model=self.config.core_model,reasoning_effort=self.config.structure_reasoning,system_prompt=STRUCTURE_REPAIR_SYSTEM_PROMPT,user_prompt="FAILED CHECKS:\n- "+"\n- ".join(failures)+"\n\nCURRENT INTERPRETATION:\n"+structure.model_dump_json(indent=2)+"\n\nSOURCE MATERIAL:\n"+text,response_format=ForegroundInterpretation)
                structure=lock_foreground_interpretation(repair); failures=structure_gate(structure,text)
            if failures: raise ValueError("; ".join(failures))
            _write_json(path,structure); self.store.record_processes(doi,structure); self.store.complete_job(key,path); return structure
        except Exception as exc: self.store.fail_job(key,str(exc)); raise
    def _assignment_chunk(self, doi, source_hash, structure, candidates, idx, paper_dir):
        path=paper_dir/"extraction"/"assignments"/f"chunk_{idx:03d}.json"; key=_job_key(doi,source_hash,f"assign:{idx}",process_id=None,model=self.config.screen_model,prompt_version="candidate-assignment-v1")
        cached=self.store.completed_job_result(key)
        if cached: return _load_model(cached,CandidateAssignmentBatch)
        self.store.start_job(key,doi,f"assign:{idx}",model=self.config.screen_model)
        prompt="LOCKED PROCESSES:\n"+json.dumps([{"process_id":p.process_id,"name":p.name,"stage":p.stage} for p in structure.processes],ensure_ascii=False)+"\n\nCANDIDATES:\n"+json.dumps(_candidate_payload(candidates),ensure_ascii=False)
        try:
            result=self.api.parse(doi=doi,stage="candidate_assignment",model=self.config.screen_model,reasoning_effort=self.config.screen_reasoning,system_prompt=ASSIGN_SYSTEM_PROMPT,user_prompt=prompt,response_format=CandidateAssignmentBatch)
            _write_json(path,result); self.store.complete_job(key,path); return result
        except Exception as exc: self.store.fail_job(key,str(exc)); raise
    def _assign(self, doi, source_hash, structure, candidates, paper_dir):
        chunks=[candidates[i:i+self.config.assignment_chunk_size] for i in range(0,len(candidates),self.config.assignment_chunk_size)]; batches=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max(1,len(chunks)),self.config.max_concurrent_requests)) as pool:
            futures=[pool.submit(self._assignment_chunk,doi,source_hash,structure,ch,i+1,paper_dir) for i,ch in enumerate(chunks)]
            for f in concurrent.futures.as_completed(futures): batches.append(f.result())
        accepted,missing=_validate_assignments(candidates,CandidateAssignmentBatch(assignments=[a for b in batches for a in b.assignments]),{p.process_id for p in structure.processes})
        if missing:
            repair=self._assignment_chunk(doi,source_hash,structure,[c for c in candidates if c.candidate_id in set(missing)],999,paper_dir)
            accepted,missing=_validate_assignments(candidates,CandidateAssignmentBatch(assignments=accepted+repair.assignments),{p.process_id for p in structure.processes})
        return accepted,missing
    def _extract_process(self, doi, source_hash, process, allowed, assigned, paper_dir):
        path=paper_dir/"extraction"/"processes"/f"{_slug(process.process_id)}.json"; key=_job_key(doi,source_hash,"process_inventory",process_id=process.process_id,model=self.config.core_model,prompt_version="candidate-flow-v1")
        cached=self.store.completed_job_result(key)
        if cached:
            cleaned,_,failures=_validate_process_extraction(process.process_id,assigned,_load_model(cached,ProcessCandidateExtraction),allowed); return cleaned,failures
        self.store.start_job(key,doi,"process_inventory",process_id=process.process_id,model=self.config.core_model)
        prompt="LOCKED PROCESS:\n"+process.model_dump_json(indent=2)+"\n\nALL LOCKED IDS:\n"+json.dumps(sorted(allowed))+"\n\nASSIGNED CANDIDATES:\n"+json.dumps(_candidate_payload(assigned),ensure_ascii=False)
        try:
            result=self.api.parse(doi=doi,stage="process_inventory",process_id=process.process_id,model=self.config.core_model,reasoning_effort=self.config.flow_reasoning,system_prompt=FLOW_CANDIDATE_SYSTEM_PROMPT,user_prompt=prompt,response_format=ProcessCandidateExtraction)
            cleaned,missing,failures=_validate_process_extraction(process.process_id,assigned,result,allowed); unresolved=set(missing)|set(cleaned.ambiguous_candidate_ids)
            repairs=0
            while unresolved and repairs<self.config.max_repair_calls_per_process:
                repairs+=1; subset=[c for c in assigned if c.candidate_id in unresolved]
                repair=self.api.parse(doi=doi,stage="process_inventory_repair",process_id=process.process_id,model=self.config.core_model,reasoning_effort=self.config.flow_reasoning,system_prompt=FLOW_CANDIDATE_SYSTEM_PROMPT,user_prompt="LOCKED PROCESS:\n"+process.model_dump_json(indent=2)+"\n\nReview ONLY these previously unresolved candidates:\n"+json.dumps(_candidate_payload(subset),ensure_ascii=False),response_format=ProcessCandidateExtraction)
                kept=[f for f in cleaned.flows if f.candidate_id not in unresolved]; ni=[x for x in cleaned.non_inventory_candidate_ids if x not in unresolved]; am=[x for x in cleaned.ambiguous_candidate_ids if x not in unresolved]
                merged=ProcessCandidateExtraction(process_id=process.process_id,flows=kept+repair.flows,non_inventory_candidate_ids=ni+repair.non_inventory_candidate_ids,ambiguous_candidate_ids=am+repair.ambiguous_candidate_ids,warnings=cleaned.warnings+repair.warnings)
                cleaned,missing,failures=_validate_process_extraction(process.process_id,assigned,merged,allowed); unresolved=set(missing)|set(cleaned.ambiguous_candidate_ids)
            _write_json(path,cleaned); self.store.complete_job(key,path); return cleaned,failures
        except Exception as exc: self.store.fail_job(key,str(exc)); raise
    def process(self, doi: str) -> dict:
        doi=doi.strip().lower(); paper_dir=self._paper_dir(doi); paper_dir.mkdir(parents=True,exist_ok=True); self.store.upsert_paper(doi,status="DISCOVERED",paper_dir=str(paper_dir))
        try: data,_,source_hash=self._acquire(doi); doc=parse_jats_bytes(data,expected_doi=doi)
        except Exception as exc:
            self.store.upsert_paper(doi,status="ACQUISITION_FAILED",paper_dir=str(paper_dir),last_error=str(exc)); self.store.record_failure(doi,"acquisition","SOURCE_ACQUISITION_FAILURE",str(exc)); return {"doi":doi,"status":"ACQUISITION_FAILED","error":str(exc)}
        self.store.upsert_paper(doi,title=doc.title,status="ACQUIRED",source_hash=source_hash,paper_dir=str(paper_dir)); _write_json(paper_dir/"source"/"parsed_summary.json",{"doi":doc.doi,"title":doc.title,"abstract":doc.abstract,"section_titles":doc.section_titles,"table_count":len(doc.tables),"inventory_candidate_count":len(doc.inventory_candidates)}); _write_json(paper_dir/"extraction"/"inventory_candidates.json",_candidate_payload(doc.inventory_candidates)); self.store.record_candidates(doi,doc.inventory_candidates)
        reject=_deterministic_screen_rejection(doc)
        if reject: self.store.upsert_paper(doi,title=doc.title,status="SCREEN_REJECTED"); return {"doi":doi,"title":doc.title,"status":"SCREEN_REJECTED","reason":reject}
        try: screen=self._screen(doi,source_hash,doc,paper_dir)
        except BudgetExceeded as exc: return {"doi":doi,"title":doc.title,"status":"BUDGET_STOP","error":str(exc)}
        except Exception as exc: self.store.upsert_paper(doi,title=doc.title,status="INFRASTRUCTURE_FAILURE",last_error=str(exc)); self.store.record_failure(doi,"screen","INFRASTRUCTURE_FAILURE",str(exc)); return {"doi":doi,"title":doc.title,"status":"INFRASTRUCTURE_FAILURE","error":str(exc)}
        if not(screen.eligible and screen.reconstructable_foreground and screen.likely_inventory): self.store.upsert_paper(doi,title=doc.title,status="SCREEN_REJECTED"); return {"doi":doi,"title":doc.title,"status":"SCREEN_REJECTED","reason":screen.reason}
        self.store.upsert_paper(doi,title=doc.title,status="SCREENED")
        try: structure=self._structure(doi,source_hash,doc,paper_dir)
        except ValueError as exc: self.store.upsert_paper(doi,title=doc.title,status="UNRESOLVED_STRUCTURE",last_error=str(exc)); self.store.record_failure(doi,"structure","PROCESS_UNDERSEGMENTATION",str(exc)); return {"doi":doi,"title":doc.title,"status":"UNRESOLVED_STRUCTURE","error":str(exc)}
        except BudgetExceeded as exc: return {"doi":doi,"title":doc.title,"status":"BUDGET_STOP","error":str(exc)}
        except Exception as exc: self.store.upsert_paper(doi,title=doc.title,status="INFRASTRUCTURE_FAILURE",last_error=str(exc)); self.store.record_failure(doi,"structure","INFRASTRUCTURE_FAILURE",str(exc)); return {"doi":doi,"title":doc.title,"status":"INFRASTRUCTURE_FAILURE","error":str(exc)}
        self.store.upsert_paper(doi,title=doc.title,status="STRUCTURE_VALIDATED")
        if not doc.inventory_candidates:
            detail="deterministic JATS enumeration produced zero plausible inventory candidates"; self.store.upsert_paper(doi,title=doc.title,status="UNRESOLVED_INVENTORY",last_error=detail); self.store.record_failure(doi,"enumeration","FLOW_ENUMERATION_FAILURE",detail); return {"doi":doi,"title":doc.title,"status":"UNRESOLVED_INVENTORY","error":detail}
        try: assignments,missing=self._assign(doi,source_hash,structure,doc.inventory_candidates,paper_dir)
        except BudgetExceeded as exc: return {"doi":doi,"title":doc.title,"status":"BUDGET_STOP","error":str(exc)}
        except Exception as exc: self.store.upsert_paper(doi,title=doc.title,status="INFRASTRUCTURE_FAILURE",last_error=str(exc)); self.store.record_failure(doi,"assignment","INFRASTRUCTURE_FAILURE",str(exc)); return {"doi":doi,"title":doc.title,"status":"INFRASTRUCTURE_FAILURE","error":str(exc)}
        _write_json(paper_dir/"extraction"/"assignments.json",{"assignments":[a.model_dump() for a in assignments]}); ambiguous=[a for a in assignments if a.disposition=="ambiguous"]
        cmap={c.candidate_id:c for c in doc.inventory_candidates}; by_process={p.process_id:[] for p in structure.processes}
        for a in assignments:
            if a.disposition=="modeled_inventory":
                for pid in a.process_ids:
                    if pid in by_process: by_process[pid].append(cmap[a.candidate_id])
        allowed=set(by_process); process_results={}; process_failures={}
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.config.max_process_workers,max(1,len(structure.processes)))) as pool:
                futures={}
                for p in structure.processes:
                    assigned=by_process[p.process_id]
                    if not assigned: process_results[p.process_id]=ProcessCandidateExtraction(process_id=p.process_id,warnings=["No deterministic candidates assigned."]); process_failures[p.process_id]=["no candidates assigned to locked process"]
                    else: futures[pool.submit(self._extract_process,doi,source_hash,p,allowed,assigned,paper_dir)]=p.process_id
                for f in concurrent.futures.as_completed(futures): process_results[futures[f]],process_failures[futures[f]]=f.result()
        except BudgetExceeded as exc: return {"doi":doi,"title":doc.title,"status":"BUDGET_STOP","error":str(exc)}
        except Exception as exc: self.store.upsert_paper(doi,title=doc.title,status="INFRASTRUCTURE_FAILURE",last_error=str(exc)); self.store.record_failure(doi,"process_inventory","INFRASTRUCTURE_FAILURE",str(exc)); return {"doi":doi,"title":doc.title,"status":"INFRASTRUCTURE_FAILURE","error":str(exc)}
        flows=[]
        for pid,r in process_results.items():
            for f in r.flows:
                if f.candidate_id in cmap: flows.append(_to_inventory_flow(pid,f,cmap[f.candidate_id]))
        dedup=[]; seen=set()
        for f in flows:
            key=(f.process_id,_normalise(f.name),f.direction,f.amount,_normalise(f.unit))
            if key not in seen: seen.add(key); dedup.append(f)
        extraction=InventoryExtraction(process_name=structure.process_name,functional_unit=structure.functional_unit,source_summary=structure.source_summary,study_context=structure.study_context,assumptions_or_warnings=structure.assumptions_or_warnings,candidate_activities=structure.candidate_activities,processes=structure.processes,flows=dedup,provenance=build_extraction_provenance(model=self.config.core_model,source_mode="text")); _write_json(paper_dir/"extraction"/"inventory.json",extraction)
        modeled={a.candidate_id for a in assignments if a.disposition=="modeled_inventory"}; reviewed={f.candidate_id for r in process_results.values() for f in r.flows}|{x for r in process_results.values() for x in r.non_inventory_candidate_ids}; unresolved=set(missing)|{a.candidate_id for a in ambiguous}|{x for r in process_results.values() for x in r.ambiguous_candidate_ids}; bad={pid:v for pid,v in process_failures.items() if v}
        qc={"doi":doi,"title":doc.title,"process_count":len(structure.processes),"candidate_count":len(doc.inventory_candidates),"modeled_candidate_count":len(modeled),"candidate_coverage":len(reviewed&modeled)/len(modeled) if modeled else 0.0,"ambiguous_or_missing_candidate_count":len(unresolved),"flow_count":len(dedup),"amount_coverage":sum(f.amount is not None for f in dedup)/len(dedup) if dedup else 0.0,"unit_coverage":sum(bool(f.unit) for f in dedup)/len(dedup) if dedup else 0.0,"process_failures":bad,"source_hash":source_hash,"pipeline_version":PIPELINE_VERSION}; _write_json(paper_dir/"extraction"/"qc.json",qc)
        status="UNRESOLVED_INVENTORY" if (unresolved or bad or not dedup) else "COMPLETE"
        if status!="COMPLETE": self.store.record_failure(doi,"inventory","FLOW_ENUMERATION_FAILURE",f"unresolved_candidates={len(unresolved)} processes_with_failures={len(bad)} flows={len(dedup)}")
        self.store.upsert_paper(doi,title=doc.title,status=status); return {"doi":doi,"title":doc.title,"status":status,**qc,"paper_api_calls":self.store.paper_api_calls(doi)}


def _write_report(state_dir,store,budget,results):
    report={"generated_at_utc":_utc_now(),"pipeline_version":PIPELINE_VERSION,"paper_results":results,"persistent_state":store.aggregate(),"this_run":budget.summary()}; path=state_dir/"reports"/"latest.json"; _write_json(path,report); print(json.dumps(report,ensure_ascii=False,indent=2)); return path

def main():
    p=argparse.ArgumentParser(description="Autonomous gated resumable Phase-1 LCA literature processor (machine-readable sources only).")
    p.add_argument("--mode",choices=["one","scale"],default="one"); p.add_argument("--doi"); p.add_argument("--issn",default=IJLCA_ONLINE_ISSN); p.add_argument("--from-year",type=int,default=2024); p.add_argument("--until-year",type=int,default=2026); p.add_argument("--discovery-limit",type=int,default=100); p.add_argument("--max-papers",type=int,default=1); p.add_argument("--state-dir",type=Path,default=Path("literature_state")); p.add_argument("--screen-model",default=os.getenv("OPENAI_SCREEN_MODEL","gpt-5-nano")); p.add_argument("--core-model",default=os.getenv("OPENAI_MODEL","gpt-5-mini")); p.add_argument("--max-concurrent-requests",type=int,default=4); p.add_argument("--max-process-workers",type=int,default=4); p.add_argument("--max-paper-workers",type=int,default=3); p.add_argument("--max-total-calls",type=int,default=120); p.add_argument("--max-calls-per-paper",type=int,default=15); p.add_argument("--max-total-tokens",type=int,default=5_000_000); p.add_argument("--max-estimated-cost-usd",type=float,default=5.0); p.add_argument("--max-repair-calls-per-process",type=int,default=1); p.add_argument("--infrastructure-retries",type=int,default=3); a=p.parse_args()
    springer=os.getenv("SPRINGER_API_KEY","").strip()
    if not springer: raise SystemExit("Missing SPRINGER_API_KEY")
    if not os.getenv("OPENAI_API_KEY","").strip(): raise SystemExit("Missing OPENAI_API_KEY")
    config=RunConfig(state_dir=a.state_dir,screen_model=a.screen_model,core_model=a.core_model,max_concurrent_requests=a.max_concurrent_requests,max_process_workers=a.max_process_workers,max_paper_workers=a.max_paper_workers,max_total_calls=a.max_total_calls,max_calls_per_paper=a.max_calls_per_paper,max_total_tokens=a.max_total_tokens,max_estimated_cost_usd=a.max_estimated_cost_usd,max_repair_calls_per_process=a.max_repair_calls_per_process,infrastructure_retries=a.infrastructure_retries); store=StateStore(config.state_dir); budget=Budget(config,store); processor=PaperProcessor(config,store,ApiRunner(config,store,budget),springer)
    if a.mode=="one":
        if not a.doi: raise SystemExit("--doi required in one mode")
        _write_report(config.state_dir,store,budget,[processor.process(a.doi)]); return
    records=list(iter_crossref_journal(a.issn,from_year=a.from_year,until_year=a.until_year,max_records=a.discovery_limit,rows=min(250,a.discovery_limit))); records.sort(key=lambda r:(rough_lca_relevance(r),r.published_year or 0,r.title),reverse=True); dois=[]
    for r in records:
        if store.paper_status(r.doi) in TERMINAL_STATUSES: continue
        dois.append(r.doi)
        if len(dois)>=a.max_papers: break
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(config.max_paper_workers,max(1,len(dois)))) as pool:
        futures={pool.submit(processor.process,d):d for d in dois}
        for f in concurrent.futures.as_completed(futures):
            doi=futures[f]
            try: results.append(f.result())
            except Exception as exc: store.record_failure(doi,"paper","INFRASTRUCTURE_FAILURE",str(exc)); store.upsert_paper(doi,status="INFRASTRUCTURE_FAILURE",last_error=str(exc)); results.append({"doi":doi,"status":"INFRASTRUCTURE_FAILURE","error":str(exc)})
    _write_report(config.state_dir,store,budget,results)

if __name__=="__main__": main()
