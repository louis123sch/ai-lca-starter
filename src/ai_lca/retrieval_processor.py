from __future__ import annotations

import json

from .autonomous_literature import (
    ASSIGN_SYSTEM_PROMPT,
    CandidateAssignment,
    CandidateAssignmentBatch,
    PaperProcessor,
    _job_key,
    _load_model,
    _validate_assignments,
    _write_json,
)
from .evidence_router import (
    build_structure_evidence,
    partition_inventory_candidates,
    route_inventory_candidates,
    routed_candidate_payload,
)


class RetrievalPaperProcessor(PaperProcessor):
    """High-recall retrieval-before-reasoning variant of the literature processor.

    The deterministic JATS enumerator remains the source of truth. In v1 the
    trusted full-context structure stage is deliberately unchanged; retrieval is
    enabled only for inventory-candidate reasoning. A bounded structure evidence
    pack is still written for measurement so structure retrieval can be promoted
    later if its frozen-corpus recall becomes adequate.
    """

    retrieval_structure_max_chars = 50_000

    def _structure(self, doi, source_hash, doc, paper_dir):  # noqa: ANN001
        pack = build_structure_evidence(
            doc,
            min(self.config.structure_max_chars, self.retrieval_structure_max_chars),
        )
        _write_json(
            paper_dir / "extraction" / "retrieval" / "structure_evidence_manifest.json",
            {
                **pack.manifest(),
                "enabled_for_reasoning": False,
                "reason": "v1 retains full-context structure reasoning until retrieval recall is validated",
            },
        )
        return super()._structure(doi, source_hash, doc, paper_dir)

    def _assignment_chunk(self, doi, source_hash, structure, candidates, idx, paper_dir):  # noqa: ANN001
        path = paper_dir / "extraction" / "assignments_retrieval" / f"chunk_{idx:03d}.json"
        key = _job_key(
            doi,
            source_hash,
            f"assign_retrieval:{idx}",
            process_id=None,
            model=self.config.screen_model,
            prompt_version="candidate-assignment-retrieval-v1",
        )
        cached = self.store.completed_job_result(key)
        if cached:
            return _load_model(cached, CandidateAssignmentBatch)

        self.store.start_job(key, doi, f"assign_retrieval:{idx}", model=self.config.screen_model)
        route_by_id = {route.candidate_id: route for route in route_inventory_candidates(candidates)}
        payload = [routed_candidate_payload(candidate, route_by_id[candidate.candidate_id]) for candidate in candidates]
        prompt = (
            "LOCKED PROCESSES:\n"
            + json.dumps(
                [{"process_id": p.process_id, "name": p.name, "stage": p.stage} for p in structure.processes],
                ensure_ascii=False,
            )
            + "\n\nROUTED CANDIDATES:\n"
            + json.dumps(payload, ensure_ascii=False)
            + "\n\nThe evidence_route is a cheap advisory signal only. Override it when the source evidence requires it."
        )
        try:
            result = self.api.parse(
                doi=doi,
                stage="candidate_assignment_retrieval",
                model=self.config.screen_model,
                reasoning_effort=self.config.screen_reasoning,
                system_prompt=ASSIGN_SYSTEM_PROMPT,
                user_prompt=prompt,
                response_format=CandidateAssignmentBatch,
            )
            _write_json(path, result)
            self.store.complete_job(key, path)
            return result
        except Exception as exc:
            self.store.fail_job(key, str(exc))
            raise

    def _assign(self, doi, source_hash, structure, candidates, paper_dir):  # noqa: ANN001
        retained, excluded, routes = partition_inventory_candidates(candidates)
        route_by_id = {route.candidate_id: route for route in routes}
        _write_json(
            paper_dir / "extraction" / "retrieval" / "candidate_routes.json",
            {
                "candidate_count": len(candidates),
                "retained_for_reasoning": len(retained),
                "safe_excluded_from_reasoning": len(excluded),
                "safe_excluded_candidate_ids": [candidate.candidate_id for candidate in excluded],
                "routes": [route.as_dict() for route in routes],
            },
        )

        deterministic_not_inventory = [
            CandidateAssignment(
                candidate_id=candidate.candidate_id,
                disposition="not_inventory",
                process_ids=[],
                rationale=(
                    "High-confidence retrieval router identified an explicit LCIA/result table row with no "
                    "competing inventory or modelling-assumption signal. Candidate remains preserved in the "
                    "audit trail. "
                    + "; ".join(route_by_id[candidate.candidate_id].reasons)
                ),
            )
            for candidate in excluded
        ]

        if retained:
            accepted, missing = super()._assign(doi, source_hash, structure, retained, paper_dir)
        else:
            accepted, missing = [], []

        combined = CandidateAssignmentBatch(assignments=accepted + deterministic_not_inventory)
        validated, full_missing = _validate_assignments(
            candidates,
            combined,
            {p.process_id for p in structure.processes},
        )
        # Missing retained candidates remain unresolved. Safe-excluded candidates are
        # explicitly represented as not_inventory and therefore cannot disappear.
        return validated, sorted(set(missing) | set(full_missing))
