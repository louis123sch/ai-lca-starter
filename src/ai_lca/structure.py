from __future__ import annotations

from .models import (
    LOCKED_PROCESS_ROLES,
    ForegroundInterpretation,
    ForegroundProcess,
    ForegroundStructure,
)


def lock_foreground_interpretation(
    interpretation: ForegroundInterpretation,
) -> ForegroundStructure:
    """Deterministically convert classified candidates into the locked process graph.

    The LLM classifies source entities. This function alone decides which roles are
    allowed to become foreground process IDs. That keeps the downstream flow pass
    independent of prompt wording and makes over-decomposition rules testable.
    """
    warnings = list(interpretation.assumptions_or_warnings)

    by_id = {}
    for candidate in interpretation.candidates:
        candidate_id = candidate.candidate_id.strip()
        if not candidate_id:
            warnings.append(f"Ignored candidate with empty identifier: {candidate.name}")
            continue
        if candidate_id in by_id:
            warnings.append(
                f"Duplicate candidate id {candidate_id!r}; retained the first source-supported classification."
            )
            continue
        by_id[candidate_id] = candidate

    accepted_ids = {
        candidate_id
        for candidate_id, candidate in by_id.items()
        if candidate.role in LOCKED_PROCESS_ROLES
    }

    processes: list[ForegroundProcess] = []
    for candidate_id, candidate in by_id.items():
        if candidate.role not in LOCKED_PROCESS_ROLES:
            continue

        parent_id = candidate.parent_candidate_id.strip() if candidate.parent_candidate_id else None
        if parent_id == candidate_id:
            warnings.append(
                f"Removed self-parent relationship for {candidate.name!r} ({candidate_id})."
            )
            parent_id = None
        elif parent_id and parent_id not in accepted_ids:
            warnings.append(
                f"Candidate {candidate.name!r} referenced non-foreground parent {parent_id!r}; "
                "locked it as a top-level foreground process instead."
            )
            parent_id = None

        processes.append(
            ForegroundProcess(
                process_id=candidate_id,
                name=candidate.name,
                parent_process_id=parent_id,
                role=candidate.role,
                stage=candidate.stage,
                reference_product=candidate.reference_product,
                reference_unit=candidate.reference_unit,
                description=None,
                classification_rationale=candidate.rationale,
                evidence=candidate.evidence,
            )
        )

    if not processes:
        raise RuntimeError(
            "No candidate was classified as an assessed product system or interconnected foreground process."
        )

    return ForegroundStructure(
        process_name=interpretation.process_name,
        functional_unit=interpretation.functional_unit,
        source_summary=interpretation.source_summary,
        study_context=interpretation.study_context,
        assumptions_or_warnings=list(dict.fromkeys(warnings)),
        candidate_activities=list(by_id.values()),
        processes=processes,
    )
