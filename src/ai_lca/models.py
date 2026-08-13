from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ProcessRole = Literal[
    "assessed_product_system",
    "interconnected_foreground_process",
    "internal_stage",
    "shared_supporting_activity",
    "background_supply",
    "descriptive_only",
]

LOCKED_PROCESS_ROLES = {
    "assessed_product_system",
    "interconnected_foreground_process",
}


class SourceEvidence(BaseModel):
    """Evidence supporting an extracted process, context field, or foreground-flow value."""

    model_config = ConfigDict(extra="forbid")

    document: str | None = Field(
        default=None,
        description="Source document name when [DOCUMENT: ...] markers are available.",
    )
    page: int | None = Field(
        default=None,
        description="PDF page number if explicitly available from [PAGE N] markers.",
    )
    paragraph: int | None = Field(
        default=None,
        description="DOCX paragraph number if explicitly available from [PARAGRAPH N] markers.",
    )
    section: str | None = None
    table: str | None = None
    evidence_text: str = Field(
        description="Short verbatim evidence supporting the extracted item."
    )


class StudyContext(BaseModel):
    """Study context that should be visible to the user and can inform candidate ranking."""

    model_config = ConfigDict(extra="forbid")

    technology_or_system: str | None = None
    operational_geography: str | None = Field(
        default=None,
        description="Country/region intended for operation, only when explicit or strongly supported by the source.",
    )
    geography_basis: Literal["explicit", "inferred", "not_identified"] = "not_identified"
    additional_geographies: list[str] = Field(
        default_factory=list,
        description="Additional country/region scenarios explicitly assessed beyond the primary/reference geography.",
    )
    geography_rationale: str | None = Field(
        default=None,
        description="Why this geography was extracted or inferred; must not add unsupported facts.",
    )
    system_boundary: str | None = None
    temporal_context: str | None = None
    evidence: list[SourceEvidence] = Field(default_factory=list)


class ProcessCandidate(BaseModel):
    """One process-like source entity classified before the foreground hierarchy is locked."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(
        description="Short stable source-derived identifier. For accepted roles this becomes the process id."
    )
    name: str
    role: ProcessRole
    parent_candidate_id: str | None = Field(
        default=None,
        description="Parent candidate only where the source explicitly supports a hierarchy.",
    )
    stage: Literal[
        "construction",
        "operation",
        "maintenance",
        "end_of_life",
        "transport",
        "other",
        "unknown",
    ] = "unknown"
    reference_product: str | None = Field(
        default=None,
        description="Reference product only when explicit or unambiguous from the source/functional unit.",
    )
    reference_unit: str | None = Field(
        default=None,
        description="Reference unit only when explicit or unambiguous from the source/functional unit.",
    )
    rationale: str = Field(
        description="Brief evidence-grounded reason for the role classification; do not add unsupported facts."
    )
    evidence: list[SourceEvidence] = Field(default_factory=list)


class ForegroundInterpretation(BaseModel):
    """Model-facing first pass: classify process-like entities before deterministic locking."""

    model_config = ConfigDict(extra="forbid")

    process_name: str | None = None
    functional_unit: str | None = Field(
        default=None,
        description="Functional unit only if explicitly stated or clearly defined in supplied text.",
    )
    source_summary: str
    study_context: StudyContext = Field(default_factory=StudyContext)
    assumptions_or_warnings: list[str] = Field(default_factory=list)
    candidates: list[ProcessCandidate] = Field(default_factory=list)


class ForegroundProcess(BaseModel):
    """One evidence-backed foreground process retained after candidate-role classification."""

    model_config = ConfigDict(extra="forbid")

    process_id: str = Field(description="Short stable identifier such as P1 or a source acronym.")
    name: str
    parent_process_id: str | None = Field(
        default=None,
        description="Parent process only when the source genuinely models this as a subprocess.",
    )
    role: Literal[
        "assessed_product_system",
        "interconnected_foreground_process",
    ] = "assessed_product_system"
    stage: Literal[
        "construction",
        "operation",
        "maintenance",
        "end_of_life",
        "transport",
        "other",
        "unknown",
    ] = "unknown"
    reference_product: str | None = None
    reference_unit: str | None = None
    description: str | None = None
    classification_rationale: str | None = None
    evidence: list[SourceEvidence] = Field(default_factory=list)


class ForegroundStructure(BaseModel):
    """Locked first-pass interpretation used by downstream flow extraction."""

    model_config = ConfigDict(extra="forbid")

    process_name: str | None = None
    functional_unit: str | None = Field(
        default=None,
        description="Functional unit only if explicitly stated or clearly defined in supplied text.",
    )
    source_summary: str
    study_context: StudyContext = Field(default_factory=StudyContext)
    assumptions_or_warnings: list[str] = Field(default_factory=list)
    candidate_activities: list[ProcessCandidate] = Field(default_factory=list)
    processes: list[ForegroundProcess] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_process_hierarchy(self) -> "ForegroundStructure":
        ids = [p.process_id for p in self.processes]
        if len(ids) != len(set(ids)):
            raise ValueError("Foreground process IDs must be unique")
        id_set = set(ids)
        for process in self.processes:
            if process.parent_process_id == process.process_id:
                raise ValueError(f"Process {process.process_id} cannot be its own parent")
            if process.parent_process_id and process.parent_process_id not in id_set:
                raise ValueError(
                    f"Parent process {process.parent_process_id!r} for {process.process_id!r} does not exist"
                )
        return self


class InventoryFlow(BaseModel):
    """One proposed foreground inventory flow attached to an identified foreground process."""

    model_config = ConfigDict(extra="forbid")

    process_id: str = Field(description="ID of one process from the first-pass foreground structure.")
    name: str = Field(description="Plain-language material, energy, transport, emission, or product flow.")
    amount: float | None = Field(
        default=None,
        description="Numeric quantity exactly supported by the supplied document; null if no quantity is stated.",
    )
    unit: str | None = Field(default=None, description="Unit exactly as stated or unambiguously normalised.")
    direction: Literal["input", "output", "emission", "unknown"] = "unknown"
    linked_process_id: str | None = Field(
        default=None,
        description=(
            "Another locked foreground process supplying this exchange, only when the source explicitly models "
            "the foreground-to-foreground link. Null for background supplies and ordinary flows."
        ),
    )
    component_or_stage: str | None = None
    basis: str | None = Field(
        default=None,
        description="Basis/denominator, e.g. per kg H2, per year, per plant, per functional unit.",
    )
    background_process_hint: str | None = Field(
        default=None,
        description=(
            "Verbatim background-database process identifier for this flow, copied only when explicitly "
            "printed in the supplied source material (e.g. a supplementary LCA-software process tree/export) "
            "in a format such as 'activity name | reference product | system model, type - location'. "
            "Null unless such an explicit identifier for this exact flow is printed in the source; never "
            "constructed or inferred from the plain-language flow name alone."
        ),
    )
    notes: str | None = None
    evidence: SourceEvidence


class FlowExtraction(BaseModel):
    """Second-pass result: flows only, constrained to the already identified process structure."""

    model_config = ConfigDict(extra="forbid")

    assumptions_or_warnings: list[str] = Field(default_factory=list)
    flows: list[InventoryFlow] = Field(default_factory=list)


class ExtractionProvenance(BaseModel):
    """Version metadata attached to a reproducible extraction."""

    model_config = ConfigDict(extra="forbid")

    extractor_version: str
    model: str
    git_sha: str | None = None
    generated_at_utc: str
    source_mode: Literal["text", "documents"]


class InventoryExtraction(BaseModel):
    """Auditable proposed foreground inventory extracted from source material."""

    model_config = ConfigDict(extra="forbid")

    process_name: str | None = None
    functional_unit: str | None = None
    source_summary: str
    study_context: StudyContext = Field(default_factory=StudyContext)
    assumptions_or_warnings: list[str] = Field(default_factory=list)
    candidate_activities: list[ProcessCandidate] = Field(default_factory=list)
    processes: list[ForegroundProcess] = Field(default_factory=list)
    flows: list[InventoryFlow] = Field(default_factory=list)
    provenance: ExtractionProvenance | None = None

    @model_validator(mode="after")
    def validate_flow_process_links(self) -> "InventoryExtraction":
        process_ids = {p.process_id for p in self.processes}
        missing = sorted({flow.process_id for flow in self.flows if flow.process_id not in process_ids})
        if missing:
            raise ValueError(f"Flows reference unknown process IDs: {', '.join(missing)}")
        linked_missing = sorted(
            {
                flow.linked_process_id
                for flow in self.flows
                if flow.linked_process_id and flow.linked_process_id not in process_ids
            }
        )
        if linked_missing:
            raise ValueError(
                f"Flows reference unknown linked foreground process IDs: {', '.join(linked_missing)}"
            )
        return self
