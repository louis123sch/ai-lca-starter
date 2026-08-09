from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class ForegroundProcess(BaseModel):
    """One evidence-backed foreground process explicitly represented by the source."""

    model_config = ConfigDict(extra="forbid")

    process_id: str = Field(description="Short stable identifier such as P1 or P1.1.")
    name: str
    parent_process_id: str | None = Field(
        default=None,
        description="Parent process only when the source genuinely models this as a subprocess.",
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
    description: str | None = None
    evidence: list[SourceEvidence] = Field(default_factory=list)


class ForegroundStructure(BaseModel):
    """First-pass interpretation: study context and the actual process hierarchy in the source."""

    model_config = ConfigDict(extra="forbid")

    process_name: str | None = None
    functional_unit: str | None = Field(
        default=None,
        description="Functional unit only if explicitly stated or clearly defined in supplied text.",
    )
    source_summary: str
    study_context: StudyContext = Field(default_factory=StudyContext)
    assumptions_or_warnings: list[str] = Field(default_factory=list)
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
    component_or_stage: str | None = None
    basis: str | None = Field(
        default=None,
        description="Basis/denominator, e.g. per kg H2, per year, per electrolyser, per functional unit.",
    )
    notes: str | None = None
    evidence: SourceEvidence


class FlowExtraction(BaseModel):
    """Second-pass result: flows only, constrained to the already identified process structure."""

    model_config = ConfigDict(extra="forbid")

    assumptions_or_warnings: list[str] = Field(default_factory=list)
    flows: list[InventoryFlow] = Field(default_factory=list)


class InventoryExtraction(BaseModel):
    """Auditable proposed foreground inventory extracted from source material."""

    model_config = ConfigDict(extra="forbid")

    process_name: str | None = None
    functional_unit: str | None = None
    source_summary: str
    study_context: StudyContext = Field(default_factory=StudyContext)
    assumptions_or_warnings: list[str] = Field(default_factory=list)
    processes: list[ForegroundProcess] = Field(default_factory=list)
    flows: list[InventoryFlow] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_flow_process_links(self) -> "InventoryExtraction":
        process_ids = {p.process_id for p in self.processes}
        missing = sorted({flow.process_id for flow in self.flows if flow.process_id not in process_ids})
        if missing:
            raise ValueError(f"Flows reference unknown process IDs: {', '.join(missing)}")
        return self
