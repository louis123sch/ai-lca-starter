from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class SourceEvidence(BaseModel):
    """Evidence supporting an extracted model element or foreground-flow value."""

    model_config = ConfigDict(extra="forbid")

    source_document: str | None = Field(
        default=None,
        description="Filename from the nearest [DOCUMENT ...] marker when multiple source documents are supplied.",
    )
    page: int | None = Field(
        default=None,
        description="PDF page number if explicitly available from [PAGE N] markers.",
    )
    section: str | None = None
    table: str | None = None
    evidence_text: str = Field(
        description="Short verbatim evidence supporting the extracted item."
    )


class DescribedOperation(BaseModel):
    """An engineering operation described inside a foreground process.

    Operations are explanatory context. They must not become separate foreground
    activities unless the paper separately models them as such.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    notes: str | None = None
    evidence: SourceEvidence


class ForegroundProcess(BaseModel):
    """A process that the source LCA itself models as a distinct foreground unit."""

    model_config = ConfigDict(extra="forbid")

    process_id: str = Field(
        default="",
        description="Stable internal ID; normalised by the application after extraction.",
    )
    name: str
    geographic_context: str | None = Field(
        default=None,
        description="Intended operating/model geography when supported by the source.",
    )
    temporal_context: str | None = Field(
        default=None,
        description="Year, study period, scenario year, or other time context when supported.",
    )
    evidence_type: Literal[
        "inventory_table",
        "system_boundary_figure",
        "separate_balance",
        "explicit_process_link",
        "explicit_text",
    ]
    reason_for_separate_process: str = Field(
        description="Why the source supports treating this as a distinct LCA foreground process."
    )
    confidence: Literal["high", "medium", "low"] = "medium"
    evidence: list[SourceEvidence] = Field(default_factory=list)
    operations: list[DescribedOperation] = Field(default_factory=list)


class TechnologyGroup(BaseModel):
    """Technology/pathway grouping used to keep related foreground processes together."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    processes: list[ForegroundProcess] = Field(default_factory=list)


class ProcessMap(BaseModel):
    """Evidence-backed map of the foreground process structure represented in a source."""

    model_config = ConfigDict(extra="forbid")

    source_summary: str
    functional_unit: str | None = Field(
        default=None,
        description="Functional unit only if explicitly stated or clearly defined in supplied text.",
    )
    system_boundary: str | None = None
    geographic_context: str | None = Field(
        default=None,
        description="Overall study/model geography, not an inferred electricity-market geography.",
    )
    temporal_context: str | None = None
    assumptions_or_warnings: list[str] = Field(default_factory=list)
    technology_groups: list[TechnologyGroup] = Field(default_factory=list)


class InventoryFlow(BaseModel):
    """One proposed foreground inventory flow assigned to a confirmed foreground process."""

    model_config = ConfigDict(extra="forbid")

    process_id: str
    technology_group: str
    process_name: str
    name: str = Field(description="Plain-language material, energy, transport, emission, or product flow.")
    amount: float | None = Field(
        default=None,
        description="Numeric quantity exactly supported by the supplied document; null if no quantity is stated.",
    )
    unit: str | None = Field(default=None, description="Unit exactly as stated or unambiguously normalised.")
    direction: Literal["input", "output", "emission", "unknown"] = "unknown"
    flow_kind: Literal[
        "material",
        "energy",
        "transport",
        "water",
        "waste",
        "product",
        "emission",
        "other",
    ] = "other"
    operation_context: str | None = Field(
        default=None,
        description="Optional descriptive operation within the process; this is not a separate foreground process.",
    )
    component_or_stage: str | None = None
    basis: str | None = Field(
        default=None,
        description="Basis/denominator, e.g. per kg H2, per year, per electrolyser, per functional unit.",
    )
    notes: str | None = None
    evidence: SourceEvidence


class InventoryExtraction(BaseModel):
    """Auditable proposed foreground inventory extracted under an approved ProcessMap."""

    model_config = ConfigDict(extra="forbid")

    functional_unit: str | None = Field(
        default=None,
        description="Functional unit only if explicitly stated or clearly defined in supplied text.",
    )
    source_summary: str = Field(description="One-sentence description of what source material was analysed.")
    assumptions_or_warnings: list[str] = Field(default_factory=list)
    flows: list[InventoryFlow] = Field(default_factory=list)
