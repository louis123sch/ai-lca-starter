from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class SourceEvidence(BaseModel):
    """Evidence supporting an extracted model element or foreground-flow value."""

    model_config = ConfigDict(extra="forbid")

    source_document: str | None = Field(
        default=None,
        description="Filename from the nearest [DOCUMENT ...] marker when an evidence corpus is supplied.",
    )
    page: int | None = Field(
        default=None,
        description="PDF page number if explicitly available from [PAGE N] markers.",
    )
    section: str | None = None
    table: str | None = None
    evidence_text: str = Field(description="Short verbatim evidence supporting the extracted item.")


class DescribedOperation(BaseModel):
    """An engineering operation described inside a foreground process."""

    model_config = ConfigDict(extra="forbid")

    name: str
    notes: str | None = None
    evidence: list[SourceEvidence] = Field(default_factory=list)


class ForegroundProcess(BaseModel):
    """A process that the evidence corpus supports as a distinct foreground unit."""

    model_config = ConfigDict(extra="forbid")

    process_id: str = Field(default="")
    name: str
    geographic_context: str | None = None
    temporal_context: str | None = None
    evidence_type: Literal[
        "inventory_table",
        "system_boundary_figure",
        "separate_balance",
        "explicit_process_link",
        "explicit_text",
    ]
    reason_for_separate_process: str
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
    """Evidence-backed foreground process structure inferred across the full document corpus."""

    model_config = ConfigDict(extra="forbid")

    source_summary: str
    functional_unit: str | None = None
    system_boundary: str | None = None
    geographic_context: str | None = None
    temporal_context: str | None = None
    assumptions_or_warnings: list[str] = Field(default_factory=list)
    technology_groups: list[TechnologyGroup] = Field(default_factory=list)


class InventoryFlow(BaseModel):
    """One proposed foreground inventory flow assigned to a confirmed foreground process."""

    model_config = ConfigDict(extra="forbid")

    process_id: str
    technology_group: str
    process_name: str
    name: str = Field(
        description=(
            "Canonical bare exchange concept only, e.g. concrete, steel, aluminium, natural gas. "
            "Do not append lifecycle-stage labels such as plant construction."
        )
    )
    source_label: str | None = None
    amount: float | None = None
    unit: str | None = None
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
    operation_context: str | None = None
    component_or_stage: str | None = Field(
        default=None,
        description="Lifecycle/component context such as plant construction or operation; never part of the exchange/search name.",
    )
    basis: str | None = None
    exchange_geography_hint: str | None = Field(
        default=None,
        description=(
            "Exchange-specific source/provenance geography explicitly supported by the evidence, e.g. Norway for a natural-gas supply route. "
            "Keep this separate from the foreground process operating geography."
        ),
    )
    supplier_technology_hint: str | None = Field(
        default=None,
        description=(
            "Explicit supply technology/provenance useful for background matching, e.g. offshore wind, North Sea natural gas, or a named production route. "
            "Do not infer it from general engineering knowledge."
        ),
    )
    interpretation_reason: str | None = Field(
        default=None,
        description="Short reviewable explanation of why this item is an LCI exchange and how any matching hints were interpreted.",
    )
    ecoinvent_search_term: str | None = Field(
        default=None,
        description="Bare concept used for ecoinvent retrieval when no source-provided background mapping is available.",
    )
    ecoinvent_activity_hint: str | None = Field(
        default=None,
        description=(
            "Source-supported background activity name. It may be an exact mapping or an intentional proxy; "
            "the relation must be stated separately in background_mapping_relation."
        ),
    )
    ecoinvent_location_hint: str | None = None
    background_mapping_relation: Literal["exact", "proxy", "uncertain"] | None = Field(
        default=None,
        description=(
            "How the source-supported background activity relates to the foreground exchange: exact, proxy, or uncertain. "
            "Do not rename the foreground exchange when a proxy is used."
        ),
    )
    background_mapping_rationale: str | None = Field(
        default=None,
        description="Short explanation of why the source evidence supports the exact/proxy/uncertain relationship.",
    )
    background_mapping_evidence: list[SourceEvidence] = Field(default_factory=list)
    notes: str | None = None
    evidence: list[SourceEvidence] = Field(default_factory=list)


class InventoryExtraction(BaseModel):
    """Auditable proposed foreground inventory extracted under an approved ProcessMap."""

    model_config = ConfigDict(extra="forbid")

    functional_unit: str | None = None
    source_summary: str
    assumptions_or_warnings: list[str] = Field(default_factory=list)
    flows: list[InventoryFlow] = Field(default_factory=list)
