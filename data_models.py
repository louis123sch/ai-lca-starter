from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


LCAItemType = Literal[
    "technosphere_flow",
    "biosphere_flow",
    "parameter",
    "reference_product",
]

EcoinventActivityType = Literal[
    "market",
    "transforming",
    "treatment",
    "transport",
    "service",
    "construction",
    "operation",
    "unknown",
]


class SourceEvidence(BaseModel):
    """Evidence supporting an extracted LCA item."""

    model_config = ConfigDict(extra="forbid")

    source_document: str | None = Field(
        default=None,
        description="Source filename. The application enforces this deterministically.",
    )
    page: int | None = Field(
        default=None,
        description="PDF page number only when explicitly available from [PAGE N] markers.",
    )
    section: str | None = None
    table: str | None = None
    evidence_text: str = Field(
        description="Short verbatim evidence supporting the extracted item and amount."
    )


class ForegroundProcessProposal(BaseModel):
    """A foreground process or technology stage clearly supported by the source."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Short canonical foreground process/stage name, e.g. SMR or CO2 capture.")
    role: str = Field(description="Short description of what this process/stage does in the model.")
    source_document: str | None = None
    evidence_text: str | None = Field(
        default=None,
        description="Short source evidence showing why this process/stage was proposed.",
    )


class DocumentUnderstanding(BaseModel):
    """High-level interpretation produced before inventory extraction."""

    model_config = ConfigDict(extra="forbid")

    technology_name: str | None = Field(
        default=None,
        description="Technology/system being described, using the source's terminology where possible.",
    )
    system_description: str = Field(
        description="Concise description of what is being modelled and the important process context."
    )
    foreground_processes: list[ForegroundProcessProposal] = Field(default_factory=list)
    geography_hints: list[str] = Field(
        default_factory=list,
        description="Geographies explicitly stated or unambiguously defined by the source/study instructions.",
    )
    interpretation_notes: list[str] = Field(
        default_factory=list,
        description="Short modelling observations needed for the extraction pass; not private chain-of-thought.",
    )


class InventoryFlow(BaseModel):
    """One proposed foreground exchange, parameter, emission, or reference product."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description=(
            "Canonical LCA concept. For technosphere flows use the underlying product/service, "
            "e.g. 'natural gas', not source phrases like 'natural gas - SMR + CCS 90%'."
        )
    )
    source_label: str | None = Field(
        default=None,
        description="Short original/source wording that the canonical name was interpreted from.",
    )
    item_type: LCAItemType
    parent_process: str | None = Field(
        default=None,
        description="Foreground process/stage this item belongs to when the source makes that relationship clear.",
    )
    amount: float | None = Field(
        default=None,
        description="Numeric quantity exactly supported by the source; null if no quantity is stated.",
    )
    unit: str | None = Field(default=None, description="Unit exactly stated or unambiguously normalised.")
    direction: Literal["input", "output", "emission", "unknown"] = "unknown"
    basis: str | None = Field(
        default=None,
        description="Basis/denominator, e.g. per kg H2, per year, per plant, per functional unit.",
    )

    ecoinvent_search_term: str | None = Field(
        default=None,
        description=(
            "Concise product/service concept to send to Brightway search. Only for plausible "
            "technosphere background links; do not fabricate an exact dataset name."
        ),
    )
    ecoinvent_activity_type_hint: EcoinventActivityType | None = Field(
        default=None,
        description="Likely ecoinvent activity class; a hint only, never an assertion that a dataset exists.",
    )
    geography_hint: str | None = Field(
        default=None,
        description="Geography kept separate from the activity/search name, e.g. GB, RER, GLO, Norway.",
    )
    supplier_technology_hint: str | None = Field(
        default=None,
        description=(
            "Specific background supply technology/provenance only when explicitly supported, "
            "e.g. offshore wind or North Sea natural gas production."
        ),
    )
    search_worthy: bool = Field(
        default=False,
        description="True only when the item should be offered to the ecoinvent technosphere matcher.",
    )
    interpretation_reason: str = Field(
        default="",
        description="Short auditable explanation of the LCA interpretation; not hidden chain-of-thought.",
    )
    notes: str | None = None
    evidence: SourceEvidence


class InventoryItemsResult(BaseModel):
    """Second-pass extraction result for one source document."""

    model_config = ConfigDict(extra="forbid")

    process_name: str | None = None
    functional_unit: str | None = None
    assumptions_or_warnings: list[str] = Field(default_factory=list)
    flows: list[InventoryFlow] = Field(default_factory=list)


class InventoryExtraction(BaseModel):
    """Auditable draft foreground interpretation assembled from one or more sources."""

    model_config = ConfigDict(extra="forbid")

    process_name: str | None = None
    technology_name: str | None = None
    functional_unit: str | None = Field(
        default=None,
        description="Functional unit only if explicitly stated or clearly defined in supplied text.",
    )
    system_description: str | None = None
    foreground_processes: list[ForegroundProcessProposal] = Field(default_factory=list)
    source_summary: str = Field(description="One-sentence description of what source material was analysed.")
    assumptions_or_warnings: list[str] = Field(default_factory=list)
    flows: list[InventoryFlow] = Field(default_factory=list)
