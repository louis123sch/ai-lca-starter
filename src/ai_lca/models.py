from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


LCAItemType = Literal[
    "technosphere_flow",
    "biosphere_flow",
    "parameter",
    "reference_product",
]


class SourceEvidence(BaseModel):
    """Evidence supporting an extracted LCA item."""

    model_config = ConfigDict(extra="forbid")

    source_document: str | None = Field(
        default=None,
        description="Filename from the nearest [SOURCE filename] marker when supplied.",
    )
    page: int | None = Field(
        default=None,
        description="PDF page number if explicitly available from [PAGE N] markers.",
    )
    section: str | None = None
    table: str | None = None
    evidence_text: str = Field(
        description="Short verbatim evidence supporting the extracted item and amount."
    )


class InventoryFlow(BaseModel):
    """One proposed LCA-relevant item extracted from source material."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Plain-language name for the extracted LCA-relevant item.")
    item_type: LCAItemType = Field(
        description=(
            "How the item should be routed: technosphere_flow for purchased/consumed goods, "
            "energy, transport, or services; biosphere_flow for direct emissions/resources; "
            "parameter for model/scaling values such as lifetime, operating hours, capacity, "
            "efficiency, yield, or load factor; reference_product for the modelled product/output."
        )
    )
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


class InventoryExtraction(BaseModel):
    """Auditable proposed foreground inventory and model information extracted from source material."""

    model_config = ConfigDict(extra="forbid")

    process_name: str | None = None
    functional_unit: str | None = Field(
        default=None,
        description="Functional unit only if explicitly stated or clearly defined in supplied text.",
    )
    source_summary: str = Field(description="One-sentence description of what source material was analysed.")
    assumptions_or_warnings: list[str] = Field(default_factory=list)
    flows: list[InventoryFlow] = Field(default_factory=list)
