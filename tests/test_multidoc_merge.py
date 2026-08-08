from ai_lca.models import (
    ForegroundProcess,
    InventoryExtraction,
    InventoryFlow,
    ProcessMap,
    SourceEvidence,
    TechnologyGroup,
)
from ai_lca.validation import normalise_process_ids, validate_inventory_against_process_map


def test_same_process_from_two_documents_is_merged_with_both_evidence_records():
    process_map = ProcessMap(
        source_summary="Two-source corpus",
        technology_groups=[
            TechnologyGroup(
                name="Methane pyrolysis",
                processes=[
                    ForegroundProcess(
                        name="Thermal plasma methane pyrolysis",
                        evidence_type="explicit_text",
                        reason_for_separate_process="Paper defines the pathway.",
                        geographic_context=None,
                        evidence=[
                            SourceEvidence(
                                source_document="paper.pdf",
                                page=4,
                                evidence_text="Thermal plasma methane pyrolysis is modelled.",
                            )
                        ],
                    ),
                    ForegroundProcess(
                        name="Thermal plasma methane pyrolysis",
                        evidence_type="inventory_table",
                        reason_for_separate_process="Supplement gives its inventory.",
                        geographic_context="United Kingdom",
                        evidence=[
                            SourceEvidence(
                                source_document="supplement.docx",
                                table="Table S2",
                                evidence_text="Inventory for thermal plasma methane pyrolysis.",
                            )
                        ],
                    ),
                ],
            )
        ],
    )

    result = normalise_process_ids(process_map)
    processes = result.technology_groups[0].processes

    assert len(processes) == 1
    assert processes[0].process_id == "p001"
    assert processes[0].geographic_context == "United Kingdom"
    assert {e.source_document for e in processes[0].evidence} == {"paper.pdf", "supplement.docx"}
    assert any("Merged repeated descriptions" in warning for warning in result.assumptions_or_warnings)


def test_same_flow_from_two_documents_merges_evidence_instead_of_duplicate_exchange():
    process_map = normalise_process_ids(
        ProcessMap(
            source_summary="Two-source corpus",
            technology_groups=[
                TechnologyGroup(
                    name="Methane pyrolysis",
                    processes=[
                        ForegroundProcess(
                            name="Thermal plasma methane pyrolysis",
                            evidence_type="inventory_table",
                            reason_for_separate_process="Combined evidence supports one process.",
                            evidence=[
                                SourceEvidence(
                                    source_document="paper.pdf",
                                    evidence_text="Thermal plasma methane pyrolysis is modelled.",
                                )
                            ],
                        )
                    ],
                )
            ],
        )
    )

    extraction = InventoryExtraction(
        source_summary="Two-source corpus",
        flows=[
            InventoryFlow(
                process_id="p001",
                technology_group="Methane pyrolysis",
                process_name="Thermal plasma methane pyrolysis",
                name="electricity",
                amount=10.0,
                unit="kWh",
                direction="input",
                flow_kind="energy",
                basis="per kg H2",
                evidence=[
                    SourceEvidence(
                        source_document="paper.pdf",
                        page=5,
                        evidence_text="The plasma system uses electricity at 10 kWh/kg H2.",
                    )
                ],
            ),
            InventoryFlow(
                process_id="p001",
                technology_group="Methane pyrolysis",
                process_name="Thermal plasma methane pyrolysis",
                name="electricity",
                amount=10.0,
                unit="kWh",
                direction="input",
                flow_kind="energy",
                basis="per kg H2",
                evidence=[
                    SourceEvidence(
                        source_document="supplement.docx",
                        table="Table S2",
                        evidence_text="Electricity | 10 kWh/kg H2",
                    )
                ],
            ),
        ],
    )

    result = validate_inventory_against_process_map(
        extraction,
        process_map,
        "[DOCUMENT paper.pdf]\n...\n[DOCUMENT supplement.docx]\n...",
    )

    assert len(result.flows) == 1
    assert {e.source_document for e in result.flows[0].evidence} == {"paper.pdf", "supplement.docx"}
    assert any("Merged repeated evidence for flow" in warning for warning in result.assumptions_or_warnings)


def test_same_process_name_with_conflicting_explicit_geographies_stays_separate():
    process_map = ProcessMap(
        source_summary="Two scenarios",
        technology_groups=[
            TechnologyGroup(
                name="Electrolysis",
                processes=[
                    ForegroundProcess(
                        name="PEM electrolysis",
                        evidence_type="explicit_text",
                        reason_for_separate_process="German scenario.",
                        geographic_context="Germany",
                        evidence=[SourceEvidence(source_document="de.pdf", evidence_text="PEM electrolysis in Germany.")],
                    ),
                    ForegroundProcess(
                        name="PEM electrolysis",
                        evidence_type="explicit_text",
                        reason_for_separate_process="French scenario.",
                        geographic_context="France",
                        evidence=[SourceEvidence(source_document="fr.pdf", evidence_text="PEM electrolysis in France.")],
                    ),
                ],
            )
        ],
    )

    result = normalise_process_ids(process_map)
    assert len(result.technology_groups[0].processes) == 2
