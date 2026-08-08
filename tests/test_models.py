from ai_lca.export import extraction_to_dataframe, process_map_to_dataframe
from ai_lca.models import (
    DescribedOperation,
    ForegroundProcess,
    InventoryExtraction,
    InventoryFlow,
    ProcessMap,
    SourceEvidence,
    TechnologyGroup,
)
from ai_lca.validation import normalise_process_ids, validate_inventory_against_process_map


def _pyrolysis_process_map() -> ProcessMap:
    return normalise_process_ids(
        ProcessMap(
            source_summary="Combined paper and supplementary evidence",
            functional_unit="1 kg H2",
            geographic_context="UK",
            technology_groups=[
                TechnologyGroup(
                    name="Methane pyrolysis",
                    processes=[
                        ForegroundProcess(
                            name="Hydrogen production by thermal plasma methane pyrolysis",
                            evidence_type="inventory_table",
                            reason_for_separate_process="The combined evidence supports one foreground inventory for this pathway.",
                            confidence="high",
                            evidence=[
                                SourceEvidence(
                                    source_document="paper.pdf",
                                    page=3,
                                    evidence_text="Thermal plasma methane pyrolysis is modelled as the hydrogen production pathway.",
                                ),
                                SourceEvidence(
                                    source_document="supplement.docx",
                                    table="Table 2",
                                    evidence_text="Inventory for thermal plasma methane pyrolysis per kg H2.",
                                ),
                            ],
                            operations=[
                                DescribedOperation(
                                    name="plasma generation",
                                    evidence=[
                                        SourceEvidence(
                                            source_document="paper.pdf",
                                            page=2,
                                            evidence_text="An electric arc generates the plasma.",
                                        )
                                    ],
                                ),
                                DescribedOperation(
                                    name="carbon separation",
                                    evidence=[
                                        SourceEvidence(
                                            source_document="supplement.docx",
                                            evidence_text="Solid carbon is separated from the product gas.",
                                        )
                                    ],
                                ),
                            ],
                        )
                    ],
                )
            ],
        )
    )


def test_process_map_keeps_operations_inside_one_foreground_process():
    process_map = _pyrolysis_process_map()
    df = process_map_to_dataframe(process_map)

    assert len(df) == 1
    assert df.iloc[0]["process_id"] == "p001"
    assert "plasma generation" in df.iloc[0]["operations_not_separate_processes"]
    assert "carbon separation" in df.iloc[0]["operations_not_separate_processes"]
    assert "paper.pdf" in df.iloc[0]["source_documents"]
    assert "supplement.docx" in df.iloc[0]["source_documents"]


def test_inventory_validation_rejects_unapproved_process_and_unsupported_voltage():
    process_map = _pyrolysis_process_map()
    extraction = InventoryExtraction(
        functional_unit="1 kg H2",
        source_summary="Combined evidence",
        flows=[
            InventoryFlow(
                process_id="p001",
                technology_group="wrong",
                process_name="wrong",
                name="electricity, medium voltage",
                amount=10.0,
                unit="kWh",
                direction="input",
                flow_kind="energy",
                basis="per kg H2",
                evidence=[
                    SourceEvidence(
                        source_document="supplement.docx",
                        evidence_text="Electricity: 10 kWh/kg H2.",
                    )
                ],
            ),
            InventoryFlow(
                process_id="p999",
                technology_group="Methane pyrolysis",
                process_name="Invented purification process",
                name="electricity",
                amount=2.0,
                unit="kWh",
                direction="input",
                flow_kind="energy",
                evidence=[SourceEvidence(evidence_text="Electricity: 2 kWh.")],
            ),
        ],
    )

    result = validate_inventory_against_process_map(
        extraction,
        process_map,
        "[DOCUMENT supplement.docx]\nElectricity: 10 kWh/kg H2.",
    )

    assert len(result.flows) == 1
    assert result.flows[0].name == "electricity"
    assert result.flows[0].technology_group == "Methane pyrolysis"
    assert result.flows[0].process_name.startswith("Hydrogen production")
    assert any("unapproved process ID" in warning for warning in result.assumptions_or_warnings)
    assert any("voltage-level" in warning for warning in result.assumptions_or_warnings)


def test_inventory_validation_rejects_detected_but_unapproved_process():
    process_map = _pyrolysis_process_map()
    second = ForegroundProcess(
        name="Separately modelled carbon treatment",
        evidence_type="inventory_table",
        reason_for_separate_process="Separate inventory table.",
        confidence="high",
        evidence=[SourceEvidence(page=4, evidence_text="Carbon treatment inventory.")],
    )
    process_map.technology_groups[0].processes.append(second)
    process_map = normalise_process_ids(process_map)

    extraction = InventoryExtraction(
        functional_unit="1 kg H2",
        source_summary="Test source",
        flows=[
            InventoryFlow(
                process_id="p002",
                technology_group="Methane pyrolysis",
                process_name="Separately modelled carbon treatment",
                name="electricity",
                amount=1.0,
                unit="kWh",
                direction="input",
                flow_kind="energy",
                evidence=[SourceEvidence(evidence_text="Electricity: 1 kWh.")],
            )
        ],
    )

    result = validate_inventory_against_process_map(
        extraction,
        process_map,
        "Electricity: 1 kWh.",
        allowed_process_ids={"p001"},
    )

    assert result.flows == []
    assert any("unapproved process ID 'p002'" in warning for warning in result.assumptions_or_warnings)


def test_only_quantified_inputs_default_to_background_matching():
    process_map = _pyrolysis_process_map()
    process = process_map.technology_groups[0].processes[0]
    extraction = InventoryExtraction(
        functional_unit="1 kg H2",
        source_summary="Test source",
        flows=[
            InventoryFlow(
                process_id=process.process_id,
                technology_group="Methane pyrolysis",
                process_name=process.name,
                name="natural gas",
                amount=4.0,
                unit="kg",
                direction="input",
                flow_kind="material",
                evidence=[SourceEvidence(evidence_text="Natural gas: 4 kg.")],
            ),
            InventoryFlow(
                process_id=process.process_id,
                technology_group="Methane pyrolysis",
                process_name=process.name,
                name="hydrogen",
                amount=1.0,
                unit="kg",
                direction="output",
                flow_kind="product",
                evidence=[SourceEvidence(evidence_text="Hydrogen output: 1 kg.")],
            ),
            InventoryFlow(
                process_id=process.process_id,
                technology_group="Methane pyrolysis",
                process_name=process.name,
                name="graphite electrode",
                amount=None,
                unit="kg",
                direction="input",
                flow_kind="material",
                evidence=[SourceEvidence(evidence_text="Graphite electrodes are consumed.")],
            ),
        ],
    )

    df = extraction_to_dataframe(extraction)
    assert list(df["include"]) == [True, False, False]
    assert list(df["background_match_eligible"]) == [True, False, False]


def test_one_flow_can_be_supported_across_multiple_documents():
    process_map = _pyrolysis_process_map()
    process = process_map.technology_groups[0].processes[0]
    extraction = InventoryExtraction(
        source_summary="Combined evidence",
        flows=[
            InventoryFlow(
                process_id=process.process_id,
                technology_group="Methane pyrolysis",
                process_name=process.name,
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
                        evidence_text="The plasma reactor is electrically heated.",
                    ),
                    SourceEvidence(
                        source_document="supplement.docx",
                        table="Table S3",
                        evidence_text="Electricity | 10 kWh/kg H2",
                    ),
                ],
            )
        ],
    )

    df = extraction_to_dataframe(extraction)
    assert len(df) == 1
    assert "paper.pdf" in df.iloc[0]["source_documents"]
    assert "supplement.docx" in df.iloc[0]["source_documents"]
    assert "The plasma reactor is electrically heated." in df.iloc[0]["evidence_text"]
    assert "Electricity | 10 kWh/kg H2" in df.iloc[0]["evidence_text"]
