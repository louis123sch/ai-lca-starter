from uuid import uuid4

import bw2data as bd
import pandas as pd

from ai_lca.brightway_writer import write_foreground_database
from ai_lca.models import ForegroundProcess, InventoryExtraction, StudyContext


def test_writer_creates_reviewed_database_in_temporary_brightway_project():
    project_name = f"ai-lca-test-{uuid4().hex}"
    bd.projects.set_current(project_name)
    try:
        bd.Database("background").write(
            {
                ("background", "electricity"): {
                    "name": "electricity supply",
                    "reference product": "electricity",
                    "unit": "kilowatt hour",
                    "location": "GLO",
                    "type": "process",
                    "exchanges": [
                        {
                            "input": ("background", "electricity"),
                            "amount": 1.0,
                            "type": "production",
                        }
                    ],
                }
            }
        )
        bd.Database("biosphere-test").write(
            {
                ("biosphere-test", "co2"): {
                    "name": "Carbon dioxide",
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                }
            }
        )

        extraction = InventoryExtraction(
            process_name="Synthetic product",
            functional_unit="1 kg product",
            source_summary="Temporary Brightway integration test",
            study_context=StudyContext(
                operational_geography="Germany",
                geography_basis="explicit",
                geography_rationale="The synthetic source explicitly states Germany.",
            ),
            processes=[
                ForegroundProcess(
                    process_id="p1",
                    name="Synthetic product production",
                    reference_product="synthetic product",
                    reference_unit="kg",
                )
            ],
        )
        inventory = pd.DataFrame(
            [
                {
                    "include": True,
                    "flow_id": 0,
                    "process_id": "p1",
                    "name": "electricity",
                    "amount": 4.0,
                    "unit": "kWh",
                    "direction": "input",
                    "linked_process_id": "",
                },
                {
                    "include": True,
                    "flow_id": 1,
                    "process_id": "p1",
                    "name": "carbon dioxide",
                    "amount": 0.2,
                    "unit": "kg",
                    "direction": "emission",
                    "linked_process_id": "",
                },
            ]
        )
        mappings = pd.DataFrame(
            [
                {
                    "flow_id": 0,
                    "database": "background",
                    "code": "electricity",
                    "name": "electricity supply",
                    "unit": "kilowatt hour",
                },
                {
                    "flow_id": 1,
                    "database": "biosphere-test",
                    "code": "co2",
                    "name": "Carbon dioxide",
                    "unit": "kilogram",
                },
            ]
        )

        report = write_foreground_database(
            project_name=project_name,
            database_name="reviewed-foreground",
            extraction=extraction,
            inventory_df=inventory,
            mapping_df=mappings,
        )

        assert report["processes_created"] == 1
        assert report["exchanges_created"] == 2
        assert report["brightway_location"] == "DE"
        assert report["paper_geography"] == "Germany"
        activity = bd.Database("reviewed-foreground").get("p1")
        assert activity["location"] == "DE"
        assert activity["ai_lca_operational_geography"] == "Germany"
        exchanges = list(activity.exchanges())
        assert len(exchanges) == 3  # production + technosphere input + biosphere emission
        assert any(exc["type"] == "production" for exc in exchanges)
        assert any(exc.input.key == ("background", "electricity") for exc in exchanges)
        assert any(exc.input.key == ("biosphere-test", "co2") for exc in exchanges)
    finally:
        if project_name in bd.projects:
            bd.projects.delete_project(project_name, delete_dir=True)
