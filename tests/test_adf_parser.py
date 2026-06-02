"""
Tests for AdfParser — pipeline JSON and dataset JSON paths. No LLM calls.
JSON payloads are built inline so tests have no external file dependencies.
"""
import json
import pytest

from src.parsers.adf_parser import AdfParser, _ADF_NS
from src.graph.models import SystemType, ConfidenceLevel


def _parse(data: dict, filename: str = "pipeline.json", artifact_type: str | None = None):
    artifact = {
        "filename": filename,
        "content": json.dumps(data),
    }
    if artifact_type:
        artifact["artifact_type"] = artifact_type
    return AdfParser().parse(artifact)


# ---------------------------------------------------------------------------
# Helpers for building ADF JSON fixtures
# ---------------------------------------------------------------------------

def _pipeline(activities: list[dict]) -> dict:
    return {"name": "TestPipeline", "properties": {"activities": activities}}


def _copy_activity(name: str, sources: list[str], sinks: list[str]) -> dict:
    return {
        "name": name,
        "type": "Copy",
        "inputs":  [{"referenceName": s, "type": "DatasetReference"} for s in sources],
        "outputs": [{"referenceName": s, "type": "DatasetReference"} for s in sinks],
        "typeProperties": {
            "source": {"type": "ParquetSource"},
            "sink":   {"type": "ParquetSink"},
        },
    }


def _dataflow_activity(name: str, df_name: str, sources: list[str], sinks: list[str]) -> dict:
    return {
        "name": name,
        "type": "ExecuteDataFlow",
        "inputs":  [{"referenceName": s, "type": "DatasetReference"} for s in sources],
        "outputs": [{"referenceName": s, "type": "DatasetReference"} for s in sinks],
        "typeProperties": {
            "dataflow": {"referenceName": df_name, "type": "DataFlowReference"},
        },
    }


def _notebook_activity(name: str, nb_path: str, sources: list[str], sinks: list[str]) -> dict:
    return {
        "name": name,
        "type": "DatabricksNotebook",
        "inputs":  [{"referenceName": s, "type": "DatasetReference"} for s in sources],
        "outputs": [{"referenceName": s, "type": "DatasetReference"} for s in sinks],
        "typeProperties": {"notebookPath": nb_path},
    }


def _dataset(name: str, ds_type: str, type_props: dict, ls_name: str = "LS1") -> dict:
    return {
        "name": name,
        "properties": {
            "linkedServiceName": {"referenceName": ls_name, "type": "LinkedServiceReference"},
            "type": ds_type,
            "typeProperties": type_props,
        },
    }


# ===========================================================================
# Copy activities
# ===========================================================================

class TestCopyActivity:

    def test_single_copy_creates_one_edge(self):
        frag = _parse(_pipeline([_copy_activity("CopyOrders", ["RawOrdersDS"], ["SilverOrdersDS"])]))
        assert len(frag.edges) == 1

    def test_source_and_target_have_adf_namespace(self):
        frag = _parse(_pipeline([_copy_activity("Copy1", ["SrcDS"], ["TgtDS"])]))
        paths = {n.full_path for n in frag.nodes}
        assert f"{_ADF_NS}SrcDS" in paths
        assert f"{_ADF_NS}TgtDS" in paths

    def test_transformation_contains_activity_name(self):
        frag = _parse(_pipeline([_copy_activity("CopyOrders", ["Src"], ["Tgt"])]))
        assert "CopyOrders" in frag.edges[0].transformation

    def test_transformation_starts_with_copy(self):
        frag = _parse(_pipeline([_copy_activity("CopyOrders", ["Src"], ["Tgt"])]))
        assert frag.edges[0].transformation.startswith("copy:")

    def test_confidence_is_high(self):
        frag = _parse(_pipeline([_copy_activity("C", ["S"], ["T"])]))
        assert frag.edges[0].confidence == 0.85
        assert frag.edges[0].confidence_level == ConfidenceLevel.HIGH

    def test_evidence_mentions_pipeline_json(self):
        frag = _parse(_pipeline([_copy_activity("C", ["S"], ["T"])]))
        assert "pipeline" in frag.edges[0].evidence.lower()

    def test_multiple_copy_activities(self):
        pipeline = _pipeline([
            _copy_activity("CopyA", ["RawA"], ["SilverA"]),
            _copy_activity("CopyB", ["RawB"], ["SilverB"]),
        ])
        frag = _parse(pipeline)
        assert len(frag.edges) == 2

    def test_nodes_deduplicated_when_dataset_shared(self):
        pipeline = _pipeline([
            _copy_activity("CopyA", ["SharedRaw"], ["SilverA"]),
            _copy_activity("CopyB", ["SharedRaw"], ["SilverB"]),
        ])
        frag = _parse(pipeline)
        paths = [n.full_path for n in frag.nodes]
        assert paths.count(f"{_ADF_NS}SharedRaw") == 1


# ===========================================================================
# Nested (container) activities
# ===========================================================================

class TestNestedActivities:

    def test_foreach_inner_activities_extracted(self):
        pipeline = _pipeline([{
            "name": "ForEachPartition",
            "type": "ForEach",
            "typeProperties": {
                "items": "@pipeline().parameters.partitions",
                "activities": [_copy_activity("InnerCopy", ["PartDS"], ["SilverDS"])],
            },
        }])
        frag = _parse(pipeline)
        assert len(frag.edges) == 1

    def test_if_condition_true_branch_extracted(self):
        pipeline = _pipeline([{
            "name": "IfFull",
            "type": "IfCondition",
            "typeProperties": {
                "expression": {"value": "@equals(1,1)", "type": "Expression"},
                "ifTrueActivities":  [_copy_activity("FullLoad",  ["Src"], ["TgtFull"])],
                "ifFalseActivities": [_copy_activity("DeltaLoad", ["Src"], ["TgtDelta"])],
            },
        }])
        frag = _parse(pipeline)
        assert len(frag.edges) == 2

    def test_switch_cases_extracted(self):
        pipeline = _pipeline([{
            "name": "SwitchEnv",
            "type": "Switch",
            "typeProperties": {
                "on": {"value": "@pipeline().parameters.env", "type": "Expression"},
                "cases": [
                    {"value": "prod",  "activities": [_copy_activity("ProdCopy", ["ProdSrc"], ["ProdTgt"])]},
                    {"value": "dev",   "activities": [_copy_activity("DevCopy",  ["DevSrc"],  ["DevTgt"])]},
                ],
                "defaultActivities": [],
            },
        }])
        frag = _parse(pipeline)
        assert len(frag.edges) == 2


# ===========================================================================
# ExecuteDataFlow activity
# ===========================================================================

class TestDataFlowActivity:

    def test_dataflow_creates_edge(self):
        pipeline = _pipeline([
            _dataflow_activity("RunTransform", "SalesDataFlow", ["RawSales"], ["SilverSales"])
        ])
        frag = _parse(pipeline)
        assert len(frag.edges) == 1

    def test_transformation_contains_dataflow_name(self):
        frag = _parse(_pipeline([
            _dataflow_activity("Run", "MyDataFlow", ["Src"], ["Tgt"])
        ]))
        assert "MyDataFlow" in frag.edges[0].transformation


# ===========================================================================
# DatabricksNotebook activity
# ===========================================================================

class TestDatabricksActivity:

    def test_notebook_activity_creates_edge(self):
        pipeline = _pipeline([
            _notebook_activity("RunNB", "/Shared/orders_etl", ["RawDS"], ["SilverDS"])
        ])
        frag = _parse(pipeline)
        assert len(frag.edges) == 1

    def test_transformation_contains_notebook_path(self):
        frag = _parse(_pipeline([
            _notebook_activity("RunNB", "/Shared/orders_etl", ["R"], ["S"])
        ]))
        assert "/Shared/orders_etl" in frag.edges[0].transformation


# ===========================================================================
# Dataset JSON parsing
# ===========================================================================

class TestDatasetParsing:

    def test_parquet_dataset_gets_adls_system(self):
        ds = _dataset("RawOrdersDS", "Parquet", {
            "location": {
                "type": "AzureBlobFSLocation",
                "fileSystem": "raw",
                "folderPath": "orders",
                "fileName": "orders.parquet",
            }
        })
        frag = _parse(ds, filename="RawOrdersDS.json")
        assert len(frag.nodes) == 1
        assert frag.nodes[0].system == SystemType.ADLS

    def test_dataset_node_uses_adf_namespace(self):
        ds = _dataset("MyDataset", "Parquet", {})
        frag = _parse(ds, filename="MyDataset.json")
        assert frag.nodes[0].full_path == f"{_ADF_NS}MyDataset"

    def test_adls_gen2_path_extracted_as_description(self):
        ds = _dataset("RawDS", "AzureBlobFSFile", {
            "location": {
                "type": "AzureBlobFSLocation",
                "fileSystem": "raw",
                "folderPath": "orders",
                "fileName": "orders.parquet",
            }
        })
        frag = _parse(ds, filename="RawDS.json")
        assert frag.nodes[0].description == "raw/orders/orders.parquet"

    def test_sql_table_dataset(self):
        ds = _dataset("SilverOrdersDS", "AzureSqlTable", {"schema": "silver", "table": "orders"})
        frag = _parse(ds, filename="SilverOrdersDS.json")
        node = frag.nodes[0]
        assert node.system == SystemType.SQL
        assert "orders" in node.description

    def test_databricks_delta_lake_dataset(self):
        ds = _dataset("DatabricksDS", "AzureDatabricksDeltaLakeDataset", {
            "database": "silver", "table": "orders"
        })
        frag = _parse(ds, filename="DatabricksDS.json")
        node = frag.nodes[0]
        assert node.system == SystemType.DATABRICKS

    def test_old_style_folder_path(self):
        ds = _dataset("OldDS", "Parquet", {"folderPath": "raw/data", "fileName": "file.parquet"})
        frag = _parse(ds, filename="OldDS.json")
        assert frag.nodes[0].description == "raw/data/file.parquet"


# ===========================================================================
# can_parse
# ===========================================================================

class TestCanParse:

    def test_explicit_artifact_type(self):
        assert AdfParser().can_parse({"filename": "x.json", "content": "{}", "artifact_type": "adf"})

    def test_pipeline_json_detected_by_content(self):
        content = json.dumps(_pipeline([]))
        assert AdfParser().can_parse({"filename": "p.json", "content": content})

    def test_dataset_json_detected_by_content(self):
        content = json.dumps(_dataset("D", "Parquet", {}))
        assert AdfParser().can_parse({"filename": "d.json", "content": content})

    def test_non_json_not_parsed(self):
        assert not AdfParser().can_parse({"filename": "file.sql", "content": "SELECT 1"})

    def test_unrelated_json_not_parsed(self):
        assert not AdfParser().can_parse({"filename": "config.json", "content": '{"foo": "bar"}'})

    def test_linked_service_parsed_without_error(self):
        ls = {"name": "LS", "properties": {"type": "AzureBlobFS", "typeProperties": {"url": "https://..."}}}
        # should not crash — returns empty fragment (no lineage from LS files)
        frag = _parse(ls, filename="LS.json")
        assert frag is not None


# ===========================================================================
# Artifact type is preserved
# ===========================================================================

def test_artifact_type_is_adf():
    frag = _parse(_pipeline([_copy_activity("C", ["S"], ["T"])]))
    assert frag.artifact_type == "adf"
