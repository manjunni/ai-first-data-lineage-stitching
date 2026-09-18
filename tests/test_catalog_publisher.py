"""
Tests for the catalog publisher — payload serialisation and dry-run path.
No real HTTP calls are made.
"""
import uuid
import pytest
from unittest.mock import patch, MagicMock

from src.catalog.publisher import (
    AlationPublisher, CollibraPublisher, DryRunPublisher,
    PublishResult, get_publisher,
    _graph_to_alation_payload, _graph_to_collibra_payload,
    _node_to_alation_obj,
)
from src.graph.stitcher import stitch_fragments
from src.graph.models import (
    LineageFragment, LineageNode, LineageEdge,
    LineageGraph, SystemType, ConfidenceLevel, ColumnMapping,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(name: str, system: SystemType = SystemType.DATABRICKS,
          full_path: str | None = None, description: str = "") -> LineageNode:
    return LineageNode(name=name, system=system,
                       full_path=full_path or name, description=description or None)


def _edge(src: LineageNode, tgt: LineageNode,
          confidence: float = 0.90,
          transformation: str = "copy",
          sme: str | None = None,
          column_mappings: list[ColumnMapping] | None = None) -> LineageEdge:
    return LineageEdge(
        source_id=src.id, target_id=tgt.id,
        confidence=confidence, confidence_level=ConfidenceLevel.HIGH,
        transformation=transformation, evidence="test",
        sme_approved_by=sme,
        column_mappings=column_mappings or [],
    )


def _graph(*items) -> LineageGraph:
    f = LineageFragment(artifact_name="t", artifact_type="t")
    for item in items:
        if isinstance(item, LineageNode):
            f.nodes.append(item)
        else:
            f.edges.append(item)
    return stitch_fragments([f])


def _full_graph() -> LineageGraph:
    raw   = _node("raw_orders",   system=SystemType.ADLS,       full_path="adls://raw/orders")
    clean = _node("silver_orders",system=SystemType.DATABRICKS, full_path="silver.orders")
    gold  = _node("gold_summary", system=SystemType.DATABRICKS, full_path="gold.summary")
    report= _node("sales_report", system=SystemType.BI,         full_path="bi.sales")
    return _graph(
        raw, clean, gold, report,
        _edge(raw,   clean, confidence=0.85, transformation="filter+cast"),
        _edge(clean, gold,  confidence=0.90, transformation="aggregated",  sme="alice@corp.com"),
        _edge(gold,  report,confidence=0.92, transformation="direct copy"),
    )


# ===========================================================================
# Alation payload serialisation
# ===========================================================================

class TestAlationPayload:

    def test_column_mappings_are_serialized(self):
        src = _node("customers", full_path="silver.customers")
        tgt = _node("orders", full_path="gold.orders")
        mapping = ColumnMapping(
            source_column="first_name",
            target_column="full_name",
            transformation="CONCAT(first_name, last_name)",
            confidence=0.85,
            confidence_level=ConfidenceLevel.HIGH,
            evidence="notebook.py",
            sme_approved_by="alice@corp.com",
        )
        p = _graph_to_alation_payload(
            _graph(src, tgt, _edge(src, tgt, column_mappings=[mapping])), "r"
        )
        assert p["lineage"][0]["column_mappings"] == [{
            "source_column": "first_name",
            "target_column": "full_name",
            "transformation": "CONCAT(first_name, last_name)",
            "confidence": 0.85,
            "confidence_level": "high",
            "evidence": "notebook.py",
            "sme_approved_by": "alice@corp.com",
        }]

    def test_column_mappings_are_empty_for_table_only_edge(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        p = _graph_to_alation_payload(_graph(src, tgt, _edge(src, tgt)), "r")
        assert p["lineage"][0]["column_mappings"] == []

    def test_edge_approval_is_inherited_by_column_mapping(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        mapping = ColumnMapping(source_column="id", target_column="id")
        p = _graph_to_alation_payload(
            _graph(src, tgt, _edge(src, tgt, sme="owner@corp.com",
                                   column_mappings=[mapping])), "r"
        )
        assert p["lineage"][0]["column_mappings"][0]["sme_approved_by"] == "owner@corp.com"

    def test_payload_has_correct_top_level_keys(self):
        g = _full_graph()
        p = _graph_to_alation_payload(g, "run-1")
        assert "run_id" in p
        assert "lineage" in p
        assert "node_count" in p
        assert "edge_count" in p

    def test_one_lineage_item_per_edge(self):
        g = _full_graph()
        p = _graph_to_alation_payload(g, "run-1")
        assert len(p["lineage"]) == len(g.edges)

    def test_lineage_item_has_source_and_target(self):
        g = _full_graph()
        p = _graph_to_alation_payload(g, "run-1")
        for item in p["lineage"]:
            assert "source_objects" in item
            assert "target_objects" in item
            assert len(item["source_objects"]) == 1
            assert len(item["target_objects"]) == 1

    def test_dataflow_external_id_contains_run_id(self):
        g = _full_graph()
        p = _graph_to_alation_payload(g, "my-run-42")
        for item in p["lineage"]:
            assert "my-run-42" in item["dataflow"]["external_id"]

    def test_transformation_in_dataflow_name(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        g = _graph(src, tgt, _edge(src, tgt, transformation="filtered, aggregated"))
        p = _graph_to_alation_payload(g, "r")
        assert "filtered" in p["lineage"][0]["dataflow"]["name"]

    def test_sme_approval_in_description(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        g = _graph(src, tgt, _edge(src, tgt, sme="bob@corp.com"))
        p = _graph_to_alation_payload(g, "r")
        assert "bob@corp.com" in p["lineage"][0]["dataflow"]["description"]

    def test_confidence_in_description(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        g = _graph(src, tgt, _edge(src, tgt, confidence=0.85))
        p = _graph_to_alation_payload(g, "r")
        assert "85%" in p["lineage"][0]["dataflow"]["description"]

    def test_node_counts_match_graph(self):
        g = _full_graph()
        p = _graph_to_alation_payload(g, "r")
        assert p["node_count"] == len(g.nodes)
        assert p["edge_count"] == len(g.edges)

    def test_empty_graph_produces_empty_lineage(self):
        g = LineageGraph()
        p = _graph_to_alation_payload(g, "r")
        assert p["lineage"] == []

    def test_dangling_edge_excluded(self):
        src = _node("a", full_path="a")
        g = LineageGraph()
        g.nodes[src.id] = src
        # Edge with target that doesn't exist in nodes
        orphan_edge = LineageEdge(
            source_id=src.id, target_id="nonexistent",
            confidence=0.9, evidence="t"
        )
        g.edges.append(orphan_edge)
        p = _graph_to_alation_payload(g, "r")
        assert len(p["lineage"]) == 0


class TestNodeToAlationObj:

    def test_adls_node_maps_to_schema(self):
        n = _node("raw", system=SystemType.ADLS, full_path="adls://raw")
        obj = _node_to_alation_obj(n)
        assert obj["object_type"] == "SCHEMA"

    def test_databricks_node_maps_to_table(self):
        n = _node("t", system=SystemType.DATABRICKS, full_path="silver.t")
        obj = _node_to_alation_obj(n)
        assert obj["object_type"] == "TABLE"

    def test_sql_node_maps_to_table(self):
        n = _node("t", system=SystemType.SQL, full_path="dbo.t")
        obj = _node_to_alation_obj(n)
        assert obj["object_type"] == "TABLE"

    def test_bi_node_maps_to_report(self):
        n = _node("r", system=SystemType.BI, full_path="bi.r")
        obj = _node_to_alation_obj(n)
        assert obj["object_type"] == "REPORT"

    def test_node_id_is_external_id(self):
        n = _node("t", full_path="t")
        obj = _node_to_alation_obj(n)
        assert obj["external_id"] == n.id

    def test_full_path_is_fqn(self):
        n = _node("orders", full_path="silver.orders")
        obj = _node_to_alation_obj(n)
        assert obj["fully_qualified_name"] == "silver.orders"

    def test_description_included(self):
        n = _node("t", full_path="t", description="important table")
        obj = _node_to_alation_obj(n)
        assert obj["description"] == "important table"


# ===========================================================================
# Collibra payload serialisation
# ===========================================================================

class TestCollibraPayload:

    def test_column_mappings_are_serialized_on_relation(self):
        src = _node("customers", full_path="silver.customers")
        tgt = _node("orders", full_path="gold.orders")
        mapping = ColumnMapping(
            source_column="customer_id",
            target_column="customer_id",
            confidence=0.9,
            confidence_level=ConfidenceLevel.HIGH,
            evidence="sqlglot",
        )
        p = _graph_to_collibra_payload(
            _graph(src, tgt, _edge(src, tgt, column_mappings=[mapping])), "r"
        )
        assert p["relations"][0]["column_mappings"][0]["source_column"] == "customer_id"
        assert p["relations"][0]["column_mappings"][0]["confidence_level"] == "high"

    def test_rejected_edge_does_not_serialize_column_mappings(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        edge = _edge(src, tgt, column_mappings=[
            ColumnMapping(source_column="id", target_column="id")
        ])
        edge.review_status = "rejected"
        p = _graph_to_collibra_payload(_graph(src, tgt, edge), "r")
        assert p["relations"] == []

    def test_payload_has_assets_and_relations(self):
        g = _full_graph()
        p = _graph_to_collibra_payload(g, "run-1")
        assert "assets" in p
        assert "relations" in p

    def test_one_asset_per_node(self):
        g = _full_graph()
        p = _graph_to_collibra_payload(g, "run-1")
        assert len(p["assets"]) == len(g.nodes)

    def test_one_relation_per_edge(self):
        g = _full_graph()
        p = _graph_to_collibra_payload(g, "run-1")
        assert len(p["relations"]) == len(g.edges)

    def test_asset_external_id_is_deterministic(self):
        """Same full_path always produces the same external_id (uuid5)."""
        g = _full_graph()
        p1 = _graph_to_collibra_payload(g, "run-1")
        p2 = _graph_to_collibra_payload(g, "run-2")  # different run, same graph
        ids_1 = {a["external_id"] for a in p1["assets"]}
        ids_2 = {a["external_id"] for a in p2["assets"]}
        assert ids_1 == ids_2

    def test_no_duplicate_assets_for_same_full_path(self):
        # Two separate node objects with the same full_path should produce one asset
        n1 = _node("orders", full_path="silver.orders")
        n2 = _node("orders", full_path="silver.orders")
        n3 = _node("summary", full_path="gold.summary")
        g = _graph(n1, n2, n3)
        p = _graph_to_collibra_payload(g, "r")
        paths = [a["full_path"] for a in p["assets"]]
        assert paths.count("silver.orders") == 1

    def test_asset_type_name_for_databricks(self):
        n = _node("t", system=SystemType.DATABRICKS, full_path="db.t")
        g = _graph(n)
        p = _graph_to_collibra_payload(g, "r")
        assert p["assets"][0]["asset_type"] == "Table"

    def test_asset_type_name_for_adls(self):
        n = _node("f", system=SystemType.ADLS, full_path="adls://f")
        g = _graph(n)
        p = _graph_to_collibra_payload(g, "r")
        assert p["assets"][0]["asset_type"] == "File"

    def test_asset_type_name_for_bi(self):
        n = _node("r", system=SystemType.BI, full_path="bi.r")
        g = _graph(n)
        p = _graph_to_collibra_payload(g, "r")
        assert p["assets"][0]["asset_type"] == "Report"

    def test_relation_references_correct_external_ids(self):
        src = _node("a", full_path="src_path")
        tgt = _node("b", full_path="tgt_path")
        g = _graph(src, tgt, _edge(src, tgt))
        p = _graph_to_collibra_payload(g, "r")
        rel = p["relations"][0]
        asset_ext_ids = {a["external_id"] for a in p["assets"]}
        assert rel["source_external_id"] in asset_ext_ids
        assert rel["target_external_id"] in asset_ext_ids

    def test_confidence_in_relation_description(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        g = _graph(src, tgt, _edge(src, tgt, confidence=0.85))
        p = _graph_to_collibra_payload(g, "r")
        assert "85%" in p["relations"][0]["description"]

    def test_sme_approved_in_relation(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        g = _graph(src, tgt, _edge(src, tgt, sme="carol@corp.com"))
        p = _graph_to_collibra_payload(g, "r")
        assert p["relations"][0]["sme_approved_by"] == "carol@corp.com"

    def test_system_boundary_in_relation(self):
        src = _node("a", system=SystemType.ADLS,       full_path="adls://a")
        tgt = _node("b", system=SystemType.DATABRICKS, full_path="db.b")
        g = _graph(src, tgt, _edge(src, tgt))
        p = _graph_to_collibra_payload(g, "r")
        assert "adls→databricks" in p["relations"][0]["system_boundary"]

    def test_empty_graph_produces_empty_payload(self):
        g = LineageGraph()
        p = _graph_to_collibra_payload(g, "r")
        assert p["assets"] == []
        assert p["relations"] == []


# ===========================================================================
# DryRunPublisher
# ===========================================================================

class TestDryRunPublisher:

    def test_preview_contains_column_mappings_for_alation_and_collibra(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        mapping = ColumnMapping(
            source_column="id",
            target_column="id",
            transformation="cast(id as bigint)",
        )
        g = _graph(src, tgt, _edge(src, tgt, column_mappings=[mapping]))
        alation = DryRunPublisher("alation").publish(g, "r").payload
        collibra = DryRunPublisher("collibra").publish(g, "r").payload
        assert alation["lineage"][0]["column_mappings"][0]["transformation"] == "cast(id as bigint)"
        assert collibra["relations"][0]["column_mappings"][0]["target_column"] == "id"

    def test_status_is_dry_run(self):
        g = _full_graph()
        result = DryRunPublisher().publish(g, "r1")
        assert result.status == "dry_run"

    def test_success_is_true(self):
        g = _full_graph()
        result = DryRunPublisher().publish(g, "r1")
        assert result.success is True

    def test_payload_included(self):
        g = _full_graph()
        result = DryRunPublisher().publish(g, "r1")
        assert result.payload is not None

    def test_node_and_edge_counts(self):
        g = _full_graph()
        result = DryRunPublisher().publish(g, "r1")
        assert result.nodes_published == len(g.nodes)
        assert result.edges_published == len(g.edges)

    def test_no_errors(self):
        g = _full_graph()
        result = DryRunPublisher().publish(g, "r1")
        assert result.errors == []

    def test_collibra_target(self):
        g = _full_graph()
        result = DryRunPublisher(target_catalog="collibra").publish(g, "r1")
        assert "collibra" in result.catalog
        assert "assets" in result.payload

    def test_alation_target(self):
        g = _full_graph()
        result = DryRunPublisher(target_catalog="alation").publish(g, "r1")
        assert "lineage" in result.payload

    def test_empty_graph_succeeds(self):
        result = DryRunPublisher().publish(LineageGraph(), "r1")
        assert result.success is True
        assert result.nodes_published == 0


# ===========================================================================
# AlationPublisher — missing config path
# ===========================================================================

class TestAlationPublisherConfig:

    def test_missing_config_returns_skipped(self):
        with patch("src.catalog.publisher.settings") as s:
            s.alation_base_url = ""
            s.alation_api_token = ""
            s.alation_lineage_parent_path = "/test"
            s.collibra_lineage_type_id = ""
            s.collibra_domain_id = ""
            result = AlationPublisher().publish(_full_graph(), "run-1")
        assert result.status == "skipped"
        assert result.errors

    def test_validate_config_lists_missing_keys(self):
        with patch("src.catalog.publisher.settings") as s:
            s.alation_base_url = ""
            s.alation_api_token = ""
            missing = AlationPublisher().validate_config()
        assert "ALATION_BASE_URL" in missing
        assert "ALATION_API_TOKEN" in missing

    def test_validate_config_empty_when_configured(self):
        with patch("src.catalog.publisher.settings") as s:
            s.alation_base_url = "https://org.alation.com"
            s.alation_api_token = "secret"
            missing = AlationPublisher().validate_config()
        assert missing == []

    def test_payload_included_even_when_skipped(self):
        with patch("src.catalog.publisher.settings") as s:
            s.alation_base_url = ""
            s.alation_api_token = ""
            s.alation_lineage_parent_path = "/test"
            s.collibra_lineage_type_id = ""
            s.collibra_domain_id = ""
            result = AlationPublisher().publish(_full_graph(), "run-1")
        assert result.payload is not None


# ===========================================================================
# CollibraPublisher — missing config path
# ===========================================================================

class TestCollibraPublisherConfig:

    def test_missing_config_returns_skipped(self):
        with patch("src.catalog.publisher.settings") as s:
            s.collibra_base_url = ""
            s.collibra_username = ""
            s.collibra_password = ""
            s.collibra_domain_id = ""
            s.collibra_lineage_type_id = ""
            result = CollibraPublisher().publish(_full_graph(), "run-1")
        assert result.status == "skipped"

    def test_validate_config_lists_missing_keys(self):
        with patch("src.catalog.publisher.settings") as s:
            s.collibra_base_url = ""
            s.collibra_username = ""
            s.collibra_password = ""
            s.collibra_domain_id = ""
            missing = CollibraPublisher().validate_config()
        assert "COLLIBRA_BASE_URL" in missing
        assert "COLLIBRA_USERNAME" in missing


# ===========================================================================
# Factory
# ===========================================================================

class TestGetPublisher:

    def test_alation_returns_alation_publisher(self):
        assert isinstance(get_publisher("alation"), AlationPublisher)

    def test_collibra_returns_collibra_publisher(self):
        assert isinstance(get_publisher("collibra"), CollibraPublisher)

    def test_dry_run_returns_dry_run_publisher(self):
        assert isinstance(get_publisher("dry_run"), DryRunPublisher)

    def test_unknown_catalog_raises(self):
        with pytest.raises(ValueError, match="Unknown catalog"):
            get_publisher("nonexistent_catalog")


# ===========================================================================
# PublishResult
# ===========================================================================

class TestPublishResult:

    def test_success_true_when_published_no_errors(self):
        r = PublishResult(status="published", catalog="alation", run_id="r")
        assert r.success is True

    def test_success_true_for_dry_run(self):
        r = PublishResult(status="dry_run", catalog="dry_run:alation", run_id="r")
        assert r.success is True

    def test_success_false_when_errors(self):
        r = PublishResult(status="published", catalog="alation", run_id="r",
                          errors=["something failed"])
        assert r.success is False

    def test_success_false_for_skipped(self):
        r = PublishResult(status="skipped", catalog="alation", run_id="r")
        assert r.success is False
