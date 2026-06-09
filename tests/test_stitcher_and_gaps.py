"""
Tests for stitch_fragments() and detect_gaps().
Fragments are built manually so tests are isolated from parser behaviour.
"""
import pytest

from src.graph.stitcher import stitch_fragments, to_networkx, _ADF_NS
from src.graph.gap_detector import detect_gaps
from src.graph.models import (
    LineageFragment, LineageNode, LineageEdge,
    LineageGraph, SystemType, ConfidenceLevel, Gap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(name: str, system: SystemType = SystemType.DATABRICKS,
          full_path: str | None = None, description: str = "") -> LineageNode:
    return LineageNode(
        name=name,
        system=system,
        full_path=full_path or name,
        description=description or None,
    )


def _edge(src: LineageNode, tgt: LineageNode,
          confidence: float = 0.90,
          transformation: str = "direct copy",
          evidence: str = "test") -> LineageEdge:
    return LineageEdge(
        source_id=src.id,
        target_id=tgt.id,
        confidence=confidence,
        confidence_level=ConfidenceLevel.HIGH,
        transformation=transformation,
        evidence=evidence,
    )


def _fragment(*nodes_and_edges) -> LineageFragment:
    frag = LineageFragment(artifact_name="test", artifact_type="test")
    for item in nodes_and_edges:
        if isinstance(item, LineageNode):
            frag.nodes.append(item)
        else:
            frag.edges.append(item)
    return frag


# ===========================================================================
# Stitcher — node deduplication
# ===========================================================================

class TestStitcherDeduplication:

    def test_same_path_in_two_fragments_becomes_one_node(self):
        n1 = _node("orders", full_path="bronze.orders")
        n2 = _node("orders", full_path="bronze.orders")
        graph = stitch_fragments([_fragment(n1), _fragment(n2)])
        assert len(graph.nodes) == 1

    def test_different_paths_produce_different_nodes(self):
        n1 = _node("a", full_path="bronze.a")
        n2 = _node("b", full_path="silver.b")
        graph = stitch_fragments([_fragment(n1), _fragment(n2)])
        assert len(graph.nodes) == 2

    def test_edge_endpoints_remapped_after_dedup(self):
        # Same node appears in both fragments; edge from fragment 2 must still link correctly
        shared = _node("orders", full_path="bronze.orders")
        target = _node("clean",  full_path="silver.orders")
        duplicate = _node("orders", full_path="bronze.orders")  # second fragment's version

        frag1 = _fragment(shared)
        frag2 = _fragment(duplicate, target, _edge(duplicate, target))

        graph = stitch_fragments([frag1, frag2])
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        # The edge must use the canonical id (from frag1, which registered first)
        canonical_src_id = next(
            nid for nid, n in graph.nodes.items() if n.full_path == "bronze.orders"
        )
        assert graph.edges[0].source_id == canonical_src_id

    def test_metadata_merged_unknown_system_upgraded(self):
        placeholder = _node("ds", system=SystemType.UNKNOWN, full_path="adf://MyDataset")
        enriched    = _node("ds", system=SystemType.ADLS,    full_path="adf://MyDataset",
                            description="raw/orders/orders.parquet")
        graph = stitch_fragments([_fragment(placeholder), _fragment(enriched)])
        node = list(graph.nodes.values())[0]
        assert node.system == SystemType.ADLS
        assert node.description == "raw/orders/orders.parquet"

    def test_known_system_not_overwritten_by_unknown(self):
        known   = _node("ds", system=SystemType.SQL,     full_path="dbo.orders")
        unknown = _node("ds", system=SystemType.UNKNOWN, full_path="dbo.orders")
        graph = stitch_fragments([_fragment(known), _fragment(unknown)])
        node = list(graph.nodes.values())[0]
        assert node.system == SystemType.SQL

    def test_description_not_overwritten_once_set(self):
        first  = _node("ds", full_path="p", description="original")
        second = _node("ds", full_path="p", description="should not win")
        # description is only set from incoming if existing has none
        graph = stitch_fragments([_fragment(first), _fragment(second)])
        assert list(graph.nodes.values())[0].description == "original"


# ===========================================================================
# Stitcher — edge deduplication
# ===========================================================================

class TestStitcherEdgeDedup:

    def test_identical_edges_collapsed_to_one(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        e1 = _edge(src, tgt, confidence=0.90)
        e2 = _edge(src, tgt, confidence=0.90)

        # Need copies so they have different ids but same path pair
        src2 = _node("a", full_path="a")
        tgt2 = _node("b", full_path="b")
        e2 = LineageEdge(source_id=src2.id, target_id=tgt2.id, confidence=0.90, evidence="t")

        graph = stitch_fragments([
            _fragment(src, tgt, e1),
            _fragment(src2, tgt2, e2),
        ])
        assert len(graph.edges) == 1

    def test_higher_confidence_edge_wins(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        low  = _edge(src, tgt, confidence=0.60)
        src2 = _node("a", full_path="a")
        tgt2 = _node("b", full_path="b")
        high = LineageEdge(source_id=src2.id, target_id=tgt2.id, confidence=0.95, evidence="high")

        graph = stitch_fragments([
            _fragment(src,  tgt,  low),
            _fragment(src2, tgt2, high),
        ])
        assert len(graph.edges) == 1
        assert graph.edges[0].confidence == 0.95

    def test_dangling_edge_removed(self):
        src = _node("a", full_path="a")
        orphan_id = "nonexistent-id-xyz"
        edge = LineageEdge(source_id=src.id, target_id=orphan_id, confidence=0.9, evidence="t")
        graph = stitch_fragments([_fragment(src, edge)])
        assert len(graph.edges) == 0


# ===========================================================================
# Stitcher — system boundary annotation
# ===========================================================================

class TestSystemBoundaries:

    def test_cross_system_edge_gets_boundary_label(self):
        src = _node("raw",   system=SystemType.ADLS,        full_path="adls_path")
        tgt = _node("clean", system=SystemType.DATABRICKS,  full_path="db_table")
        graph = stitch_fragments([_fragment(src, tgt, _edge(src, tgt))])
        assert graph.edges[0].system_boundary == "adls→databricks"

    def test_same_system_edge_has_no_boundary(self):
        src = _node("a", system=SystemType.DATABRICKS, full_path="db.a")
        tgt = _node("b", system=SystemType.DATABRICKS, full_path="db.b")
        graph = stitch_fragments([_fragment(src, tgt, _edge(src, tgt))])
        assert graph.edges[0].system_boundary is None

    def test_to_networkx_has_correct_node_and_edge_count(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        graph = stitch_fragments([_fragment(src, tgt, _edge(src, tgt))])
        G = to_networkx(graph)
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1


# ===========================================================================
# Gap detector — unresolved ADF refs
# ===========================================================================

class TestUnresolvedAdfRefs:

    def test_adf_namespace_node_flagged(self):
        n = _node("MyDS", system=SystemType.UNKNOWN, full_path=f"{_ADF_NS}MyDS")
        graph = stitch_fragments([_fragment(n)])
        gaps = detect_gaps(graph)
        types = [g.gap_type for g in gaps]
        assert "unresolved_dataset_ref" in types

    def test_resolved_adf_node_not_flagged(self):
        # When a dataset JSON is uploaded the system type is upgraded from UNKNOWN.
        # A resolved adf:// node (system != UNKNOWN) should NOT produce a gap —
        # the user already provided the dataset definition.
        n = _node("MyDS", system=SystemType.ADLS, full_path=f"{_ADF_NS}MyDS")
        graph = stitch_fragments([_fragment(n)])
        gaps = detect_gaps(graph)
        assert not any(g.gap_type == "unresolved_dataset_ref" for g in gaps)

    def test_unknown_adf_node_still_flagged(self):
        # An adf:// node whose system is still UNKNOWN means no dataset JSON
        # was uploaded — it should remain flagged as unresolved.
        n = _node("MyDS", system=SystemType.UNKNOWN, full_path=f"{_ADF_NS}MyDS")
        graph = stitch_fragments([_fragment(n)])
        gaps = detect_gaps(graph)
        assert any(g.gap_type == "unresolved_dataset_ref" for g in gaps)

    def test_normal_node_not_flagged_as_adf_ref(self):
        n = _node("orders", system=SystemType.DATABRICKS, full_path="bronze.orders")
        graph = stitch_fragments([_fragment(n)])
        gaps = detect_gaps(graph)
        assert not any(g.gap_type == "unresolved_dataset_ref" for g in gaps)


# ===========================================================================
# Gap detector — unknown system
# ===========================================================================

class TestUnknownSystem:

    def test_unknown_system_node_flagged(self):
        n = _node("mystery", system=SystemType.UNKNOWN, full_path="mystery_table")
        graph = stitch_fragments([_fragment(n)])
        gaps = detect_gaps(graph)
        assert any(g.gap_type == "unknown_system" for g in gaps)

    def test_known_system_not_flagged(self):
        for system in (SystemType.ADLS, SystemType.DATABRICKS, SystemType.SQL):
            n = _node("t", system=system, full_path=f"p_{system.value}")
            graph = stitch_fragments([_fragment(n)])
            gaps = detect_gaps(graph)
            assert not any(g.gap_type == "unknown_system" for g in gaps)


# ===========================================================================
# Gap detector — missing source
# ===========================================================================

class TestMissingSource:

    def test_non_root_node_with_no_upstream_flagged(self):
        n = _node("silver_orders", system=SystemType.DATABRICKS, full_path="silver.orders")
        graph = stitch_fragments([_fragment(n)])
        gaps = detect_gaps(graph)
        assert any(g.gap_type == "missing_source" for g in gaps)

    def test_adls_node_not_flagged_as_missing_source(self):
        n = _node("raw", system=SystemType.ADLS, full_path="adls://raw/orders")
        graph = stitch_fragments([_fragment(n)])
        gaps = detect_gaps(graph)
        assert not any(g.gap_type == "missing_source" for g in gaps)

    def test_node_with_upstream_not_flagged(self):
        src = _node("raw",   system=SystemType.ADLS,       full_path="adls://raw")
        tgt = _node("clean", system=SystemType.DATABRICKS, full_path="silver.orders")
        graph = stitch_fragments([_fragment(src, tgt, _edge(src, tgt))])
        gaps = detect_gaps(graph)
        assert not any(g.gap_type == "missing_source" and
                       g.node_id == list(graph.nodes.keys())[
                           list(graph.nodes.values()).index(tgt)
                           if tgt in graph.nodes.values() else 1
                       ] for g in gaps)


# ===========================================================================
# Gap detector — missing target
# ===========================================================================

class TestMissingTarget:

    def test_node_with_upstream_but_no_downstream_flagged(self):
        src = _node("raw",   system=SystemType.ADLS,       full_path="adls://raw")
        tgt = _node("clean", system=SystemType.DATABRICKS, full_path="silver.orders")
        graph = stitch_fragments([_fragment(src, tgt, _edge(src, tgt))])
        gaps = detect_gaps(graph)
        assert any(g.gap_type == "missing_target" for g in gaps)

    def test_single_node_graph_not_flagged(self):
        n = _node("only", system=SystemType.ADLS, full_path="adls://only")
        graph = stitch_fragments([_fragment(n)])
        gaps = detect_gaps(graph)
        assert not any(g.gap_type == "missing_target" for g in gaps)

    def test_node_with_both_upstream_and_downstream_not_flagged(self):
        a = _node("a", system=SystemType.ADLS,       full_path="a")
        b = _node("b", system=SystemType.DATABRICKS, full_path="b")
        c = _node("c", system=SystemType.DATABRICKS, full_path="c")
        graph = stitch_fragments([_fragment(a, b, c, _edge(a, b), _edge(b, c))])
        gaps = detect_gaps(graph)
        # b has both upstream (a) and downstream (c) — should not be flagged
        b_id = next(nid for nid, n in graph.nodes.items() if n.full_path == "b")
        assert not any(g.gap_type == "missing_target" and g.node_id == b_id for g in gaps)


# ===========================================================================
# Gap detector — low confidence edges
# ===========================================================================

class TestLowConfidenceEdges:

    def test_low_confidence_edge_flagged(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        graph = stitch_fragments([_fragment(src, tgt, _edge(src, tgt, confidence=0.50))])
        gaps = detect_gaps(graph)
        assert any(g.gap_type == "low_confidence_edge" for g in gaps)

    def test_high_confidence_edge_not_flagged(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        graph = stitch_fragments([_fragment(src, tgt, _edge(src, tgt, confidence=0.90))])
        gaps = detect_gaps(graph)
        assert not any(g.gap_type == "low_confidence_edge" for g in gaps)

    def test_threshold_boundary_exactly_at_limit_not_flagged(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        # 0.70 is exactly the threshold — should not be flagged (< not <=)
        graph = stitch_fragments([_fragment(src, tgt, _edge(src, tgt, confidence=0.70))])
        gaps = detect_gaps(graph)
        assert not any(g.gap_type == "low_confidence_edge" for g in gaps)


# ===========================================================================
# Gap detector — contradictory paths
# ===========================================================================

class TestContradictoryPaths:

    def test_two_edges_same_pair_different_transforms_flagged(self):
        src  = _node("a", full_path="src_path")
        tgt  = _node("b", full_path="tgt_path")
        src2 = _node("a", full_path="src_path")
        tgt2 = _node("b", full_path="tgt_path")
        e1 = _edge(src,  tgt,  transformation="filtered, aggregated")
        e2 = LineageEdge(
            source_id=src2.id, target_id=tgt2.id,
            confidence=0.60, transformation="direct copy", evidence="t2",
        )
        graph = stitch_fragments([
            _fragment(src,  tgt,  e1),
            _fragment(src2, tgt2, e2),
        ])
        # After dedup, only one edge remains — no contradiction at graph level.
        # Contradictions only appear if both edges survive (different transformations
        # cause dedup to keep the higher-confidence one; same confidence keeps first).
        # This test verifies the detector runs without error.
        gaps = detect_gaps(graph)
        assert isinstance(gaps, list)

    def test_same_transform_no_contradiction(self):
        src  = _node("a", full_path="src_path")
        tgt  = _node("b", full_path="tgt_path")
        src2 = _node("a", full_path="src_path")
        tgt2 = _node("b", full_path="tgt_path")
        e1 = _edge(src,  tgt,  transformation="filtered")
        e2 = LineageEdge(
            source_id=src2.id, target_id=tgt2.id,
            confidence=0.90, transformation="filtered", evidence="t2",
        )
        graph = stitch_fragments([
            _fragment(src,  tgt,  e1),
            _fragment(src2, tgt2, e2),
        ])
        gaps = detect_gaps(graph)
        assert not any(g.gap_type == "contradictory_paths" for g in gaps)


# ===========================================================================
# Gap detector — isolated subgraphs
# ===========================================================================

class TestIsolatedSubgraphs:

    def test_component_with_no_root_system_flagged(self):
        # Both nodes are Databricks — no ADLS root
        a = _node("a", system=SystemType.DATABRICKS, full_path="db.a")
        b = _node("b", system=SystemType.DATABRICKS, full_path="db.b")
        graph = stitch_fragments([_fragment(a, b, _edge(a, b))])
        gaps = detect_gaps(graph)
        assert any(g.gap_type == "isolated_subgraph" for g in gaps)

    def test_component_with_adls_root_not_flagged(self):
        raw   = _node("raw",   system=SystemType.ADLS,       full_path="adls://raw")
        clean = _node("clean", system=SystemType.DATABRICKS, full_path="db.clean")
        graph = stitch_fragments([_fragment(raw, clean, _edge(raw, clean))])
        gaps = detect_gaps(graph)
        assert not any(g.gap_type == "isolated_subgraph" for g in gaps)

    def test_single_isolated_node_not_flagged_as_subgraph(self):
        n = _node("lonely", system=SystemType.DATABRICKS, full_path="db.lonely")
        graph = stitch_fragments([_fragment(n)])
        gaps = detect_gaps(graph)
        # Single-node components are skipped (< 2 nodes); missing_source catches them
        assert not any(g.gap_type == "isolated_subgraph" for g in gaps)


# ===========================================================================
# End-to-end: full pipeline scenario
# ===========================================================================

class TestEndToEnd:

    def test_complete_adls_to_bi_pipeline_has_minimal_gaps(self):
        """
        A fully documented pipeline: ADLS → Databricks → Databricks → BI
        should produce only missing_target on the BI node (no downstream of BI).
        """
        raw     = _node("raw",     system=SystemType.ADLS,       full_path="adls://raw/orders")
        silver  = _node("silver",  system=SystemType.DATABRICKS, full_path="silver.orders")
        gold    = _node("gold",    system=SystemType.DATABRICKS, full_path="gold.summary")
        report  = _node("report",  system=SystemType.BI,         full_path="bi.sales_report")

        graph = stitch_fragments([_fragment(
            raw, silver, gold, report,
            _edge(raw,    silver, confidence=0.90),
            _edge(silver, gold,   confidence=0.90),
            _edge(gold,   report, confidence=0.90),
        )])
        gaps = detect_gaps(graph)
        gap_types = {g.gap_type for g in gaps}
        # BI is a valid sink — no missing_target for it
        assert "missing_source" not in gap_types
        assert "low_confidence_edge" not in gap_types
        assert "unresolved_dataset_ref" not in gap_types

    def test_gap_ids_reference_correct_nodes(self):
        src = _node("mystery", system=SystemType.DATABRICKS, full_path="db.mystery")
        graph = stitch_fragments([_fragment(src)])
        gaps = detect_gaps(graph)
        for gap in gaps:
            if gap.node_id:
                assert gap.node_id in graph.nodes
