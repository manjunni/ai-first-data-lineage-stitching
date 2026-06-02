"""Tests for Mermaid and Graphviz diagram generators."""
import pytest
from src.visualization.diagram_generator import (
    generate_mermaid, generate_graphviz_dot, generate_summary_html,
    _ADF_NS, _safe_id,
)
from src.graph.stitcher import stitch_fragments
from src.graph.models import (
    LineageFragment, LineageNode, LineageEdge,
    LineageGraph, SystemType, ConfidenceLevel, Gap,
)
from src.scoring.confidence import score_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(name: str, system: SystemType = SystemType.DATABRICKS,
          full_path: str | None = None) -> LineageNode:
    return LineageNode(name=name, system=system, full_path=full_path or name)


def _edge(src: LineageNode, tgt: LineageNode,
          confidence: float = 0.90,
          level: ConfidenceLevel = ConfidenceLevel.HIGH,
          transformation: str = "copy",
          sme: str | None = None) -> LineageEdge:
    return LineageEdge(
        source_id=src.id, target_id=tgt.id,
        confidence=confidence, confidence_level=level,
        transformation=transformation, evidence="test",
        sme_approved_by=sme,
    )


def _graph(*items):
    f = LineageFragment(artifact_name="t", artifact_type="t")
    for item in items:
        if isinstance(item, LineageNode):
            f.nodes.append(item)
        else:
            f.edges.append(item)
    return stitch_fragments([f])


# ===========================================================================
# Mermaid
# ===========================================================================

class TestMermaid:

    def test_starts_with_flowchart(self):
        g = _graph(_node("a"), _node("b"))
        assert generate_mermaid(g).startswith("flowchart LR")

    def test_node_ids_present(self):
        n = _node("orders", full_path="bronze.orders")
        g = _graph(n)
        mmd = generate_mermaid(g)
        assert _safe_id(list(g.nodes.keys())[0]) in mmd

    def test_node_name_in_output(self):
        n = _node("raw_orders", full_path="raw_orders")
        g = _graph(n)
        assert "raw_orders" in generate_mermaid(g)

    def test_system_type_in_node_label(self):
        n = _node("t", system=SystemType.ADLS, full_path="adls://t")
        g = _graph(n)
        assert "adls" in generate_mermaid(g)

    def test_subgraph_per_system(self):
        adls = _node("raw",   system=SystemType.ADLS,       full_path="adls://raw")
        db   = _node("clean", system=SystemType.DATABRICKS, full_path="db.clean")
        g = _graph(adls, db, _edge(adls, db))
        mmd = generate_mermaid(g)
        assert "subgraph ADLS" in mmd
        assert "subgraph DATABRICKS" in mmd

    def test_classDef_for_each_system(self):
        n = _node("a", system=SystemType.SQL, full_path="sql.a")
        g = _graph(n)
        mmd = generate_mermaid(g)
        assert "classDef sql" in mmd

    def test_edge_label_contains_confidence(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        g = _graph(src, tgt, _edge(src, tgt, confidence=0.90))
        mmd = generate_mermaid(g)
        assert "90%" in mmd

    def test_edge_label_contains_transformation(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        g = _graph(src, tgt, _edge(src, tgt, transformation="filtered, aggregated"))
        mmd = generate_mermaid(g)
        assert "filtered" in mmd

    def test_class_assignment_per_node(self):
        n = _node("t", system=SystemType.BI, full_path="bi.t")
        g = _graph(n)
        mmd = generate_mermaid(g)
        assert "class " in mmd and "bi" in mmd

    def test_gap_node_gets_gap_class(self):
        n = _node("bad", full_path="bad_node")
        g = _graph(n)
        node_id = list(g.nodes.keys())[0]
        gap = Gap(gap_type="missing_source", node_id=node_id, description="test")
        mmd = generate_mermaid(g, gaps=[gap])
        assert f"class {_safe_id(node_id)} gap" in mmd

    def test_classDef_gap_defined(self):
        n = _node("a", full_path="a")
        g = _graph(n)
        mmd = generate_mermaid(g, gaps=[])
        assert "classDef gap" in mmd

    def test_adf_unresolved_marker_in_label(self):
        n = _node("MyDS", system=SystemType.UNKNOWN, full_path=f"{_ADF_NS}MyDS")
        g = _graph(n)
        assert "unresolved" in generate_mermaid(g)

    def test_linkStyle_lines_match_edge_count(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        tgt2 = _node("c", full_path="c")
        g = _graph(src, tgt, tgt2, _edge(src, tgt), _edge(src, tgt2))
        mmd = generate_mermaid(g)
        link_styles = [l for l in mmd.splitlines() if l.strip().startswith("linkStyle")]
        assert len(link_styles) == len(g.edges)

    def test_medium_confidence_uses_dashed_arrow(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        g = _graph(src, tgt, _edge(src, tgt, confidence=0.70, level=ConfidenceLevel.MEDIUM))
        mmd = generate_mermaid(g)
        assert ".-" in mmd or "-." in mmd

    def test_empty_graph_does_not_crash(self):
        g = LineageGraph()
        mmd = generate_mermaid(g)
        assert "flowchart LR" in mmd


# ===========================================================================
# Graphviz DOT
# ===========================================================================

class TestGraphvizDot:

    def test_starts_with_digraph(self):
        g = _graph(_node("a"))
        assert generate_graphviz_dot(g).startswith("digraph lineage {")

    def test_rankdir_LR(self):
        g = _graph(_node("a"))
        assert "rankdir=LR" in generate_graphviz_dot(g)

    def test_node_id_in_output(self):
        n = _node("orders", full_path="orders")
        g = _graph(n)
        node_id = list(g.nodes.keys())[0]
        assert _safe_id(node_id) in generate_graphviz_dot(g)

    def test_cluster_per_system(self):
        adls = _node("raw",   system=SystemType.ADLS,       full_path="adls://raw")
        db   = _node("clean", system=SystemType.DATABRICKS, full_path="db.clean")
        g = _graph(adls, db, _edge(adls, db))
        dot = generate_graphviz_dot(g)
        assert "cluster_adls" in dot
        assert "cluster_databricks" in dot

    def test_node_fillcolor_set(self):
        n = _node("t", system=SystemType.ADLS, full_path="adls://t")
        g = _graph(n)
        assert "fillcolor" in generate_graphviz_dot(g)

    def test_edge_label_contains_confidence(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        g = _graph(src, tgt, _edge(src, tgt, confidence=0.85))
        dot = generate_graphviz_dot(g)
        assert "85%" in dot

    def test_high_confidence_edge_is_solid(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        g = _graph(src, tgt, _edge(src, tgt, confidence=0.90, level=ConfidenceLevel.HIGH))
        dot = generate_graphviz_dot(g)
        assert "style=solid" in dot

    def test_medium_confidence_edge_is_dashed(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        g = _graph(src, tgt, _edge(src, tgt, confidence=0.70, level=ConfidenceLevel.MEDIUM))
        dot = generate_graphviz_dot(g)
        assert "style=dashed" in dot

    def test_low_confidence_edge_is_dotted(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        g = _graph(src, tgt, _edge(src, tgt, confidence=0.40, level=ConfidenceLevel.LOW))
        dot = generate_graphviz_dot(g)
        assert "style=dotted" in dot

    def test_confirmed_edge_is_bold(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        g = _graph(src, tgt, _edge(src, tgt, confidence=0.95, level=ConfidenceLevel.CONFIRMED, sme="alice"))
        dot = generate_graphviz_dot(g)
        assert "style=bold" in dot

    def test_gap_node_gets_red_fill(self):
        n = _node("bad", full_path="bad")
        g = _graph(n)
        node_id = list(g.nodes.keys())[0]
        gap = Gap(gap_type="missing_source", node_id=node_id, description="test")
        dot = generate_graphviz_dot(g, gaps=[gap])
        assert "#FF3333" in dot

    def test_ends_with_closing_brace(self):
        g = _graph(_node("a"))
        assert generate_graphviz_dot(g).strip().endswith("}")

    def test_empty_graph_does_not_crash(self):
        g = LineageGraph()
        dot = generate_graphviz_dot(g)
        assert "digraph lineage" in dot


# ===========================================================================
# Summary HTML
# ===========================================================================

class TestSummaryHtml:

    def test_returns_html_string(self):
        g = _graph(_node("a"))
        scores = score_graph(g)
        from src.scoring.confidence import graph_confidence_summary
        s = graph_confidence_summary(g, scores)
        html = generate_summary_html(g, s)
        assert "<table" in html

    def test_node_count_in_html(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        g = _graph(src, tgt, _edge(src, tgt))
        scores = score_graph(g)
        from src.scoring.confidence import graph_confidence_summary
        s = graph_confidence_summary(g, scores)
        html = generate_summary_html(g, s)
        assert "2" in html   # 2 nodes

    def test_gap_count_in_html(self):
        g = _graph(_node("a"))
        scores = score_graph(g)
        from src.scoring.confidence import graph_confidence_summary
        s = graph_confidence_summary(g, scores)
        gaps = [Gap(gap_type="missing_source", node_id="x", description="d")]
        html = generate_summary_html(g, s, gaps=gaps)
        assert "1" in html
