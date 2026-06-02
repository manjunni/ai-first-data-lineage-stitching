"""Tests for confidence scoring and graph_confidence_summary."""
import pytest
from src.scoring.confidence import score_graph, graph_confidence_summary, _ADF_NS
from src.graph.stitcher import stitch_fragments
from src.graph.models import (
    LineageFragment, LineageNode, LineageEdge,
    LineageGraph, SystemType, ConfidenceLevel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(name: str, system: SystemType = SystemType.DATABRICKS,
          full_path: str | None = None) -> LineageNode:
    return LineageNode(name=name, system=system, full_path=full_path or name)


def _edge(src: LineageNode, tgt: LineageNode,
          evidence: str = "", confidence: float = 0.0,
          sme_approved_by: str | None = None,
          transformation: str | None = None) -> LineageEdge:
    return LineageEdge(
        source_id=src.id, target_id=tgt.id,
        evidence=evidence, confidence=confidence,
        sme_approved_by=sme_approved_by,
        transformation=transformation,
    )


def _graph(src: LineageNode, tgt: LineageNode, edge: LineageEdge) -> LineageGraph:
    f = LineageFragment(artifact_name="t", artifact_type="t")
    f.nodes.extend([src, tgt])
    f.edges.append(edge)
    return stitch_fragments([f])


# ===========================================================================
# Base score from evidence
# ===========================================================================

class TestBaseScores:

    def test_sqlglot_evidence_scores_090(self):
        src, tgt = _node("a"), _node("b")
        g = _graph(src, tgt, _edge(src, tgt, evidence="parsed from x.sql via sqlglot"))
        scores = score_graph(g)
        assert list(scores.values())[0] == 0.90

    def test_ast_evidence_scores_090(self):
        src, tgt = _node("a"), _node("b")
        g = _graph(src, tgt, _edge(src, tgt, evidence="parsed from pipeline.py via ast"))
        scores = score_graph(g)
        assert list(scores.values())[0] == 0.90

    def test_regex_evidence_scores_085(self):
        src, tgt = _node("a"), _node("b")
        g = _graph(src, tgt, _edge(src, tgt, evidence="parsed from nb.ipynb via regex"))
        scores = score_graph(g)
        assert list(scores.values())[0] == 0.85

    def test_adf_pipeline_evidence_scores_085(self):
        src, tgt = _node("a"), _node("b")
        g = _graph(src, tgt, _edge(src, tgt, evidence="parsed from p.json via ADF pipeline JSON"))
        scores = score_graph(g)
        assert list(scores.values())[0] == 0.85

    def test_llm_evidence_scores_060(self):
        src, tgt = _node("a"), _node("b")
        g = _graph(src, tgt, _edge(src, tgt, evidence="LLM fallback from x.sql"))
        scores = score_graph(g)
        assert list(scores.values())[0] == 0.60

    def test_fallback_evidence_scores_060(self):
        src, tgt = _node("a"), _node("b")
        g = _graph(src, tgt, _edge(src, tgt, evidence="fallback"))
        scores = score_graph(g)
        assert list(scores.values())[0] == 0.60

    def test_no_evidence_but_parser_confidence_preserved(self):
        src, tgt = _node("a"), _node("b")
        # Parser set confidence=0.85 with no textual evidence
        g = _graph(src, tgt, _edge(src, tgt, evidence="", confidence=0.85))
        scores = score_graph(g)
        assert list(scores.values())[0] == 0.85

    def test_sme_approved_scores_095_regardless_of_evidence(self):
        src, tgt = _node("a"), _node("b")
        g = _graph(src, tgt, _edge(src, tgt, evidence="LLM fallback", sme_approved_by="alice@corp.com"))
        scores = score_graph(g)
        assert list(scores.values())[0] == 0.95


# ===========================================================================
# Penalties
# ===========================================================================

class TestPenalties:

    def test_cross_system_boundary_penalty(self):
        src = _node("a", system=SystemType.ADLS,       full_path="adls://raw")
        tgt = _node("b", system=SystemType.DATABRICKS, full_path="db.silver")
        f = LineageFragment(artifact_name="t", artifact_type="t")
        f.nodes.extend([src, tgt])
        f.edges.append(_edge(src, tgt, evidence="parsed from x.sql via sqlglot"))
        g = stitch_fragments([f])
        scores = score_graph(g)
        # 0.90 base − 0.05 cross-system = 0.85
        assert list(scores.values())[0] == 0.85

    def test_no_penalty_when_sme_confirmed(self):
        src = _node("a", system=SystemType.ADLS,       full_path="adls://raw")
        tgt = _node("b", system=SystemType.DATABRICKS, full_path="db.silver")
        f = LineageFragment(artifact_name="t", artifact_type="t")
        f.nodes.extend([src, tgt])
        f.edges.append(_edge(src, tgt, evidence="sqlglot", sme_approved_by="alice"))
        g = stitch_fragments([f])
        scores = score_graph(g)
        # SME-confirmed → 0.95, no cross-system penalty applied
        assert list(scores.values())[0] == 0.95

    def test_unknown_system_node_penalty(self):
        src = _node("a", system=SystemType.UNKNOWN, full_path="mystery")
        tgt = _node("b", system=SystemType.DATABRICKS, full_path="db.b")
        f = LineageFragment(artifact_name="t", artifact_type="t")
        f.nodes.extend([src, tgt])
        f.edges.append(_edge(src, tgt, evidence="parsed from x via sqlglot"))
        g = stitch_fragments([f])
        scores = score_graph(g)
        # 0.90 base − 0.05 cross-system (unknown→databricks) − 0.10 unknown node = 0.75
        assert list(scores.values())[0] == 0.75

    def test_adf_unresolved_penalty(self):
        src = _node("MyDS", system=SystemType.UNKNOWN, full_path=f"{_ADF_NS}MyDS")
        tgt = _node("b",    system=SystemType.DATABRICKS, full_path="db.b")
        f = LineageFragment(artifact_name="t", artifact_type="t")
        f.nodes.extend([src, tgt])
        # ADF pipeline JSON evidence = 0.85 base
        f.edges.append(_edge(src, tgt, evidence="parsed from p.json via ADF pipeline JSON"))
        g = stitch_fragments([f])
        scores = score_graph(g)
        # 0.85 − 0.05 cross-system − 0.15 adf unresolved − 0.10 unknown node = 0.55
        assert list(scores.values())[0] == 0.55

    def test_score_clamped_to_zero_minimum(self):
        src = _node("a", system=SystemType.UNKNOWN, full_path=f"{_ADF_NS}a")
        tgt = _node("b", system=SystemType.UNKNOWN, full_path=f"{_ADF_NS}b")
        f = LineageFragment(artifact_name="t", artifact_type="t")
        f.nodes.extend([src, tgt])
        f.edges.append(_edge(src, tgt, evidence=""))  # 0.20 base, multiple penalties
        g = stitch_fragments([f])
        scores = score_graph(g)
        assert list(scores.values())[0] >= 0.0


# ===========================================================================
# Confidence level assignment
# ===========================================================================

class TestConfidenceLevels:

    def _make(self, confidence: float, sme: str | None = None) -> ConfidenceLevel:
        src, tgt = _node("a"), _node("b")
        # Pass no evidence so _base_from_evidence falls back to the parser-set confidence value
        e = _edge(src, tgt, evidence="", confidence=confidence, sme_approved_by=sme)
        g = _graph(src, tgt, e)
        score_graph(g)
        return g.edges[0].confidence_level

    def test_sme_approved_is_confirmed(self):
        assert self._make(0.95, sme="alice") == ConfidenceLevel.CONFIRMED

    def test_high_score_is_high(self):
        assert self._make(0.90) == ConfidenceLevel.HIGH

    def test_medium_score_is_medium(self):
        assert self._make(0.70) == ConfidenceLevel.MEDIUM

    def test_low_score_is_low(self):
        assert self._make(0.50) == ConfidenceLevel.LOW

    def test_boundary_085_is_high(self):
        assert self._make(0.85) == ConfidenceLevel.HIGH

    def test_boundary_065_is_medium(self):
        assert self._make(0.65) == ConfidenceLevel.MEDIUM


# ===========================================================================
# Edge updates in place
# ===========================================================================

def test_score_graph_updates_edge_confidence_in_place():
    src, tgt = _node("a"), _node("b")
    g = _graph(src, tgt, _edge(src, tgt, evidence="parsed from x.sql via sqlglot"))
    score_graph(g)
    assert g.edges[0].confidence == 0.90
    assert g.edges[0].confidence_level == ConfidenceLevel.HIGH


def test_score_graph_returns_edge_id_map():
    src, tgt = _node("a"), _node("b")
    g = _graph(src, tgt, _edge(src, tgt, evidence="sqlglot"))
    scores = score_graph(g)
    assert g.edges[0].id in scores


# ===========================================================================
# graph_confidence_summary
# ===========================================================================

class TestSummary:

    def test_empty_graph_returns_zeroed_summary(self):
        g = LineageGraph()
        s = graph_confidence_summary(g, {})
        assert s["total_edges"] == 0
        assert s["avg_confidence"] is None

    def test_avg_confidence_computed(self):
        src, tgt = _node("a"), _node("b")
        g = _graph(src, tgt, _edge(src, tgt, evidence="sqlglot"))
        scores = score_graph(g)
        s = graph_confidence_summary(g, scores)
        assert s["avg_confidence"] == 0.90

    def test_sme_confirmed_count(self):
        src, tgt = _node("a"), _node("b")
        g = _graph(src, tgt, _edge(src, tgt, evidence="sqlglot", sme_approved_by="bob"))
        scores = score_graph(g)
        s = graph_confidence_summary(g, scores)
        assert s["sme_confirmed"] == 1

    def test_cross_system_count(self):
        src = _node("a", system=SystemType.ADLS,       full_path="adls://a")
        tgt = _node("b", system=SystemType.DATABRICKS, full_path="db.b")
        f = LineageFragment(artifact_name="t", artifact_type="t")
        f.nodes.extend([src, tgt])
        f.edges.append(_edge(src, tgt, evidence="sqlglot"))
        g = stitch_fragments([f])
        scores = score_graph(g)
        s = graph_confidence_summary(g, scores)
        assert s["cross_system_edges"] == 1

    def test_by_level_keys_present(self):
        src, tgt = _node("a"), _node("b")
        g = _graph(src, tgt, _edge(src, tgt, evidence="sqlglot"))
        scores = score_graph(g)
        s = graph_confidence_summary(g, scores)
        assert set(s["by_level"].keys()) == {level.value for level in ConfidenceLevel}
