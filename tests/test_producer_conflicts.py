"""
Conflicting producers: two artifacts write the same table (margin v1 vs v2).
The agent flags the conflict with evidence and resolves it only on SME input.
"""
import json
from unittest.mock import patch

import pytest

import src.parsers.notebook_parser as nb_mod
import src.sme.response_processor as rp
from src.graph.gap_detector import detect_gaps
from src.graph.models import LineageFragment
from src.graph.stitcher import stitch_fragments
from src.parsers.adf_parser import AdfParser
from src.parsers.notebook_parser import NotebookParser
from src.parsers.uc_catalog_parser import UcCatalogParser
from src.scoring.confidence import score_graph
from src.sme.question_generator import generate_questions
from src.visualization.diagram_generator import generate_mermaid

V1 = '''# Notebook path: /Repos/x/gold/margin_v1
sales = spark.table("c.silver.sales")
cost = spark.table("c.silver.cost")
fact = (sales.join(cost, "k").groupBy("s").agg(F.sum("x")))
fact.write.saveAsTable("c.gold.fact")
'''
V2 = '''# Notebook path: /Repos/x/gold/margin_v2
sales = spark.table("c.silver.sales")
cost = spark.table("c.silver.cost")
sup = spark.table("c.bronze.supplier")
fact = (sales.join(cost, "k").join(sup, "k").groupBy("s").agg(F.sum("x")))
fact.write.saveAsTable("c.gold.fact")
'''
PIPELINE = {"name": "pl_nightly", "properties": {"activities": [
    {"name": "nb_gold_margin", "type": "DatabricksNotebook",
     "typeProperties": {"notebookPath": "/Repos/x/gold/margin_v1"}}]}}
UC = {"export": {"tool": "unity-catalog-metadata-export"}, "tables": [{
    "name": "fact", "full_name": "c.gold.fact",
    "last_write": {"notebook_path": "/Repos/x/gold/margin_v2", "job_id": 771204,
                   "timestamp": "2026-06-09T03:55:20Z"}}]}


@pytest.fixture(autouse=True)
def _no_llm():
    empty = lambda content, filename: LineageFragment(artifact_name=filename, artifact_type="llm")
    with patch.object(nb_mod, "_parse_with_llm_nb", empty), \
         patch.object(rp, "_get_llm", side_effect=Exception("no llm")):
        yield


def _graph(include_v2=True):
    arts = [
        ("gold_margin_v1.py", V1, NotebookParser()),
        ("pl.json", json.dumps(PIPELINE), AdfParser()),
        ("uc_gold.json", json.dumps(UC), UcCatalogParser()),
    ]
    if include_v2:
        arts.append(("gold_margin_v2.py", V2, NotebookParser()))
    frags = [p.parse({"filename": n, "content": c}) for n, c, p in arts]
    g = stitch_fragments(frags)
    score_graph(g)
    return g


def _into_fact(g):
    return {g.nodes[e.source_id].full_path: e for e in g.edges
            if g.nodes[e.target_id].full_path == "c.gold.fact"}


def test_conflict_is_recorded_with_evidence_per_claim():
    g = _graph()
    [c] = g.producer_conflicts
    claims = {cl.artifact: cl for cl in c.claims}
    assert claims["gold_margin_v1.py"].orchestrated_by == "adf:pl_nightly/nb_gold_margin"
    assert not claims["gold_margin_v1.py"].matches_catalog_last_write
    assert claims["gold_margin_v2.py"].matches_catalog_last_write
    assert claims["gold_margin_v2.py"].orchestrated_by is None
    assert c.catalog_last_write["job_id"] == 771204


def test_single_writer_is_not_a_conflict():
    assert _graph(include_v2=False).producer_conflicts == []


def test_gap_reports_conflict_without_choosing():
    g = _graph()
    [gap] = [x for x in detect_gaps(g) if x.gap_type == "conflicting_producers"]
    assert "Evidence disagrees" in gap.description
    assert "only this version reads supplier" in gap.description
    [q] = generate_questions([gap], g, enrich_with_llm=False)
    assert "ADF schedules gold_margin_v1.py" in q["question"]
    assert "Unity Catalog shows gold_margin_v2.py wrote the table last" in q["question"]


def test_contested_edges_are_medium_and_unattributed_until_resolved():
    g = _graph()
    edges = _into_fact(g)
    assert all(e.confidence <= 0.70 for e in edges.values())
    # Shared edges have no single orchestrator while the producers disagree
    assert edges["c.silver.sales"].orchestrated_by is None
    assert "⚡ conflict" in generate_mermaid(g)


def _answer(g, text):
    [gap] = [x for x in detect_gaps(g) if x.gap_type == "conflicting_producers"]
    gaps, _ = rp.apply_responses_to_graph(
        [{"gap_id": gap.id, "raw_answer": text, "sme_email": "sme@x"}], [gap], g)
    return gaps[0]


def test_sme_picks_v2_keeps_its_inputs():
    g = _graph()
    gap = _answer(g, "v2 is live since May, v1 is deprecated")
    assert gap.resolved
    edges = _into_fact(g)
    assert set(edges) == {"c.silver.sales", "c.silver.cost", "c.bronze.supplier"}
    assert all(e.produced_by == ["gold_margin_v2.py"] for e in edges.values())
    score_graph(g)
    assert all(e.confidence > 0.70 for e in edges.values())
    assert not [x for x in detect_gaps(g) if x.gap_type == "conflicting_producers"]


def test_sme_picks_v1_drops_v2_only_lineage_and_inherits_orchestrator():
    g = _graph()
    assert _answer(g, "The live one is gold_margin_v1.py").resolved
    edges = _into_fact(g)
    assert set(edges) == {"c.silver.sales", "c.silver.cost"}
    assert edges["c.silver.sales"].orchestrated_by == "adf:pl_nightly/nb_gold_margin"


def test_both_keeps_everything():
    g = _graph()
    assert _answer(g, "Both — they write different partitions").resolved
    assert len(_into_fact(g)) == 3
    assert g.open_conflicts() == []


def test_ambiguous_reply_leaves_conflict_open():
    g = _graph()
    assert not _answer(g, "Not sure, ask the finance team").resolved
    assert len(g.open_conflicts()) == 1
