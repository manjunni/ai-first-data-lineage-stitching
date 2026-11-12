"""
Declared external producers and stewardship-required landing data.
"""
import json
from unittest.mock import patch

import pytest

import src.parsers.notebook_parser as nb_mod
import src.sme.response_processor as rp
from src.catalog.publisher import _graph_to_alation_payload, _graph_to_collibra_payload
from src.graph.gap_detector import detect_gaps
from src.graph.models import LineageFragment, SystemType, ConfidenceLevel
from src.graph.stewardship import declared_external_from_comment, producer_from_text
from src.graph.stitcher import stitch_fragments
from src.parsers.adf_parser import AdfParser
from src.parsers.notebook_parser import NotebookParser
from src.parsers.uc_catalog_parser import UcCatalogParser
from src.scoring.confidence import score_graph
from src.sme.question_generator import generate_questions
from src.visualization.diagram_generator import generate_mermaid


@pytest.fixture(autouse=True)
def _no_llm():
    empty = lambda content, filename: LineageFragment(artifact_name=filename, artifact_type="llm")
    with patch.object(nb_mod, "_parse_with_llm_nb", empty), \
         patch.object(rp, "_get_llm", side_effect=Exception("no llm")):
        yield


def _dataset(name, fs, folder, annotations):
    return {"filename": f"{name}.json", "content": json.dumps({"name": name, "properties": {
        "linkedServiceName": {"referenceName": "ls"}, "type": "Parquet", "annotations": annotations,
        "typeProperties": {"location": {"fileSystem": fs, "folderPath": folder}}}})}


NB = '''# Notebook path: /Repos/x/bronze/reference_ingest
spark.read.parquet("abfss://raw@st.dfs.core.windows.net/sap/articles").write.saveAsTable("c.bronze.sap_articles")
spark.read.csv("abfss://landing@st.dfs.core.windows.net/supplier_prices").write.saveAsTable("c.bronze.supplier_price_list")
spark.table("c.ref.fx_rate_daily").write.saveAsTable("c.silver.fx")
spark.table("c.ref.vat_rate").write.saveAsTable("c.silver.vat")
spark.table("c.bronze.xref_by_job").write.saveAsTable("c.silver.xref")
'''
UC = {"export": {"tool": "unity-catalog-metadata-export"}, "tables": [
    {"name": "supplier_price_list", "full_name": "c.bronze.supplier_price_list",
     "comment": "Weekly contract prices as uploaded by Category Buying."},
    {"name": "store_id_xref", "full_name": "c.bronze.store_id_xref",
     "comment": "Store identifier crosswalk. Maintained by Master Data team."},
    {"name": "fx_rate_daily", "full_name": "c.ref.fx_rate_daily",
     "comment": "Daily ECB-based FX rates. Loaded by Group Treasury FX feed (Logic App), external to Data Platform."},
    {"name": "vat_rate", "full_name": "c.ref.vat_rate", "owner": "svc-tax-refdata",
     "comment": "VAT rates per country and code. Maintained by Group Tax; quarterly CSV load.",
     "last_write": {"operation": "WRITE", "job_id": None, "notebook_path": None}},
    {"name": "xref_by_job", "full_name": "c.bronze.xref_by_job",
     "comment": "Crosswalk. Maintained by Master Data team.",
     "last_write": {"operation": "WRITE", "job_id": 612377, "notebook_path": None}},
    {"name": "unused_ext", "full_name": "c.ref.unused_ext",
     "comment": "Loaded by Someone Else, external to Data Platform."},
]}


def _graph():
    arts = [
        (_dataset("ds_sap", "raw", "sap/articles",
                  ["declared_producer:SAP CDC extraction (Group Integration team)"]), AdfParser()),
        (_dataset("ds_supplier", "landing", "supplier_prices", ["owner:category-buying"]), AdfParser()),
        ({"filename": "nb.py", "content": NB}, NotebookParser()),
        ({"filename": "uc.json", "content": json.dumps(UC)}, UcCatalogParser()),
    ]
    g = stitch_fragments([p.parse(a) for a, p in arts])
    score_graph(g)
    return g


def _node(g, name):
    return next(n for n in g.nodes.values() if n.name == name or n.full_path == name)


def test_text_helpers():
    assert producer_from_text("Weekly prices as uploaded by Category Buying.") == "Category Buying"
    assert declared_external_from_comment("Maintained by Master Data team.") is None
    assert declared_external_from_comment(
        "Loaded by Group Treasury FX feed (Logic App), external to Data Platform."
    ) == "Group Treasury FX feed (Logic App)"


def test_adf_declared_producer_becomes_external_source():
    g = _graph()
    ext = _node(g, "SAP CDC extraction (Group Integration team)")
    assert ext.system == SystemType.EXTERNAL
    [edge] = [e for e in g.edges if e.source_id == ext.id]
    assert g.nodes[edge.target_id].full_path == "abfss://raw@st.dfs.core.windows.net/sap/articles"
    assert edge.confidence_level == ConfidenceLevel.MEDIUM


def test_uc_comment_declares_external_only_when_explicit_and_used():
    g = _graph()
    fx = _node(g, "c.ref.fx_rate_daily")
    sources = [g.nodes[e.source_id] for e in g.edges if e.target_id == fx.id]
    assert [s.system for s in sources] == [SystemType.EXTERNAL]
    # Not pulled into the graph just because it is declared
    assert not any(n.full_path == "c.ref.unused_ext" for n in g.nodes.values())
    # "Maintained by…" without "external" is not a declaration
    assert not any("master-data" in n.full_path for n in g.nodes.values())


def test_uc_table_loaded_outside_databricks_is_declared_external():
    g = _graph()
    vat = _node(g, "c.ref.vat_rate")
    [src] = [g.nodes[e.source_id] for e in g.edges if e.target_id == vat.id]
    assert src.system == SystemType.EXTERNAL and src.name == "Group Tax"
    # Same comment shape, but a Databricks job wrote it last → producer is a
    # platform artifact not yet uploaded, so it stays a missing-source gap.
    xref = _node(g, "c.bronze.xref_by_job")
    assert not any(e.target_id == xref.id for e in g.edges)
    assert any(x.gap_type == "missing_source" and x.node_id == xref.id for x in detect_gaps(g))


def test_undeclared_landing_data_needs_stewardship_with_suggested_steward():
    g = _graph()
    gaps = [x for x in detect_gaps(g) if x.gap_type == "stewardship_required"]
    [gap] = gaps
    assert g.nodes[gap.node_id].name == "ds_supplier"
    assert "Suggested steward: Category Buying" in gap.description
    # Declared landing data and external nodes are roots, not missing sources
    assert not any(x.gap_type == "missing_source" and g.nodes[x.node_id].system in
                   (SystemType.EXTERNAL, SystemType.ADLS) for x in detect_gaps(g))
    [q] = generate_questions(gaps, g, enrich_with_llm=False)
    assert "Category Buying is responsible" in q["question"]
    assert "stewardship required" in generate_mermaid(g)


def test_catalog_payloads_carry_stewardship_status():
    g = _graph()
    assets = {a["name"]: a for a in _graph_to_collibra_payload(g, "r")["assets"]}
    assert assets["ds_supplier"]["stewardship_status"] == "required"
    assert assets["ds_supplier"]["suggested_steward"] == "Category Buying"
    assert assets["SAP CDC extraction (Group Integration team)"]["stewardship_status"] == "declared_external"
    assert assets["SAP CDC extraction (Group Integration team)"]["asset_type"] == "System"
    alation = _graph_to_alation_payload(g, "r")["lineage"]
    objs = [o for item in alation for o in item["source_objects"] if o["name"] == "ds_supplier"]
    assert objs and objs[0]["stewardship_status"] == "required"


def _reply(g, text):
    [gap] = [x for x in detect_gaps(g) if x.gap_type == "stewardship_required"]
    gaps, _ = rp.apply_responses_to_graph(
        [{"gap_id": gap.id, "raw_answer": text, "sme_email": "buyer@x"}], [gap], g)
    return gaps[0]


def test_dont_know_keeps_stewardship_open():
    g = _graph()
    assert not _reply(g, "No idea, sorry").resolved
    assert len([x for x in detect_gaps(g) if x.gap_type == "stewardship_required"]) == 1


def test_attestation_registers_confirmed_external_producer():
    g = _graph()
    assert _reply(g, "Dropped manually on the SFTP share every Monday by Category Buying.").resolved
    ext = _node(g, "Category Buying (dropped manually on the SFTP share every Monday)")
    assert ext.system == SystemType.EXTERNAL
    [edge] = [e for e in g.edges if e.source_id == ext.id]
    score_graph(g)
    assert edge.confidence_level == ConfidenceLevel.CONFIRMED
    assert not [x for x in detect_gaps(g) if x.gap_type == "stewardship_required"]
    assets = {a["name"]: a for a in _graph_to_collibra_payload(g, "r")["assets"]}
    assert assets[ext.name]["stewardship_status"] == "attested"
    assert "stewardship_status" not in assets["ds_supplier"]


def test_sftp_copy_source_is_external_system():
    pl = {"name": "pl", "properties": {"activities": [{
        "name": "cp", "type": "Copy",
        "typeProperties": {"source": {"storeSettings": {"type": "SftpReadSettings"}},
                           "sink": {"storeSettings": {"type": "AzureBlobFSWriteSettings"}}},
        "inputs": [{"referenceName": "ds_src"}], "outputs": [{"referenceName": "ds_dst"}]}]}}
    frag = AdfParser().parse({"filename": "pl.json", "content": json.dumps(pl)})
    systems = {n.full_path: n.system for n in frag.nodes}
    assert systems["adf://ds_src"] == SystemType.EXTERNAL
