"""
Cross-system stitching: ADF as orchestrator between ADLS and Databricks.

Covers the AST notebook evaluator, ADF dataset/pipeline handling, the UC
catalog parser and the stitcher passes that join them.
"""
import json
from unittest.mock import patch

import pytest

import src.parsers.adf_parser as adf_mod
import src.parsers.notebook_parser as nb_mod
from src.graph.models import LineageFragment, SystemType
from src.graph.paths import adls_location_key, normalise_location
from src.graph.stitcher import stitch_fragments
from src.parsers.adf_parser import AdfParser, _adf_expr_to_template
from src.parsers.notebook_parser import NotebookParser
from src.parsers.pyspark_ast import evaluate_python
from src.parsers.uc_catalog_parser import UcCatalogParser
from src.scoring.confidence import score_graph

ACCT = "abfss://{c}@stlake.dfs.core.windows.net"


@pytest.fixture(autouse=True)
def _no_llm():
    def empty(content, filename):
        return LineageFragment(artifact_name=filename, artifact_type="llm")
    with patch.object(nb_mod, "_parse_with_llm_nb", empty), \
         patch.object(adf_mod, "_parse_with_llm_adf", empty):
        yield


def _edges(frag_or_graph, nodes=None):
    nodes = nodes or ({n.id: n for n in frag_or_graph.nodes}
                      if isinstance(frag_or_graph, LineageFragment) else frag_or_graph.nodes)
    return {(nodes[e.source_id].full_path, nodes[e.target_id].full_path) for e in frag_or_graph.edges}


# ---------------------------------------------------------------------------
# AST evaluator
# ---------------------------------------------------------------------------

def test_helper_function_is_inlined_at_call_sites():
    src = '''
BASE = "abfss://{c}@stlake.dfs.core.windows.net"
def autoload(container, folder, target):
    df = spark.readStream.format("cloudFiles").load(f"{BASE.format(c=container)}/{folder}")
    return df.writeStream.toTable(f"cat.bronze.{target}")
autoload("raw", "sap/articles", "sap_articles")
autoload("landing", "prices", "price_list")
'''
    writes = {(w.target.text, tuple(w.df.sources)) for w in evaluate_python(src).writes}
    assert ("cat.bronze.sap_articles", ("abfss://raw@stlake.dfs.core.windows.net/sap/articles",)) in writes
    assert ("cat.bronze.price_list", ("abfss://landing@stlake.dfs.core.windows.net/prices",)) in writes


def test_fstring_with_empty_widget_becomes_placeholder_with_case():
    src = '''
dbutils.widgets.text("country", "")
country = dbutils.widgets.get("country").lower()
df = (spark.read.format("parquet").load(f"abfss://raw@stlake.dfs.core.windows.net/pos/{country.upper()}"))
(df.write.saveAsTable(f"cat.bronze.pos_{country}_tx"))
'''
    [w] = evaluate_python(src).writes
    assert w.target.text == "cat.bronze.pos_{country:lower}_tx"
    assert w.df.sources == ["abfss://raw@stlake.dfs.core.windows.net/pos/{country:upper}"]


def test_widget_default_list_comprehension_and_union_loop():
    src = '''
dbutils.widgets.text("countries", "NL,BE")
cs = [c.strip().lower() for c in dbutils.widgets.get("countries").split(",")]
frames = [spark.table(f"cat.bronze.pos_{c}") for c in cs]
pos = frames[0]
for f in frames[1:]:
    pos = pos.unionByName(f)
pos.write.saveAsTable("cat.silver.sales")
'''
    [w] = evaluate_python(src).writes
    assert w.df.sources == ["cat.bronze.pos_nl", "cat.bronze.pos_be"]
    assert w.df.soft  # depends on a runtime-overridable widget default


def test_join_in_parenthesised_chain_keeps_all_parents():
    src = '''
a = spark.table("s.a")
b = spark.table("s.b")
c = (a.join(b, "k").withColumn("x", F.lit(1)))
lines = some_lib_fn(c, col="x")
(lines.select("k").write.saveAsTable("g.out"))
'''
    [w] = evaluate_python(src).writes
    assert w.df.sources == ["s.a", "s.b"]
    assert "some_lib_fn" in w.df.opaque


def test_notebook_py_records_notebook_path_header():
    content = "# Databricks notebook source\n# Notebook path: /Repos/x/silver/foo\n" \
              'spark.table("a.b.c").write.saveAsTable("a.b.d")\n'
    frag = NotebookParser().parse({"filename": "nb.py", "content": content})
    assert frag.raw_metadata["notebook_path"] == "/Repos/x/silver/foo"
    assert _edges(frag) == {("a.b.c", "a.b.d")}


# ---------------------------------------------------------------------------
# ADF parser
# ---------------------------------------------------------------------------

def test_adf_expression_to_template():
    expr = "@concat('pos/', toUpper(dataset().country_code), '/', formatDateTime(dataset().run_date,'yyyy/MM/dd'))"
    assert _adf_expr_to_template(expr) == "pos/{country_code:upper}/{run_date}"


def test_dataset_with_expression_folder_path_is_parsed():
    ds = {"name": "ds_pos", "properties": {
        "linkedServiceName": {"referenceName": "ls"}, "type": "Parquet",
        "annotations": ["owner:trading"],
        "typeProperties": {"location": {"type": "AzureBlobFSLocation", "fileSystem": "raw", "folderPath": {
            "value": "@concat('pos/', toUpper(dataset().country_code))", "type": "Expression"}}}}}
    frag = AdfParser().parse({"filename": "ds.json", "content": json.dumps(ds)})
    [node] = frag.nodes
    assert node.system == SystemType.ADLS
    assert node.description == "raw/pos/{country_code:upper}"
    assert node.owner == "trading"


def _pipeline():
    return {"name": "pl_x", "properties": {"activities": [
        {"name": "lkp", "type": "Lookup", "typeProperties": {
            "source": {"type": "AzureSqlSource", "sqlReaderQuery": "SELECT 1"},
            "dataset": {"referenceName": "ds_ctl"}}},
        {"name": "cp", "type": "Copy",
         "typeProperties": {"source": {"storeSettings": {"type": "SftpReadSettings"}},
                            "sink": {"storeSettings": {"type": "AzureBlobFSWriteSettings"}}},
         "inputs": [{"referenceName": "ds_src"}], "outputs": [{"referenceName": "ds_pos"}]},
        {"name": "nb_ingest", "type": "DatabricksNotebook",
         "typeProperties": {"notebookPath": "/Repos/x/bronze/pos_ingest"}},
    ]}}


def test_pipeline_does_not_model_adf_as_a_data_system():
    frag = AdfParser().parse({"filename": "pl.json", "content": json.dumps(_pipeline())})
    systems = {n.full_path: n.system for n in frag.nodes}
    assert SystemType.ADF not in systems.values()
    assert systems["adf://ds_pos"] == SystemType.ADLS        # from sink storeSettings
    assert "adf://ds_ctl" not in systems                     # Lookup = control metadata
    [edge] = frag.edges
    assert edge.orchestrated_by == "adf:pl_x/cp"
    [orch] = frag.raw_metadata["orchestrations"]
    assert orch["notebook_path"] == "/Repos/x/bronze/pos_ingest"
    assert frag.raw_metadata["control_lookups"][0]["dataset"] == "ds_ctl"


# ---------------------------------------------------------------------------
# UC catalog parser
# ---------------------------------------------------------------------------

def _uc(*names):
    return {"filename": "uc.json", "content": json.dumps({
        "export": {"tool": "unity-catalog-metadata-export"},
        "tables": [{"name": n.split(".")[-1], "full_name": n, "owner": "svc", "comment": f"c {n}"} for n in names]})}


def test_uc_catalog_parser_emits_nodes_only():
    art = _uc("cat.bronze.t1")
    assert UcCatalogParser().can_parse(art)
    frag = UcCatalogParser().parse(art)
    assert [n.full_path for n in frag.nodes] == ["cat.bronze.t1"]
    assert not frag.edges and frag.raw_metadata["catalog_only"]


# ---------------------------------------------------------------------------
# Stitcher — end-to-end over a miniature of the demo pack
# ---------------------------------------------------------------------------

def test_location_keys_match_across_shapes():
    assert adls_location_key("abfss://raw@acct.dfs.core.windows.net/sap/articles/") == "raw/sap/articles"
    assert normalise_location("raw/sap/articles") == "raw/sap/articles"
    assert adls_location_key("abfss://raw@a.dfs.core.windows.net/pos/{country:upper}/{year}/{month}") == \
        normalise_location("raw/pos/{country_code:upper}/{run_date}") == "raw/pos/{*}"


def _wave1_mini():
    ds_pos = {"name": "ds_pos", "properties": {
        "linkedServiceName": {"referenceName": "ls"}, "type": "Parquet",
        "typeProperties": {"location": {"fileSystem": "raw", "folderPath": {
            "value": "@concat('pos/', toUpper(dataset().country_code), '/', formatDateTime(dataset().run_date,'yyyy/MM/dd'))",
            "type": "Expression"}}}}}
    ds_sap = {"name": "ds_sap", "properties": {
        "linkedServiceName": {"referenceName": "ls"}, "type": "Parquet",
        "typeProperties": {"location": {"fileSystem": "raw", "folderPath": "sap/articles"}}}}
    pl_nightly = {"name": "pl_nightly", "properties": {"activities": [
        {"name": "nb_ref", "type": "DatabricksNotebook",
         "typeProperties": {"notebookPath": "/Repos/x/bronze/reference_ingest"}}]}}
    nb_pos = '''# Notebook path: /Repos/x/bronze/pos_ingest
dbutils.widgets.text("country", "")
dbutils.widgets.text("run_date", "")
country = dbutils.widgets.get("country").lower()
year, month, day = dbutils.widgets.get("run_date").split("-")
src = f"abfss://raw@stlake.dfs.core.windows.net/pos/{country.upper()}/{year}/{month}/{day}"
df = (spark.read.format("parquet").load(src))
(df.write.saveAsTable(f"cat.bronze.pos_{country}_tx"))
'''
    nb_ref = '''# Notebook path: /Repos/x/bronze/reference_ingest
spark.read.parquet("abfss://raw@stlake.dfs.core.windows.net/sap/articles").write.saveAsTable("cat.bronze.sap_articles")
'''
    return [
        {"filename": "ds_pos.json", "content": json.dumps(ds_pos)},
        {"filename": "ds_sap.json", "content": json.dumps(ds_sap)},
        {"filename": "pl_pos.json", "content": json.dumps(_pipeline())},
        {"filename": "pl_nightly.json", "content": json.dumps(pl_nightly)},
        {"filename": "databricks_bronze_pos_ingest.py", "content": nb_pos},
        {"filename": "databricks_bronze_reference_ingest.py", "content": nb_ref},
        _uc("cat.bronze.pos_nl_tx", "cat.bronze.pos_es_tx", "cat.bronze.sap_articles", "cat.bronze.unused"),
    ]


def _stitch(artifacts):
    parsers = [NotebookParser(), UcCatalogParser(), AdfParser()]
    frags = [next(p for p in parsers if p.can_parse(a)).parse(a) for a in artifacts]
    graph = stitch_fragments(frags)
    score_graph(graph)
    return graph


def test_adls_dataset_and_notebook_path_become_one_node():
    g = _stitch(_wave1_mini())
    sap = [n for n in g.nodes.values() if "sap/articles" in n.full_path]
    assert len(sap) == 1
    assert sap[0].system == SystemType.ADLS
    assert sap[0].full_path == "abfss://raw@stlake.dfs.core.windows.net/sap/articles"
    assert sap[0].name == "ds_sap"


def test_no_node_is_typed_as_adf():
    g = _stitch(_wave1_mini())
    assert all(n.system != SystemType.ADF for n in g.nodes.values())


def test_templated_paths_expand_against_catalog_tables():
    g = _stitch(_wave1_mini())
    edges = _edges(g)
    base = "abfss://raw@stlake.dfs.core.windows.net/pos"
    assert (f"{base}/NL", "cat.bronze.pos_nl_tx") in edges
    assert (f"{base}/ES", "cat.bronze.pos_es_tx") in edges
    assert ("adf://ds_src", f"{base}/NL") in edges        # ADF copy reaches the same folder
    assert not any("{" in n.full_path for n in g.nodes.values())
    # Unconnected catalog-only tables are not shown
    assert "cat.bronze.unused" not in {n.full_path for n in g.nodes.values()}
    # Pattern-matched edges stay at medium confidence
    nl = next(e for e in g.edges if g.nodes[e.target_id].full_path == "cat.bronze.pos_nl_tx")
    assert nl.confidence < 0.85


def test_notebook_edges_are_tagged_with_the_adf_activity_that_runs_them():
    g = _stitch(_wave1_mini())
    by_target = {g.nodes[e.target_id].full_path: e for e in g.edges}
    assert by_target["cat.bronze.sap_articles"].orchestrated_by == "adf:pl_nightly/nb_ref"
    assert by_target["cat.bronze.pos_nl_tx"].orchestrated_by == "adf:pl_x/nb_ingest"
    assert by_target["abfss://raw@stlake.dfs.core.windows.net/pos/NL"].orchestrated_by == "adf:pl_x/cp"
