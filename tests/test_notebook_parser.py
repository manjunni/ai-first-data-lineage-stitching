"""
Tests for NotebookParser — regex + sqlglot path only (no LLM calls).
Notebooks are built programmatically using nbformat so tests have no binary files.
"""
import json
import nbformat
import pytest

from src.parsers.notebook_parser import NotebookParser
from src.graph.models import SystemType, ConfidenceLevel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_notebook(*cells: tuple[str, str]) -> str:
    """Build a minimal .ipynb JSON string. Each cell is (cell_type, source)."""
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(src) if ctype == "code"
        else nbformat.v4.new_markdown_cell(src)
        for ctype, src in cells
    ]
    return nbformat.writes(nb)


def _parse(*cells: tuple[str, str], filename: str = "test.ipynb"):
    parser = NotebookParser()
    artifact = {"filename": filename, "content": _make_notebook(*cells)}
    return parser.parse(artifact)


# ---------------------------------------------------------------------------
# spark.read.table / df.write.saveAsTable — same cell
# ---------------------------------------------------------------------------

def test_read_table_write_save_as_table():
    code = """
df = spark.read.table("bronze.raw_orders")
df.write.mode("overwrite").saveAsTable("silver.orders")
"""
    frag = _parse(("code", code))
    assert len(frag.edges) == 1
    src = next(n for n in frag.nodes if n.id == frag.edges[0].source_id)
    tgt = next(n for n in frag.nodes if n.id == frag.edges[0].target_id)
    assert src.full_path == "bronze.raw_orders"
    assert tgt.full_path == "silver.orders"


def test_spark_table_write_insert_into():
    code = """
raw = spark.table("hive_metastore.bronze.events")
raw.write.insertInto("hive_metastore.silver.events")
"""
    frag = _parse(("code", code))
    assert len(frag.edges) == 1


def test_adls_read_write():
    code = """
df = spark.read.parquet("abfss://raw@myaccount.dfs.core.windows.net/orders")
df.write.format("delta").save("abfss://silver@myaccount.dfs.core.windows.net/orders")
"""
    frag = _parse(("code", code))
    assert len(frag.edges) == 1
    src = next(n for n in frag.nodes if n.id == frag.edges[0].source_id)
    assert src.system == SystemType.ADLS


# ---------------------------------------------------------------------------
# Cross-cell variable tracking
# ---------------------------------------------------------------------------

def test_cross_cell_simple_chain():
    frag = _parse(
        ("code", 'raw = spark.read.table("bronze.orders")'),
        ("code", 'raw.write.saveAsTable("silver.orders")'),
    )
    assert len(frag.edges) == 1


def test_cross_cell_transform_chain():
    frag = _parse(
        ("code", 'raw = spark.read.table("bronze.events")'),
        ("code", 'clean = raw.filter("status = \'active\'")'),
        ("code", 'clean.write.saveAsTable("silver.events")'),
    )
    assert len(frag.edges) == 1
    src = next(n for n in frag.nodes if n.id == frag.edges[0].source_id)
    assert src.full_path == "bronze.events"


def test_transformation_reflects_operations():
    frag = _parse(
        ("code", 'df = spark.read.table("bronze.sales")'),
        ("code", 'agg = df.groupBy("region").agg({"amount": "sum"})'),
        ("code", 'agg.write.saveAsTable("gold.sales_by_region")'),
    )
    assert len(frag.edges) == 1
    transform = frag.edges[0].transformation
    assert "aggregated" in transform


def test_filter_in_transformation():
    frag = _parse(
        ("code", 'df = spark.read.table("bronze.orders")'),
        ("code", 'clean = df.filter(df.amount > 0).distinct()'),
        ("code", 'clean.write.saveAsTable("silver.orders")'),
    )
    transform = frag.edges[0].transformation
    assert "filtered" in transform
    assert "deduplicated" in transform


# ---------------------------------------------------------------------------
# %sql magic cells
# ---------------------------------------------------------------------------

def test_sql_magic_cell():
    frag = _parse(("code", "%sql\nINSERT INTO silver.orders SELECT * FROM bronze.raw_orders"))
    assert len(frag.edges) == 1
    tgt = next(n for n in frag.nodes if n.id == frag.edges[0].target_id)
    assert "silver" in tgt.full_path


def test_sql_magic_create_table():
    sql = "%sql\nCREATE TABLE main.gold.summary AS SELECT dept, COUNT(*) FROM main.silver.emp GROUP BY dept"
    frag = _parse(("code", sql))
    assert len(frag.edges) == 1


# ---------------------------------------------------------------------------
# spark.sql("...") calls
# ---------------------------------------------------------------------------

def test_spark_sql_call():
    code = 'spark.sql("INSERT INTO silver.a SELECT * FROM bronze.b")'
    frag = _parse(("code", code))
    assert len(frag.edges) == 1


def test_spark_sql_triple_quoted():
    code = '''spark.sql("""
    INSERT INTO gold.summary
    SELECT region, SUM(amount) FROM silver.sales GROUP BY region
""")'''
    frag = _parse(("code", code))
    assert len(frag.edges) == 1


# ---------------------------------------------------------------------------
# Multi-line chained calls (backslash and open-paren continuation)
# ---------------------------------------------------------------------------

def test_multiline_backslash_read():
    code = (
        "df = spark.read \\\n"
        '    .format("delta") \\\n'
        '    .load("abfss://bronze@storage.dfs.core.windows.net/orders")\n'
        'df.write.saveAsTable("silver.orders")'
    )
    frag = _parse(("code", code))
    assert len(frag.edges) == 1
    src = next(n for n in frag.nodes if n.id == frag.edges[0].source_id)
    assert src.system == SystemType.ADLS


def test_multiline_paren_write():
    code = (
        'df = spark.read.table("bronze.events")\n'
        "(df\n"
        '    .filter("active = true")\n'
        '    .write.saveAsTable("silver.events"))'
    )
    frag = _parse(("code", code))
    # At minimum a node for bronze.events should be present
    assert any(n.full_path == "bronze.events" for n in frag.nodes)


# ---------------------------------------------------------------------------
# DeltaTable patterns
# ---------------------------------------------------------------------------

def test_delta_table_for_name():
    code = """
from delta.tables import DeltaTable
dt = DeltaTable.forName(spark, "main.silver.orders")
dt.toDF().write.saveAsTable("main.gold.orders")
"""
    frag = _parse(("code", code))
    assert any(n.full_path == "main.silver.orders" for n in frag.nodes)


# ---------------------------------------------------------------------------
# System type inference
# ---------------------------------------------------------------------------

def test_three_part_unity_catalog_is_databricks():
    code = """
df = spark.read.table("main.silver.customers")
df.write.saveAsTable("main.gold.customers")
"""
    frag = _parse(("code", code))
    for node in frag.nodes:
        assert node.system == SystemType.DATABRICKS


# ---------------------------------------------------------------------------
# Confidence and evidence
# ---------------------------------------------------------------------------

def test_regex_path_confidence_is_high():
    code = """
df = spark.read.table("bronze.a")
df.write.saveAsTable("silver.a")
"""
    frag = _parse(("code", code))
    assert frag.edges[0].confidence == 0.85
    assert frag.edges[0].confidence_level == ConfidenceLevel.HIGH


def test_evidence_mentions_regex():
    code = """
df = spark.read.table("bronze.a")
df.write.saveAsTable("silver.a")
"""
    frag = _parse(("code", code))
    assert "regex" in frag.edges[0].evidence


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------

def test_empty_notebook_returns_empty_fragment():
    frag = _parse()
    assert frag is not None
    assert frag.artifact_type == "notebook"


def test_markdown_cells_ignored():
    frag = _parse(
        ("markdown", "# This is a heading"),
        ("code", 'df = spark.read.table("bronze.a")\ndf.write.saveAsTable("silver.a")'),
    )
    assert len(frag.edges) == 1


def test_nodes_deduplicated_across_cells():
    frag = _parse(
        ("code", 'a = spark.read.table("bronze.shared")\na.write.saveAsTable("silver.x")'),
        ("code", 'b = spark.read.table("bronze.shared")\nb.write.saveAsTable("silver.y")'),
    )
    paths = [n.full_path for n in frag.nodes]
    assert paths.count("bronze.shared") == 1
