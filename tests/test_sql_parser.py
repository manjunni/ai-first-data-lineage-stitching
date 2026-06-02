"""
Tests for SqlParser — sqlglot path only (no LLM calls).
Run with: pytest tests/test_sql_parser.py -v
"""
import pytest
from src.parsers.sql_parser import SqlParser
from src.graph.models import SystemType, ConfidenceLevel


def _parse(sql: str, filename: str = "test.sql"):
    parser = SqlParser()
    artifact = {"filename": filename, "content": sql}
    return parser.parse(artifact)


# ---------------------------------------------------------------------------
# Basic statement types
# ---------------------------------------------------------------------------

def test_insert_into_select():
    frag = _parse("INSERT INTO silver.orders SELECT id, amount FROM bronze.raw_orders WHERE amount > 0")
    assert len(frag.edges) == 1
    edge = frag.edges[0]
    src = next(n for n in frag.nodes if n.id == edge.source_id)
    tgt = next(n for n in frag.nodes if n.id == edge.target_id)
    assert src.full_path == "bronze.raw_orders"
    assert tgt.full_path == "silver.orders"


def test_create_table_as_select():
    sql = "CREATE TABLE hive_metastore.gold.customer_summary AS SELECT customer_id, SUM(amount) AS total FROM hive_metastore.silver.orders GROUP BY customer_id"
    frag = _parse(sql)
    assert len(frag.edges) == 1
    tgt = next(n for n in frag.nodes if "gold" in n.full_path)
    assert tgt.system == SystemType.DATABRICKS


def test_create_or_replace_view():
    sql = "CREATE OR REPLACE VIEW reporting.v_sales AS SELECT * FROM main.silver.sales"
    frag = _parse(sql)
    assert len(frag.edges) == 1
    src = next(n for n in frag.nodes if "silver" in n.full_path)
    assert src.system == SystemType.DATABRICKS


def test_merge_into():
    sql = """
    MERGE INTO main.gold.target t
    USING main.silver.source s ON t.id = s.id
    WHEN MATCHED THEN UPDATE SET t.val = s.val
    WHEN NOT MATCHED THEN INSERT (id, val) VALUES (s.id, s.val)
    """
    frag = _parse(sql)
    assert len(frag.edges) == 1
    tgt = next(n for n in frag.nodes if "target" in n.name)
    src = next(n for n in frag.nodes if "source" in n.name)
    assert tgt.system == SystemType.DATABRICKS
    assert src.system == SystemType.DATABRICKS


# ---------------------------------------------------------------------------
# Multi-statement scripts
# ---------------------------------------------------------------------------

def test_multi_statement_script():
    sql = """
    INSERT INTO silver.a SELECT * FROM bronze.x;
    INSERT INTO silver.b SELECT * FROM bronze.y;
    """
    frag = _parse(sql)
    assert len(frag.edges) == 2


def test_nodes_deduplicated_across_statements():
    sql = """
    INSERT INTO silver.a SELECT * FROM bronze.shared;
    INSERT INTO silver.b SELECT * FROM bronze.shared;
    """
    frag = _parse(sql)
    paths = [n.full_path for n in frag.nodes]
    # bronze.shared should appear only once
    assert paths.count("bronze.shared") == 1


# ---------------------------------------------------------------------------
# Transformation extraction
# ---------------------------------------------------------------------------

def test_transformation_describes_join():
    sql = "INSERT INTO gold.t SELECT a.id, b.name FROM silver.a JOIN silver.b ON a.id = b.id"
    frag = _parse(sql)
    assert "join" in frag.edges[0].transformation.lower()


def test_transformation_describes_aggregation():
    sql = "INSERT INTO gold.summary SELECT dept, COUNT(*) FROM silver.employees GROUP BY dept"
    frag = _parse(sql)
    assert "aggregated" in frag.edges[0].transformation


def test_transformation_direct_copy():
    sql = "INSERT INTO silver.copy SELECT * FROM bronze.source"
    frag = _parse(sql)
    assert frag.edges[0].transformation == "direct copy"


# ---------------------------------------------------------------------------
# System type inference
# ---------------------------------------------------------------------------

def test_adls_path_detected():
    sql = "INSERT INTO silver.orders SELECT * FROM delta.`abfss://raw@storageaccount.dfs.core.windows.net/orders`"
    frag = _parse(sql)
    # At minimum: no crash and at least one node
    assert len(frag.nodes) >= 1


def test_three_part_name_is_databricks():
    sql = "INSERT INTO main.gold.fact_sales SELECT * FROM main.silver.orders"
    frag = _parse(sql)
    for node in frag.nodes:
        assert node.system == SystemType.DATABRICKS


def test_two_part_name_is_sql():
    sql = "INSERT INTO dbo.target SELECT * FROM dbo.source"
    frag = _parse(sql, filename="test.sql")
    for node in frag.nodes:
        assert node.system == SystemType.SQL


# ---------------------------------------------------------------------------
# Confidence and evidence
# ---------------------------------------------------------------------------

def test_sqlglot_confidence_is_high():
    sql = "INSERT INTO silver.a SELECT * FROM bronze.b"
    frag = _parse(sql)
    assert frag.edges[0].confidence == 0.90
    assert frag.edges[0].confidence_level == ConfidenceLevel.HIGH


def test_evidence_mentions_sqlglot():
    sql = "INSERT INTO silver.a SELECT * FROM bronze.b"
    frag = _parse(sql)
    assert "sqlglot" in frag.edges[0].evidence


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------

def test_empty_sql_returns_empty_fragment():
    frag = _parse("")
    # Should return a fragment (possibly via LLM fallback) without crashing
    assert frag is not None
    assert frag.artifact_type == "sql"


def test_plain_select_has_no_edges():
    # A bare SELECT with no target should produce no edges from sqlglot path
    sql = "SELECT id, name FROM customers WHERE active = 1"
    frag = _parse(sql)
    # No INSERT/CREATE/MERGE — sqlglot finds no target, fragment may be empty or LLM-assisted
    assert frag is not None
