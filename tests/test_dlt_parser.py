"""
Tests for DltParser — AST path (Python DLT) and sqlglot path (SQL DLT).
No LLM calls.
"""
import pytest
from src.parsers.dlt_parser import DltParser
from src.graph.models import SystemType, ConfidenceLevel


def _parse_py(code: str, filename: str = "pipeline.py"):
    parser = DltParser()
    return parser.parse({"filename": filename, "content": code})


def _parse_sql(sql: str, filename: str = "pipeline.sql"):
    parser = DltParser()
    return parser.parse({"filename": filename, "content": sql})


# ===========================================================================
# Python DLT — @dlt.table
# ===========================================================================

class TestPythonDlt:

    def test_bare_decorator_function_name_is_target(self):
        code = """
import dlt

@dlt.table
def silver_orders():
    return dlt.read("bronze_orders")
"""
        frag = _parse_py(code)
        assert len(frag.edges) == 1
        tgt = next(n for n in frag.nodes if n.id == frag.edges[0].target_id)
        assert tgt.full_path == "silver_orders"
        assert tgt.system == SystemType.DLT

    def test_decorator_with_name_kwarg_overrides_function_name(self):
        code = """
import dlt

@dlt.table(name="silver_customers", comment="cleaned customers")
def _silver_customers():
    return dlt.read("bronze_customers")
"""
        frag = _parse_py(code)
        tgt = next(n for n in frag.nodes if n.id == frag.edges[0].target_id)
        assert tgt.full_path == "silver_customers"

    def test_dlt_view_decorator(self):
        code = """
import dlt

@dlt.view
def active_orders():
    return dlt.read("silver_orders").filter("status = 'active'")
"""
        frag = _parse_py(code)
        assert len(frag.edges) == 1
        tgt = next(n for n in frag.nodes if n.id == frag.edges[0].target_id)
        assert tgt.full_path == "active_orders"

    def test_dlt_read_source_is_dlt_system_type(self):
        code = """
import dlt

@dlt.table
def gold_summary():
    return dlt.read("silver_orders")
"""
        frag = _parse_py(code)
        src = next(n for n in frag.nodes if n.id == frag.edges[0].source_id)
        assert src.system == SystemType.DLT

    def test_dlt_read_stream(self):
        code = """
import dlt

@dlt.table
def silver_events():
    return dlt.read_stream("bronze_events")
"""
        frag = _parse_py(code)
        assert len(frag.edges) == 1
        src = next(n for n in frag.nodes if n.id == frag.edges[0].source_id)
        assert src.full_path == "bronze_events"

    def test_external_spark_read_table(self):
        code = """
import dlt

@dlt.table
def bronze_orders():
    return spark.read.table("hive_metastore.raw.orders")
"""
        frag = _parse_py(code)
        assert len(frag.edges) == 1
        src = next(n for n in frag.nodes if n.id == frag.edges[0].source_id)
        assert src.full_path == "hive_metastore.raw.orders"
        assert src.system == SystemType.DATABRICKS

    def test_external_adls_read(self):
        code = """
import dlt

@dlt.table
def bronze_raw():
    return (spark.read
        .format("parquet")
        .load("abfss://raw@storage.dfs.core.windows.net/orders"))
"""
        frag = _parse_py(code)
        assert len(frag.edges) == 1
        src = next(n for n in frag.nodes if n.id == frag.edges[0].source_id)
        assert src.system == SystemType.ADLS

    def test_multiple_dlt_sources(self):
        code = """
import dlt
from pyspark.sql.functions import col

@dlt.table
def gold_order_summary():
    orders = dlt.read("silver_orders")
    customers = dlt.read("silver_customers")
    return orders.join(customers, "customer_id")
"""
        frag = _parse_py(code)
        assert len(frag.edges) == 2
        src_paths = {n.full_path for n in frag.nodes if n.id in {e.source_id for e in frag.edges}}
        assert "silver_orders" in src_paths
        assert "silver_customers" in src_paths

    def test_multiple_dlt_tables_in_one_file(self):
        code = """
import dlt

@dlt.table
def silver_orders():
    return dlt.read("bronze_raw_orders")

@dlt.table
def gold_summary():
    return dlt.read("silver_orders")
"""
        frag = _parse_py(code)
        assert len(frag.edges) == 2

    def test_transformation_ops_extracted(self):
        code = """
import dlt

@dlt.table
def silver_orders():
    df = dlt.read("bronze_orders")
    return df.filter("amount > 0").groupBy("region").agg({"amount": "sum"})
"""
        frag = _parse_py(code)
        transform = frag.edges[0].transformation
        assert "filtered" in transform
        assert "aggregated" in transform

    def test_no_decorator_produces_no_edges(self):
        code = """
def helper():
    return spark.read.table("some_table")
"""
        frag = _parse_py(code)
        assert len(frag.edges) == 0

    def test_confidence_is_high(self):
        code = """
import dlt

@dlt.table
def silver_a():
    return dlt.read("bronze_a")
"""
        frag = _parse_py(code)
        assert frag.edges[0].confidence == 0.90
        assert frag.edges[0].confidence_level == ConfidenceLevel.HIGH

    def test_evidence_mentions_ast(self):
        code = """
import dlt

@dlt.table
def silver_a():
    return dlt.read("bronze_a")
"""
        frag = _parse_py(code)
        assert "ast" in frag.edges[0].evidence

    def test_nodes_deduplicated_across_tables(self):
        code = """
import dlt

@dlt.table
def silver_a():
    return dlt.read("shared_bronze")

@dlt.table
def silver_b():
    return dlt.read("shared_bronze")
"""
        frag = _parse_py(code)
        paths = [n.full_path for n in frag.nodes]
        assert paths.count("shared_bronze") == 1

    def test_mixed_dlt_and_external_sources(self):
        code = """
import dlt

@dlt.table
def gold_enriched():
    orders = dlt.read("silver_orders")
    ref = spark.read.table("main.reference.products")
    return orders.join(ref, "product_id")
"""
        frag = _parse_py(code)
        assert len(frag.edges) == 2
        systems = {n.system for n in frag.nodes if n.id in {e.source_id for e in frag.edges}}
        assert SystemType.DLT in systems
        assert SystemType.DATABRICKS in systems


# ===========================================================================
# SQL DLT
# ===========================================================================

class TestSqlDlt:

    def test_create_live_table(self):
        sql = """
CREATE OR REFRESH LIVE TABLE silver_orders
AS SELECT order_id, customer_id, amount
FROM LIVE.bronze_raw_orders
WHERE amount > 0;
"""
        frag = _parse_sql(sql)
        assert len(frag.edges) == 1
        src = next(n for n in frag.nodes if n.id == frag.edges[0].source_id)
        tgt = next(n for n in frag.nodes if n.id == frag.edges[0].target_id)
        assert src.system == SystemType.DLT
        assert tgt.full_path == "silver_orders"

    def test_create_live_view(self):
        sql = """
CREATE OR REFRESH LIVE VIEW active_orders
AS SELECT * FROM LIVE.silver_orders WHERE status = 'active';
"""
        frag = _parse_sql(sql)
        assert len(frag.edges) == 1

    def test_streaming_live_table(self):
        sql = """
CREATE OR REFRESH STREAMING LIVE TABLE silver_events
AS SELECT * FROM STREAM(LIVE.bronze_events);
"""
        frag = _parse_sql(sql)
        assert len(frag.edges) == 1
        src = next(n for n in frag.nodes if n.id == frag.edges[0].source_id)
        assert src.system == SystemType.DLT

    def test_external_non_live_source(self):
        sql = """
CREATE OR REFRESH LIVE TABLE bronze_customers
AS SELECT * FROM main.raw.customers;
"""
        frag = _parse_sql(sql)
        assert len(frag.edges) == 1
        src = next(n for n in frag.nodes if n.id == frag.edges[0].source_id)
        assert src.full_path == "main.raw.customers"
        assert src.system == SystemType.DATABRICKS

    def test_multi_statement_sql_file(self):
        sql = """
CREATE OR REFRESH LIVE TABLE silver_orders
AS SELECT * FROM LIVE.bronze_orders;

CREATE OR REFRESH LIVE TABLE gold_summary
AS SELECT region, SUM(amount) AS total FROM LIVE.silver_orders GROUP BY region;
"""
        frag = _parse_sql(sql)
        assert len(frag.edges) == 2

    def test_live_source_node_has_no_live_prefix_in_path(self):
        sql = "CREATE OR REFRESH LIVE TABLE silver AS SELECT * FROM LIVE.bronze;"
        frag = _parse_sql(sql)
        for node in frag.nodes:
            assert "LIVE." not in node.full_path
            assert "_dlt_" not in node.full_path

    def test_artifact_type_is_dlt(self):
        sql = "CREATE OR REFRESH LIVE TABLE t AS SELECT * FROM LIVE.s;"
        frag = _parse_sql(sql)
        assert frag.artifact_type == "dlt"


# ===========================================================================
# can_parse
# ===========================================================================

class TestCanParse:

    def test_explicit_artifact_type(self):
        parser = DltParser()
        assert parser.can_parse({"filename": "x.py", "content": "", "artifact_type": "dlt"})

    def test_py_file_with_dlt_import(self):
        parser = DltParser()
        assert parser.can_parse({"filename": "pipeline.py", "content": "import dlt\n"})

    def test_py_file_with_dlt_decorator(self):
        parser = DltParser()
        assert parser.can_parse({"filename": "pipeline.py", "content": "@dlt.table\ndef foo(): pass"})

    def test_sql_file_with_live_table(self):
        parser = DltParser()
        assert parser.can_parse({"filename": "pipeline.sql", "content": "CREATE OR REFRESH LIVE TABLE t AS SELECT 1"})

    def test_ipynb_is_not_parsed(self):
        parser = DltParser()
        assert not parser.can_parse({"filename": "notebook.ipynb", "content": "import dlt"})

    def test_plain_py_without_dlt_is_not_parsed(self):
        parser = DltParser()
        assert not parser.can_parse({"filename": "utils.py", "content": "def helper(): pass"})
