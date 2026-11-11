"""Generates the Harborline demo input artifacts (notebooks need valid nbformat JSON).
Run from the repo root:  python demo/scripts/_build_artifacts.py
"""
import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = Path(__file__).resolve().parent.parent / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)


def write_nb(name: str, cells: list) -> None:
    nb = new_notebook(cells=cells)
    nb.metadata["language_info"] = {"name": "python"}
    nbformat.validate(nb)
    (OUT / name).write_text(nbformat.writes(nb), encoding="utf-8")


# ---------------------------------------------------------------------------
# 01  Databricks notebook — bronze -> silver (deterministic: regex + sqlglot)
# ---------------------------------------------------------------------------
write_nb("01_nb_bronze_to_silver.ipynb", [
    new_markdown_cell(
        "# 01 - Bronze to Silver (Harborline Retail)\n"
        "Owner: Data Engineering (Marcus Oyelaran). Cleans orders and customers for the net revenue model."
    ),
    new_code_cell(
        "from pyspark.sql import functions as F\n"
        "\n"
        "# Orders: drop cancelled orders, derive net_amount (gross less discount)\n"
        "orders_bronze = spark.read.table(\"harborline.bronze.orders_raw\")\n"
        "o1 = orders_bronze.filter(F.col(\"order_status\") != \"CANCELLED\")\n"
        "o2 = o1.withColumn(\"net_amount\", F.col(\"gross_amount\") - F.col(\"discount_amount\"))\n"
        "orders_clean = o2.select(\"order_id\", \"customer_id\", \"order_date\", \"gross_amount\", \"discount_amount\", \"net_amount\")\n"
        "orders_clean.write.mode(\"overwrite\").saveAsTable(\"harborline.silver.orders_clean\")\n"
    ),
    new_code_cell(
        "%sql\n"
        "CREATE OR REPLACE TABLE harborline.silver.customers_clean AS\n"
        "SELECT customer_id,\n"
        "       INITCAP(TRIM(full_name)) AS full_name,\n"
        "       UPPER(country_code)      AS country_code,\n"
        "       segment                  AS customer_segment\n"
        "FROM harborline.bronze.customers_raw\n"
        "WHERE is_active = true\n"
    ),
])

# ---------------------------------------------------------------------------
# 02  Databricks notebook — finance adjustments. Table names come from a config
#     dict, so the deterministic regex parser finds nothing and the agent must
#     fall back to the LLM (lower confidence -> human review).
# ---------------------------------------------------------------------------
write_nb("02_nb_apply_finance_adjustments.ipynb", [
    new_markdown_cell(
        "# 02 - Apply Finance Adjustments\n"
        "Owner: Finance Systems (Elena Brandt). Applies manual credit notes / accrual adjustments to silver orders.\n"
        "Table names are driven by the job config so the same notebook runs in dev, test and prod."
    ),
    new_code_cell(
        "from pyspark.sql import functions as F\n"
        "\n"
        "CONFIG = {\n"
        "    \"orders_table\":      \"harborline.silver.orders_clean\",\n"
        "    \"adjustments_table\": \"harborline.finance.revenue_adjustments\",\n"
        "    \"output_table\":      \"harborline.silver.orders_adjusted\",\n"
        "}\n"
    ),
    new_code_cell(
        "orders = spark.table(CONFIG[\"orders_table\"])\n"
        "adjustments = spark.table(CONFIG[\"adjustments_table\"])\n"
        "\n"
        "adjusted = (\n"
        "    orders.join(adjustments, \"order_id\", \"left\")\n"
        "          .withColumn(\n"
        "              \"adjusted_net_amount\",\n"
        "              F.col(\"net_amount\") + F.coalesce(F.col(\"adjustment_amount\"), F.lit(0)),\n"
        "          )\n"
        ")\n"
        "adjusted.write.mode(\"overwrite\").saveAsTable(CONFIG[\"output_table\"])\n"
    ),
])

# ---------------------------------------------------------------------------
# 05..10  ADF dataset + pipeline JSON
#   Datasets are numbered BEFORE the pipeline on purpose: the stitcher only links
#   an adf:// dataset to a matching Databricks/SQL table node when the dataset
#   file is processed before the pipeline file that references it.
# ---------------------------------------------------------------------------
def dump(name: str, obj: dict) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def copy_activity(name, src, sink, src_type, sink_type):
    return {
        "name": name,
        "type": "Copy",
        "dependsOn": [],
        "policy": {"timeout": "0.12:00:00", "retry": 2},
        "typeProperties": {"source": {"type": src_type}, "sink": {"type": sink_type}},
        "inputs": [{"referenceName": src, "type": "DatasetReference"}],
        "outputs": [{"referenceName": sink, "type": "DatasetReference"}],
    }


def adls_dataset(name, folder):
    return {
        "name": name,
        "properties": {
            "linkedServiceName": {"referenceName": "ls_adls_harborlake", "type": "LinkedServiceReference"},
            "type": "Parquet",
            "typeProperties": {
                "location": {"type": "AzureBlobFSLocation", "fileSystem": "pos-export", "folderPath": folder}
            },
        },
    }


def sql_dataset(name, table):
    return {
        "name": name,
        "properties": {
            "linkedServiceName": {"referenceName": "ls_azuresql_salesdb", "type": "LinkedServiceReference"},
            "type": "AzureSqlTable",
            "typeProperties": {"schema": "dbo", "table": table},
        },
    }


def delta_dataset(name, table):
    # For Unity Catalog the "database" holds <catalog>.<schema> in this demo export.
    return {
        "name": name,
        "properties": {
            "linkedServiceName": {"referenceName": "ls_databricks_harborline", "type": "LinkedServiceReference"},
            "type": "AzureDatabricksDeltaLakeDataset",
            "typeProperties": {"database": "harborline.bronze", "table": table},
        },
    }


dump("05_adf_ds_pos_export_orders.json", adls_dataset("pos_export_orders", "orders"))
dump("06_adf_ds_pos_export_customers.json", adls_dataset("pos_export_customers", "customers"))
dump("07_adf_ds_salesdb_orders.json", sql_dataset("salesdb_orders", "orders"))
dump("08_adf_ds_salesdb_customers.json", sql_dataset("salesdb_customers", "customers"))
dump("09_adf_ds_bronze_orders_raw.json", delta_dataset("bronze_orders_raw", "orders_raw"))

dump("10_adf_pl_ingest_sales_daily.json", {
    "name": "pl_ingest_sales_daily",
    "properties": {
        "description": "Nightly: POS exports -> SQL sales DW, then sales DW orders -> Databricks bronze.",
        "activities": [
            copy_activity("Copy_Orders_ToDW", "pos_export_orders", "salesdb_orders", "ParquetSource", "AzureSqlSink"),
            copy_activity("Copy_Customers_ToDW", "pos_export_customers", "salesdb_customers", "ParquetSource", "AzureSqlSink"),
            copy_activity("Copy_Orders_ToLakehouse", "salesdb_orders", "bronze_orders_raw", "AzureSqlSource", "DatabricksDeltaLakeSink"),
        ],
        "annotations": ["domain:sales", "owner:data-engineering"],
    },
})
print("built notebooks + ADF json in", OUT)
