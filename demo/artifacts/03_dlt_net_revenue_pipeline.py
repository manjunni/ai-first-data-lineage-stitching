# Databricks notebook source
# Delta Live Tables pipeline: net revenue model (Harborline Retail)
# Owner: Data Engineering. Target schema: harborline.silver / harborline.gold
import dlt
from pyspark.sql import functions as F


@dlt.table(
    name="silver_order_enriched",
    comment="Orders enriched with customer segment, finance adjustments and refunds",
)
def silver_order_enriched():
    orders = spark.read.table("harborline.silver.orders_adjusted")
    customers = spark.read.table("harborline.silver.customers_clean")
    returns = spark.read.table("harborline.bronze.returns_raw")
    return (
        orders.join(customers, "customer_id", "left")
              .join(returns, "order_id", "left")
              .withColumn(
                  "net_revenue",
                  F.col("adjusted_net_amount") - F.coalesce(F.col("refund_amount"), F.lit(0)),
              )
    )


@dlt.table(
    name="gold_net_revenue_by_segment",
    comment="Net revenue by customer segment and month - feeds the executive dashboard",
)
def gold_net_revenue_by_segment():
    return (
        dlt.read("silver_order_enriched")
           .groupBy("customer_segment", F.date_trunc("month", F.col("order_date")).alias("order_month"))
           .agg(F.sum("net_revenue").alias("net_revenue"))
    )
