from pyspark.sql.functions import sum, count, avg, max

customer_orders = spark.table("curated.customer_orders")

customer_summary = customer_orders.groupBy("customer_id").agg(
    count("order_id").alias("total_orders"),
    sum("amount").alias("total_spent"),
    avg("amount").alias("avg_order_value"),
    max("order_date").alias("last_order_date")
)

customer_summary.write.mode("overwrite").saveAsTable("curated.customer_summary")
