summary = spark.table("curated.customer_segment")
orders = spark.table("curated.customer_orders")

dashboard = summary.join(
    orders.select("customer_id", "full_name", "email"),
    on="customer_id",
    how="inner"
).dropDuplicates(["customer_id"])

dashboard.write.mode("overwrite").saveAsTable("mart.customer_dashboard")
