from pyspark.sql.functions import when, col

df = spark.table("curated.customer_summary")

segmented = df.withColumn(
    "segment",
    when(col("total_spent") > 1000, "High Value")
    .when(col("total_spent") >= 500, "Medium Value")
    .otherwise("Low Value")
)

segmented.write.mode("overwrite").saveAsTable("curated.customer_segment")
