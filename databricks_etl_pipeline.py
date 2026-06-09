# Databricks notebook source

# COMMAND ----------
# Step 1: Create schemas
spark.sql("""
CREATE SCHEMA IF NOT EXISTS staging
""")

spark.sql("""
CREATE SCHEMA IF NOT EXISTS curated
""")

# COMMAND ----------
# Step 2: staging customers_clean
spark.sql("""
CREATE OR REPLACE TABLE customers_clean AS
SELECT
    id AS customer_id,
    UPPER(first_name) AS first_name,
    UPPER(last_name) AS last_name,
    LOWER(email) AS email,
    created_at
FROM bronze.customers
WHERE email IS NOT NULL
""")

# COMMAND ----------
# Step 2: staging orders_clean
spark.sql("""
CREATE OR REPLACE TABLE orders_clean AS
SELECT
    order_id,
    customer_id,
    amount,
    status,
    created_at
FROM bronze.orders
WHERE status IN ('completed', 'shipped')
""")

# COMMAND ----------
# Step 3: curated layer
spark.sql("""
CREATE OR REPLACE TABLE customer_orders AS
SELECT
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS full_name,
    c.email,
    o.order_id,
    o.amount,
    o.created_at AS order_date
FROM customers_clean c
JOIN orders_clean o
    ON c.customer_id = o.customer_id
""")
