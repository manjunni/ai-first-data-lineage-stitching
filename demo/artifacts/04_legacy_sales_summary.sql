-- Legacy on-prem reporting load (SQL Server, scheduled by SQL Agent).
-- Feeds the monthly Revenue Pack that Finance still reconciles against.
-- Owner: Finance Systems.
INSERT INTO reporting.daily_sales_summary
SELECT o.order_date,
       c.segment AS customer_segment,
       SUM(o.gross_amount - o.discount_amount) AS net_revenue
FROM dbo.orders o
JOIN dbo.customers c ON o.customer_id = c.customer_id
WHERE o.order_status <> 'CANCELLED'
GROUP BY o.order_date, c.segment;
