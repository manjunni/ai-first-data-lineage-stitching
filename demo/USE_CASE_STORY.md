# Use case: "Which number is right?" Net revenue lineage for audit

**Client (fictional): Harborline Retail Group.** All names, systems and figures are invented for the demo.

## The situation
Harborline is mid-migration from an on-prem SQL Server sales warehouse to an Azure Databricks lakehouse. For two quarters, two reports have claimed to show **net revenue**:

| Report | Owner | Built on |
|---|---|---|
| **Monthly Revenue Pack** (SSRS) | Finance | Legacy SQL Server load, still running |
| **Executive Net Revenue** (Power BI) | CFO office | New lakehouse (ADF, notebooks, Delta Live Tables) |

At quarter close the two disagree and nobody can say why. Internal Audit has asked for **documented, end-to-end lineage for the net revenue KPI** within ten working days.

## Why the existing tooling can't answer it
Lineage is in five places and no tool sees across them:
- the **legacy SQL script** (SQL Server Agent job)
- **Azure Data Factory** pipelines and datasets
- **Databricks notebooks** (one of them config-driven, so static parsers see nothing)
- a **Delta Live Tables** pipeline
- **people's heads**: how customers and returns are loaded, where Finance's manual adjustments come from, which reports sit downstream

Each tool shows its own island. The joins between islands (SQL to lakehouse, ADLS to Databricks, Databricks to BI) are exactly where the answer lives.

## Cast
| Persona | Role in the demo | App role |
|---|---|---|
| Priya Nair, Head of Data Governance | Sponsor; signs off lineage for Audit | Data Owner |
| Tomasz Kowalski, Sales Domain Steward | Reviews domain lineage | Domain Steward |
| Marcus Oyelaran, Data Engineering Lead | Knows how data is loaded | SME |
| Elena Brandt, Finance Systems | Owns manual adjustments and the legacy report | SME |
| Jordan Ellis, BI Lead | Knows what sits on top of the gold table | SME |
| Chloe Park, Data Analyst | Can review, cannot sign off | Analyst |

## What the agent does with it (the demo beats)
1. **Reads 10 artifacts of 4 technology types** in one upload: a notebook, a config-driven notebook, a DLT pipeline, a legacy SQL script, ADF datasets and an ADF pipeline.
2. **Stitches across boundaries**: 14 nodes, 13 edges, with `adls→sql`, `sql→databricks` and `databricks→dlt` boundaries labelled. ADF dataset files are merged into the matching SQL and Databricks tables automatically.
3. **Shows column-level lineage** where it can parse it, e.g. `net_amount = gross_amount - discount_amount`.
4. **Falls back to the LLM** for the config-driven notebook (deterministic parsers find nothing). Those edges come back at **60% confidence**, flagged in red.
5. **Finds 7 gaps** and turns them into targeted questions for the right people: 3 "where does this come from?", 2 "does this feed anything?", 2 "is this link right?".
6. **Takes answers in the form people actually give them**: typed, a Word reply, an Excel reply, or one email-style reply for everything. Answers add the missing ADLS sources, the Power BI dataset and the SSRS report.
7. **Gates sign-off by role**: the Analyst cannot sign off; only the Data Owner's approval allows publishing; a Data Owner decision becomes read-only; rejections need a reason.
8. **Publishes a dry-run payload** for Alation or Collibra, including column mappings, and shows a full **audit trail**.

## The reveal
With the graph complete, the two reports are visibly different pipelines from the same `dbo.orders`:

- **Legacy path:** `dbo.orders` -> `reporting.daily_sales_summary` -> *Monthly Revenue Pack*. Net revenue = `gross_amount - discount_amount`. No returns, no finance adjustments.
- **Lakehouse path:** `dbo.orders` -> `orders_raw` -> `orders_clean` -> `orders_adjusted` (+ finance adjustments) -> `silver_order_enriched` (- refunds from returns) -> `gold_net_revenue_by_segment` -> *Executive Net Revenue*.

The disagreement is not a data error; it is two different definitions. Nobody could see that until the lineage was stitched. (The pack does not contain data values, so do not quote a variance figure; use the client's real numbers in a real engagement.)

## Why this matters to the stakeholders
- **Governance:** audit-ready evidence in days, with who approved what and when.
- **Engineering:** the agent drafts; people confirm. Engineers answer a handful of precise questions instead of being interviewed.
- **Finance:** the manual Excel adjustment path becomes visible and owned.
- **Trust in AI:** every LLM-derived edge is visibly lower-confidence and needs human approval before publishing.

## Pitch line
*"Catalogs show lineage inside each system. The agent connects the systems, asks the humans only what it can't know, and leaves an audit trail."*
