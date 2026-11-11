# Demo pack: Harborline net revenue lineage

Everything needed to demo the Data Lineage Stitching Agent against a fictional client. Read `USE_CASE_STORY.md` first for the narrative.

```
demo/
  USE_CASE_STORY.md          business story, cast, beats, the reveal
  .env.demo.example          settings + demo users
  artifacts/                 the 10 files you upload (numbered; keep that order)
  sme_answers/               A1..A7 answers (txt/docx/xlsx) + playbook + email example
  scripts/
    validate_demo.py         headless end-to-end check (run before every demo)
    demo_driver.py           scripts the dull parts against the API
    run_offline_api.py       Plan B: real backend with a canned LLM
    fake_llm.py              the canned LLM (used by the two scripts above)
  expected/                  what you should see: graph JSON, questions, publish payload, backup diagram PNGs
```

## Setup (10 minutes, do it the day before)
1. `cp demo/.env.demo.example .env`, fill in the three `AZURE_OPENAI_*` values and set the passwords. The file may contain only known keys; the UI and API both fail on start without the three Azure values.
2. `uv venv .venv --python 3.12 && uv pip install -r requirements.txt`
3. Terminal 1: `uvicorn src.api.main:app --port 8000`. Terminal 2: `streamlit run app.py`.
4. `python demo/scripts/validate_demo.py` must end with `ALL CHECKS PASSED` (it uses the canned LLM, so it checks the pack, not Azure).
5. Do one full rehearsal with the real LLM and compare with `expected/`.

## Pre-flight (15 minutes before)
- [ ] Demo machine has internet: the graph diagrams load Mermaid and viz.js from cdn.jsdelivr.net. If they show as raw text, use `expected/lineage_*.png`.
- [ ] Sign-in works for all four users. Note the Streamlit banners use pale text on pale backgrounds; check legibility on the projector.
- [ ] Do not refresh the browser during the demo: the Dashboard lists no runs (see Known issues), so a refresh loses the active run.
- [ ] Azure OpenAI answers within a few seconds. If not, switch to Plan B (below).

## 20-minute script
Sign in as **Marcus (SME)** unless stated. Expected values are from `expected/`.

| # | Do | Say / show | Expect |
|---|---|---|---|
| 1 | **New Run**, select all 10 files in `artifacts/`, Start Run | "Four technology types, one upload." | 14 nodes, 13 edges, 7 open gaps, status waiting for SME |
| 2 | **Lineage Graph** | Boundaries `adls→sql`, `sql→databricks`, `databricks→dlt`. Red nodes/edges = gaps. | Matches `expected/lineage_before_sme.png` |
| 3 | **Table & Column Lineage Approval** tab, expand `orders_clean ← orders_raw` | Column mappings, e.g. `net_amount`. | 8 mappings, 85% |
| 4 | Expand `orders_adjusted ← revenue_adjustments` | "Parsers found nothing in this config-driven notebook, so the LLM did. 60%, and it's flagged." | 60%, evidence "LLM fallback" |
| 5 | **Gaps & Questions** | Three kinds of question, each addressed to the right person. | 3 missing source, 2 missing target, 2 low confidence |
| 6 | **Submit Responses**: answer 3 by typing, 2 by uploading `.docx`/`.xlsx`, 2 as you like | Match by the table name in quotes (order varies). Files are in `sme_answers/`. | "Responses submitted" |
| 7 | **Lineage Graph** again | ADLS sources, Power BI and SSRS added; the `dbo.customers` answer reused the existing node. | 18 nodes, 18 edges, 0 gaps. Matches `expected/lineage_after_sme.png` |
| 8 | **The reveal** (see story) | Two pipelines from `dbo.orders`; different definitions. | |
| 9 | Sign in as **Chloe (Analyst)**, approve one edge; then **Priya (Data Owner)**, approve an LLM edge, reject one with a reason | Analyst can review but cannot sign off; Data Owner decision is then read-only. | Publish still disabled |
| 10 | Bulk-approve the rest: `python demo/scripts/demo_driver.py approve --exclude-evidence` (as Data Owner) | **Say it is a demo shortcut** for the dull edges. | |
| 11 | **Publish** (dry_run) then **Audit Trail** | Payload with column mappings; every step, who and when. | Publish succeeds |

To script steps 1, 6 and 10 instead of clicking, use `demo_driver.py start | questions | answer [--only A1 A2] [--from-files] | approve | status | publish`. `start` and `answer` work on the same run the UI would use only if you keep the UI in the same session; simplest is to do all of one run in the UI or all in the driver.

## Plan B: Azure OpenAI is down or slow
Run `python demo/scripts/run_offline_api.py` instead of uvicorn. Same real backend; only LLM calls are canned (fallback parsing of the finance notebook, question wording, answer extraction). **Tell the audience**; it will not generalise beyond this pack.

## Honest Q&A prep
- *Does the LLM see our code?* Yes, for artifacts the deterministic parsers cannot handle, and SME reply text. There is no redaction step today.
- *Is it auditable?* Every step is logged with a timestamp, and review decisions carry reviewer and reason. The audit trail is held in memory (LangGraph MemorySaver), so a restart loses it; production needs a persistent store.
- *Who can sign off?* Only a Data Owner; enforced in UI and API. Credentials come from an environment variable, not an identity provider.
- *Do the emails go out?* Only with Azure Communication Services configured. Without it the questions show "skipped" and you answer in the UI.
- *How accurate is it?* Parser edges are deterministic (85-90%); LLM edges start at 60% and need a human. The pack's numbers are for a controlled example, not a benchmark.

## Known issues found while building this pack (verified unless noted)
1. **Dashboard shows no runs**: `GET /runs` returns `[]` even when runs exist. Don't refresh the browser.
2. **Stitching depends on upload order**: an ADF dataset links to its Databricks/SQL table only if the dataset file is processed before the pipeline file that references it. The artifacts are numbered for this; keep the order.
3. **SME questions never flip to answered**, so "questions pending" still shows 7 after the run is complete.
4. **SME-added edges have no system boundary label** (only edges from the initial stitch do).
5. **Approving an edge overwrites its confidence with 95%**, hiding the original parser/LLM score.
6. **Only ADLS counts as a source system**: a SQL source with no upstream gets a "where does this come from?" question. The pack avoids this by giving the SQL tables an ADLS feed.
7. Diagrams need internet (CDN); `.env` unknown keys and missing Azure values crash both processes; LangGraph prints harmless "Deserializing unregistered type" warnings.
8. *From code reading, not run:* the answer handler for an isolated sub-graph links the SME's source to every orphan node in the graph, so phrase such answers carefully.

## Regenerating
`python demo/scripts/_build_artifacts.py` and `_build_sme_files.py` rebuild the inputs; `validate_demo.py --snapshot` refreshes `expected/` (the PNGs are rendered from the `.mmd` files with `mmdc`).
