# Data Lineage Stitching Agent

**Organisation:** Capgemini — Insights & Data  
**Programme Mission:** Create accurate, complete, and catalog-ready data lineage across systems using an AI agent supported by human expertise.  
**Key Stakeholders:** Ralf Teschner, Greg Solomon

---

## What This Project Is

A **human-in-the-loop AI agent** that extracts, stitches, and publishes cross-system data lineage for the Azure → Spark → Databricks → BI stack. It parses SQL scripts, Databricks notebooks, Delta Live Tables pipelines, and Azure Data Factory JSON definitions, stitches the fragments into a unified lineage graph, detects gaps, emails SMEs for input, scores confidence, generates diagrams, and publishes to Alation or Collibra.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph (stateful, interrupt/resume for SME wait) |
| LLM | Azure OpenAI GPT-4o (via LangChain) |
| SQL parsing | sqlglot (primary), LLM fallback |
| Notebook/DLT parsing | nbformat + regex / Python ast module |
| ADF parsing | JSON traversal |
| Graph operations | networkx |
| API | FastAPI + uvicorn |
| UI | Streamlit |
| Email | Azure Communication Services |
| Catalog | Alation REST API v1 / Collibra REST API v2 |
| Checkpointing | LangGraph MemorySaver (swap for Cosmos DB in prod) |
| Runtime | Python 3.12, Azure Container Apps |

---

## Repository Layout

```
src/
├── agents/
│   ├── lineage_agent.py    — LangGraph graph: all 11 nodes, edges, routing
│   └── state.py            — LineageState TypedDict (shared across all nodes)
├── parsers/
│   ├── base_parser.py      — BaseParser abstract interface
│   ├── sql_parser.py       — sqlglot AST → LineageFragment + LLM fallback
│   ├── notebook_parser.py  — nbformat + regex (spark.read/write) + LLM fallback
│   ├── dlt_parser.py       — Python ast (@dlt.table) + sqlglot (LIVE TABLE) + LLM fallback
│   └── adf_parser.py       — ADF pipeline/dataset JSON traversal + LLM fallback
├── graph/
│   ├── models.py           — Pydantic models: LineageNode, LineageEdge, LineageFragment, LineageGraph, Gap
│   ├── stitcher.py         — Two-pass merge: node dedup by full_path, edge dedup by (src,tgt), boundary annotation
│   └── gap_detector.py     — 7 gap types: missing_source/target, unknown_system, unresolved_dataset_ref, low_confidence_edge, contradictory_paths, isolated_subgraph
├── sme/
│   ├── question_generator.py  — Template per gap type + parallel LLM enrichment
│   ├── email_handler.py       — ACS grouped send, [REF:gap-id] reply parsing
│   └── response_processor.py — LLM structured extraction + per-gap-type graph update
├── scoring/
│   └── confidence.py       — Multi-signal scoring: evidence type + cross-system penalty + ADF/unknown penalties; graph_confidence_summary()
├── visualization/
│   └── diagram_generator.py — Mermaid (subgraphs, classDef, linkStyle) + Graphviz DOT (clusters, colours) + HTML summary
├── catalog/
│   └── publisher.py        — AlationPublisher (Integration API v1), CollibraPublisher (REST API v2), DryRunPublisher, PublishResult
├── api/
│   ├── main.py             — FastAPI: 14 routes, async agent calls, email webhook
│   └── models.py           — Pydantic request/response models
└── ui/
    ├── app.py              — Streamlit 8-page app (Dashboard → Publish)
    ├── api_client.py       — Typed requests wrapper for all FastAPI endpoints
    └── components.py       — Mermaid renderer, status badges, workflow timeline, legend
config/
└── settings.py             — Pydantic Settings (all env vars)
tests/                      — 343 tests, all passing
```

---

## Running the Project

### 1. Environment

```bash
cp .env.example .env
# Fill in Azure OpenAI, ACS email, Alation/Collibra credentials
```

### 2. Install dependencies

```bash
uv venv .venv --python 3.12
uv pip install -r requirements.txt
```

### 3. Start the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

### 4. Start the UI

```bash
streamlit run src/ui/app.py
# Or: API_BASE_URL=http://localhost:8000 streamlit run src/ui/app.py
```

### 5. Run tests

```bash
pytest tests/ -q
```

---

## Agent Workflow

```
ingest_artifacts → parse_artifacts → stitch_graph → detect_gaps
                                                          ↓
                                        [gaps?] ──yes──→ generate_sme_questions
                                                              → send_emails
                                                              → ⏸ INTERRUPT
                                                          ↑       (wait for SME)
                                        POST /runs/{id}/responses resumes here
                                                              → process_responses
                                                              → stitch_graph (loop)
                                        [no gaps / max iterations reached]
                                                          ↓
                                          score_and_finalize → generate_visuals
                                                                → publish_to_catalog
```

The agent pauses at `await_sme_responses` (LangGraph `interrupt`). The FastAPI `POST /runs/{id}/responses` endpoint or the email webhook resumes it with `Command(resume=responses)`.

---

## Key Design Decisions

### Parsers
- **sqlglot over regex for SQL** — handles dialect variants (Spark SQL, T-SQL) and multi-statement files reliably.
- **Python `ast` over regex for DLT** — correctly handles multi-line function bodies and complex decorator arguments.
- **`adf://` namespace** — ADF pipeline parsers use `adf://DatasetName` as node identifiers; dataset JSON parsers emit the same identifier, so the stitcher connects them automatically via `full_path` deduplication even though files are uploaded separately.
- **LLM as fallback, not primary** — every parser tries a deterministic path first; LLM activates only when that fails or yields no edges.

### Stitcher
- **Two-pass merge**: Pass 1 builds the canonical `full_path → node_id` registry and merges metadata (UNKNOWN system is upgraded when a richer fragment arrives). Pass 2 remaps edge endpoints and deduplicates by `(source_id, target_id)` — the higher-confidence edge wins.
- **Dangling edge removal**: edges whose endpoints aren't in the node registry are silently dropped rather than crashing at graph traversal time.

### Confidence scoring
Scores are assigned at finalization (not at parse time) so they reflect the full graph context:
- Base from evidence string: `sqlglot/ast` → 0.90, `regex/adf` → 0.85, `llm` → 0.60
- Penalties: cross-system boundary (−0.05), `adf://` placeholder (−0.15), UNKNOWN system (−0.10)
- SME-confirmed edges are locked at 0.95 (`ConfidenceLevel.CONFIRMED`), no penalties applied.

### SME email flow
- Questions are generated from **per-gap-type templates** (fast, no LLM cost); LLM enriches wording in parallel via `ThreadPoolExecutor`.
- Emails are **grouped by SME** — one email per person, not one per gap.
- Replies are parsed via **`[REF:gap-uuid]: answer` regex** — deterministic, no LLM needed for matching.
- Structured answer extraction (LLM) and graph update are in `response_processor.py`, with a handler per gap type.

### Catalog publishing
- `DryRunPublisher` serialises to the target format without HTTP calls — used as default and for preview.
- Collibra asset `external_id` is a **deterministic `uuid5`** from `node.full_path` — repeat runs update rather than duplicate assets.
- `validate_config()` on each publisher exposes missing env vars before a run starts.

---

## API Routes

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + catalog credential readiness |
| POST | `/runs` | Upload artifacts, start run |
| GET | `/runs` | List runs |
| GET | `/runs/{id}` | Run summary |
| GET | `/runs/{id}/lineage` | Full graph (nodes + edges) |
| GET | `/runs/{id}/gaps` | Gap list (filterable) |
| GET | `/runs/{id}/questions` | SME questions (filterable) |
| POST | `/runs/{id}/responses` | Submit SME replies, resume run |
| POST | `/webhooks/email-reply` | Inbound email webhook (ACS/SendGrid) |
| GET | `/runs/{id}/visuals` | Mermaid + Graphviz + HTML |
| GET | `/runs/{id}/confidence` | Confidence statistics |
| GET | `/runs/{id}/audit` | Audit trail |
| GET | `/runs/{id}/publish-preview` | Dry-run payload preview |
| POST | `/runs/{id}/publish` | Publish to catalog |

---

## Adding a New Parser

1. Create `src/parsers/my_parser.py` implementing `BaseParser`:
   ```python
   class MyParser(BaseParser):
       def can_parse(self, artifact: dict) -> bool: ...
       def parse(self, artifact: dict) -> LineageFragment: ...
   ```
2. Register it in `_PARSERS` in `src/agents/lineage_agent.py`.
3. Add tests in `tests/test_my_parser.py`.

The parser should use `_get_or_create_node`, `_infer_system`, and `_normalise_llm_node` from `src/parsers/sql_parser.py` to keep node construction consistent.

---

## Connecting Real Catalog APIs

Set the relevant env vars in `.env`:

**Alation:**
```
ALATION_BASE_URL=https://your-org.alationcloud.com
ALATION_API_TOKEN=your-token
```

**Collibra:**
```
COLLIBRA_BASE_URL=https://your-org.collibra.com
COLLIBRA_USERNAME=your-username
COLLIBRA_PASSWORD=your-password
COLLIBRA_DOMAIN_ID=your-domain-uuid
COLLIBRA_LINEAGE_TYPE_ID=00000000-0000-0000-0000-000000003115
```

Verify with `GET /health` — `catalog_config.alation_ready` and `catalog_config.collibra_ready` will be `true` once credentials are present.
