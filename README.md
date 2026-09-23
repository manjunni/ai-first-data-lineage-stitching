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
│   ├── file_handlers.py       — File format extraction: .txt, .docx, .xlsx, .pdf, .json (Phase 1)
│   ├── question_generator.py  — Template per gap type + parallel LLM enrichment
│   ├── email_handler.py       — ACS grouped send, [REF:gap-id] reply parsing + generic text parser (Phase 2)
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
tests/                      — 382 tests, all passing (28 file handlers + 8 response extraction + 346 existing)
```

---

## Running the Project

### 1. Environment

```bash
cp .env.example .env
# Fill in Azure OpenAI, ACS email, Alation/Collibra credentials
```

Configure role-based human approval users with `LINEAGE_USERS_JSON`. Passwords
are read from the environment and are not stored in lineage data:

```dotenv
LINEAGE_USERS_JSON={"owner@company.com":{"password":"change-me","role":"Data Owner"},"steward@company.com":{"password":"change-me","role":"Domain Steward"},"sme@company.com":{"password":"change-me","role":"SME"},"analyst@company.com":{"password":"change-me","role":"Analyst"}}
```

The Streamlit app requires login before showing the agent. Missing-lineage edge
approvals use the hierarchy **Data Owner > Domain Steward > SME > Analyst**.
Every authenticated role may make an initial decision; a higher-ranked role may
override a lower-ranked decision, but a lower-ranked role cannot override a
higher-ranked one.

Catalog publishing requires every lineage edge to be approved by a **Data
Owner**. SME, Domain Steward, or Analyst approval alone is not sufficient. The
Streamlit Publish action is disabled until this condition is met, and the API
enforces the same authorization rule for direct publish requests. Payload
previews remain available before approval. Once a Data Owner approves an edge,
that decision becomes read-only and cannot be changed.

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
streamlit run app.py
# Or: API_BASE_URL=http://localhost:8000 streamlit run app.py
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
- Publishing is enabled only after every lineage edge has an approved review from a `Data Owner`. This is enforced in both the Streamlit UI and the publish API; Analyst, SME, and Domain Steward approvals do not authorize publication.
- Alation lineage items, Collibra relations, and dry-run publish previews include a
  structured `column_mappings` list for each table edge. Each mapping preserves
  source and target columns, derived-column transformation logic, independent
  confidence and evidence, and SME approval metadata. Table-only edges receive
  an empty list, while rejected edges are excluded from all catalog payloads.
  Edge-level SME approval is inherited by its column mappings in API and
  catalog responses; mapping-level approval metadata takes precedence when
  present.

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

---

## Enhancement: Lineage Gap Fixing Using Multi-Format Input (Issue #35)

### Overview

The agent now accepts gap-fixing input from SMEs in multiple file formats (.txt, .docx, .xlsx, .pdf, .json) in addition to email responses. This enables flexible workflows where SMEs can submit gap fixes as structured documents, spreadsheets, or plain text files via the `POST /runs/{id}/responses` endpoint.

**Status**: ✅ Complete (382 tests passing, zero regressions)

### Supported File Formats

| Format | Handler | Status | Example |
|--------|---------|--------|---------|
| .txt | PlainTextHandler | ✅ Native | `[REF:gap-001]: Source is sales_db.customer` |
| .docx | DocxHandler | ✅ Optional | Word documents (paragraphs + tables) |
| .xlsx | ExcelHandler | ✅ Optional | Excel spreadsheets (all sheets + rows) |
| .pdf | PdfHandler | ✅ Optional | PDF documents (text extraction) |
| .json | JsonHandler | ✅ Native | Structured JSON data |

**Optional Dependencies** (graceful degradation):
- `python-docx>=0.8.10` — for .docx support
- `openpyxl>=3.0.0` — for .xlsx support
- `PyPDF2>=3.0.0` — for .pdf support

If optional libraries are not installed, the agent provides clear error messages with installation instructions. Core functionality (.txt, .json) is always available.

### Architecture

The multi-format enhancement consists of two phases:

#### Phase 1: File Format Handlers (`src/sme/file_handlers.py`)

Extracts plain text from multiple file formats using a handler registry pattern:

```python
from src.sme.file_handlers import extract_text_from_file, get_supported_formats

# Extract text from any supported format
text = extract_text_from_file("response.xlsx", file_bytes)

# Get available formats at runtime
formats = get_supported_formats()  # ['.txt', '.docx', '.xlsx', '.pdf', '.json']
```

**Handler Classes:**
- `PlainTextHandler` — reads .txt files (UTF-8 + Latin-1 fallback)
- `DocxHandler` — parses .docx using python-docx (paragraphs + tables)
- `ExcelHandler` — parses .xlsx using openpyxl (all sheets + rows)
- `PdfHandler` — extracts text from .pdf using PyPDF2
- `JsonHandler` — validates and formats .json files
- `BaseFileHandler` — abstract base for easy extension

#### Phase 2: Generic Response Parsing (`src/sme/email_handler.py`)

Introduced `parse_responses_from_text()` — a generic text parser that works with any text source (email bodies, extracted files, direct input):

```python
from src.sme.email_handler import parse_responses_from_text

# Parse gap responses from any text source
responses = parse_responses_from_text(
    text="[REF:gap-001]: Sales data from ADLS\n[REF:gap-002]: Flows to Databricks",
    sme_email="analyst@company.com",
    run_id="run-abc123"
)

# Returns:
# [
#   {"gap_id": "gap-001", "raw_answer": "Sales data from ADLS", ...},
#   {"gap_id": "gap-002", "raw_answer": "Flows to Databricks", ...}
# ]
```

**Key Features:**
- ✅ Extracts `[REF:gap-id]: answer` patterns from any text
- ✅ Supports flexible gap-id formats (UUIDs, simple IDs like "gap-001", custom IDs)
- ✅ Falls back to unstructured response if no `[REF:...]` markers found
- ✅ Auto-extracts `run_id` from text if not provided (searches for `[Lineage Agent] Run {id}`)
- ✅ Returns standard response dict format (compatible with existing response processor)

The existing `parse_email_responses()` now uses this generic parser internally, maintaining 100% backward compatibility.

### Usage Examples

#### Example 1: Structured .txt File

**File: response.txt**
```
[REF:gap-001]: The source table is sales_db.customer_transactions in SQL Server

[REF:gap-002]: Data flows to Databricks in the bronze layer as adls://lakehouse/bronze/customer

[REF:gap-003]: Transformation happens in the silver layer catalog.schema.silver_customer
```

**Python Code:**
```python
# Read file
with open("response.txt", "rb") as f:
    file_bytes = f.read()

# Extract and parse
from src.sme.file_handlers import extract_text_from_file
from src.sme.email_handler import parse_responses_from_text

text = extract_text_from_file("response.txt", file_bytes)
responses = parse_responses_from_text(text, sme_email="sme@company.com", run_id="run-123")

# Pass to agent
from src.sme.response_processor import apply_responses_to_graph
updated_gaps, updated_graph = apply_responses_to_graph(responses, gaps, graph)
```

#### Example 2: Structured Excel File

**File: responses.xlsx** (Sheet: "Gap Fixes")
```
| Gap ID | Answer |
|--------|--------|
| gap-001 | sales_db.customer_transactions |
| gap-002 | adls://lakehouse/bronze/customer |
| gap-003 | catalog.schema.silver_customer |
```

Excel handler converts to text and extracts in the same way as .txt files.

#### Example 3: Narrative .docx File

**File: response.docx**
```
Subject: Lineage Gap Responses

Gap 1 [REF:gap-001]: 
The customer data originates from our SQL Server OLTP system in sales_db.customer_transactions.

Gap 2 [REF:gap-002]: 
This data is extracted daily to our ADLS raw zone at adls://lakehouse/raw/customer and 
then processed into the bronze layer at adls://lakehouse/bronze/customer.

Gap 3 [REF:gap-003]: 
In Databricks, we apply transformations and load into the silver layer at 
catalog.schema.silver_customer for reporting.
```

Word handler extracts paragraphs and parses [REF:...] markers.

### Response Dict Format

All sources (email, .txt, .xlsx, .docx, .pdf, .json) produce identical response dicts:

```python
{
    "gap_id": "gap-001",                              # From [REF:gap-001]
    "sme_email": "analyst@company.com",              # Provided by caller
    "run_id": "run-abc123",                          # Auto-extracted or provided
    "raw_answer": "sales_db.customer_transactions",  # Extracted answer text
    "raw_text": "[REF:gap-001]: ..."                 # Full text for LLM context
}
```

This format is **identical to email responses**, so the existing response processor requires **zero changes**.

### Testing

**382 tests passing** (28 file handlers + 8 response extraction + 346 existing):

```bash
# Test file handlers (Phase 1)
pytest tests/test_sme_file_handlers.py -v

# Test response extraction (Phase 2)
pytest tests/test_sme_flow.py::TestParseResponsesFromText -v

# All tests
pytest tests/ -q
```

**Test Coverage:**
- File format detection and extraction
- Error handling (unsupported formats, corrupted files, missing libraries)
- Response parsing ([REF:...] patterns, multiline answers, run_id extraction)
- Edge cases (UTF-8/Latin-1 encoding, multi-sheet Excel, multi-line documents)
- Backward compatibility (existing email flow)

### Backward Compatibility

✅ **100% Backward Compatible:**
- Email flow unchanged — `parse_email_responses()` still works exactly as before
- Response processor unchanged — no modifications needed to graph update logic
- All 346 existing tests pass without modification
- No breaking changes to API or agent workflow

### Extensibility

Adding a new file format is straightforward:

```python
# Create handler class
class MyFormatHandler(BaseFileHandler):
    def can_handle(self, filename: str) -> bool:
        return filename.lower().endswith('.myformat')
    
    def extract_text(self, file_content: bytes) -> str:
        # Parse and return text
        data = parse_myformat(file_content)
        return str(data)

# Register in _HANDLERS list in file_handlers.py
_HANDLERS = [
    PlainTextHandler(),
    DocxHandler(),
    ExcelHandler(),
    PdfHandler(),
    JsonHandler(),
    MyFormatHandler(),  # NEW
]

# Done! `get_supported_formats()` now includes '.myformat'
```

### Error Handling

Clear, actionable error messages:

```python
# Unsupported file type
try:
    text = extract_text_from_file("file.unknown", bytes)
except FileHandlerError as e:
    print(e)
    # "Unsupported file type: .unknown. Supported types: .txt, .docx, .xlsx, .pdf, .json"

# Missing optional dependency
try:
    text = extract_text_from_file("file.docx", bytes)
except FileHandlerError as e:
    print(e)
    # "python-docx not installed. Install with: pip install python-docx"
```

### Implementation Details

**Module Structure:**
```
src/sme/
├── file_handlers.py         — Phase 1: File extraction (195 lines)
├── email_handler.py         — Phase 2: Response parsing (refactored, +45 lines)
└── response_processor.py    — Existing: Graph updates (unchanged)

tests/
├── test_sme_file_handlers.py — Phase 1: 28 tests
├── test_sme_flow.py          — Phase 2: +8 tests
├── test_api.py               — Phase 3: +8 tests (TestSubmitResponsesFiles)
└── test_*                    — Existing: 338 tests (all passing)
```

**Dependencies:**
- Phase 1 handlers use standard extraction libraries (python-docx, openpyxl, PyPDF2)
- Phase 2 parser uses existing regex-based extraction (no new LLM calls)
- Response processor integration is seamless (same dict format)

### Phase 3: API Integration ✅ COMPLETE

**Endpoint: `POST /runs/{run_id}/responses/files`**

The new file upload endpoint accepts gap fixes in multiple formats with a single unified interface:

```python
# Upload text directly
POST /runs/{run_id}/responses/files
Content-Type: application/x-www-form-urlencoded

text=[REF:gap-001]: Data comes from sales_db
sme_email=john.smith@corp.com

# Upload files (single or multiple)
POST /runs/{run_id}/responses/files
Content-Type: multipart/form-data

files: response.xlsx (binary)
files: response.docx (binary)
sme_email: jane.doe@corp.com

# Combine files + text
POST /runs/{run_id}/responses/files
Content-Type: multipart/form-data

files: sources.txt (binary)
text: [REF:gap-002]: Additional info here
sme_email: alex.wilson@corp.com
```

**Implementation Details:**
- `src/api/models.py` — Added `SmeFileUploadPayload` model (text, sme_email, run_id fields)
- `src/api/main.py` — Added new endpoint that:
  - Accepts optional multipart file uploads (up to 10 MB each)
  - Accepts optional direct text input
  - Routes file uploads through `extract_text_from_file()` → `parse_responses_from_text()`
  - Merges file + text responses into single list before resuming agent
  - Validates run is in "emails_sent" status (same guard as text-only endpoint)
  - Returns detailed error messages for unsupported formats or corrupted files

**Error Handling:**
| Status | Scenario |
|--------|----------|
| 400 | No files + no text (both required) |
| 400 | File format not supported (e.g., .xyz, .bin) |
| 400 | Corrupted file (parsing fails after extraction) |
| 409 | Run not in "emails_sent" status |
| 404 | Run not found |
| 500 | Missing optional dependency (graceful with install instructions) |

**Tests:**
- 8 new tests in `tests/test_api.py::TestSubmitResponsesFiles` covering:
  - Single file upload (`.txt`)
  - Multiple file uploads (parallel processing)
  - Direct text submission
  - Combined files + text
  - Validation: no input, unsupported format, wrong run status
  - Response merging and agent resumption

**Backward Compatibility:**
- Existing `POST /runs/{run_id}/responses` endpoint unchanged (text-only JSON)
- New endpoint is additive; existing workflows unaffected
- Both endpoints share response processor logic

**Response Format (Unified):**
```python
[
  {
    "gap_id": "gap-001",
    "sme_email": "sme@corp.com",
    "run_id": "run-uuid",
    "raw_answer": "Data comes from sales_db",
    "raw_text": "[REF:gap-001]: Data comes from sales_db",
    "source": "file"  # or "text" or "email"
  },
  ...
]
```

### Phase 4 Implementation Summary

The Streamlit UI now supports the complete multi-format SME response workflow:

- The **Submit Responses** page accepts multiple `.txt`, `.docx`, `.xlsx`, `.pdf`, and `.json` files.
- SMEs can add supplemental response text and reuse the responder identity for submitted metadata.
- Uploaded content is previewed as parsed gap IDs and answers before submission.
- File type and parsing errors are displayed without silently submitting invalid input.
- Valid responses are submitted through `POST /runs/{run_id}/responses/files`.
- The existing inline question-and-answer form remains available for direct responses.
- Each pending question now has its own document uploader above its text input, so the responder can answer by typing, uploading a document, or combining both.
- The UI API client now provides multipart response submission support.
- The repository-root `app.py` is the documented Streamlit entrypoint.

The Phase 4 implementation adds focused UI client coverage; the full test suite now passes **392 tests**.

### Future Enhancements (Phase 4+)

**Phase 4: Streamlit UI** ✅ Complete
- Multi-file uploader on the Submit Responses page for `.txt`, `.docx`, `.xlsx`, `.pdf`, and `.json`
- Optional supplemental text input alongside uploaded files
- Live preview of parsed gap IDs and answers before submission
- Format filtering and validation feedback, using the same backend file handlers
- Existing inline question-answer workflow remains available

**Phase 5: Documentation & Examples**
- Sample files (one per format) with [REF:...] patterns
- Field mapping templates for Excel/CSV
- Video walkthrough of file upload workflow

### Column-Level Lineage

The lineage graph supports optional column-level mappings attached to each
table-level edge. Each mapping includes:

- source and target column names
- derived-column transformation logic where available
- independent confidence score and confidence level
- extraction evidence and SME approval metadata

The SQL parser extracts direct and derived mappings with `sqlglot`. PySpark
notebook and script parsing also extracts common DataFrame mappings from
`select`, `withColumn`, and column alias expressions, including derived-column
transformation text. The structured LLM fallback can return mappings for
artifacts where deterministic extraction is unavailable. Existing artifacts
without mappings remain valid table-level lineage.

In the Streamlit **Lineage Graph** page, the **Tables and Datasets** tab lists
the available table and dataset nodes. The **Table & Column Lineage** tab
contains one table-formatted lineage section with columns for target table,
source table, system boundary, confidence, and mapping count. Each table row
is expandable; opening it reveals source-to-target mappings,
the system boundary (for example, `ADLS → Databricks`), transformation
expressions, evidence, and column-level confidence. Table and column
confidence are scored independently.

An expanded relationship renders a tree such as:

```text
customer_orders
├── customer_id ← customers_clean.customer_id
├── full_name ← CONCAT(first_name, last_name)
├── email ← customers_clean.email
└── amount ← orders_clean.amount
```

### Confidence Review

The **Lineage Graph** page provides review before lineage is approved or
published. The page shows the pending Data Owner approval count at the top.
Each edge can be expanded to inspect source and target paths, system boundary,
transformation, parser/LLM evidence, per-edge confidence, and column mappings.
The expanded relationship also displays approval status, reviewer identity, and
reviewer role, with approve/reject controls for authorized reviewers. Once a
Data Owner approves an edge, its approval and rejection controls are disabled
and the lineage becomes read-only. Rejected edges are excluded from catalog
publishing and are recorded in the audit trail. Aggregate average, minimum,
and maximum confidence metrics are not shown in the UI.

---

## Architecture Diagram
<img width="760" height="440" alt="image" src="https://github.com/user-attachments/assets/64670f4f-e157-4510-b0a2-c9fa8ec579dc" />
