from __future__ import annotations

import asyncio
import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command

from src.agents.lineage_agent import build_graph
from src.api.models import (
    RunSummary, RunListItem,
    LineageGraphResponse, NodeResponse, EdgeResponse,
    GapListResponse, GapResponse,
    QuestionListResponse, QuestionResponse,
    SmeResponsePayload, EmailWebhookPayload,
    ConfidenceSummaryResponse,
    VisualsResponse,
    PublishRequest, PublishResponse,
    HealthResponse,
)
from src.catalog.publisher import get_publisher
from src.scoring.confidence import score_graph, graph_confidence_summary
from src.visualization.diagram_generator import (
    generate_mermaid, generate_graphviz_dot, generate_summary_html,
)
from src.sme.email_handler import parse_email_responses
from config.settings import settings

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Data Lineage Stitching Agent",
    version="0.2.0",
    description="AI-powered cross-system data lineage extraction, stitching, and catalog publishing.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangGraph agent — lazy-initialised so tests can patch before first use.
# Swap MemorySaver for a CosmosDB checkpointer in production.
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = build_graph()
    return _agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _require_state(run_id: str) -> dict:
    """Fetch the LangGraph state for a run or raise 404."""
    state = _get_agent().get_state(_config(run_id))
    if not state or not state.values:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return state.values


def _require_status(values: dict, *allowed: str, run_id: str = "") -> None:
    """Raise 409 if the run is not in one of the allowed statuses."""
    status = values.get("status", "")
    if status not in allowed:
        raise HTTPException(
            status_code=409,
            detail=f"Run '{run_id}' has status '{status}'. Expected one of: {list(allowed)}",
        )


def _run_summary(run_id: str, values: dict) -> RunSummary:
    graph = values.get("lineage_graph")
    gaps = values.get("gaps", [])
    return RunSummary(
        run_id=run_id,
        status=values.get("status", "unknown"),
        iteration=values.get("iteration", 0),
        node_count=len(graph.nodes) if graph else 0,
        edge_count=len(graph.edges) if graph else 0,
        gaps_open=sum(1 for g in gaps if not g.resolved),
        gaps_total=len(gaps),
        sme_questions_pending=sum(
            1 for q in values.get("sme_questions", []) if not q.get("answered")
        ),
        catalog_published=values.get("catalog_published", False),
        target_catalog=values.get("target_catalog", "dry_run"),
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Liveness check. Also reports whether catalog credentials are configured."""
    from src.catalog.publisher import AlationPublisher, CollibraPublisher
    return HealthResponse(
        status="ok",
        version="0.2.0",
        catalog_config={
            "alation_ready":  not AlationPublisher().validate_config(),
            "collibra_ready": not CollibraPublisher().validate_config(),
        },
    )


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------

@app.post("/runs", response_model=RunSummary, status_code=201, tags=["runs"])
async def start_run(
    files: list[UploadFile] = File(...),
    artifact_types: str = Form(default=""),
    catalog: str = Form(default="dry_run"),
):
    """
    Start a new lineage run by uploading one or more artifact files.

    Supported artifact types: `.sql`, `.ipynb`, `.py` (DLT), `.json` (ADF).
    Use `artifact_types` as a comma-separated list matching file order to
    hint the parser (e.g. `sql,notebook,dlt,adf`).

    The run is executed asynchronously. If gaps are found, the agent pauses
    and waits for SME responses; the status will be `"emails_sent"`.
    Poll `GET /runs/{run_id}` to track progress.
    """
    run_id = str(uuid.uuid4())
    type_hints = [t.strip() for t in artifact_types.split(",")] if artifact_types else []

    artifacts = []
    for i, upload in enumerate(files):
        content = await upload.read()
        artifacts.append({
            "filename":      upload.filename,
            "content":       content.decode("utf-8", errors="replace"),
            "artifact_type": type_hints[i] if i < len(type_hints) else None,
        })

    initial_state = {
        "run_id":          run_id,
        "artifacts":       artifacts,
        "fragments":       [],
        "lineage_graph":   None,
        "gaps":            [],
        "sme_questions":   [],
        "sme_responses":   [],
        "iteration":       0,
        "confidence_scores": {},
        "visuals":         None,
        "audit_trail":     [],
        "target_catalog":  catalog,
        "status":          "started",
        "catalog_published": False,
    }

    # Run agent in a thread to avoid blocking the event loop
    await asyncio.to_thread(_get_agent().invoke, initial_state, _config(run_id))

    return _run_summary(run_id, _require_state(run_id))


@app.get("/runs", response_model=list[RunListItem], tags=["runs"])
def list_runs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
):
    """
    List all runs known to the in-memory checkpointer.
    Use `status` to filter (e.g. `emails_sent`, `published`).
    """
    items = []
    for state_snapshot in _get_agent().get_state_history({"configurable": {"thread_id": "*"}}):
        values = state_snapshot.values
        run_id = state_snapshot.config.get("configurable", {}).get("thread_id", "")
        if not run_id or not values:
            continue
        if status and values.get("status") != status:
            continue
        graph = values.get("lineage_graph")
        gaps = values.get("gaps", [])
        items.append(RunListItem(
            run_id=run_id,
            status=values.get("status", "unknown"),
            node_count=len(graph.nodes) if graph else 0,
            edge_count=len(graph.edges) if graph else 0,
            gaps_open=sum(1 for g in gaps if not g.resolved),
        ))
        if len(items) >= limit:
            break
    return items


@app.get("/runs/{run_id}", response_model=RunSummary, tags=["runs"])
def get_run(run_id: str):
    """Get the current summary of a lineage run."""
    return _run_summary(run_id, _require_state(run_id))


# ---------------------------------------------------------------------------
# Lineage graph
# ---------------------------------------------------------------------------

@app.get("/runs/{run_id}/lineage", response_model=LineageGraphResponse, tags=["lineage"])
def get_lineage(run_id: str):
    """Return the full stitched lineage graph (nodes and edges)."""
    values = _require_state(run_id)
    graph = values.get("lineage_graph")
    if not graph:
        raise HTTPException(status_code=404, detail="Lineage graph not yet built for this run")

    nodes = [
        NodeResponse(
            id=n.id, name=n.name, system=n.system.value,
            full_path=n.full_path, description=n.description, owner=n.owner,
        )
        for n in graph.nodes.values()
    ]
    edges = [
        EdgeResponse(
            id=e.id, source_id=e.source_id, target_id=e.target_id,
            transformation=e.transformation, system_boundary=e.system_boundary,
            confidence=e.confidence, confidence_level=e.confidence_level.value,
            evidence=e.evidence, sme_approved_by=e.sme_approved_by,
        )
        for e in graph.edges
    ]
    return LineageGraphResponse(
        run_id=run_id, node_count=len(nodes), edge_count=len(edges),
        nodes=nodes, edges=edges,
    )


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------

@app.get("/runs/{run_id}/gaps", response_model=GapListResponse, tags=["gaps"])
def get_gaps(
    run_id: str,
    resolved: Optional[bool] = Query(None, description="Filter: true=resolved, false=open"),
):
    """Return gaps detected in the lineage graph for this run."""
    values = _require_state(run_id)
    all_gaps = values.get("gaps", [])

    filtered = all_gaps
    if resolved is not None:
        filtered = [g for g in all_gaps if g.resolved == resolved]

    gap_responses = [
        GapResponse(
            id=g.id, gap_type=g.gap_type, description=g.description,
            node_id=g.node_id, edge_id=g.edge_id,
            sme_email=g.sme_email, resolved=g.resolved,
        )
        for g in filtered
    ]
    return GapListResponse(
        run_id=run_id,
        total=len(all_gaps),
        open=sum(1 for g in all_gaps if not g.resolved),
        resolved=sum(1 for g in all_gaps if g.resolved),
        gaps=gap_responses,
    )


# ---------------------------------------------------------------------------
# SME questions
# ---------------------------------------------------------------------------

@app.get("/runs/{run_id}/questions", response_model=QuestionListResponse, tags=["sme"])
def get_questions(
    run_id: str,
    answered: Optional[bool] = Query(None, description="Filter: true=answered, false=pending"),
):
    """Return SME questions generated for gaps in this run."""
    values = _require_state(run_id)
    all_questions = values.get("sme_questions", [])

    filtered = all_questions
    if answered is not None:
        filtered = [q for q in all_questions if q.get("answered", False) == answered]

    q_responses = [
        QuestionResponse(
            gap_id=q.get("gap_id", ""),
            gap_type=q.get("gap_type", ""),
            question=q.get("question", ""),
            sme_email=q.get("sme_email"),
            answered=q.get("answered", False),
            send_status=q.get("send_status"),
        )
        for q in filtered
    ]
    return QuestionListResponse(
        run_id=run_id,
        total=len(all_questions),
        pending=sum(1 for q in all_questions if not q.get("answered", False)),
        questions=q_responses,
    )


# ---------------------------------------------------------------------------
# SME responses — resume the paused agent
# ---------------------------------------------------------------------------

@app.post("/runs/{run_id}/responses", response_model=RunSummary, tags=["sme"])
async def submit_sme_responses(run_id: str, payload: SmeResponsePayload):
    """
    Submit SME email responses to resume a paused run.

    The run must have status `emails_sent` (i.e. waiting for SME input).
    Pass the list of parsed reply dicts produced by `parse_email_responses()`,
    or structured dicts with at least `gap_id` and `raw_answer`.
    """
    values = _require_state(run_id)
    _require_status(values, "emails_sent", run_id=run_id)

    await asyncio.to_thread(
        _get_agent().invoke, Command(resume=payload.responses), _config(run_id)
    )
    return _run_summary(run_id, _require_state(run_id))


@app.post("/webhooks/email-reply", status_code=202, tags=["sme"])
async def receive_email_reply(payload: EmailWebhookPayload):
    """
    Webhook endpoint for inbound SME email replies.

    Integrates with Azure Communication Services Event Grid or SendGrid
    inbound parse. Extracts the run_id from the email subject/body, parses
    structured [REF:gap_id] answers, and resumes the matching run.

    Returns 202 Accepted immediately — processing is best-effort.
    """
    raw = [{
        "from":    payload.sender,
        "subject": payload.subject,
        "body":    payload.body_text(),
    }]
    parsed = parse_email_responses(raw)
    if not parsed:
        return {"accepted": True, "matched_run": None}

    run_id = parsed[0].get("run_id")
    if not run_id:
        return {"accepted": True, "matched_run": None, "reason": "could not extract run_id"}

    try:
        values = _require_state(run_id)
        if values.get("status") == "emails_sent":
            await asyncio.to_thread(
                _get_agent().invoke, Command(resume=parsed), _config(run_id)
            )
            return {"accepted": True, "matched_run": run_id, "responses_parsed": len(parsed)}
    except HTTPException:
        pass

    return {"accepted": True, "matched_run": run_id, "reason": "run not in awaiting state"}


# ---------------------------------------------------------------------------
# Visuals
# ---------------------------------------------------------------------------

@app.get("/runs/{run_id}/visuals", response_model=VisualsResponse, tags=["lineage"])
def get_visuals(run_id: str):
    """
    Return generated diagrams (Mermaid and Graphviz DOT) for a run.
    If the agent has not yet run the visualization step, regenerates on demand.
    """
    values = _require_state(run_id)
    stored = values.get("visuals")
    graph = values.get("lineage_graph")

    if stored:
        mermaid  = stored.get("mermaid", "")
        graphviz = stored.get("graphviz", "")
    elif graph:
        gaps = values.get("gaps", [])
        mermaid  = generate_mermaid(graph, gaps=gaps)
        graphviz = generate_graphviz_dot(graph, gaps=gaps)
    else:
        raise HTTPException(status_code=404, detail="No lineage graph available yet")

    # Build summary HTML on demand
    summary_html = None
    if graph:
        scores = values.get("confidence_scores", {})
        summary = graph_confidence_summary(graph, scores)
        gaps = values.get("gaps", [])
        summary_html = generate_summary_html(graph, summary, gaps=gaps)

    return VisualsResponse(
        run_id=run_id,
        mermaid=mermaid,
        graphviz=graphviz,
        summary_html=summary_html,
    )


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

@app.get("/runs/{run_id}/confidence", response_model=ConfidenceSummaryResponse, tags=["lineage"])
def get_confidence(run_id: str):
    """Return confidence statistics for the lineage graph."""
    values = _require_state(run_id)
    graph = values.get("lineage_graph")
    if not graph:
        raise HTTPException(status_code=404, detail="Lineage graph not yet built")

    scores = values.get("confidence_scores") or score_graph(graph)
    summary = graph_confidence_summary(graph, scores)
    return ConfidenceSummaryResponse(run_id=run_id, **summary)


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

@app.get("/runs/{run_id}/audit", tags=["runs"])
def get_audit(run_id: str):
    """Return the full ordered audit trail for a run."""
    values = _require_state(run_id)
    return {"run_id": run_id, "audit_trail": values.get("audit_trail", [])}


# ---------------------------------------------------------------------------
# Catalog publishing
# ---------------------------------------------------------------------------

@app.get("/runs/{run_id}/publish-preview", response_model=PublishResponse, tags=["catalog"])
def publish_preview(
    run_id: str,
    catalog: str = Query("alation", description="Target catalog format: alation | collibra"),
):
    """
    Dry-run preview: serialise the lineage graph into the catalog's API payload
    without making any HTTP calls. Use this to inspect the payload before
    configuring real catalog credentials.
    """
    values = _require_state(run_id)
    graph = values.get("lineage_graph")
    if not graph:
        raise HTTPException(status_code=404, detail="No lineage graph available for this run")

    from src.catalog.publisher import DryRunPublisher
    publisher = DryRunPublisher(target_catalog=catalog)
    result = publisher.publish(graph, run_id)

    return _publish_result_to_response(run_id, result)


@app.post("/runs/{run_id}/publish", response_model=PublishResponse, tags=["catalog"])
async def publish_run(run_id: str, body: PublishRequest = PublishRequest()):
    """
    Publish the finalised lineage graph to the target catalog.

    The run must be in status `finalized`, `visuals_ready`, or `published`.
    Set `catalog` to `"alation"`, `"collibra"`, or `"dry_run"` (default).

    Returns the publish result including any errors and catalog URLs.
    """
    values = _require_state(run_id)
    _require_status(
        values, "finalized", "visuals_ready", "published", "publish_failed",
        run_id=run_id,
    )

    graph = values.get("lineage_graph")
    if not graph:
        raise HTTPException(status_code=404, detail="No lineage graph to publish")

    publisher = get_publisher(body.catalog)
    result = await asyncio.to_thread(publisher.publish, graph, run_id)

    return _publish_result_to_response(run_id, result)


def _publish_result_to_response(run_id: str, result) -> PublishResponse:
    payload_summary = None
    if result.payload:
        p = result.payload
        if "lineage" in p:
            payload_summary = {"format": "alation", "lineage_items": len(p["lineage"])}
        elif "assets" in p:
            payload_summary = {
                "format": "collibra",
                "assets": len(p["assets"]),
                "relations": len(p.get("relations", [])),
            }
    return PublishResponse(
        run_id=run_id,
        catalog=result.catalog,
        status=result.status,
        success=result.success,
        nodes_published=result.nodes_published,
        edges_published=result.edges_published,
        errors=result.errors,
        catalog_urls=result.catalog_urls,
        payload_summary=payload_summary,
    )
