from datetime import datetime, timezone
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from src.agents.state import LineageState
from src.parsers.sql_parser import SqlParser
from src.parsers.notebook_parser import NotebookParser
from src.parsers.dlt_parser import DltParser
from src.parsers.adf_parser import AdfParser
from src.graph.stitcher import stitch_fragments
from src.graph.gap_detector import detect_gaps
from src.sme.question_generator import generate_questions
from src.sme.email_handler import send_sme_questions
from src.sme.response_processor import apply_responses_to_graph
from src.scoring.confidence import score_graph
from src.visualization.diagram_generator import generate_mermaid, generate_graphviz_dot
from src.catalog.publisher import get_publisher
from config.settings import settings

_PARSERS = [SqlParser(), NotebookParser(), DltParser(), AdfParser()]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def ingest_artifacts(state: LineageState) -> dict:
    _log = [{"step": "ingest", "count": len(state["artifacts"]), "ts": _now()}]
    return {
        "status": "ingested",
        "audit_trail": _log,
    }


def parse_artifacts(state: LineageState) -> dict:
    fragments = []
    audit = []
    for artifact in state["artifacts"]:
        parser = next((p for p in _PARSERS if p.can_parse(artifact)), None)
        if parser:
            fragment = parser.parse(artifact)
            fragments.append(fragment)
            audit.append({"step": "parse", "artifact": artifact["filename"], "parser": type(parser).__name__, "ts": _now()})
        else:
            audit.append({"step": "parse", "artifact": artifact["filename"], "parser": None, "warning": "no parser found", "ts": _now()})

    return {
        "fragments": fragments,
        "status": "parsed",
        "audit_trail": audit,
    }


def stitch_graph_node(state: LineageState) -> dict:
    graph = stitch_fragments(state["fragments"])
    # Score immediately so edge.confidence is consistent with the confidence tab
    # from the first render — avoids the edge table showing raw parser values
    # while the confidence tab shows re-scored (lower) values.
    score_graph(graph)
    return {
        "lineage_graph": graph,
        "status": "stitched",
        "audit_trail": [{"step": "stitch", "nodes": len(graph.nodes), "edges": len(graph.edges), "ts": _now()}],
    }


def detect_gaps_node(state: LineageState) -> dict:
    gaps = detect_gaps(state["lineage_graph"])
    # Carry forward resolved status: gaps with the same (type, node, edge) that
    # were already resolved in a prior iteration should stay resolved rather than
    # being reset to False by fresh Gap objects produced by detect_gaps().
    resolved_signatures = {
        (g.gap_type, g.node_id, g.edge_id)
        for g in state.get("gaps", [])
        if g.resolved
    }
    for gap in gaps:
        if (gap.gap_type, gap.node_id, gap.edge_id) in resolved_signatures:
            gap.resolved = True
    open_count = sum(1 for g in gaps if not g.resolved)
    return {
        "gaps": gaps,
        "status": "gaps_detected",
        "audit_trail": [{"step": "gap_detection", "gaps_found": len(gaps), "open": open_count, "ts": _now()}],
    }


def generate_sme_questions_node(state: LineageState) -> dict:
    existing = state.get("sme_questions", [])
    # Skip gaps already asked — gap IDs change each pass, so match by signature.
    asked_signatures = {
        (q.get("gap_type"), q.get("node_id"), q.get("edge_id"))
        for q in existing
    }
    open_gaps = [
        g for g in state["gaps"]
        if not g.resolved and (g.gap_type, g.node_id, g.edge_id) not in asked_signatures
    ]
    new_questions = generate_questions(open_gaps, state["lineage_graph"])
    return {
        "sme_questions": existing + new_questions,
        "status": "questions_generated",
        "audit_trail": [{"step": "question_gen", "questions": len(new_questions), "ts": _now()}],
    }


def send_emails_node(state: LineageState) -> dict:
    existing = state.get("sme_questions", [])
    unanswered = [q for q in existing if not q.get("answered")]
    sent = send_sme_questions(unanswered)
    # Update send_status in-place — do NOT append, to avoid duplicating questions.
    sent_by_gap_id = {q.get("gap_id"): q for q in sent}
    updated = [sent_by_gap_id.get(q.get("gap_id"), q) for q in existing]
    return {
        "sme_questions": updated,
        "status": "emails_sent",
        "audit_trail": [{"step": "send_email", "sent": len(sent), "ts": _now()}],
    }


def await_sme_responses_node(state: LineageState) -> dict:
    """
    Interrupt the graph here and wait for SME email replies.
    The FastAPI layer resumes the graph by calling:
        graph.invoke(Command(resume=responses), config)
    where responses is a list of already-parsed reply dicts (gap_id, raw_answer, …).
    Both the /responses API endpoint and the email webhook pre-parse via
    parse_email_responses() before resuming, so we must NOT parse again here.
    """
    responses = interrupt("Waiting for SME email responses")
    parsed = responses if isinstance(responses, list) else []
    iteration = state.get("iteration", 0) + 1
    return {
        "sme_responses": parsed,
        "iteration": iteration,
        "status": "responses_received",
        "audit_trail": [{"step": "sme_response", "responses": len(parsed), "iteration": iteration, "ts": _now()}],
    }


def process_responses_node(state: LineageState) -> dict:
    """
    Apply SME answers back into the lineage graph.
    Uses LLM to extract structured answers from free-text replies,
    then updates nodes/edges in the graph accordingly.
    """
    updated_gaps, updated_graph = apply_responses_to_graph(
        responses=state["sme_responses"],
        gaps=state["gaps"],
        graph=state["lineage_graph"],
    )
    resolved_count = sum(1 for g in updated_gaps if g.resolved)

    return {
        "gaps":          updated_gaps,
        "lineage_graph": updated_graph,
        "status":        "responses_processed",
        "audit_trail":   [{"step": "process_responses", "resolved": resolved_count, "ts": _now()}],
    }


def score_and_finalize_node(state: LineageState) -> dict:
    scores = score_graph(state["lineage_graph"])
    return {
        "confidence_scores": scores,
        "status": "finalized",
        "audit_trail": [{"step": "scoring", "edges_scored": len(scores), "ts": _now()}],
    }


def generate_visuals_node(state: LineageState) -> dict:
    graph = state["lineage_graph"]
    visuals = {
        "mermaid": generate_mermaid(graph),
        "graphviz": generate_graphviz_dot(graph),
    }
    return {
        "visuals": visuals,
        "status": "visuals_ready",
        "audit_trail": [{"step": "visualize", "ts": _now()}],
    }


def publish_to_catalog_node(state: LineageState) -> dict:
    catalog = state.get("target_catalog", "dry_run")
    publisher = get_publisher(catalog)
    result = publisher.publish(state["lineage_graph"], state["run_id"])
    return {
        "catalog_published": result.success,
        "status": "published" if result.success else "publish_failed",
        "audit_trail": [{"step": "publish", "catalog": catalog, "success": result.success,
                         "errors": result.errors, "ts": _now()}],
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _should_loop_to_sme(state: LineageState) -> str:
    open_gaps = [g for g in state.get("gaps", []) if not g.resolved]
    iteration = state.get("iteration", 0)
    if open_gaps and iteration < settings.max_sme_iterations:
        return "sme"
    return "finalize"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    workflow = StateGraph(LineageState)

    workflow.add_node("ingest_artifacts", ingest_artifacts)
    workflow.add_node("parse_artifacts", parse_artifacts)
    workflow.add_node("stitch_graph", stitch_graph_node)
    workflow.add_node("detect_gaps", detect_gaps_node)
    workflow.add_node("generate_sme_questions", generate_sme_questions_node)
    workflow.add_node("send_emails", send_emails_node)
    workflow.add_node("await_sme_responses", await_sme_responses_node)
    workflow.add_node("process_responses", process_responses_node)
    workflow.add_node("score_and_finalize", score_and_finalize_node)
    workflow.add_node("generate_visuals", generate_visuals_node)
    workflow.add_node("publish_to_catalog", publish_to_catalog_node)

    workflow.set_entry_point("ingest_artifacts")
    workflow.add_edge("ingest_artifacts", "parse_artifacts")
    workflow.add_edge("parse_artifacts", "stitch_graph")
    workflow.add_edge("stitch_graph", "detect_gaps")

    workflow.add_conditional_edges(
        "detect_gaps",
        _should_loop_to_sme,
        {"sme": "generate_sme_questions", "finalize": "score_and_finalize"},
    )

    workflow.add_edge("generate_sme_questions", "send_emails")
    workflow.add_edge("send_emails", "await_sme_responses")
    workflow.add_edge("await_sme_responses", "process_responses")
    workflow.add_edge("process_responses", "detect_gaps")  # skip re-stitch to preserve SME-added edges

    workflow.add_edge("score_and_finalize", "generate_visuals")
    workflow.add_edge("generate_visuals", END)
    # publish_to_catalog is intentionally NOT in the automatic workflow —
    # publishing is triggered explicitly via POST /runs/{id}/publish from the UI.

    return workflow.compile(checkpointer=checkpointer or MemorySaver())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
