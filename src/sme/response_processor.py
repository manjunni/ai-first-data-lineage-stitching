from __future__ import annotations

import json
import uuid
from typing import Optional

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.models import (
    Gap, LineageGraph, LineageNode, LineageEdge,
    SystemType, ConfidenceLevel,
)
from config.settings import settings

_ADF_NS = "adf://"

_EXTRACT_SYSTEM_PROMPT = """\
You are a data lineage expert parsing a subject matter expert's reply to a lineage question.
Given the original question and the SME's answer, extract structured information.

Return ONLY valid JSON with these fields:
{
  "answer_type": "<one of: source_path | target_path | system_type | confirmation | correction | no_information>",
  "primary_value": "<the main extracted value, e.g. a path, table name, or system name — null if not applicable>",
  "system_type": "<one of: adls | databricks | sql | dlt | bi | unknown — the system the value belongs to>",
  "confirmed": <true | false | null — whether the SME confirmed an existing link>,
  "notes": "<any extra context from the reply — null if none>"
}

Rules:
- answer_type "source_path"  : SME provided the origin/source of data
- answer_type "target_path"  : SME provided where the data goes downstream
- answer_type "system_type"  : SME clarified which platform a node belongs to
- answer_type "confirmation" : SME confirmed (or denied) an existing edge
- answer_type "correction"   : SME provided a different relationship than documented
- answer_type "no_information": SME does not know or the reply is non-informative

For primary_value, extract the most specific value possible:
  - ADLS: full abfss:// path or container/folder/file
  - Databricks: catalog.schema.table or schema.table
  - SQL: schema.table or database.schema.table
  - System type only: just the system name ("databricks", "adls", etc.)"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_responses_to_graph(
    responses: list[dict],
    gaps: list[Gap],
    graph: LineageGraph,
) -> tuple[list[Gap], LineageGraph]:
    """
    Interpret SME email responses and update the lineage graph accordingly.

    For each response that matches a gap by gap_id:
    1. Use LLM to extract a structured answer from the free-text reply.
    2. Apply the answer to the graph (update nodes/edges).
    3. Mark the gap as resolved.

    Returns (updated_gaps, updated_graph).
    """
    gap_index = {g.id: g for g in gaps}

    # Initialise LLM once — gracefully degrade if credentials are unavailable
    try:
        llm = _get_llm()
    except Exception:
        llm = None

    for response in responses:
        gap_id = response.get("gap_id")
        if not gap_id or gap_id not in gap_index:
            continue

        gap = gap_index[gap_id]
        raw_answer = response.get("raw_answer", "")
        sme_email = response.get("sme_email", "")

        if llm is not None:
            question = _find_question_for_gap(gap_id, response)
            structured = _extract_structured_answer(llm, question, raw_answer, gap)
            _apply_to_graph(gap, structured, graph, sme_email, raw_answer)

        # Mark the gap resolved regardless of whether the LLM was available —
        # the SME has provided an answer and the loop should not ask again.
        gap.resolved = True

        # Credit the SME on the node/edge they answered about.
        # The resolving responder should always be the owner/approver recorded
        # for the fixed gap, even when the node or edge had a previous default
        # value from a fallback UI identity.
        if sme_email:
            if gap.node_id:
                node = graph.nodes.get(gap.node_id)
                if node:
                    node.owner = sme_email
            if gap.edge_id:
                edge = next((e for e in graph.edges if e.id == gap.edge_id), None)
                if edge:
                    edge.sme_approved_by = sme_email

    return list(gap_index.values()), graph


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

def _extract_structured_answer(
    llm: AzureChatOpenAI,
    question: str,
    raw_answer: str,
    gap: Gap,
) -> dict:
    prompt = (
        f"Gap type: {gap.gap_type}\n"
        f"Gap description: {gap.description}\n\n"
        f"Question asked:\n{question or '(question text not available)'}\n\n"
        f"SME reply:\n{raw_answer}"
    )
    try:
        response = llm.invoke([
            SystemMessage(content=_EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        return json.loads(response.content)
    except Exception:
        # Fall back to a minimal structure so graph update can still proceed
        return {
            "answer_type": "no_information",
            "primary_value": None,
            "system_type": "unknown",
            "confirmed": None,
            "notes": raw_answer[:200],
        }


# ---------------------------------------------------------------------------
# Graph update — one handler per gap type
# ---------------------------------------------------------------------------

def _apply_to_graph(
    gap: Gap,
    structured: dict,
    graph: LineageGraph,
    sme_email: str,
    raw_answer: str,
) -> None:
    answer_type = structured.get("answer_type", "no_information")
    if answer_type == "no_information":
        return

    handler = _HANDLERS.get(gap.gap_type)
    if handler:
        handler(gap, structured, graph, sme_email, raw_answer)


def _handle_missing_source(
    gap: Gap, s: dict, graph: LineageGraph, sme_email: str, raw_answer: str,
) -> None:
    """SME provided the source → add a new node + edge to the graph."""
    path = s.get("primary_value")
    system = _parse_system(s.get("system_type"))
    target_node = graph.nodes.get(gap.node_id or "")
    if not target_node or not path:
        return

    src_node = _find_or_create_node(path, system, graph)
    _add_sme_edge(src_node, target_node, graph, sme_email,
                  transformation=f"SME confirmed: {raw_answer[:120]}")


def _handle_missing_target(
    gap: Gap, s: dict, graph: LineageGraph, sme_email: str, raw_answer: str,
) -> None:
    """SME provided the downstream target → add a new node + edge."""
    path = s.get("primary_value")
    system = _parse_system(s.get("system_type"))
    source_node = graph.nodes.get(gap.node_id or "")
    if not source_node or not path:
        return

    tgt_node = _find_or_create_node(path, system, graph)
    _add_sme_edge(source_node, tgt_node, graph, sme_email,
                  transformation=f"SME confirmed: {raw_answer[:120]}")


def _handle_unknown_system(
    gap: Gap, s: dict, graph: LineageGraph, sme_email: str, raw_answer: str,
) -> None:
    """SME clarified which system a node belongs to — update in place."""
    node = graph.nodes.get(gap.node_id or "")
    if not node:
        return
    system = _parse_system(s.get("system_type"))
    if system != SystemType.UNKNOWN:
        node.system = system
    if s.get("primary_value"):
        node.description = s["primary_value"]
        node.owner = sme_email


def _handle_unresolved_dataset_ref(
    gap: Gap, s: dict, graph: LineageGraph, sme_email: str, raw_answer: str,
) -> None:
    """SME provided the actual storage path for an adf:// placeholder."""
    node = graph.nodes.get(gap.node_id or "")
    if not node:
        return
    if s.get("primary_value"):
        node.description = s["primary_value"]
    system = _parse_system(s.get("system_type"))
    if system != SystemType.UNKNOWN:
        node.system = system
    node.owner = sme_email


def _handle_low_confidence_edge(
    gap: Gap, s: dict, graph: LineageGraph, sme_email: str, raw_answer: str,
) -> None:
    """SME confirmed (or denied) a low-confidence edge."""
    edge = next((e for e in graph.edges if e.id == gap.edge_id), None)
    if not edge:
        return

    confirmed = s.get("confirmed")
    if confirmed is True:
        edge.sme_approved_by = sme_email
        edge.confidence = 0.95
        edge.confidence_level = ConfidenceLevel.CONFIRMED
        edge.audit_trail.append({
            "action": "sme_confirmed",
            "by": sme_email,
            "note": raw_answer[:200],
        })
    elif confirmed is False:
        # SME denied — mark edge as low confidence with audit note
        edge.confidence = 0.10
        edge.confidence_level = ConfidenceLevel.LOW
        edge.audit_trail.append({
            "action": "sme_denied",
            "by": sme_email,
            "note": raw_answer[:200],
        })
        # If SME provided a correction, add it as a note on the edge
        if s.get("notes"):
            edge.transformation = f"CORRECTION: {s['notes'][:120]}"


def _handle_isolated_subgraph(
    gap: Gap, s: dict, graph: LineageGraph, sme_email: str, raw_answer: str,
) -> None:
    """SME provided the root data source for an isolated component."""
    path = s.get("primary_value")
    system = _parse_system(s.get("system_type"))
    if not path:
        return

    src_node = _find_or_create_node(path, system, graph)
    # Connect source to all orphan nodes in the gap description
    for node_id, node in graph.nodes.items():
        if node_id == src_node.id:
            continue
        from src.graph.stitcher import to_networkx
        G = to_networkx(graph)
        if G.in_degree(node_id) == 0 and node.system not in (SystemType.ADLS,):
            _add_sme_edge(src_node, node, graph, sme_email,
                          transformation=f"SME confirmed origin: {raw_answer[:80]}")


def _handle_contradictory_paths(
    gap: Gap, s: dict, graph: LineageGraph, sme_email: str, raw_answer: str,
) -> None:
    """SME picked the correct transformation — update the surviving edge."""
    edge = next((e for e in graph.edges if e.id == gap.edge_id), None)
    if not edge:
        return
    if s.get("primary_value"):
        edge.transformation = s["primary_value"]
    edge.sme_approved_by = sme_email
    edge.confidence = 0.95
    edge.confidence_level = ConfidenceLevel.CONFIRMED
    edge.audit_trail.append({
        "action": "contradiction_resolved",
        "by": sme_email,
        "note": raw_answer[:200],
    })


_HANDLERS = {
    "missing_source":          _handle_missing_source,
    "missing_target":          _handle_missing_target,
    "unknown_system":          _handle_unknown_system,
    "unresolved_dataset_ref":  _handle_unresolved_dataset_ref,
    "low_confidence_edge":     _handle_low_confidence_edge,
    "isolated_subgraph":       _handle_isolated_subgraph,
    "contradictory_paths":     _handle_contradictory_paths,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_system(system_str: Optional[str]) -> SystemType:
    try:
        return SystemType(system_str or "unknown")
    except ValueError:
        return SystemType.UNKNOWN


def _find_or_create_node(path: str, system: SystemType, graph: LineageGraph) -> LineageNode:
    for node in graph.nodes.values():
        if node.full_path == path:
            return node
    node = LineageNode(
        id=str(uuid.uuid4()),
        name=path.split("/")[-1].split(".")[-1],
        system=system,
        full_path=path,
    )
    graph.nodes[node.id] = node
    return node


def _add_sme_edge(
    src: LineageNode,
    tgt: LineageNode,
    graph: LineageGraph,
    sme_email: str,
    transformation: str,
) -> None:
    # Avoid creating duplicate edges
    existing = next(
        (e for e in graph.edges if e.source_id == src.id and e.target_id == tgt.id),
        None,
    )
    if existing:
        existing.sme_approved_by = sme_email
        existing.confidence = 0.95
        existing.confidence_level = ConfidenceLevel.CONFIRMED
        return

    graph.edges.append(LineageEdge(
        id=str(uuid.uuid4()),
        source_id=src.id,
        target_id=tgt.id,
        transformation=transformation,
        confidence=0.95,
        confidence_level=ConfidenceLevel.CONFIRMED,
        sme_approved_by=sme_email,
        evidence=f"SME-provided: {sme_email}",
        audit_trail=[{"action": "sme_added", "by": sme_email}],
    ))


def _find_question_for_gap(gap_id: str, response: dict) -> Optional[str]:
    """Best-effort: the response dict may carry the original question text."""
    return response.get("question")


def _get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        azure_deployment=settings.azure_openai_deployment,
        api_version="2024-08-01-preview",
        temperature=0,
    )
