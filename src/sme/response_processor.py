from __future__ import annotations

import json
import re
import uuid
from typing import Callable, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.models import (
    Gap, LineageGraph, LineageNode, LineageEdge,
    SystemType, ConfidenceLevel,
)
from src.graph.stewardship import external_path, make_external_node, producer_from_text
from src.llm_client import create_chat_model
from src.llm_usage import make_usage_event

_ADF_NS = "adf://"

# Storage URIs an SME may paste into a reply (ADLS Gen2 / Blob).
_STORAGE_URI_RE = re.compile(r"\b(?:abfss?|wasbs?)://[^\s,;'\"()<>]+", re.IGNORECASE)

# Free-text system names the LLM may return instead of the enum value.
_SYSTEM_ALIASES = {
    "adls": SystemType.ADLS, "adls gen2": SystemType.ADLS, "adlsgen2": SystemType.ADLS,
    "azure data lake": SystemType.ADLS, "azure data lake storage": SystemType.ADLS,
    "azure data lake storage gen2": SystemType.ADLS, "data lake": SystemType.ADLS,
    "blob": SystemType.ADLS, "azure blob storage": SystemType.ADLS,
    "databricks": SystemType.DATABRICKS, "azure databricks": SystemType.DATABRICKS,
    "dlt": SystemType.DLT, "delta live tables": SystemType.DLT,
    "adf": SystemType.ADF, "azure data factory": SystemType.ADF, "data factory": SystemType.ADF,
    "sql": SystemType.SQL, "azure sql": SystemType.SQL, "sql server": SystemType.SQL,
    "azure sql database": SystemType.SQL, "synapse": SystemType.SQL,
    "bi": SystemType.BI, "power bi": SystemType.BI, "powerbi": SystemType.BI,
    "ssrs": SystemType.BI, "tableau": SystemType.BI,
    "external": SystemType.EXTERNAL, "external system": SystemType.EXTERNAL,
}

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
    usage_callback: Optional[Callable[[dict], None]] = None,
) -> tuple[list[Gap], LineageGraph]:
    """
    Interpret SME email responses and update the lineage graph accordingly.

    Responses use deterministic handlers where possible and LLM extraction for
    gap types that need free-text interpretation.

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

        if gap.gap_type == "stewardship_required":
            if usage_callback:
                usage_callback(make_usage_event(
                    "sme_response_processing", "deterministic", "success", gap_id=gap.id,
                ))
            producer = _producer_from_answer(raw_answer)
            if producer and llm is not None and not producer_from_text(raw_answer):
                # No explicit "… by <who>" phrase — let the extractor name the producer.
                producer = llm_structured_value(
                    llm, gap, response, raw_answer, usage_callback=usage_callback,
                ) or producer
            if producer:
                _attest_external_producer(gap, producer, graph, sme_email, raw_answer)
            gap.resolved = bool(producer)
            continue

        if gap.gap_type == "conflicting_producers":
            # Matched deterministically against the known claims — no LLM needed.
            # Stays open (and is asked again) if the reply names no single producer.
            _handle_conflicting_producers(gap, {}, graph, sme_email, raw_answer)
            if usage_callback:
                usage_callback(make_usage_event(
                    "sme_response_processing", "deterministic", "success", gap_id=gap.id,
                ))
            gap.resolved = not any(c.target_id == gap.node_id for c in graph.open_conflicts())
            continue

        if llm is not None:
            question = _find_question_for_gap(gap_id, response)
            structured = _extract_structured_answer(
                llm, question, raw_answer, gap, usage_callback=usage_callback,
            )
            _apply_to_graph(gap, structured, graph, sme_email, raw_answer)
        elif usage_callback:
            usage_callback(make_usage_event(
                "sme_response_processing", "deterministic", "fallback",
                gap_id=gap.id, error_type="LLMUnavailable",
            ))

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
    llm: ChatOpenAI,
    question: str,
    raw_answer: str,
    gap: Gap,
    usage_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    prompt = (
        f"Gap type: {gap.gap_type}\n"
        f"Gap description: {gap.description}\n\n"
        f"Question asked:\n{question or '(question text not available)'}\n\n"
        f"SME reply:\n{raw_answer}"
    )
    response = None
    try:
        response = llm.invoke([
            SystemMessage(content=_EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
    except Exception as exc:
        if usage_callback:
            usage_callback(make_usage_event(
                "sme_response_processing", "llm", "failed", llm_called=True,
                gap_id=gap.id, error_type=type(exc).__name__,
            ))
        return {
            "answer_type": "no_information",
            "primary_value": None,
            "system_type": "unknown",
            "confirmed": None,
            "notes": raw_answer[:200],
        }

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
        structured = json.loads(content)
    except Exception as exc:
        if usage_callback:
            usage_callback(make_usage_event(
                "sme_response_processing", "llm", "fallback", response=response,
                llm_called=True, gap_id=gap.id, error_type=type(exc).__name__,
            ))
        # Fall back to a minimal structure so graph update can still proceed
        return {
            "answer_type": "no_information",
            "primary_value": None,
            "system_type": "unknown",
            "confirmed": None,
            "notes": raw_answer[:200],
        }
    if usage_callback:
        usage_callback(make_usage_event(
            "sme_response_processing", "llm", "success", response=response,
            llm_called=True, gap_id=gap.id,
        ))
    return structured


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
    target_node = graph.nodes.get(gap.node_id or "")
    if not target_node:
        return
    resolved = _resolve_endpoint(s, raw_answer, graph, exclude=target_node)
    if not resolved:
        return

    src_node = _find_or_create_node(*resolved, graph)
    _add_sme_edge(src_node, target_node, graph, sme_email,
                  transformation=f"SME confirmed: {raw_answer[:120]}")


def _handle_missing_target(
    gap: Gap, s: dict, graph: LineageGraph, sme_email: str, raw_answer: str,
) -> None:
    """SME provided the downstream target → add a new node + edge."""
    source_node = graph.nodes.get(gap.node_id or "")
    if not source_node:
        return
    resolved = _resolve_endpoint(s, raw_answer, graph, exclude=source_node)
    if not resolved:
        return

    tgt_node = _find_or_create_node(*resolved, graph)
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


_DONT_KNOW = re.compile(
    r"\b(don'?t know|do not know|no idea|not sure|unknown|unclear|n/?a|ask (?:someone|the)\b)",
    re.IGNORECASE,
)


def llm_structured_value(
    llm,
    gap: Gap,
    response: dict,
    raw_answer: str,
    usage_callback: Optional[Callable[[dict], None]] = None,
) -> Optional[str]:
    """Producer name via the LLM extractor, if it returns a usable value."""
    try:
        s = _extract_structured_answer(
            llm, _find_question_for_gap(gap.id, response), raw_answer, gap,
            usage_callback=usage_callback,
        )
    except Exception:
        return None
    if s.get("answer_type") == "no_information":
        return None
    val = (s.get("primary_value") or "").strip()
    return val or None


def _producer_from_answer(answer: str) -> Optional[str]:
    """
    Deterministic producer extraction from an attestation reply:
    "Uploaded manually via SFTP by Category Buying" → "Category Buying (manual SFTP upload)".
    Falls back to the first sentence. Returns None for "don't know" replies.
    """
    text = " ".join((answer or "").split())
    if not text or _DONT_KNOW.search(text):
        return None
    first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0].rstrip(".")
    who = producer_from_text(first)
    if not who:
        m = _BY_TEAM.search(first)
        who = m.group(1).strip() if m else None
    if who and who.lower() != first.lower():
        how = re.sub(r"\s*\bby\s+(?:the\s+)?" + re.escape(who) + r".*$", "", first,
                     flags=re.IGNORECASE).strip(" ,")
        how = how[:1].lower() + how[1:]
        return f"{who} ({how})" if how else who
    return first[:80]


# "… every Monday by Category Buying" — a capitalised team/system name after "by".
_BY_TEAM = re.compile(
    r"\bby\s+(?:the\s+)?([A-Z][\w&/\-]*(?:\s+[A-Z][\w&/\-]*)*"
    r"(?:\s+(?:team|department|dept|group|squad|function))?(?:\s*\([^)]*\))?)"
)


def _attest_external_producer(
    gap: Gap, producer: str, graph: LineageGraph, sme_email: str, raw_answer: str,
) -> None:
    """Record an SME-attested external producer → landing node edge (CONFIRMED)."""
    target = graph.nodes.get(gap.node_id or "")
    if target is None:
        return
    path = external_path(producer)
    ext = next((n for n in graph.nodes.values() if n.full_path == path), None)
    if ext is None:
        ext = make_external_node(producer, description=f"attested by {sme_email or 'SME'}")
        graph.nodes[ext.id] = ext
    graph.edges.append(LineageEdge(
        source_id=ext.id,
        target_id=target.id,
        transformation="attested external producer",
        system_boundary=f"{SystemType.EXTERNAL.value}→{target.system.value}",
        confidence=0.95,
        confidence_level=ConfidenceLevel.CONFIRMED,
        evidence="sme_approved attestation",
        sme_approved_by=sme_email or "sme",
        audit_trail=[{"action": "stewardship_attested", "by": sme_email, "note": raw_answer[:200]}],
    ))


def _match_claim(conflict, text: str):
    """
    Deterministically map an SME reply to one producer claim. Matches on the
    artifact filename/stem, the notebook path or its last segment, and finally
    on a distinguishing token like ``v2``. Returns the claim, "all", or None.
    """
    t = (text or "").lower()
    if re.search(r"\bboth\b|\ball of them\b", t) and not re.search(r"\bnot both\b", t):
        return "all"

    def keys(c):
        out = {c.artifact.lower(), re.sub(r"\.\w+$", "", c.artifact.lower())}
        if c.notebook_path:
            out |= {c.notebook_path.lower(), c.notebook_path.rstrip("/").split("/")[-1].lower()}
        return {k for k in out if k}

    # Tokens that identify exactly one claim (e.g. "v2" vs "v1").
    tokens = {c.artifact: set(re.split(r"[^a-z0-9]+", " ".join(keys(c)))) for c in conflict.claims}
    unique = {}
    for c in conflict.claims:
        others = set().union(*(v for k, v in tokens.items() if k != c.artifact))
        unique[c.artifact] = {tok for tok in tokens[c.artifact] - others if tok}

    def mentioned(clause: str) -> list:
        full = [c for c in conflict.claims if any(k in clause for k in keys(c))]
        if full:
            return full
        return [c for c in conflict.claims
                if any(re.search(rf"\b{re.escape(tok)}\b", clause) for tok in unique[c.artifact])]

    # Judge each clause separately: "v2 is live, v1 is deprecated" names both,
    # but only one positively.
    positive, negative = [], []
    for clause in re.split(r"[.;,\n]|\bbut\b|\band\b|\bwhile\b|\bwhereas\b", t):
        hits = mentioned(clause)
        if not hits:
            continue
        (negative if _NEGATIVE.search(clause) else positive).extend(hits)

    neg = {c.artifact for c in negative}
    pos = list({c.artifact: c for c in positive if c.artifact not in neg}.values())
    if len(pos) == 1:
        return pos[0]
    if not pos and neg:
        remaining = [c for c in conflict.claims if c.artifact not in neg]
        if len(remaining) == 1:
            return remaining[0]
    return None


_NEGATIVE = re.compile(
    r"\b(deprecated|obsolete|stale|old|legacy|retired|decommissioned|unused|dead|"
    r"not (?:live|used|in use|running|active)|no longer|disabled|replaced|superseded)\b"
)


def _handle_conflicting_producers(
    gap: Gap, s: dict, graph: LineageGraph, sme_email: str, raw_answer: str,
) -> None:
    """
    SME named the live producer. Drop lineage that only the stale producer(s)
    claim, keep edges the live one produces, and inherit its orchestrator.
    """
    conflict = next((c for c in graph.producer_conflicts if c.target_id == gap.node_id), None)
    if conflict is None or conflict.resolved_artifact:
        return
    choice = _match_claim(conflict, f"{s.get('primary_value') or ''} {raw_answer}")
    if choice is None:
        return  # ambiguous reply — leave the conflict open

    conflict.resolved_by = sme_email
    if choice == "all":
        conflict.resolved_artifact = "__all__"
        return

    conflict.resolved_artifact = choice.artifact
    claimants = {c.artifact for c in conflict.claims}
    kept = []
    for e in graph.edges:
        if e.target_id != conflict.target_id or not (set(e.produced_by) & claimants):
            kept.append(e)
            continue
        if choice.artifact not in e.produced_by:
            continue  # only the stale version claims this edge
        e.produced_by = [p for p in e.produced_by if p == choice.artifact or p not in claimants]
        e.orchestrated_by = choice.orchestrated_by
        e.audit_trail.append({
            "action": "producer_conflict_resolved",
            "by": sme_email,
            "live_producer": choice.artifact,
            "note": raw_answer[:200],
        })
        kept.append(e)
    graph.edges = kept


_HANDLERS = {
    "conflicting_producers":   _handle_conflicting_producers,
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
    key = " ".join((system_str or "").strip().lower().replace("_", " ").split())
    if not key:
        return SystemType.UNKNOWN
    try:
        return SystemType(key)
    except ValueError:
        return _SYSTEM_ALIASES.get(key, SystemType.UNKNOWN)


def _resolve_endpoint(
    s: dict, raw_answer: str, graph: LineageGraph, exclude: LineageNode,
) -> Optional[tuple[str, SystemType]]:
    """
    Decide the (path, system) of the node the SME named for a missing source/target.

    The LLM's extraction is used unless it is unusable, in which case a storage
    URI quoted verbatim in the reply wins. Unusable means: no value, a bare
    system name ("Azure Data Lake Storage"), or the very node the gap is about
    (an answer like "lands in ADLS ... loads it into bronze.returns_raw").
    """
    path = (s.get("primary_value") or "").strip().rstrip(".")
    system = _parse_system(s.get("system_type"))

    uri_match = _STORAGE_URI_RE.search(raw_answer or "")
    uri = uri_match.group(0).rstrip(".)") if uri_match else None

    is_uri = "://" in path
    is_system_name = _parse_system(path) != SystemType.UNKNOWN
    is_self = bool(path) and path.lower() in {
        exclude.full_path.lower(), exclude.name.lower(),
        exclude.full_path.lower().split(".", 1)[-1],
    }

    if uri and (not path or is_system_name or is_self
                or (not is_uri and system in (SystemType.ADLS, SystemType.UNKNOWN))):
        path = uri
    if not path or (is_self and path != uri):
        return None
    if _parse_system(path) != SystemType.UNKNOWN and "://" not in path:
        return None  # only a system name and no location to attach it to

    if path.lower().startswith(("abfss://", "abfs://", "wasbs://", "wasb://")):
        system = SystemType.ADLS
    return path, system


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


def _get_llm() -> ChatOpenAI:
    return create_chat_model(temperature=0)
